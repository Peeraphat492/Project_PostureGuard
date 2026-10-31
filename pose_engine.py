"""MediaPipe Pose wrapper with landmark gating, outlier rejection, One Euro smoothing and a
posture-only skeleton renderer.

Frames are always processed un-mirrored (as the camera delivers them) so MediaPipe's anatomical
left/right labels are correct; mirroring is a display concern handled by the caller.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Optional, Tuple

import cv2
import mediapipe as mp

from config import CONFIG


@dataclass(frozen=True)
class LandmarkPoint:
    x: float                 # normalized 0..1 (image width)
    y: float                 # normalized 0..1 (image height)
    z: float
    visibility: float        # combined confidence (min of visibility and presence when available)
    width: float = 1.0       # frame size, so geometry can be done in real pixel space
    height: float = 1.0

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def pxy(self) -> Tuple[float, float]:
        """Pixel coordinates: angles and ratios must use these, not normalized x/y (aspect ratio)."""
        return (self.x * self.width, self.y * self.height)


Landmarks = Dict[str, LandmarkPoint]

_PL = mp.solutions.pose.PoseLandmark
TRACKED = {
    "nose": _PL.NOSE,
    "left_eye": _PL.LEFT_EYE,
    "right_eye": _PL.RIGHT_EYE,
    "left_ear": _PL.LEFT_EAR,
    "right_ear": _PL.RIGHT_EAR,
    "left_shoulder": _PL.LEFT_SHOULDER,
    "right_shoulder": _PL.RIGHT_SHOULDER,
    "left_hip": _PL.LEFT_HIP,
    "right_hip": _PL.RIGHT_HIP,
    # Arms are tracked only for the Active Break game; they are never drawn on the posture skeleton.
    "left_elbow": _PL.LEFT_ELBOW,
    "right_elbow": _PL.RIGHT_ELBOW,
    "left_wrist": _PL.LEFT_WRIST,
    "right_wrist": _PL.RIGHT_WRIST,
    # Hand points (index / pinky knuckles) locate the hand itself for touching game targets.
    "left_index": _PL.LEFT_INDEX,
    "right_index": _PL.RIGHT_INDEX,
    "left_pinky": _PL.LEFT_PINKY,
    "right_pinky": _PL.RIGHT_PINKY,
}


def side_overrides() -> Dict[str, float]:
    """Per-landmark confidence thresholds for the side camera."""
    return {"left_hip": CONFIG.side_hip_visibility_threshold, "right_hip": CONFIG.side_hip_visibility_threshold}


def mirror_landmarks(lms: Optional[Landmarks]) -> Optional[Landmarks]:
    """Landmarks of a horizontally flipped frame (names stay anatomical)."""
    if lms is None:
        return None
    return {name: replace(p, x=1.0 - p.x) for name, p in lms.items()}


# ---------------------------------------------------------------------------
# One Euro filter (Casiez et al. 2012): low jitter when still, low lag when moving
# ---------------------------------------------------------------------------

class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev = 0.0
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.x_prev is None or self.t_prev is None or t <= self.t_prev:
            self.x_prev, self.t_prev, self.dx_prev = x, t, 0.0
            return x
        dt = t - self.t_prev
        dx = (x - self.x_prev) / dt
        dx_hat = self.dx_prev + self._alpha(self.d_cutoff, dt) * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self.x_prev + self._alpha(cutoff, dt) * (x - self.x_prev)
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


@dataclass
class _Track:
    fx: OneEuroFilter
    fy: OneEuroFilter
    last: Optional[Tuple[float, float]] = None   # last accepted (filtered) pixel position
    visible: bool = False                         # gating state with hysteresis
    outliers: int = 0                             # consecutive rejected jumps


class LandmarkStabilizer:
    """Turns raw per-frame landmarks into a dict of *reliable*, smoothed landmarks.

    A landmark is dropped (not drawn, not measured) when its confidence is low, it lies outside
    the frame, or it jumps implausibly far in one frame. Smoothing runs in pixel space.
    """

    def __init__(self, visibility_threshold: float, overrides: Optional[Dict[str, float]] = None):
        self.threshold = visibility_threshold
        self.overrides = dict(overrides or {})  # per-landmark thresholds, e.g. partly hidden seated hips
        self.tracks: Dict[str, _Track] = {}

    def reset(self):
        self.tracks.clear()

    def _track(self, name: str) -> _Track:
        if name not in self.tracks:
            make = lambda: OneEuroFilter(CONFIG.landmark_filter_min_cutoff, CONFIG.landmark_filter_beta,
                                         CONFIG.landmark_filter_d_cutoff)
            self.tracks[name] = _Track(make(), make())
        return self.tracks[name]

    def update(self, raw: Optional[Landmarks], t: float) -> Optional[Landmarks]:
        if not raw:
            self.reset()
            return None
        out: Landmarks = {}
        margin = CONFIG.landmark_edge_margin
        for name, p in raw.items():
            track = self._track(name)
            in_frame = -margin <= p.x <= 1 + margin and -margin <= p.y <= 1 + margin
            threshold = self.overrides.get(name, self.threshold)
            gate = threshold if not track.visible else threshold - CONFIG.landmark_visibility_hysteresis
            track.visible = in_frame and p.visibility >= gate
            if not track.visible:
                del self.tracks[name]  # forget history so it restarts cleanly when it reappears
                continue

            px, py = p.x * p.width, p.y * p.height
            if track.last is not None:
                jump = math.hypot(px - track.last[0], py - track.last[1])
                if jump > CONFIG.landmark_max_jump_ratio * math.hypot(p.width, p.height):
                    track.outliers += 1
                    if track.outliers <= CONFIG.landmark_outlier_max_frames:
                        continue  # implausible jump: hide this landmark for now
                    track.fx.x_prev = track.fy.x_prev = None  # it persisted: accept the new position
                track.outliers = 0
            sx, sy = track.fx(px, t), track.fy(py, t)
            track.last = (sx, sy)
            out[name] = replace(p, x=sx / p.width, y=sy / p.height)
        return out


# ---------------------------------------------------------------------------
# Posture-only skeleton
# ---------------------------------------------------------------------------

LINE_BGR = (212, 182, 6)        # cyan #06B6D4
JOINT_BGR = (255, 255, 255)
BAD_BGR = (68, 68, 239)         # red  #EF4444

FRONT_LINES = (
    ("left_ear", "left_eye"), ("left_eye", "nose"), ("nose", "right_eye"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
)
FRONT_JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_hip", "right_hip")


def _mid(a: LandmarkPoint, b: LandmarkPoint) -> LandmarkPoint:
    return replace(a, x=(a.x + b.x) / 2, y=(a.y + b.y) / 2)


class PoseEngine:
    def __init__(self, visibility_threshold: Optional[float] = None, overrides: Optional[Dict[str, float]] = None):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=CONFIG.pose_model_complexity,
            smooth_landmarks=CONFIG.mediapipe_smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=CONFIG.min_detection_confidence,
            min_tracking_confidence=CONFIG.min_tracking_confidence,
        )
        threshold = CONFIG.landmark_visibility_threshold if visibility_threshold is None else visibility_threshold
        self.stabilizer = LandmarkStabilizer(threshold, overrides)

    def close(self):
        self.pose.close()

    def reset(self):
        self.stabilizer.reset()

    @staticmethod
    def _raw_landmarks(result, width: int, height: int) -> Optional[Landmarks]:
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks.landmark
        out: Landmarks = {}
        for name, idx in TRACKED.items():
            p = lm[idx.value]
            confidence = p.visibility
            if p.HasField("presence"):
                confidence = min(confidence, p.presence)
            out[name] = LandmarkPoint(p.x, p.y, p.z, confidence, float(width), float(height))
        return out

    def process(self, frame_bgr, t: Optional[float] = None):
        """Run pose on an un-mirrored BGR frame. Returns (raw MediaPipe result, reliable smoothed landmarks)."""
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.pose.process(rgb)
        raw = self._raw_landmarks(result, width, height)
        return result, self.stabilizer.update(raw, time.monotonic() if t is None else t)

    # ---- drawing (on the same un-mirrored frame the landmarks came from)

    @staticmethod
    def _scale(frame) -> Tuple[int, int]:
        width = frame.shape[1]
        return max(2, round(width / 400)), max(3, round(width / 260))

    @staticmethod
    def _px(p: LandmarkPoint) -> Tuple[int, int]:
        return int(round(p.x * p.width)), int(round(p.y * p.height))

    def _lines(self, frame, lms: Landmarks, pairs: Iterable[Tuple[str, str]], color, thickness):
        for a, b in pairs:
            if a in lms and b in lms:   # never connect a landmark that is hidden/unreliable
                cv2.line(frame, self._px(lms[a]), self._px(lms[b]), color, thickness, cv2.LINE_AA)

    def _joints(self, frame, lms: Landmarks, names: Iterable[str], radius: int):
        for name in names:
            if name in lms:
                c = self._px(lms[name])
                cv2.circle(frame, c, radius + 2, LINE_BGR, -1, cv2.LINE_AA)
                cv2.circle(frame, c, radius, JOINT_BGR, -1, cv2.LINE_AA)

    def draw_front(self, frame, lms: Optional[Landmarks], bad_regions=()):
        if not lms:
            return frame
        line, radius = self._scale(frame)
        pts = dict(lms)
        if "left_shoulder" in pts and "right_shoulder" in pts:
            pts["neck"] = _mid(pts["left_shoulder"], pts["right_shoulder"])
        if "left_hip" in pts and "right_hip" in pts:
            pts["pelvis"] = _mid(pts["left_hip"], pts["right_hip"])
        center = (("neck", "nose"), ("neck", "pelvis"))
        self._lines(frame, pts, FRONT_LINES + center, LINE_BGR, line)
        if "shoulder" in bad_regions:
            self._lines(frame, pts, (("left_shoulder", "right_shoulder"),), BAD_BGR, line + 2)
        if "back" in bad_regions:
            self._lines(frame, pts, center, BAD_BGR, line + 2)
        self._joints(frame, pts, FRONT_JOINTS, radius)
        return frame

    def draw_side(self, frame, lms: Optional[Landmarks], side: Optional[str], bad_regions=()):
        """Draw only the camera-facing body side so near/far landmarks are never cross-connected."""
        if not lms or side not in ("left", "right"):
            return frame
        line, radius = self._scale(frame)
        ear, shoulder, hip = f"{side}_ear", f"{side}_shoulder", f"{side}_hip"
        neck, torso = (ear, shoulder), (shoulder, hip)
        if hip in lms:  # thin vertical reference through the hip to visualise torso lean
            hx, hy = self._px(lms[hip])
            top = int(hy - 0.35 * frame.shape[0])
            cv2.line(frame, (hx, hy), (hx, max(0, top)), (200, 200, 200), 1, cv2.LINE_AA)
        self._lines(frame, lms, (("nose", ear), neck, torso), LINE_BGR, line)
        if "neck" in bad_regions:
            self._lines(frame, lms, (neck,), BAD_BGR, line + 2)
        if "back" in bad_regions:
            self._lines(frame, lms, (torso,), BAD_BGR, line + 2)
        self._joints(frame, lms, ("nose", ear, shoulder, hip), radius)
        return frame
