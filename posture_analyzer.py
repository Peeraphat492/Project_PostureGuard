"""Automatic posture analysis against ergonomic reference criteria (no personal calibration).

Each camera produces a FeatureSet of geometric posture indicators measured in image pixel space.
Indicators are compared with configurable reference limits (config.py, documented in README):
the user's current or first posture is never taken as "correct".

Front camera (un-mirrored image, signs are image directions):
  shoulder_tilt_deg      shoulder line vs horizontal; > 0 when the image-left shoulder is lower
  head_tilt_deg          eye line (or ear line) vs horizontal; > 0 when the image-left side is lower
  side_lean_deg          hip-centre -> shoulder-centre vs vertical (only when both hips are visible)
  neck_lateral_deg       shoulder-centre -> head-centre vs vertical ("upper-body alignment")
Side camera (signed along the direction the person faces):
  forward_head_deg       angle between the neck line (shoulder -> ear) and the trunk axis (hip -> shoulder);
                         > 0 when the head sits in front of the trunk line. Measured RELATIVE to the trunk, so
                         leaning or reclining the whole body and a slightly rolled camera do not change it.
  head_flexion_deg       head pitch: ear -> eye line below horizontal (approximates the Frankfort plane, which
                         is level when the head is upright), minus head_pitch_neutral_deg. Separate from forward
                         head: it measures where the head points, not where it sits.
  torso_lean_deg         hip -> shoulder vs vertical; > 0 forward, < 0 reclined
Ear, eye, shoulder and hip are always taken from the same basis -- the body midline when both sides are
visible, otherwise the same (camera-facing) side -- so a slightly oblique camera does not mix a lateral
offset into the forward direction. Not a clinical craniovertebral angle (no C7 landmark is available).

A single 2-D camera view cannot measure spinal curvature; these are surface-landmark proxies and the
resulting score is an application-level ergonomic indicator, not a medical assessment.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import CONFIG
from geometry import clamp, distance, midpoint, signed_lean_deg, signed_tilt_deg
from pose_engine import LandmarkPoint

Landmarks = Dict[str, LandmarkPoint]
Point = Tuple[float, float]


@dataclass
class FeatureSet:
    valid: bool = False
    values: Dict[str, float] = field(default_factory=dict)
    issue: str = ""     # camera guidance: why it is invalid, or a soft placement warning when valid
    held: bool = False  # last reliable values re-used through a brief drop-out


@dataclass(frozen=True)
class Metric:
    name: str           # unique, human readable
    key: str            # feature key
    camera: str         # "front" | "side"
    label: str          # posture label it contributes to (also the session duration key)
    component: str      # score component: Neck | Back | Shoulder | Alignment
    limit_attr: str     # CONFIG attribute holding the reference limit ("" = limit of 1.0)
    mode: str           # "abs" (either direction), "positive" or "negative" (one-sided)
    reason: str


METRICS = (
    Metric("Shoulder tilt", "shoulder_tilt_deg", "front", "Uneven Shoulder", "Shoulder",
           "shoulder_tilt_max_deg", "abs", "Shoulders are not level"),
    Metric("Head tilt", "head_tilt_deg", "front", "Leaning Left/Right", "Alignment",
           "head_tilt_max_deg", "abs", "Head is tilted sideways"),
    Metric("Side lean", "side_lean_deg", "front", "Leaning Left/Right", "Alignment",
           "side_lean_max_deg", "abs", "Torso is leaning sideways"),
    Metric("Upper-body alignment", "neck_lateral_deg", "front", "Leaning Left/Right", "Alignment",
           "upper_body_alignment_max_deg", "abs", "Head is not centred over the shoulders"),
    Metric("Forward head", "forward_head_deg", "side", "Forward Head", "Neck",
           "forward_head_max_deg", "positive", "Head is forward of the trunk line"),
    Metric("Neck flexion", "head_flexion_deg", "side", "Forward Head", "Neck",
           "neck_flexion_max_deg", "positive", "Head is tipped down"),
    Metric("Trunk forward lean", "torso_lean_deg", "side", "Slouching", "Back",
           "torso_forward_max_deg", "positive", "Trunk is leaning forward"),
    Metric("Trunk recline", "torso_lean_deg", "side", "Slouching", "Back",
           "torso_recline_max_deg", "negative", "Reclined beyond a supported position"),
    Metric("Hunched posture", "slouch_index", "side", "Slouching", "Back",
           "", "positive", "Head and trunk are both drifting forward"),
)
LABEL_ORDER = ("Forward Head", "Slouching", "Uneven Shoulder", "Leaning Left/Right")
COMPONENTS = ("Neck", "Back", "Shoulder", "Alignment")
CORE_FRONT = ("shoulder_tilt_deg", "neck_lateral_deg")
CORE_SIDE = ("forward_head_deg", "torso_lean_deg")


def metric_limit(metric: Metric) -> float:
    return getattr(CONFIG, metric.limit_attr) if metric.limit_attr else 1.0


def deviation_ratio(metric: Metric, value: float) -> float:
    """Value as a multiple of its reference limit (0 = neutral, 1 = at the limit)."""
    if metric.mode == "positive":
        value = max(0.0, value)
    elif metric.mode == "negative":
        value = max(0.0, -value)
    return abs(value) / max(metric_limit(metric), 1e-9)


def indicator_score(ratio: float) -> float:
    """Transparent piecewise-linear score: 100 at neutral, 80 at the limit, 60 at 1.5x, 0 at 3x the limit."""
    if ratio <= 1.0:
        return 100.0 - 20.0 * ratio
    return clamp(80.0 - 40.0 * (ratio - 1.0), 0.0, 100.0)


def score_band(risk: str) -> Tuple[float, float]:
    """Score range that matches a classification, derived from the indicator score mapping so that number,
    colour and status always agree: GOOD >= score at the limit (80), POOR < score at metric_poor_ratio (60)."""
    good_min = indicator_score(1.0)
    poor_below = indicator_score(CONFIG.metric_poor_ratio)
    return {"GOOD": (good_min, 100.0), "WARNING": (poor_below, good_min - 1), "POOR": (0.0, poor_below - 1)}[risk]


@dataclass
class PostureResult:
    labels: List[str]
    status: str
    risk_level: str                       # GOOD | WARNING | POOR | N/A (insufficient data)
    score: Optional[float]                # None when reliable scoring is not possible
    components: Dict[str, Optional[float]]
    front_features: Dict[str, float]
    side_features: Dict[str, float]
    bad_regions_front: List[str]
    bad_regions_side: List[str]
    reasons: List[str]
    # name -> (value, limit, deviation ratio, flagged)
    metrics: Dict[str, Tuple[float, float, float, bool]] = field(default_factory=dict)
    raw_score: Optional[float] = None  # weighted indicator mean before it is held inside the status band
    guidance: List[str] = field(default_factory=list)  # camera placement hints
    partial: bool = False  # only one camera gave reliable data; the other camera's indicators are missing


class EMA:
    """Exponential moving average per feature that forgets stale values after the pose was lost."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.data: Dict[str, float] = {}
        self.missing = 0

    def miss(self):
        self.missing += 1

    def update(self, values: Dict[str, float]) -> Dict[str, float]:
        if self.missing > CONFIG.feature_reset_after_missing_frames:
            self.data.clear()
        self.missing = 0
        # Optional features (e.g. hips) that disappeared must not come back with old values.
        self.data = {k: v for k, v in self.data.items() if k in values}
        out = {}
        for k, v in values.items():
            self.data[k] = v if k not in self.data else self.alpha * v + (1 - self.alpha) * self.data[k]
            out[k] = self.data[k]
        return out


class PostureAnalyzer:
    def __init__(self):
        self.front_ema = EMA(CONFIG.ema_alpha)
        self.side_ema = EMA(CONFIG.ema_alpha)
        self.side_choice: Optional[str] = None  # body side the side camera is tracking
        self.facing: Optional[float] = None     # +1 facing image-right, -1 facing image-left
        self._last_valid: Dict[str, Tuple[float, FeatureSet]] = {}
        self._flags: Dict[str, List] = {}       # metric name -> [active, pending_since]
        self._severe: Dict[str, List] = {}      # metric name -> [active, pending_since] for severe deviations

    def reset_smoothing(self):
        """Forget smoothing, held readings and warning state (e.g. after an Active Break)."""
        self.front_ema = EMA(CONFIG.ema_alpha)
        self.side_ema = EMA(CONFIG.ema_alpha)
        self._last_valid.clear()
        self._flags.clear()
        self._severe.clear()

    @staticmethod
    def _present(lms: Landmarks, *names: str) -> bool:
        # Landmarks normally arrive already gated by LandmarkStabilizer; this is a safety net for raw input.
        def floor(name: str) -> float:
            base = CONFIG.side_hip_visibility_threshold if name.endswith("_hip") else \
                min(CONFIG.landmark_visibility_threshold, CONFIG.side_landmark_visibility_threshold)
            return base - CONFIG.landmark_visibility_hysteresis
        return all(name in lms and lms[name].visibility >= floor(name) for name in names)

    def _hold(self, camera: str, fs: FeatureSet, now: float) -> FeatureSet:
        """Bridge brief drop-outs with the last reliable reading instead of flickering to 'unreliable'."""
        if fs.valid:
            self._last_valid[camera] = (now, fs)
            return fs
        last = self._last_valid.get(camera)
        if last is not None and now - last[0] <= CONFIG.measurement_hold_seconds:
            return FeatureSet(True, last[1].values, last[1].issue, held=True)
        return fs

    # ------------------------------------------------------------------ front camera

    def extract_front(self, lms: Optional[Landmarks], now: Optional[float] = None) -> FeatureSet:
        now = time.monotonic() if now is None else now
        fs = self._front(lms)
        if not fs.valid:
            self.front_ema.miss()
        return self._hold("front", fs, now)

    def _front(self, lms: Optional[Landmarks]) -> FeatureSet:
        if not lms:
            return FeatureSet(issue="Front camera: no person detected — sit facing the camera")
        if not self._present(lms, "nose", "left_shoulder", "right_shoulder"):
            return FeatureSet(issue="Front camera: keep your head and both shoulders in view")
        ls, rs, nose = lms["left_shoulder"].pxy, lms["right_shoulder"].pxy, lms["nose"].pxy
        width = lms["nose"].width
        mid = midpoint(ls, rs)
        shoulder_width = distance(ls, rs)
        tilt = signed_tilt_deg(ls, rs)
        if abs(tilt) > 45 or nose[1] >= mid[1]:
            return FeatureSet(issue="Front camera looks rotated — mount it upright and level")
        if shoulder_width < CONFIG.front_min_shoulder_width_ratio * width:
            return FeatureSet(issue="Front camera: move closer (upper body too small)")

        head_center, head_frontal = nose, True
        if self._present(lms, "left_ear", "right_ear"):
            le, re = lms["left_ear"].pxy, lms["right_ear"].pxy
            ear_dist = max(distance(le, re), 1e-6)
            head_center = midpoint(le, re)  # robust to head turning, unlike the nose
            head_frontal = abs(nose[0] - head_center[0]) / ear_dist <= CONFIG.front_max_head_yaw_ratio
            if shoulder_width / ear_dist < CONFIG.front_min_shoulder_head_ratio:
                return FeatureSet(issue="Front camera: turn your body to face the camera")

        values = {
            "shoulder_tilt_deg": tilt,
            "neck_lateral_deg": signed_lean_deg(mid, head_center, 1.0),
        }
        if head_frontal:  # roll of a turned head is not measurable from the front
            if self._present(lms, "left_eye", "right_eye"):
                values["head_tilt_deg"] = signed_tilt_deg(lms["left_eye"].pxy, lms["right_eye"].pxy)
            elif self._present(lms, "left_ear", "right_ear"):
                values["head_tilt_deg"] = signed_tilt_deg(lms["left_ear"].pxy, lms["right_ear"].pxy)
        if self._present(lms, "left_hip", "right_hip"):
            hip_mid = midpoint(lms["left_hip"].pxy, lms["right_hip"].pxy)
            if hip_mid[1] - mid[1] > 0.5 * shoulder_width:  # hips really below the shoulders
                values["side_lean_deg"] = signed_lean_deg(hip_mid, mid, 1.0)
        issue = "" if head_frontal else "Front camera: face the camera to measure head tilt"
        return FeatureSet(True, self.front_ema.update(values), issue)

    # ------------------------------------------------------------------ side camera

    def _side_points(self, lms: Landmarks, prefer: str) -> Dict[str, Optional[Point]]:
        """Ear/eye/shoulder/hip from ONE consistent basis. When ear, shoulder and hip are visible on both
        sides (camera somewhat oblique) all points are body-midline midpoints; otherwise all come from the
        tracked (camera-facing) side, falling back per joint to the other side, which in a true side view
        projects almost on top of it. Mixing a lateral ear with midline shoulders would turn the ear's
        sideways offset into a fake forward displacement on an oblique camera."""
        other = "right" if prefer == "left" else "left"
        midline = all(self._present(lms, f"left_{j}", f"right_{j}") for j in ("ear", "shoulder", "hip"))
        points: Dict[str, Optional[Point]] = {}
        for joint in ("ear", "eye", "shoulder", "hip"):
            a, b = f"{prefer}_{joint}", f"{other}_{joint}"
            if midline and self._present(lms, a, b):
                points[joint] = midpoint(lms[a].pxy, lms[b].pxy)
            else:
                points[joint] = next((lms[n].pxy for n in (a, b) if self._present(lms, n)), None)
        return points

    def _side_score(self, lms: Landmarks, side: str) -> float:
        names = (f"{side}_ear", f"{side}_shoulder", f"{side}_hip")
        return sum(lms[n].visibility for n in names if n in lms) / len(names)

    def extract_side(self, lms: Optional[Landmarks], now: Optional[float] = None) -> FeatureSet:
        now = time.monotonic() if now is None else now
        fs = self._side(lms)
        if not fs.valid:
            self.side_ema.miss()
        return self._hold("side", fs, now)

    def _side(self, lms: Optional[Landmarks]) -> FeatureSet:
        if not lms:
            return FeatureSet(issue="Side camera: no person detected — place it at your side")

        # Track the camera-facing side with hysteresis so the reading does not jump between sides.
        scores = {s: self._side_score(lms, s) for s in ("left", "right")}
        best = max(scores, key=scores.get)
        if self.side_choice is None or scores[best] > scores[self.side_choice] + CONFIG.side_switch_margin:
            if best != self.side_choice:
                self.side_ema = EMA(CONFIG.ema_alpha)
            self.side_choice = best
        side = self.side_choice

        points = self._side_points(lms, side)
        ear, eye, shoulder, hip = points["ear"], points["eye"], points["shoulder"], points["hip"]
        if ear is None:
            return FeatureSet(issue="Side camera: ear not visible — keep your head in view (hair or a hood can hide it)")
        if shoulder is None:
            return FeatureSet(issue="Side camera: shoulder not visible — move the camera back")
        if hip is None:
            return FeatureSet(issue="Side camera: hip not visible — move the camera back or lower so your hip is in view")
        height = next(p.height for p in lms.values())
        torso_len = distance(shoulder, hip)
        if not (ear[1] < shoulder[1] < hip[1]):
            return FeatureSet(issue="Side camera looks rotated — mount it upright")
        if torso_len < CONFIG.side_min_torso_height_ratio * height:
            return FeatureSet(issue="Side camera: move closer (torso too small in view)")

        issue = ""
        if self._present(lms, "left_shoulder", "right_shoulder"):
            separation = abs(lms["left_shoulder"].pxy[0] - lms["right_shoulder"].pxy[0]) / torso_len
            if separation >= CONFIG.side_oblique_max_ratio:
                return FeatureSet(issue="Side camera is too far in front of or behind you — place it at your side (90°)")
            if separation >= CONFIG.side_oblique_warn_ratio:
                issue = "Side camera is angled — move it level with your side (90°) for accurate readings"

        # Facing direction: the face is in front of the ear. Keep the last value when unclear.
        width = next(p.width for p in lms.values())
        face = lms["nose"].pxy if self._present(lms, "nose") else eye  # nose, else the eye, is in front
        if face is not None and abs(face[0] - ear[0]) > 0.01 * width:
            self.facing = 1.0 if face[0] > ear[0] else -1.0
        if self.facing is None:
            self.facing = 1.0 if ear[0] >= shoulder[0] else -1.0
        forward = self.facing

        torso_lean = signed_lean_deg(hip, shoulder, forward)
        neck_inclination = signed_lean_deg(shoulder, ear, forward)  # vs vertical (gravity)
        values = {
            "torso_lean_deg": torso_lean,
            # Head position relative to the trunk: subtracting the trunk angle removes whole-body lean/recline
            # and any camera roll (both angles shift equally), leaving neck protraction/flexion vs the torso.
            "forward_head_deg": neck_inclination - torso_lean,
        }
        if eye is not None and (eye[0] - ear[0]) * forward > 0.01 * width:  # pitch needs a clear profile
            pitch = math.degrees(math.atan2(eye[1] - ear[1], (eye[0] - ear[0]) * forward))
            values["head_flexion_deg"] = pitch - CONFIG.head_pitch_neutral_deg
        return FeatureSet(True, self.side_ema.update(values), issue)

    # ------------------------------------------------------------------ classification

    @staticmethod
    def _debounced(store: Dict[str, List], name: str, ratio: float, now: float,
                   level: float = 1.0, enter: Optional[float] = None) -> bool:
        """Flag only sustained deviations; clear only after a sustained return (with value hysteresis)."""
        state = store.setdefault(name, [False, None])
        active, since = state
        crossing = ratio < level * CONFIG.metric_hysteresis_ratio if active else ratio > level
        hold = CONFIG.metric_exit_seconds if active else (CONFIG.metric_enter_seconds if enter is None else enter)
        if crossing:
            since = now if since is None else since
            if now - since >= hold:
                active, since = not active, None
        else:
            since = None
        state[0], state[1] = active, since
        return active

    def classify(self, front: FeatureSet, side: FeatureSet, persistent_bad_seconds: float = 0.0,
                 now: Optional[float] = None) -> PostureResult:
        now = time.monotonic() if now is None else now
        guidance = [fs.issue for fs in (front, side) if fs.issue]
        front_ok = front.valid and all(k in front.values for k in CORE_FRONT)
        side_ok = side.valid and all(k in side.values for k in CORE_SIDE)
        if not (front_ok or side_ok):
            self._flags.clear()
            self._severe.clear()
            return PostureResult(
                labels=[], status="Insufficient Data", risk_level="N/A", score=None,
                components={c: None for c in COMPONENTS},
                front_features=front.values, side_features=side.values,
                bad_regions_front=[], bad_regions_side=[],
                reasons=guidance or ["Waiting for reliable landmarks from both cameras"], guidance=guidance,
            )

        # If one camera is unreliable, keep evaluating the indicators the other camera still measures;
        # the missing camera's indicators (and components) stay unavailable rather than counted as good.
        values = {}
        if front_ok:
            values.update(front.values)
        if side_ok:
            values.update(side.values)
            torso_ratio = max(0.0, values["torso_lean_deg"]) / CONFIG.torso_forward_max_deg
            neck_ratio = max(0.0, values["forward_head_deg"]) / CONFIG.forward_head_max_deg
            values["slouch_index"] = min(torso_ratio, neck_ratio) / CONFIG.slouch_combined_ratio
        for metric in METRICS:
            if metric.key not in values:  # unavailable now: don't keep a stale warning
                self._flags.pop(metric.name, None)
                self._severe.pop(metric.name, None)

        metrics: Dict[str, Tuple[float, float, float, bool]] = {}
        component_scores: Dict[str, List[float]] = {c: [] for c in COMPONENTS}
        flagged_labels, reasons = set(), []
        severe_any, worst_flagged = False, 0.0
        for metric in METRICS:
            if metric.key not in values:
                continue  # unavailable indicators are left out, never scored as perfect
            value = values[metric.key]
            ratio = deviation_ratio(metric, value)
            flagged = self._debounced(self._flags, metric.name, ratio, now)
            severe = self._debounced(self._severe, metric.name, ratio, now,
                                     level=CONFIG.metric_poor_ratio, enter=CONFIG.poor_severe_seconds)
            if flagged:
                severe_any = severe_any or severe
                worst_flagged = max(worst_flagged, ratio)
            metrics[metric.name] = (value, metric_limit(metric), ratio, flagged)
            component_scores[metric.component].append(indicator_score(ratio))
            if flagged:
                flagged_labels.add(metric.label)
                unit = "" if metric.key == "slouch_index" else f" ({abs(value):.1f}°, limit {metric_limit(metric):.0f}°)"
                reasons.append(f"{metric.reason}{unit}")

        # Each component shows its worst indicator; the total is the weighted mean of available components.
        components = {c: (round(min(s)) if s else None) for c, s in component_scores.items()}
        weights = {"Neck": CONFIG.neck_weight, "Back": CONFIG.back_weight,
                   "Shoulder": CONFIG.shoulder_weight, "Alignment": CONFIG.alignment_weight}
        available = [c for c in COMPONENTS if components[c] is not None]
        raw_score = sum(components[c] * weights[c] for c in available) / sum(weights[c] for c in available)

        labels = [label for label in LABEL_ORDER if label in flagged_labels]
        partial = not (front_ok and side_ok)
        if not labels:
            labels = ["Good Posture"]
            reasons = ["All measured indicators are within the ergonomic reference ranges"]
            risk, status = "GOOD", "Good Posture"
        else:
            # Severity uses magnitude AND duration: severe deviations become POOR after poor_severe_seconds;
            # a long-lasting deviation becomes POOR only if it is not minor. Minor deviations stay WARNING.
            persistent_poor = (persistent_bad_seconds >= CONFIG.poor_posture_seconds
                               and worst_flagged >= CONFIG.poor_persistent_min_ratio)
            risk = "POOR" if severe_any or persistent_poor else "WARNING"
            status = " + ".join(labels)
        # The displayed score stays inside the band of the classification (GOOD >= 80, WARNING 60-79,
        # POOR < 60), so a flagged indicator is never hidden behind a "green" weighted average.
        score = clamp(raw_score, *score_band(risk))

        bad_front = (["shoulder"] if "Uneven Shoulder" in labels else []) + \
                    (["back"] if "Leaning Left/Right" in labels else [])
        bad_side = (["neck"] if "Forward Head" in labels else []) + (["back"] if "Slouching" in labels else [])
        return PostureResult(
            labels=labels, status=status, risk_level=risk, score=round(score), components=components,
            front_features=front.values, side_features=side.values,
            bad_regions_front=bad_front, bad_regions_side=bad_side,
            reasons=(([f"{'Side' if front_ok else 'Front'} camera unavailable — "
                       f"showing {'front' if front_ok else 'side'}-camera indicators only"] if partial else [])
                     + reasons),
            metrics=metrics, guidance=guidance, partial=partial, raw_score=round(raw_score),
        )
