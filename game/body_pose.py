"""Screen-space body pose helpers for the Active Break game.

Coordinates are the normalized (0..1) landmark positions of the *displayed* front-camera frame
(already mirrored), so "left" always means the left side of the screen, which matches what the
player sees. Thresholds are deliberately forgiving so the game stays playable while seated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

Point = Tuple[float, float]

CORE_VISIBILITY = 0.4   # nose + shoulders must be reasonably visible
ARM_VISIBILITY = 0.3    # wrists/elbows are often partly out of frame

LEAN_TILT_DEG = 10.0    # shoulder line tilt that counts as a sideways lean
LEAN_HEAD_OFFSET = 0.30  # head offset from shoulder centre, in shoulder widths
LEAN_SHIFT = 0.40        # head shift from the player's neutral position, in shoulder widths
DUCK_DROP = 0.22         # head drop from neutral, in shoulder widths (converted to y units)
DUCK_MIN_DROP = 0.05     # ...but never less than 5% of the frame height


@dataclass(frozen=True)
class BodyPose:
    nose: Point
    left_shoulder: Point    # screen-left shoulder (smaller x)
    right_shoulder: Point   # screen-right shoulder
    wrists: Tuple[Point, ...] = ()
    elbows: Tuple[Point, ...] = ()
    aspect: float = 16 / 9  # frame width / height, to compare x and y distances

    @classmethod
    def from_landmarks(cls, lms: Optional[Dict], aspect: float = 16 / 9) -> Optional["BodyPose"]:
        if not lms:
            return None
        for name in ("nose", "left_shoulder", "right_shoulder"):
            if name not in lms or lms[name].visibility < CORE_VISIBILITY:
                return None
        a, b = lms["left_shoulder"].xy, lms["right_shoulder"].xy
        left, right = (a, b) if a[0] <= b[0] else (b, a)

        def visible(*names):
            return tuple(lms[n].xy for n in names if n in lms and lms[n].visibility >= ARM_VISIBILITY)

        return cls(
            nose=lms["nose"].xy,
            left_shoulder=left,
            right_shoulder=right,
            wrists=visible("left_wrist", "right_wrist"),
            elbows=visible("left_elbow", "right_elbow"),
            aspect=aspect,
        )

    # ------------------------------------------------------------------ geometry

    @property
    def shoulder_mid(self) -> Point:
        return ((self.left_shoulder[0] + self.right_shoulder[0]) / 2,
                (self.left_shoulder[1] + self.right_shoulder[1]) / 2)

    @property
    def shoulder_width(self) -> float:
        """Shoulder distance in x-normalized units."""
        dx = self.right_shoulder[0] - self.left_shoulder[0]
        dy = (self.right_shoulder[1] - self.left_shoulder[1]) / self.aspect
        return max(math.hypot(dx, dy), 1e-3)

    @property
    def tilt_deg(self) -> float:
        """Positive when the screen-left shoulder is lower, i.e. the body leans to the screen-left."""
        dx = (self.right_shoulder[0] - self.left_shoulder[0]) * self.aspect
        dy = self.left_shoulder[1] - self.right_shoulder[1]
        return math.degrees(math.atan2(dy, max(dx, 1e-6)))

    @property
    def head_offset(self) -> float:
        """Head position relative to the shoulder centre, in shoulder widths (negative = screen-left)."""
        return (self.nose[0] - self.shoulder_mid[0]) / self.shoulder_width

    # ------------------------------------------------------------------ actions

    def lean(self, neutral_x: Optional[float] = None) -> int:
        """-1 when leaning to the screen-left, +1 to the screen-right, 0 when roughly upright."""
        left = self.tilt_deg >= LEAN_TILT_DEG or self.head_offset <= -LEAN_HEAD_OFFSET
        right = self.tilt_deg <= -LEAN_TILT_DEG or self.head_offset >= LEAN_HEAD_OFFSET
        if neutral_x is not None:
            shift = (self.nose[0] - neutral_x) / self.shoulder_width
            left = left or shift <= -LEAN_SHIFT
            right = right or shift >= LEAN_SHIFT
        if left == right:
            return 0
        return -1 if left else 1

    def ducked(self, neutral_y: float) -> bool:
        drop = self.nose[1] - neutral_y
        return drop >= max(DUCK_DROP * self.shoulder_width * self.aspect, DUCK_MIN_DROP)

    def arms_raised(self) -> bool:
        """Both hands above the head (elbows above the head count when the hands leave the top of the frame,
        like arms_wide). Hands that are visible but below the head never count."""
        if len(self.wrists) == 2:
            return all(w[1] < self.nose[1] for w in self.wrists)
        return len(self.elbows) == 2 and all(e[1] < self.nose[1] for e in self.elbows)

    @property
    def arms_visible(self) -> bool:
        return len(self.wrists) == 2 or len(self.elbows) == 2

    def arms_wide(self) -> bool:
        """Arms stretched out sideways at roughly shoulder height (elbows count if hands leave the frame)."""
        sw = self.shoulder_width
        band = 0.6 * sw * self.aspect  # vertical tolerance in y units
        mid_y = self.shoulder_mid[1]

        def wide(points, spread_needed):
            return (
                len(points) == 2
                and abs(points[0][0] - points[1][0]) >= spread_needed * sw
                and all(abs(p[1] - mid_y) <= band for p in points)
            )

        return wide(self.wrists, 1.9) or wide(self.elbows, 1.5)
