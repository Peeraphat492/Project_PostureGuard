import math
from typing import Tuple

Point = Tuple[float, float]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def signed_tilt_deg(a: Point, b: Point) -> float:
    """Tilt of the line between two points (any order) from horizontal, in image pixel space.

    Positive when the point further left in the image is lower (larger y). Range (-90, 90].
    """
    left, right = (a, b) if a[0] <= b[0] else (b, a)
    return math.degrees(math.atan2(left[1] - right[1], max(right[0] - left[0], 1e-9)))


def signed_lean_deg(bottom: Point, top: Point, forward_x: float = 1.0) -> float:
    """Angle of the segment bottom->top from vertical, in image pixel space.

    Positive when `top` is displaced in the direction of `forward_x` (+1 = image-right, -1 = image-left).
    """
    return math.degrees(math.atan2((top[0] - bottom[0]) * forward_x, bottom[1] - top[1]))
