"""Coin Collection: touch coins with your hands and dodge incoming bombs with your body.

Positions are normalized to the displayed (mirrored) front-camera frame. Distances are measured
in frame-height units (x is scaled by the aspect ratio) so circles stay round.

Bombs fly at the player's relaxed head position. A bomb hits only when its path actually passes through the
head hit-box (an ellipse around the nose), so any move that takes the head out of the path counts as a dodge:
ducking under a bomb from the side, stepping aside from a bomb from above, or moving away from a side bomb.
Untracked frames are never treated as a hit: the bomb is judged on the last tracked head position if it is
recent, otherwise it ends without a hit or points (leaving the frame is never rewarded).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .body_pose import DUCK_DROP, DUCK_MIN_DROP, LEAN_SHIFT, BodyPose
from config import CONFIG

Point = Tuple[float, float]

COIN_RADIUS = 0.06          # frame-height units
HAND_RADIUS = 0.06          # generous hand hit-box
COIN_LIFETIME = 8.0         # an uncollected coin moves somewhere else
COIN_AREA = ((0.10, 0.90), (0.12, 0.62))  # x range, y range for spawning
BOMB_RADIUS = 0.07
BOMB_TRAVEL_SECONDS = 2.0   # time from spawn to impact
BOMB_DANGER_FROM = 0.45     # the bomb is drawn as dangerous during the last 55% of the approach
TRACKING_GRACE = 0.5        # an untracked bomb is judged on the last tracked head if it is at most this old
FIRST_BOMB_DELAY = 4.0
BOMB_GAP = (2.5, 4.0)
NEUTRAL_SMOOTHING = 0.08    # how fast the "upright" head position follows the player
FEEDBACK_SECONDS = 0.9

DODGE_FOR_SIDE = {"left": "lean_right", "right": "lean_left", "top": "duck"}
DODGE_TEXT = {"lean_right": "Lean RIGHT", "lean_left": "Lean LEFT", "duck": "Duck DOWN"}


@dataclass
class Coin:
    x: float
    y: float
    spawned_at: float


@dataclass
class Bomb:
    side: str               # "left" / "right" (comes from that screen edge) or "top"
    start: Point
    target: Point
    spawned_at: float
    reach: Point = (0.1, 0.1)         # head hit-box half-width / half-height (frame-height units)
    dodged: bool = False
    hit: bool = False                 # the path passed through the tracked head (a confirmed collision)
    seen_at: Optional[float] = None   # last frame with a tracked head during this bomb's flight
    seen_head: Optional[Point] = None
    checked_pos: Optional[Point] = None  # bomb position at that frame (the path is checked from here)

    @property
    def dodge(self) -> str:
        return DODGE_FOR_SIDE[self.side]

    def progress(self, now: float) -> float:
        return max(0.0, min(1.0, (now - self.spawned_at) / BOMB_TRAVEL_SECONDS))

    def position(self, now: float) -> Point:
        t = self.progress(now)
        ease = t * t * (3 - 2 * t)
        return (self.start[0] + (self.target[0] - self.start[0]) * ease,
                self.start[1] + (self.target[1] - self.start[1]) * ease)


@dataclass
class Feedback:
    text: str
    good: bool
    x: float
    y: float
    at: float


@dataclass
class CoinGame:
    rng: random.Random = field(default_factory=random.Random)
    aspect: float = 16 / 9
    score: int = 0
    coins_collected: int = 0
    bombs_avoided: int = 0
    bomb_hits: int = 0
    coin: Optional[Coin] = None
    bomb: Optional[Bomb] = None
    feedback: List[Feedback] = field(default_factory=list)
    neutral: Optional[Point] = None   # the player's relaxed head position
    scale: Optional[float] = None     # the player's relaxed shoulder width, in frame-height units
    started_at: Optional[float] = None
    next_bomb_at: float = 0.0

    # ------------------------------------------------------------------ helpers

    def _distance(self, a: Point, b: Point) -> float:
        return math.hypot((a[0] - b[0]) * self.aspect, a[1] - b[1])

    def _spawn_coin(self, now: float, pose: Optional[BodyPose]):
        avoid = list(pose.hands) + [pose.nose] if pose else []
        (x0, x1), (y0, y1) = COIN_AREA
        for _ in range(25):
            x, y = self.rng.uniform(x0, x1), self.rng.uniform(y0, y1)
            if all(self._distance((x, y), p) > 0.2 for p in avoid):
                break
        self.coin = Coin(x, y, now)

    def _spawn_bomb(self, now: float, side: Optional[str] = None):
        side = side or self.rng.choice(("left", "right", "top"))
        hx, hy = self.neutral
        start = {"left": (-0.08, hy), "right": (1.08, hy), "top": (hx, -0.1)}[side]
        # Head hit-box sized from the relaxed pose (frozen for the flight, so leaning toward the camera does not
        # grow it): clearing it takes the same head movement as the duck / sideways-shift rules of BodyPose.
        reach = (LEAN_SHIFT * self.scale, max(DUCK_DROP * self.scale, DUCK_MIN_DROP))
        self.bomb = Bomb(side, start, (hx, hy), now, reach)

    def _touches(self, head: Point, a: Point, b: Point, reach: Point) -> bool:
        """True when the bomb's path from a to b passes through the head hit-box (an ellipse around the nose)."""
        rx, ry = reach

        def local(p: Point) -> Point:  # hit-box space: the ellipse becomes the unit circle
            return (p[0] - head[0]) * self.aspect / rx, (p[1] - head[1]) / ry

        (ax, ay), (bx, by) = local(a), local(b)
        dx, dy = bx - ax, by - ay
        length = dx * dx + dy * dy
        k = 0.0 if length == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / length))
        return math.hypot(ax + k * dx, ay + k * dy) < 1.0

    def _say(self, text: str, good: bool, where: Point, now: float):
        self.feedback.append(Feedback(text, good, where[0], where[1], now))

    # ------------------------------------------------------------------ public

    def instruction(self) -> str:
        if self.bomb is not None:
            origin = "above" if self.bomb.side == "top" else f"the {self.bomb.side}"
            return f"Bomb from {origin}! {DODGE_TEXT[self.bomb.dodge]}"
        return "Touch the coins with your hands"

    def update(self, now: float, pose: Optional[BodyPose]):
        if self.started_at is None:
            self.started_at = now
            self.next_bomb_at = now + FIRST_BOMB_DELAY
        self.feedback = [f for f in self.feedback if now - f.at < FEEDBACK_SECONDS]
        if pose is not None:
            self.aspect = pose.aspect

        if pose is not None and (self.neutral is None or self.bomb is None):
            scale = pose.shoulder_width * pose.aspect
            if self.neutral is None:
                self.neutral, self.scale = pose.nose, scale
            else:
                k = NEUTRAL_SMOOTHING
                self.neutral = (self.neutral[0] + (pose.nose[0] - self.neutral[0]) * k,
                                self.neutral[1] + (pose.nose[1] - self.neutral[1]) * k)
                self.scale += (scale - self.scale) * k

        # Coins
        if self.coin is None or now - self.coin.spawned_at > COIN_LIFETIME:
            self._spawn_coin(now, pose)
        elif pose is not None:
            for hand in pose.hands:   # the hand itself, not the wrist behind it
                if self._distance(hand, (self.coin.x, self.coin.y)) <= COIN_RADIUS + HAND_RADIUS:
                    self.score += CONFIG.coin_points
                    self.coins_collected += 1
                    self._say(f"+{CONFIG.coin_points}", True, (self.coin.x, self.coin.y), now)
                    self._spawn_coin(now, pose)
                    break

        # Bombs
        if self.bomb is None:
            if now >= self.next_bomb_at and self.neutral is not None:
                self._spawn_bomb(now)
            return
        bomb = self.bomb
        pos = bomb.position(now)
        if pose is not None:
            # Check the path flown since the last tracked frame (so a fast bomb cannot skip over the head).
            recent = bomb.seen_at is not None and now - bomb.seen_at <= TRACKING_GRACE
            bomb.hit = bomb.hit or self._touches(pose.nose, bomb.checked_pos if recent else pos, pos, bomb.reach)
            bomb.seen_at, bomb.seen_head, bomb.checked_pos = now, pose.nose, pos
        if bomb.progress(now) >= 1.0:
            tracked = bomb.seen_at is not None and now - bomb.seen_at <= TRACKING_GRACE
            if tracked and not bomb.hit:  # briefly untracked at impact: the head is assumed to have stayed put
                bomb.hit = self._touches(bomb.seen_head, bomb.checked_pos, bomb.target, bomb.reach)
            if bomb.hit:
                self.score = max(0, self.score - CONFIG.bomb_hit_penalty)
                self.bomb_hits += 1
                self._say(f"Hit! -{CONFIG.bomb_hit_penalty}", False, bomb.target, now)
            elif tracked:
                bomb.dodged = True
                self.score += CONFIG.bomb_avoid_points
                self.bombs_avoided += 1
                self._say(f"Dodged! +{CONFIG.bomb_avoid_points}", True, bomb.target, now)
            else:  # not seen near the impact: neither a hit nor a dodge can be confirmed
                self._say("Not tracked", False, bomb.target, now)
            self.bomb = None
            self.next_bomb_at = now + self.rng.uniform(*BOMB_GAP)
