"""Coin Collection: touch coins with your hands and dodge incoming bombs with your body.

Positions are normalized to the displayed (mirrored) front-camera frame. Distances are measured
in frame-height units (x is scaled by the aspect ratio) so circles stay round.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .body_pose import BodyPose
from config import CONFIG

Point = Tuple[float, float]

COIN_RADIUS = 0.06          # frame-height units
HAND_RADIUS = 0.06          # generous hand hit-box
COIN_LIFETIME = 8.0         # an uncollected coin moves somewhere else
COIN_AREA = ((0.10, 0.90), (0.12, 0.62))  # x range, y range for spawning
BOMB_RADIUS = 0.07
BOMB_TRAVEL_SECONDS = 2.0   # time from spawn to impact
BOMB_DANGER_FROM = 0.45     # dodges count during the last 55% of the approach
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
    dodged: bool = False

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
    started_at: Optional[float] = None
    next_bomb_at: float = 0.0

    # ------------------------------------------------------------------ helpers

    def _distance(self, a: Point, b: Point) -> float:
        return math.hypot((a[0] - b[0]) * self.aspect, a[1] - b[1])

    def _spawn_coin(self, now: float, pose: Optional[BodyPose]):
        avoid = list(pose.wrists) + [pose.nose] if pose else []
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
        self.bomb = Bomb(side, start, (hx, hy), now)

    def _did_dodge(self, dodge: str, pose: BodyPose) -> bool:
        if dodge == "duck":
            return pose.ducked(self.neutral[1])
        direction = pose.lean(self.neutral[0])
        return direction == (1 if dodge == "lean_right" else -1)

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
            if self.neutral is None:
                self.neutral = pose.nose
            else:
                k = NEUTRAL_SMOOTHING
                self.neutral = (self.neutral[0] + (pose.nose[0] - self.neutral[0]) * k,
                                self.neutral[1] + (pose.nose[1] - self.neutral[1]) * k)

        # Coins
        if self.coin is None or now - self.coin.spawned_at > COIN_LIFETIME:
            self._spawn_coin(now, pose)
        elif pose is not None:
            for wrist in pose.wrists:
                if self._distance(wrist, (self.coin.x, self.coin.y)) <= COIN_RADIUS + HAND_RADIUS:
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
        if pose is not None and self.bomb.progress(now) >= BOMB_DANGER_FROM and self._did_dodge(self.bomb.dodge, pose):
            self.bomb.dodged = True
        if self.bomb.progress(now) >= 1.0:
            if self.bomb.dodged:
                self.score += CONFIG.bomb_avoid_points
                self.bombs_avoided += 1
                self._say(f"Dodged! +{CONFIG.bomb_avoid_points}", True, self.bomb.target, now)
            else:
                self.score = max(0, self.score - CONFIG.bomb_hit_penalty)
                self.bomb_hits += 1
                self._say(f"Hit! -{CONFIG.bomb_hit_penalty}", False, self.bomb.target, now)
            self.bomb = None
            self.next_bomb_at = now + self.rng.uniform(*BOMB_GAP)
