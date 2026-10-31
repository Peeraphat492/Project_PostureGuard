"""Stretch Challenge: follow one stretch at a time and hold it to score."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .body_pose import BodyPose
from config import CONFIG

HOLD_GRACE_SECONDS = 0.3   # brief wobbles out of the pose don't reset the hold
TRACKING_LOSS_GRACE = 1.0  # a hold survives a tracking drop-out this long (paused: the lost time doesn't count)
MAX_STEP_SECONDS = 0.5     # camera stalls don't fast-forward the skip timer
ARM_POSES = ("raise_arms", "arms_wide")
CELEBRATE_SECONDS = 0.9    # pause on the success message before the next stretch


@dataclass(frozen=True)
class Exercise:
    key: str
    name: str
    hint: str


EXERCISES = (
    Exercise("raise_arms", "Raise Both Arms", "Lift both hands above your head"),
    Exercise("arms_wide", "Arms Wide", "Stretch both arms out to the sides"),
    Exercise("lean_left", "Lean Left", "Bend your upper body to the left"),
    Exercise("lean_right", "Lean Right", "Bend your upper body to the right"),
)


def pose_matches(key: str, pose: Optional[BodyPose]) -> bool:
    if pose is None:
        return False
    if key == "raise_arms":
        return pose.arms_raised()
    if key == "arms_wide":
        return pose.arms_wide()
    if key == "lean_left":
        return pose.lean() == -1
    if key == "lean_right":
        return pose.lean() == 1
    return False


class StretchGame:
    def __init__(self):
        self.index = 0
        self.score = 0
        self.completed = 0
        self.exercise_started: Optional[float] = None
        self.hold_started: Optional[float] = None
        self.last_match: Optional[float] = None
        self.celebrate_until = 0.0
        self.message = ""          # short feedback such as "+10" or "Skipped"
        self.message_good = True
        self.matching = False
        self.status = "not_visible"   # "detected" | "adjust" | "not_visible" (shown to the player)
        self.attempt_seconds = 0.0    # time with a tracked body on this exercise (the skip timeout counts this)
        self.lost_since: Optional[float] = None
        self._last_update: Optional[float] = None

    @property
    def exercise(self) -> Exercise:
        return EXERCISES[self.index % len(EXERCISES)]

    @property
    def next_exercise(self) -> Exercise:
        return EXERCISES[(self.index + 1) % len(EXERCISES)]

    def hold_progress(self, now: float) -> float:
        if self.hold_started is None:
            return 0.0
        end = self.lost_since if self.lost_since is not None else now   # frozen while tracking is lost
        return max(0.0, min(1.0, (end - self.hold_started) / CONFIG.stretch_hold_seconds))

    def celebrating(self, now: float) -> bool:
        return now < self.celebrate_until

    def _advance(self, now: float):
        self.index += 1
        self.exercise_started = now
        self.hold_started = None
        self.last_match = None
        self.attempt_seconds = 0.0
        self.lost_since = None

    def update(self, now: float, pose: Optional[BodyPose]):
        dt = 0.0 if self._last_update is None else max(0.0, min(now - self._last_update, MAX_STEP_SECONDS))
        self._last_update = now
        if self.exercise_started is None:
            self.exercise_started = now
        if self.celebrating(now):
            return

        if pose is None:
            # Tracking lost: never a match, never a skip. A hold in progress is paused, then dropped if the
            # body stays out of view for longer than TRACKING_LOSS_GRACE.
            self.matching = False
            self.status = "not_visible"
            if self.hold_started is not None:
                if self.lost_since is None:
                    self.lost_since = now
                elif now - self.lost_since > TRACKING_LOSS_GRACE:
                    self.hold_started = self.lost_since = None
            return
        if self.lost_since is not None:
            if self.hold_started is not None:
                self.hold_started += now - self.lost_since   # the lost time doesn't count toward the hold
            self.lost_since = None
        self.attempt_seconds += dt

        self.matching = pose_matches(self.exercise.key, pose)
        if self.matching:
            self.status = "detected"
        elif self.exercise.key in ARM_POSES and not pose.arms_visible:
            self.status = "not_visible"
        else:
            self.status = "adjust"
        if self.matching:
            self.last_match = now
            if self.hold_started is None:
                self.hold_started = now
            if now - self.hold_started >= CONFIG.stretch_hold_seconds:
                self.score += CONFIG.stretch_points
                self.completed += 1
                self.message, self.message_good = f"+{CONFIG.stretch_points}", True
                self.celebrate_until = now + CELEBRATE_SECONDS
                self._advance(now + CELEBRATE_SECONDS)
            return

        if self.hold_started is not None and (self.last_match is None or now - self.last_match > HOLD_GRACE_SECONDS):
            self.hold_started = None
        if self.hold_started is None and self.attempt_seconds >= CONFIG.stretch_exercise_timeout_seconds:
            self.message, self.message_good = "Skipped", False
            self.celebrate_until = now + CELEBRATE_SECONDS
            self._advance(now + CELEBRATE_SECONDS)
