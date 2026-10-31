"""Popup notification rules. Qt-free, driven by explicit (monotonic) timestamps so it can be tested.

Only one popup exists at a time: the posture warning or the Active Break reminder.

Posture warning episode:
  * starts with the first confirmed WARNING/POOR frame (the analyzer's debounced classification);
  * only confirmed bad time counts toward `popup_trigger_seconds`. Frames without posture evidence pause the
    count, they never add to it: Insufficient Data, and a GOOD result from cameras that cannot measure the
    episode's problem (e.g. Forward Head is measured by the side camera; while only the front camera works,
    "GOOD" says nothing about it);
  * ends after GOOD posture has been confirmed continuously for `popup_recovery_seconds` by the cameras that
    measured the problem, or after `popup_inactive_reset_seconds` without any posture evidence (the user left);
  * gets one popup. Dismiss and the automatic close never reopen it; Snooze lets it warn again once the snooze
    has expired if the bad posture continues.
A visible popup closes after `popup_visible_seconds`, or as soon as the episode ends (posture recovered).
Snooze suppresses posture popups (including those of new episodes) for `popup_snooze_seconds`.
Timing uses the elapsed time between frames. A gap longer than `camera_frame_timeout_seconds` (stall, sleep) is
time without measurements: only that much of it counts as posture time, the rest as time without evidence.
"""
from __future__ import annotations

from config import CONFIG
from posture_analyzer import METRICS

BAD_RISKS = ("WARNING", "POOR")
CAMERAS = frozenset(("front", "side"))


def result_evidence(result) -> tuple:
    """update() arguments for a PostureResult (None = no usable frames): risk, reliable, the cameras that measured
    this frame and the cameras whose indicators are flagged."""
    if result is None or result.risk_level == "N/A":
        return "N/A", False, frozenset(), frozenset()
    cameras = frozenset(m.camera for m in METRICS if m.name in result.metrics)
    problems = frozenset(m.camera for m in METRICS if m.name in result.metrics and result.metrics[m.name][3])
    return result.risk_level, True, cameras, problems


class PopupNotifier:
    def __init__(self):
        self.visible: str | None = None   # "posture" / "break" while a popup is on screen
        self.shown_at = 0.0
        self.snoozed_until = float("-inf")
        self.reset_episode()

    def reset_episode(self):
        """Forget the current bad-posture episode (recovery, user away, cameras stopped, Active Break)."""
        self.bad_seconds = 0.0     # confirmed bad-posture time in this episode
        self.good_seconds = 0.0    # continuous confirmed GOOD time
        self.idle_seconds = 0.0    # continuous time without posture evidence
        self.problem_cameras = set()  # cameras that measured this episode's problems
        self.notified = False      # this episode already had its popup
        self._last = None

    def update(self, now: float, risk: str, reliable: bool, cameras=CAMERAS, problems=()) -> str | None:
        """Advance with one evaluated frame. Returns "show" (open the posture popup), "close" or None.
        cameras: cameras that measured this frame; problems: cameras with a flagged indicator (result_evidence)."""
        gap = 0.0 if self._last is None else max(0.0, now - self._last)
        self._last = now
        step = min(gap, CONFIG.camera_frame_timeout_seconds)   # posture time; the rest of a long gap is unmeasured
        bad = reliable and risk in BAD_RISKS
        if bad:
            self.problem_cameras |= set(problems)
        recovered = reliable and risk == "GOOD" and self.problem_cameras <= set(cameras)
        if bad:
            self.bad_seconds += step
            self.good_seconds = 0.0
            self.idle_seconds = gap - step
        elif recovered:
            self.good_seconds += step
            self.idle_seconds = gap - step
        else:
            self.good_seconds = 0.0  # no evidence interrupts the recovery confirmation too
            self.idle_seconds += gap

        active = self.bad_seconds > 0 or self.notified
        if active and (self.good_seconds >= CONFIG.popup_recovery_seconds
                       or self.idle_seconds >= CONFIG.popup_inactive_reset_seconds):
            was_posture = self.visible == "posture"
            self.reset_episode()
            self._last = now
            if was_posture:
                return "close"
        if self.visible is not None:
            return "close" if now - self.shown_at >= CONFIG.popup_visible_seconds else None
        if (bad and not self.notified and self.bad_seconds >= CONFIG.popup_trigger_seconds
                and now >= self.snoozed_until):
            self.notified = True
            self.opened("posture", now)
            return "show"
        return None

    def opened(self, kind: str, now: float):
        self.visible, self.shown_at = kind, now

    def closed(self):
        self.visible = None

    def snooze(self, now: float):
        """No posture popup for popup_snooze_seconds; if the bad posture continues, it warns again afterwards."""
        self.snoozed_until = now + CONFIG.popup_snooze_seconds
        self.notified = False

    def break_popup_allowed(self) -> bool:
        """The Active Break reminder waits until no other popup is on screen."""
        return self.visible is None
