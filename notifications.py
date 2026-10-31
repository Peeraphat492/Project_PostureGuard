"""Popup notification rules. Qt-free, driven by explicit (monotonic) timestamps so it can be tested.

Only one popup exists at a time: the posture warning or the Active Break reminder.

Posture warning episode:
  * starts with the first confirmed WARNING/POOR frame (the analyzer's debounced classification);
  * only confirmed bad time counts toward `popup_trigger_seconds`. Unreliable data (Insufficient Data) and
    short GOOD moments pause the count, they never add to it;
  * ends only after GOOD posture has been confirmed continuously for `popup_recovery_seconds`;
  * gets at most one popup. Dismiss, Snooze and the automatic close never reopen it.
A visible popup closes after `popup_visible_seconds`, or as soon as the episode ends (posture recovered).
Snooze suppresses posture popups (including those of new episodes) for `popup_snooze_seconds`.
"""
from __future__ import annotations

from config import CONFIG

BAD_RISKS = ("WARNING", "POOR")


class PopupNotifier:
    def __init__(self):
        self.visible: str | None = None   # "posture" / "break" while a popup is on screen
        self.shown_at = 0.0
        self.snoozed_until = float("-inf")
        self.reset_episode()

    def reset_episode(self):
        """Forget the current bad-posture episode (recovery, cameras stopped, Active Break)."""
        self.bad_seconds = 0.0     # confirmed bad-posture time in this episode
        self.good_seconds = 0.0    # continuous confirmed GOOD time
        self.notified = False      # this episode already had its popup
        self._last = None

    def update(self, now: float, risk: str, reliable: bool) -> str | None:
        """Advance with one evaluated frame. Returns "show" (open the posture popup), "close" or None."""
        dt = 0.0 if self._last is None else max(0.0, min(now - self._last, CONFIG.session_max_tick_seconds))
        self._last = now
        bad = reliable and risk in BAD_RISKS
        if bad:
            self.bad_seconds += dt
            self.good_seconds = 0.0
        elif reliable and risk == "GOOD":
            self.good_seconds += dt
        else:
            self.good_seconds = 0.0  # unreliable data interrupts the recovery confirmation too

        if self.good_seconds >= CONFIG.popup_recovery_seconds and (self.bad_seconds or self.notified):
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
        self.snoozed_until = now + CONFIG.popup_snooze_seconds

    def break_popup_allowed(self) -> bool:
        """The Active Break reminder waits until no other popup is on screen."""
        return self.visible is None
