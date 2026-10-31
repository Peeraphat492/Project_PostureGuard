from __future__ import annotations

import csv
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict

from config import CONFIG


@dataclass
class SessionTracker:
    """Session statistics.

    Time model (all in seconds, wall-clock time measured between processed frames):
      elapsed               session duration, excluding time the cameras were stopped (pause/resume)
      monitored_seconds     valid posture analysis: both cameras reliable. This is the ONLY time used for
                            the Good/Warning/Poor percentages, the condition durations and the average score.
      partial_seconds       only one camera reliable (its indicators are shown, but not counted in statistics)
      insufficient_seconds  cameras open but no reliable measurement / no frames
      break_seconds         Active Break game
      paused_total          cameras stopped
    Good + Warning + Poor durations add up to monitored_seconds. Condition durations (Forward Head,
    Slouching, ...) can overlap: several conditions may be present in the same moment.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)  # unique key in the history database
    started_at: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    durations: Dict[str, float] = field(default_factory=lambda: {
        "Good Posture": 0.0,
        "Forward Head": 0.0,
        "Slouching": 0.0,
        "Uneven Shoulder": 0.0,
        "Leaning Left/Right": 0.0,
        "Warning": 0.0,
        "Poor": 0.0,
    })
    monitored_seconds: float = 0.0
    partial_seconds: float = 0.0
    insufficient_seconds: float = 0.0
    break_seconds: float = 0.0
    score_time_sum: float = 0.0      # integral of the posture score over monitored time
    current_bad_since: float | None = None
    last_break_at: float = field(default_factory=time.time)
    paused_at: float | None = None   # set while the cameras are stopped
    paused_total: float = 0.0        # stopped time excluded from the session duration

    def reset(self):
        fresh = SessionTracker()
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)

    @property
    def elapsed(self) -> float:
        end = self.paused_at if self.paused_at is not None else time.time()
        return max(0.0, end - self.started_at - self.paused_total)

    @property
    def paused(self) -> bool:
        return self.paused_at is not None

    def pause(self):
        """Cameras stopped: freeze the session clock; all collected statistics are kept."""
        if self.paused_at is None:
            self.paused_at = time.time()
            self.current_bad_since = None

    def resume(self):
        """Cameras restarted: continue the same session without counting the stopped time."""
        if self.paused_at is None:
            return
        now = time.time()
        stopped = now - self.paused_at
        self.paused_total += stopped
        self.last_break_at += stopped  # the sitting-too-long timer pauses too
        self.last_update = now
        self.current_bad_since = None
        self.paused_at = None

    @property
    def persistent_bad_seconds(self) -> float:
        return 0.0 if self.current_bad_since is None else max(0.0, time.time() - self.current_bad_since)

    def preclassify_tick(self, currently_bad: bool):
        if currently_bad:
            if self.current_bad_since is None:
                self.current_bad_since = time.time()
        else:
            self.current_bad_since = None

    def tick(self, state: str, labels=(), risk_level: str = "", score: float | None = None):
        """Attribute the time since the previous tick to one state.

        state: "monitoring" (both cameras reliable), "partial", "insufficient" or "break".
        Time-based (not frame-based), so a changing frame rate does not change the statistics.
        """
        now = time.time()
        # Clamp so camera stalls / window dragging do not create fake duration spikes.
        dt = max(0.0, min(now - self.last_update, CONFIG.session_max_tick_seconds))
        self.last_update = now
        if state == "partial":
            self.partial_seconds += dt
            return
        if state == "break":
            self.break_seconds += dt
            return
        if state != "monitoring":
            self.insufficient_seconds += dt
            return

        self.monitored_seconds += dt
        if list(labels) == ["Good Posture"]:
            self.durations["Good Posture"] += dt
        else:
            for label in labels or []:
                if label in self.durations:
                    self.durations[label] += dt
        if risk_level == "WARNING":
            self.durations["Warning"] += dt
        elif risk_level == "POOR":
            self.durations["Poor"] += dt
        if score is not None:  # zero is a valid score and is included
            self.score_time_sum += float(score) * dt

    def should_break_remind(self) -> bool:
        """True once every break_reminder_minutes of sitting since the session start or the last break."""
        if CONFIG.break_reminder_minutes <= 0:
            return False
        now = time.time()
        if now - self.last_break_at >= CONFIG.break_reminder_minutes * 60:
            self.last_break_at = now
            return True
        return False

    def mark_break_taken(self):
        """An Active Break just ended: restart the sitting-too-long timer and the bad-posture timer."""
        self.last_break_at = time.time()
        self.current_bad_since = None

    @property
    def average_score(self) -> int:
        """Time-weighted mean posture score over valid monitoring time."""
        if self.monitored_seconds <= 1e-9:
            return 0
        return round(self.score_time_sum / self.monitored_seconds)

    def snapshot(self) -> dict:
        """Current statistics for the session history, computed exactly like the dashboard values."""
        d, p = self.durations, self.percentages()
        stopped = self.paused_total + (time.time() - self.paused_at if self.paused_at is not None else 0.0)
        return {
            "id": self.session_id,
            "started_at": self.started_at,
            "session_seconds": self.elapsed,
            "monitored_seconds": self.monitored_seconds,
            "partial_seconds": self.partial_seconds,
            "insufficient_seconds": self.insufficient_seconds,
            "break_seconds": self.break_seconds,
            "stopped_seconds": stopped,
            # time-weighted mean over monitored time (the dashboard shows it rounded); None = no valid data
            "average_score": self.score_time_sum / self.monitored_seconds if self.monitored_seconds > 1e-9 else None,
            "good_percent": p["Good Posture"],
            "warning_percent": p["Warning"],
            "poor_percent": p["Poor"],
            "good_seconds": d["Good Posture"],
            "warning_seconds": d["Warning"],
            "poor_seconds": d["Poor"],
            "forward_head_seconds": d["Forward Head"],
            "slouching_seconds": d["Slouching"],
            "uneven_shoulder_seconds": d["Uneven Shoulder"],
            "leaning_seconds": d["Leaning Left/Right"],
        }

    def percentages(self):
        # Denominator = valid monitoring time (Good + Warning + Poor == monitored_seconds).
        good = self.durations["Good Posture"]
        warning = self.durations["Warning"]
        poor = self.durations["Poor"]
        total = good + warning + poor
        if total <= 1e-9:
            return {"Good Posture": 0.0, "Warning": 0.0, "Poor": 0.0}
        return {
            "Good Posture": round(good / total * 100, 1),
            "Warning": round(warning / total * 100, 1),
            "Poor": round(poor / total * 100, 1),
        }

    def save_csv(self, folder: str | os.PathLike | None = None) -> str:
        if folder is None:
            folder = Path(__file__).resolve().parent / "logs"
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"posture_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        p = self.percentages()

        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["session_seconds", round(self.elapsed, 2)])
            w.writerow(["average_posture_score", self.average_score])  # time-weighted over monitored time
            w.writerow(["good_percent", p["Good Posture"]])
            w.writerow(["warning_percent", p["Warning"]])
            w.writerow(["poor_percent", p["Poor"]])
            for k, v in self.durations.items():
                w.writerow([f"duration_{k}", round(v, 2)])
            # Added rows (older readers can ignore them): where the session time went.
            w.writerow(["monitored_seconds", round(self.monitored_seconds, 2)])
            w.writerow(["partial_data_seconds", round(self.partial_seconds, 2)])
            w.writerow(["insufficient_data_seconds", round(self.insufficient_seconds, 2)])
            w.writerow(["active_break_seconds", round(self.break_seconds, 2)])
            w.writerow(["stopped_seconds", round(self.paused_total, 2)])
        return str(path)
