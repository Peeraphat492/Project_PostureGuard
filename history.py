"""Local session history: one SQLite row per monitoring session (numbers only, never camera images).

* Location: a per-user application data folder (macOS: ~/Library/Application Support/Posture Guard), never the
  app bundle. POSTURE_GUARD_DATA_DIR overrides it (tests, portable installs).
* One app run = one session (Stop/Open Cameras pause and resume it). Its row is keyed by the session's unique id
  and upserted (INSERT ... ON CONFLICT DO UPDATE), so repeated saves never create duplicates.
* Writes run on a background thread in a transaction; the UI never waits on the disk except for the final
  save when the app closes. WAL journaling keeps the file consistent if the app is killed mid-write.
* A session still marked open when the app starts again (crash / forced quit) is closed with its last saved
  values and flagged as recovered. Existing rows are never deleted.
* Database problems are logged and reported through `error`; the app keeps monitoring without history.
"""
from __future__ import annotations

import logging
import os
import queue
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from config import CONFIG

log = logging.getLogger("posture_guard")

SCHEMA_VERSION = 1
STAT_COLUMNS = (
    "session_seconds", "monitored_seconds", "partial_seconds", "insufficient_seconds", "break_seconds",
    "stopped_seconds", "average_score", "good_percent", "warning_percent", "poor_percent",
    "good_seconds", "warning_seconds", "poor_seconds",
    "forward_head_seconds", "slouching_seconds", "uneven_shoulder_seconds", "leaning_seconds",
)
ISSUES = (  # (label, column) of the posture conditions shown on the dashboard
    ("Forward Head", "forward_head_seconds"),
    ("Slouching", "slouching_seconds"),
    ("Uneven Shoulder", "uneven_shoulder_seconds"),
    ("Leaning", "leaning_seconds"),
)
FILTERS = (("all", "All Sessions"), ("today", "Today"), ("7d", "Last 7 Days"), ("30d", "Last 30 Days"))

_CREATE = f"""
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started_at  REAL NOT NULL,      -- Unix time (shown in local time)
    updated_at  REAL NOT NULL,
    ended_at    REAL,               -- NULL while the session is open
    recovered   INTEGER NOT NULL DEFAULT 0,
    {", ".join(f"{c} REAL" for c in STAT_COLUMNS)}
);
CREATE INDEX IF NOT EXISTS sessions_started ON sessions(started_at);
"""


def default_data_dir() -> Path:
    override = os.environ.get("POSTURE_GUARD_DATA_DIR")
    if override:
        return Path(override)
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "Posture Guard"
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", home)) / "Posture Guard"
    return Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share")) / "posture-guard"


def filter_since(kind: str, now: Optional[datetime] = None) -> Optional[float]:
    """Start (Unix time) of a history filter, using local calendar days: today / last 7 / last 30 days."""
    days = {"today": 0, "7d": 6, "30d": 29}.get(kind)
    if days is None:
        return None
    now = now or datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return (midnight - timedelta(days=days)).timestamp()


@dataclass
class SessionRecord:
    id: str
    started_at: float
    updated_at: float
    ended_at: Optional[float]
    recovered: bool
    session_seconds: float = 0.0
    monitored_seconds: float = 0.0
    partial_seconds: float = 0.0
    insufficient_seconds: float = 0.0
    break_seconds: float = 0.0
    stopped_seconds: float = 0.0
    average_score: Optional[float] = None
    good_percent: float = 0.0
    warning_percent: float = 0.0
    poor_percent: float = 0.0
    good_seconds: float = 0.0
    warning_seconds: float = 0.0
    poor_seconds: float = 0.0
    forward_head_seconds: float = 0.0
    slouching_seconds: float = 0.0
    uneven_shoulder_seconds: float = 0.0
    leaning_seconds: float = 0.0

    @property
    def issues(self) -> List[Tuple[str, float]]:
        return [(label, getattr(self, column) or 0.0) for label, column in ISSUES]

    @property
    def top_issue(self) -> Optional[Tuple[str, float]]:
        """Posture condition detected for the longest time (None when none was detected)."""
        label, seconds = max(self.issues, key=lambda item: item[1])
        return (label, seconds) if seconds > 0 else None

    @property
    def score(self) -> Optional[int]:
        return None if self.average_score is None else round(self.average_score)


def summarize(records: List[SessionRecord]) -> dict:
    """Totals over sessions: count, valid monitoring time, average score weighted by monitoring time."""
    monitored = sum(r.monitored_seconds for r in records)
    scored = [(r.average_score, r.monitored_seconds) for r in records if r.average_score is not None]
    weight = sum(w for _, w in scored)
    return {
        "count": len(records),
        "monitored_seconds": monitored,
        "average_score": round(sum(s * w for s, w in scored) / weight) if weight > 0 else None,
    }


class HistoryStore:
    def __init__(self, path: Optional[os.PathLike] = None):
        self.path = Path(path) if path else default_data_dir() / "history.sqlite3"
        self.error: Optional[str] = None
        self._queue: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_CREATE)
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version > SCHEMA_VERSION:
                    raise sqlite3.DatabaseError(f"database schema {version} is newer than this app ({SCHEMA_VERSION})")
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                # Sessions left open by a crash / forced quit keep their last saved values.
                recovered = conn.execute(
                    "UPDATE sessions SET ended_at = updated_at, recovered = 1 WHERE ended_at IS NULL").rowcount
            if recovered:
                log.warning("Recovered %d unfinished session(s) in %s", recovered, self.path)
        except (OSError, sqlite3.Error) as exc:
            self._fail("open", exc)
            return
        self._thread = threading.Thread(target=self._writer, name="history-writer", daemon=True)
        self._thread.start()

    @property
    def available(self) -> bool:
        return self.error is None

    @contextmanager
    def _connect(self):
        """Short-lived connection; the block is one transaction (commit on success, rollback on error)."""
        conn = sqlite3.connect(self.path, timeout=5.0)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _fail(self, action: str, exc: BaseException):
        self.error = f"Session history is unavailable ({action} failed: {exc}). Data file: {self.path}"
        log.error("History %s failed for %s", action, self.path, exc_info=exc)

    # ------------------------------------------------------------------ writing

    def save(self, snapshot: dict, final: bool = False):
        """Queue an upsert of a session snapshot (SessionTracker.snapshot()). Sessions without enough valid
        monitoring are not stored. final=True marks the session as finished (app closing)."""
        if self._thread is None or snapshot.get("monitored_seconds", 0.0) < CONFIG.history_min_monitored_seconds:
            return
        self._queue.put((dict(snapshot), time.time(), final))

    def flush(self, timeout: float = 2.0) -> bool:
        """Wait until queued saves are written. Returns False on timeout."""
        if self._thread is None:
            return True
        done = threading.Event()
        self._queue.put(done)
        return done.wait(timeout)

    def close(self, timeout: float = 3.0):
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout)
            self._thread = None

    def _writer(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            if isinstance(item, threading.Event):
                item.set()
                continue
            try:
                self._upsert(*item)
            except (OSError, sqlite3.Error) as exc:
                self._fail("save", exc)

    def _upsert(self, snap: dict, now: float, final: bool):
        columns = ("id", "started_at", "updated_at", "ended_at") + STAT_COLUMNS
        values = [snap["id"], snap["started_at"], now, now if final else None] + [snap.get(c) for c in STAT_COLUMNS]
        # recovered = 0: a session that is saving is alive (it may have been flagged by another app instance)
        updates = ", ".join(f"{c} = excluded.{c}" for c in ("updated_at", "ended_at") + STAT_COLUMNS) + ", recovered = 0"
        with self._connect() as conn:  # one transaction: committed on success, rolled back on error
            conn.execute(
                f"INSERT INTO sessions ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}", values)

    # ------------------------------------------------------------------ reading

    def sessions(self, since: Optional[float] = None) -> List[SessionRecord]:
        """Stored sessions, newest first (optionally only those started at or after `since`)."""
        if self.error is not None and self._thread is None:
            return []
        try:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                if since is None:
                    rows = conn.execute("SELECT * FROM sessions ORDER BY started_at DESC").fetchall()
                else:
                    rows = conn.execute("SELECT * FROM sessions WHERE started_at >= ? ORDER BY started_at DESC",
                                        (since,)).fetchall()
        except (OSError, sqlite3.Error) as exc:
            self._fail("read", exc)
            return []
        records = [r for r in (self._record(row) for row in rows) if r is not None]
        if len(records) < len(rows):
            log.warning("History: skipped %d incomplete or damaged session row(s) in %s", len(rows) - len(records), self.path)
        return records

    @staticmethod
    def _record(row) -> Optional[SessionRecord]:
        """Row -> record. Rows without a valid start time or monitoring time are skipped (never shown as zeros);
        other missing values mean 'not recorded' (0 s / 0 %, or no score). Rows are never modified."""
        def number(value):
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None
        values = {k: row[k] for k in row.keys() if k in SessionRecord.__dataclass_fields__}
        if number(values.get("started_at")) is None or number(values.get("monitored_seconds")) is None:
            return None
        for key in STAT_COLUMNS + ("updated_at", "ended_at"):
            v = number(values.get(key))
            values[key] = v if v is not None or key in ("average_score", "ended_at") else 0.0
        values["updated_at"] = values["updated_at"] or values["started_at"]
        values["recovered"] = bool(values.get("recovered"))
        return SessionRecord(**values)
