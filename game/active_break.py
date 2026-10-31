"""Active Break controller.

Runs a fixed, non-random sequence -- Stretch Challenge, Coin Collection, Stretch Challenge,
Coin Collection, ... -- until active_break_total_seconds of play time is used up, with a short
automatic transition screen between games and a results screen at the end.

Pure Python: time is passed in, pose comes from the existing camera/pose pipeline.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Union

from .body_pose import BodyPose
from .coin_game import CoinGame
from config import CONFIG
from .stretch_game import StretchGame

PHASE_NAMES = {"stretch": "Stretch Challenge", "coin": "Coin Collection"}
COMPLETE_TITLES = {"stretch": "Stretch Complete!", "coin": "Coin Collection Complete!"}
SEQUENCE = ("stretch", "coin")


@dataclass(frozen=True)
class Phase:
    kind: str
    duration: float


def build_schedule(total: float, stretch: float, coin: float) -> List[Phase]:
    """Alternate stretch/coin phases until the total play time is filled (the last one may be shorter)."""
    phases: List[Phase] = []
    played = 0.0
    while played < total - 1e-6:
        kind = SEQUENCE[len(phases) % len(SEQUENCE)]
        base = stretch if kind == "stretch" else coin
        if base <= 0:
            break
        duration = min(base, total - played)
        phases.append(Phase(kind, duration))
        played += duration
    return phases


class ActiveBreak:
    def __init__(self, now: float, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.schedule = build_schedule(
            CONFIG.active_break_total_seconds,
            CONFIG.stretch_duration_seconds,
            CONFIG.coin_game_duration_seconds,
        )
        self.total_seconds = sum(p.duration for p in self.schedule)
        self.phase_index = -1
        self.game: Optional[Union[StretchGame, CoinGame]] = None
        self.finished_games: List[Union[StretchGame, CoinGame]] = []
        self.played = 0.0
        self.phase_started = 0.0
        self.results_until = 0.0
        self.state = "transition"   # "transition" | "stretch" | "coin" | "results"
        self.state_until = now + CONFIG.active_break_transition_seconds
        self.transition_title = "Get Ready!"
        first = PHASE_NAMES[self.schedule[0].kind] if self.schedule else "Results"
        self.transition_subtitle = f"First: {first}"

    # ------------------------------------------------------------------ flow

    @property
    def phase(self) -> Optional[Phase]:
        return self.schedule[self.phase_index] if 0 <= self.phase_index < len(self.schedule) else None

    @property
    def playing(self) -> bool:
        return self.state in ("stretch", "coin")

    def update(self, now: float, pose: Optional[BodyPose]):
        if self.state == "transition":
            if now >= self.state_until:
                self._start_next_phase(now)
        elif self.playing:
            self.game.update(now, pose)
            if now - self.phase_started >= self.phase.duration:
                self._end_phase(now)

    def finish_early(self, now: float):
        if self.state == "results":
            return
        if self.playing:
            self.played += min(now - self.phase_started, self.phase.duration)
            self.finished_games.append(self.game)
            self.game = None
        self._show_results(now)

    def finished(self, now: float) -> bool:
        return self.state == "results" and now >= self.results_until

    def _start_next_phase(self, now: float):
        self.phase_index += 1
        if self.phase is None:
            self._show_results(now)
            return
        self.game = StretchGame() if self.phase.kind == "stretch" else CoinGame(rng=self.rng)
        self.state = self.phase.kind
        self.phase_started = now

    def _end_phase(self, now: float):
        self.played += self.phase.duration
        self.finished_games.append(self.game)
        self.game = None
        upcoming = self.phase_index + 1
        if upcoming >= len(self.schedule):
            self._show_results(now)
            return
        self.state = "transition"
        self.state_until = now + CONFIG.active_break_transition_seconds
        self.transition_title = COMPLETE_TITLES[self.phase.kind]
        self.transition_subtitle = f"Next: {PHASE_NAMES[self.schedule[upcoming].kind]}"

    def _show_results(self, now: float):
        self.state = "results"
        self.results_until = now + CONFIG.active_break_results_seconds

    # ------------------------------------------------------------------ timers

    def phase_remaining(self, now: float) -> float:
        return max(0.0, self.phase.duration - (now - self.phase_started)) if self.playing else 0.0

    def total_remaining(self, now: float) -> float:
        played = self.played + (min(now - self.phase_started, self.phase.duration) if self.playing else 0.0)
        return max(0.0, self.total_seconds - played)

    def transition_remaining(self, now: float) -> float:
        return max(0.0, self.state_until - now) if self.state == "transition" else 0.0

    def results_remaining(self, now: float) -> float:
        return max(0.0, self.results_until - now) if self.state == "results" else 0.0

    @property
    def round_label(self) -> str:
        return f"Round {max(1, self.phase_index + 1)} of {len(self.schedule)}"

    # ------------------------------------------------------------------ results

    def _games(self):
        return self.finished_games + ([self.game] if self.game is not None else [])

    @property
    def score(self) -> int:
        return sum(g.score for g in self._games())

    @property
    def stretches_completed(self) -> int:
        return sum(g.completed for g in self._games() if isinstance(g, StretchGame))

    @property
    def coins_collected(self) -> int:
        return sum(g.coins_collected for g in self._games() if isinstance(g, CoinGame))

    @property
    def bombs_avoided(self) -> int:
        return sum(g.bombs_avoided for g in self._games() if isinstance(g, CoinGame))

    @property
    def bomb_hits(self) -> int:
        return sum(g.bomb_hits for g in self._games() if isinstance(g, CoinGame))
