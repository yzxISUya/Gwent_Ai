"""
Prioritized Fictitious Self-Play (PFSP) Opponent Pool & Elo Rating System.
Maintains built-in Heuristic Bots (Easy, Normal, Hard) plus historical snapshots
of the learning agent to prevent policy cycling and catastrophic forgetting.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import numpy as np

from gwent_rl.engine import GwentEngine, HeuristicAI
from gwent_rl.engine.rng import RngState


@dataclass
class LeagueMember:
    name: str
    policy_fn: Callable[[GwentEngine], int]
    elo: float = 1200.0
    games_played: int = 0
    wins_against_learner: int = 0
    losses_against_learner: int = 0


class PFSPLeague:
    def __init__(self, seed: int = 42, max_snapshots: int = 12):
        self.rng = RngState(seed)
        self.max_snapshots = max_snapshots
        self.learner_elo: float = 1200.0
        self.members: Dict[str, LeagueMember] = {}

        # Register built-in Heuristic AI baselines (empirically calibrated via 400-game Duplicate Round-Robin on 2x2 Core Pool)
        diff_configs = [("easy", 932.0, 101), ("normal", 1200.0, 202), ("hard", 1328.0, 303)]
        for diff, init_elo, seed_offset in diff_configs:
            bot = HeuristicAI(difficulty=diff, seed=seed + seed_offset)
            self.members[f"heuristic_{diff}"] = LeagueMember(
                name=f"heuristic_{diff}",
                policy_fn=bot.select_action,
                elo=init_elo,
            )

    def add_snapshot(self, name: str, policy_fn: Callable[[GwentEngine], int]) -> None:
        snap_keys = [k for k in self.members if k.startswith("snap_")]
        if len(snap_keys) >= self.max_snapshots:
            oldest = snap_keys[0]
            del self.members[oldest]
        self.members[name] = LeagueMember(
            name=name,
            policy_fn=policy_fn,
            elo=self.learner_elo,
        )

    def sample_opponent(self) -> LeagueMember:
        """
        PFSP weighting: prioritize opponents where learner win rate is near 50%
        or where learner is currently losing: f(p) = (1 - p)^1.5 + 0.15.
        """
        keys = list(self.members.keys())
        weights = []
        for k in keys:
            m = self.members[k]
            total = m.wins_against_learner + m.losses_against_learner
            if total < 5:
                w = 1.0
            else:
                learner_wr = m.losses_against_learner / total
                w = ((1.0 - learner_wr) ** 1.5) + 0.15
            weights.append(w)

        w_sum = max(1e-6, sum(weights))
        r = self.rng.next_float() * w_sum
        acc = 0.0
        for k, w in zip(keys, weights):
            acc += w
            if r <= acc:
                return self.members[k]
        return self.members[keys[-1]]

    def record_result(self, opponent_name: str, learner_won: Optional[bool], k_factor: float = 16.0) -> None:
        if opponent_name not in self.members:
            return
        m = self.members[opponent_name]
        m.games_played += 1
        if learner_won is True:
            m.losses_against_learner += 1
            score = 1.0
        elif learner_won is False:
            m.wins_against_learner += 1
            score = 0.0
        else:
            score = 0.5

        elo_diff = float(np.clip((m.elo - self.learner_elo) / 400.0, -10.0, 10.0))
        expected = 1.0 / (1.0 + 10.0 ** elo_diff)
        delta = k_factor * (score - expected)
        self.learner_elo += delta
        if not opponent_name.startswith("heuristic_"):
            m.elo -= delta
