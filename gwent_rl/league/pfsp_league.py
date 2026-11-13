"""
Prioritized Fictitious Self-Play (PFSP) Opponent Pool & Elo Rating System.
Maintains built-in Heuristic Bots (Easy, Normal, Hard), permanent milestone
anchors, and a sliding window of recent historical policy snapshots to prevent
policy cycling and catastrophic forgetting.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional
import numpy as np

from gwent_rl.engine import GwentEngine, HeuristicAI
from gwent_rl.engine.rng import RngState


@dataclass
class LeagueMember:
    name: str
    policy_fn: Optional[Callable[[GwentEngine], int]] = None
    state_dict: Optional[Dict[str, Any]] = None
    deterministic: bool = False
    is_anchor: bool = False
    elo: float = 1200.0
    games_played: int = 0
    wins_against_learner: int = 0
    losses_against_learner: int = 0


class PFSPLeague:
    def __init__(self, seed: int = 42, max_snapshots: int = 12, max_anchors: int = 4):
        self.rng = RngState(seed)
        self.max_snapshots = max_snapshots
        self.max_anchors = max_anchors
        self.learner_elo: float = 1200.0
        self.members: Dict[str, LeagueMember] = {}

        # Register built-in Heuristic AI baselines (calibrated on Core Pool)
        diff_configs = [("easy", 932.0, 101), ("normal", 1200.0, 202), ("hard", 1328.0, 303)]
        for diff, init_elo, seed_offset in diff_configs:
            bot = HeuristicAI(difficulty=diff, seed=seed + seed_offset)
            self.members[f"heuristic_{diff}"] = LeagueMember(
                name=f"heuristic_{diff}",
                policy_fn=bot.select_action,
                is_anchor=True,
                elo=init_elo,
            )

    def add_snapshot(
        self,
        name: str,
        policy_fn: Optional[Callable[[GwentEngine], int]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
        deterministic: bool = False,
        is_anchor: bool = False,
    ) -> None:
        anchor_snaps = [k for k, m in self.members.items() if k.startswith("snap_") and m.is_anchor]
        recent_snaps = [k for k, m in self.members.items() if k.startswith("snap_") and not m.is_anchor]

        if is_anchor and len(anchor_snaps) >= self.max_anchors:
            # Promote to recent pool eviction rather than dropping earliest anchor, or drop oldest anchor only if needed
            is_anchor = False

        max_recent = max(1, self.max_snapshots - len(anchor_snaps))
        while not is_anchor and len(recent_snaps) >= max_recent and recent_snaps:
            oldest = recent_snaps.pop(0)
            del self.members[oldest]

        self.members[name] = LeagueMember(
            name=name,
            policy_fn=policy_fn,
            state_dict=state_dict,
            deterministic=deterministic,
            is_anchor=is_anchor,
            elo=self.learner_elo,
        )

    def sample_opponent(self) -> LeagueMember:
        """
        PFSP weighting with Laplace smoothing and strong-baseline floor:
        f(p) = (1 - p)^1.5 + 0.15 where p = (wins + 1) / (total + 2).
        Keeps heuristic_hard & heuristic_normal adequately represented to prevent forgetting.
        """
        keys = list(self.members.keys())
        weights = []
        for k in keys:
            m = self.members[k]
            total = m.wins_against_learner + m.losses_against_learner
            learner_wr = (m.losses_against_learner + 1.0) / (total + 2.0)
            w = ((1.0 - learner_wr) ** 1.5) + 0.15
            if k == "heuristic_hard":
                w = max(w, 0.35)
            elif k == "heuristic_normal":
                w = max(w, 0.25)
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
