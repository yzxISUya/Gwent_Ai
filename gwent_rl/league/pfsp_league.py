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
    ema_learner_wr: float = 0.5


class PFSPLeague:
    def __init__(
        self,
        seed: int = 42,
        max_snapshots: int = 12,
        max_anchors: int = 4,
        ema_alpha: float = 0.02,
    ):
        self.rng = RngState(seed)
        self.max_snapshots = max_snapshots
        self.max_anchors = max_anchors
        self.ema_alpha = ema_alpha
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
            # Keep the earliest foundational anchor (index 0) and demote the next oldest anchor
            # into the sliding recent pool so milestone anchors span the full training horizon.
            demote_idx = 1 if len(anchor_snaps) > 1 else 0
            demoted_key = anchor_snaps.pop(demote_idx)
            self.members[demoted_key].is_anchor = False
            recent_snaps.append(demoted_key)

        max_recent = max(1, self.max_snapshots - len(anchor_snaps))
        while len(recent_snaps) >= max_recent and recent_snaps:
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
        PFSP weighting with EMA sliding win-rate (preventing early-training losses from
        permanently skewing weights of long-lived anchors) and strong-baseline floor:
        f(p) = (1 - p)^1.5 + 0.15 where p is the EMA learner win rate.
        Keeps heuristic_hard & heuristic_normal adequately represented to prevent forgetting.
        """
        keys = list(self.members.keys())
        weights = []
        for k in keys:
            m = self.members[k]
            learner_wr = float(np.clip(m.ema_learner_wr, 0.0, 1.0))
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

        m.ema_learner_wr = (1.0 - self.ema_alpha) * m.ema_learner_wr + self.ema_alpha * score

        elo_diff = float(np.clip((m.elo - self.learner_elo) / 400.0, -10.0, 10.0))
        expected = 1.0 / (1.0 + 10.0 ** elo_diff)
        delta = k_factor * (score - expected)
        self.learner_elo += delta
        if not opponent_name.startswith("heuristic_"):
            m.elo -= delta
