"""
Standard RL Environment wrappers for Gwent:
- `GwentSingleAgentEnv`: Single-agent perspective against a configurable opponent
  (HeuristicAI, historical policy checkpoint, or random bot). Compatible with Gymnasium API
  and tailored for Masked PPO + PFSP League self-play training.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    gym = None  # type: ignore
    spaces = None  # type: ignore
    _HAS_GYM = False

from gwent_rl.engine import (
    GwentEngine,
    HeuristicAI,
    PRESET_LIST,
    CORE_PRESET_IDS,
    OBS_DIM,
    ACTION_DIM,
)
from gwent_rl.engine.rng import RngState

_BaseEnv = gym.Env if _HAS_GYM else object


class GwentSingleAgentEnv(_BaseEnv):
    """
    Single-agent RL environment where the learning agent plays as `agent_player` (0 or 1,
    or randomized each episode) against an `opponent_policy` callable `fn(engine) -> action`.
    Defaults to the Stage-1 core 2-deck pool (`nr-temeria` & `ng-white-flame`, 2x2=4 combinations).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        opponent: str | Callable[[GwentEngine], int] = "normal",
        seed: int = 42,
        shaping_weight: float = 0.15,
        preset_ids: Optional[Tuple[str, ...] | list[str]] = None,
    ):
        super().__init__()
        self.rng = RngState(seed)
        self.preset_ids = list(preset_ids) if preset_ids is not None else list(CORE_PRESET_IDS)
        self.shaping_weight = shaping_weight
        self.engine = GwentEngine(seed=seed)
        self.agent_player: int = 0
        self.episode_Count: int = 0

        if isinstance(opponent, str):
            bot = HeuristicAI(difficulty=opponent, seed=seed + 777)
            self.opponent_fn: Callable[[GwentEngine], int] = bot.select_action
        else:
            self.opponent_fn = opponent

        if _HAS_GYM and spaces is not None:
            self.observation_space = spaces.Dict(
                {
                    "obs": spaces.Box(low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32),
                    "action_mask": spaces.MultiBinary(ACTION_DIM),
                }
            )
            self.action_space = spaces.Discrete(ACTION_DIM)

    def set_opponent(self, opponent: str | Callable[[GwentEngine], int]) -> None:
        if isinstance(opponent, str):
            bot = HeuristicAI(difficulty=opponent, seed=self.rng.rand_int(1_000_000))
            self.opponent_fn = bot.select_action
        else:
            self.opponent_fn = opponent

    def _potential(self) -> float:
        p = self.agent_player
        opp = 1 - p
        gem_diff = (self.engine.gems[p] - self.engine.gems[opp]) * 0.25
        card_diff = (len(self.engine.players[p].hand) - len(self.engine.players[opp].hand)) * 0.04
        return gem_diff + card_diff

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            self.rng = RngState(seed)
        self.episode_Count += 1

        p0 = (options or {}).get("preset0") or self.preset_ids[self.rng.rand_int(len(self.preset_ids))]
        p1 = (options or {}).get("preset1") or self.preset_ids[self.rng.rand_int(len(self.preset_ids))]
        self.agent_player = (options or {}).get("agent_player", self.rng.rand_int(2))

        game_seed = self.rng.rand_int(2**31 - 1)
        self.engine.reset(seed=game_seed, preset0=p0, preset1=p1)

        # Advance opponent turns until it's agent_player's turn or game over
        self._run_opponent_until_agent()
        obs_dict = self._get_obs()
        return obs_dict, {"agent_player": self.agent_player, "preset0": p0, "preset1": p1}

    def _run_opponent_until_agent(self) -> None:
        while not self.engine.done and self.engine.turn != self.agent_player:
            opp_action = self.opponent_fn(self.engine)
            self.engine.step(opp_action)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        obs = self.engine.encode_observation(pov=self.agent_player)
        mask = np.array(self.engine.get_legal_mask(), dtype=bool)
        return {"obs": obs, "action_mask": mask}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        prev_pot = self._potential()
        self.engine.step(int(action))
        self._run_opponent_until_agent()

        done = self.engine.done
        reward = 0.0
        if done:
            if self.engine.winner == self.agent_player:
                reward = 1.0
            elif self.engine.winner == (1 - self.agent_player):
                reward = -1.0
            else:
                reward = 0.0
        elif self.shaping_weight > 0.0:
            new_pot = self._potential()
            reward = self.shaping_weight * (new_pot - prev_pot)

        obs_dict = self._get_obs()
        info = {
            "winner": self.engine.winner,
            "scores": list(self.engine.scores),
            "gems": list(self.engine.gems),
            "round": self.engine.round,
        }
        return obs_dict, float(reward), done, False, info
