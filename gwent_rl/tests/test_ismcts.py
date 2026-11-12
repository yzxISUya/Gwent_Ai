"""
Quick verification of ISMCTS (Information-Set Monte-Carlo Tree Search)
and GwentSingleAgentEnv.
"""
from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gwent_rl.engine import GwentEngine, HeuristicAI
from gwent_rl.env import GwentSingleAgentEnv
from gwent_rl.agents.ismcts import ISMCTSAgent


def test_env_and_ismcts() -> None:
    env = GwentSingleAgentEnv(opponent="normal", seed=123)
    obs_dict, info = env.reset()
    assert obs_dict["obs"].shape == (689,)
    assert obs_dict["action_mask"].shape == (91,)
    steps = 0
    done = False
    while not done:
        mask = obs_dict["action_mask"]
        legal = [i for i, ok in enumerate(mask) if ok]
        obs_dict, reward, done, _, info = env.step(legal[0])
        steps += 1
    print(f"GwentSingleAgentEnv episode finished in {steps} agent steps | winner={info['winner']} | final reward={reward}")

    # Test ISMCTS search
    eng = GwentEngine(seed=42, preset0="nr-temeria", preset1="ng-white-flame")
    ismcts = ISMCTSAgent(num_simulations=32, seed=42)
    best_a, probs, val = ismcts.search(eng)
    print(f"ISMCTS root search: best_action={best_a} ({probs[best_a]*100:.1f}% visits) | root_val={val:+.3f}")


if __name__ == "__main__":
    test_env_and_ismcts()
