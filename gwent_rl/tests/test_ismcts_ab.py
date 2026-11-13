"""
Unit tests (non-cheating determinization invariants) and A/B strength benchmark
for Masked PPO (0-ply) vs Masked PPO + SO-IS-MCTS against Heuristic Hard.
"""
from __future__ import annotations
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from gwent_rl.engine import ALL_DECK_PAIRS, ALL_PRESET_IDS, GwentEngine, HeuristicAI
from gwent_rl.engine.rng import RngState
from gwent_rl.agents.ismcts import (
    GwentISMCTS,
    OnnxEvaluator,
    compute_opp_unseen_counts,
    determinize_engine,
)


def test_determinization_invariants():
    rng = RngState(12345)
    for g in range(12):
        eng = GwentEngine(
            seed=700 + g,
            preset0=ALL_PRESET_IDS[g % len(ALL_PRESET_IDS)],
            preset1=ALL_PRESET_IDS[(g + 3) % len(ALL_PRESET_IDS)],
        )
        for step_i in range(40):
            if eng.done:
                break
            pov = eng.turn
            opp = eng.players[1 - pov]
            unseen = compute_opp_unseen_counts(eng, pov)
            total_unseen = sum(unseen)
            expected_opp_hidden = len(opp.hand) + len(opp.deck) + len(opp.mulligan_discards)
            assert total_unseen == expected_opp_hidden, (
                f"Unseen multiset mismatch: {total_unseen} != {expected_opp_hidden} at step {step_i}"
            )

            sim = determinize_engine(eng, pov, rng)
            sim_opp = sim.players[1 - pov]
            assert len(sim_opp.hand) == len(opp.hand)
            assert len(sim_opp.deck) == len(opp.deck) + len(opp.mulligan_discards)
            assert sim_opp.mulligan_discards == []
            assert eng.encode_observation(pov).shape == sim.encode_observation(pov).shape
            assert np.allclose(eng.encode_observation(pov), sim.encode_observation(pov)), (
                "Determinization mutated POV observation!"
            )

            legal = eng.get_legal_actions()
            eng.step(legal[step_i % len(legal)])
    print("[OK] test_determinization_invariants passed (12 games, multiset & zero-leakage checks)")


def run_ab_benchmark(num_simulations: int = 96, max_depth: int = 6, games: int = 256):
    onnx_path = os.path.join(os.path.dirname(__file__), "..", "..", "gwent_web", "public", "models", "ppo_v1.onnx")
    ev = OnnxEvaluator(os.path.abspath(onnx_path))
    mcts = GwentISMCTS(
        eval_fn=ev,
        num_simulations=num_simulations,
        max_depth=max_depth,
        c_puct=1.0,
        max_children=5,
        fpu_reduction=0.15,
        skip_prob_threshold=0.96,
        seed=2026,
    )
    bot = HeuristicAI(difficulty="hard", seed=999)

    results = {"0ply": [], "ismcts": []}
    by_faction = {"0ply": {}, "ismcts": {}}
    search_ms_list = []
    nn_evals_list = []

    total_pairs_needed = max(1, games // 2)
    t0 = time.perf_counter()

    for mode in ("0ply", "ismcts"):
        for g_pair in range(total_pairs_needed):
            pair_idx = g_pair % len(ALL_DECK_PAIRS)
            round_idx = g_pair // len(ALL_DECK_PAIRS)
            pA, pB = ALL_DECK_PAIRS[pair_idx]
            seed = 60000 + pair_idx * 17 + round_idx * 1009
            for agent_seat in (0, 1):
                eng = GwentEngine(seed=seed, preset0=pA, preset1=pB)
                agent_preset = pA if agent_seat == 0 else pB
                af = agent_preset.split("-")[0]
                steps = 0
                while not eng.done and steps < 250:
                    if eng.turn == agent_seat:
                        if mode == "0ply":
                            obs = eng.encode_observation(agent_seat)
                            mask = np.array(eng.get_legal_mask(), dtype=bool)
                            probs, _ = ev(obs, mask)
                            legal = eng.get_legal_actions()
                            act = max(legal, key=lambda a: float(probs[a]))
                        else:
                            act, _, _, st = mcts.search(eng)
                            if st["sims"] > 0:
                                search_ms_list.append(st["elapsed_ms"])
                                nn_evals_list.append(st["nn_evals"])
                        eng.step(act)
                    else:
                        eng.step(bot.select_action(eng))
                    steps += 1
                sc = 1.0 if eng.winner == agent_seat else (0.5 if eng.winner is None else 0.0)
                results[mode].append(sc)
                by_faction[mode].setdefault(af, []).append(sc)

    dt = time.perf_counter() - t0
    for mode in ("0ply", "ismcts"):
        scores = np.array(results[mode], dtype=float)
        wr = float(scores.mean() * 100.0)
        se = float(scores.std(ddof=1) / np.sqrt(len(scores)) * 100.0)
        f_str = {k: f"{np.mean(v)*100:.1f}%" for k, v in sorted(by_faction[mode].items())}
        print(f"[{mode:6s}] {len(scores)} games vs Hard: {wr:.2f}% ± {se:.2f}% | Factions={f_str}")
    if search_ms_list:
        print(
            f"[IS-MCTS Cost] sims={num_simulations}, max_depth={max_depth} | "
            f"mean={np.mean(search_ms_list):.1f}ms/move (p95={np.percentile(search_ms_list, 95):.1f}ms), "
            f"mean_nn_evals={np.mean(nn_evals_list):.1f}/move | total_bench={dt:.1f}s"
        )


if __name__ == "__main__":
    test_determinization_invariants()
    run_ab_benchmark(num_simulations=96, max_depth=6, games=256)
