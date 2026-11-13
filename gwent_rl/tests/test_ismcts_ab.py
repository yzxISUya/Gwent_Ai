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
    SpectatorOnnxEvaluator,
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
    import argparse

    parser = argparse.ArgumentParser(description="IS-MCTS invariants & A/B benchmarks")
    parser.add_argument("--spectator-ab", action="store_true", help="Compare PPO leaf vs Spectator leaf in IS-MCTS")
    parser.add_argument("--games", type=int, default=128, help="Games per matchup for A/B benchmark")
    parser.add_argument("--sims", type=int, default=96)
    parser.add_argument("--depth", type=int, default=6)
    args = parser.parse_args()

    test_determinization_invariants()
    if args.spectator_ab:
        ppo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "gwent_web", "public", "models", "ppo_v1.onnx"))
        spec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "gwent_web", "public", "models", "spectator_v1.onnx"))
        ev = OnnxEvaluator(ppo_path)
        spec_ev = SpectatorOnnxEvaluator(spec_path)
        bot_hard = HeuristicAI(difficulty="hard", seed=999)

        def _play_match(mode_a: str, mode_b: str, n_games: int, seed_base: int) -> None:
            mcts_ppo = GwentISMCTS(eval_fn=ev, num_simulations=args.sims, max_depth=args.depth, c_puct=1.0, max_children=5, fpu_reduction=0.15, skip_prob_threshold=0.96, seed=2026)
            mcts_spec = GwentISMCTS(eval_fn=ev, spectator_eval_fn=spec_ev, num_simulations=args.sims, max_depth=args.depth, c_puct=1.0, max_children=5, fpu_reduction=0.15, skip_prob_threshold=0.96, seed=2026)

            def _act(mode: str, eng: GwentEngine, ms_acc: list[float]) -> int:
                if mode == "0ply":
                    obs = eng.encode_observation(eng.turn)
                    mask = np.array(eng.get_legal_mask(), dtype=bool)
                    probs, _ = ev(obs, mask)
                    legal = eng.get_legal_actions()
                    return max(legal, key=lambda a: float(probs[a]))
                if mode == "hard":
                    return bot_hard.select_action(eng)
                m = mcts_ppo if mode == "ismcts-ppo" else mcts_spec
                a, _, _, st = m.search(eng)
                if st["sims"] > 0:
                    ms_acc.append(st["elapsed_ms"])
                return a

            scores = []
            ms_a = []
            t0 = time.perf_counter()
            n_pairs = max(1, n_games // 2)
            for gp in range(n_pairs):
                pA, pB = ALL_DECK_PAIRS[gp % len(ALL_DECK_PAIRS)]
                seed = seed_base + (gp % len(ALL_DECK_PAIRS)) * 17 + (gp // len(ALL_DECK_PAIRS)) * 1009
                for seat_a in (0, 1):
                    eng = GwentEngine(seed=seed, preset0=pA, preset1=pB)
                    steps = 0
                    while not eng.done and steps < 250:
                        if eng.turn == seat_a:
                            eng.step(_act(mode_a, eng, ms_a))
                        else:
                            eng.step(_act(mode_b, eng, []))
                        steps += 1
                    scores.append(1.0 if eng.winner == seat_a else (0.5 if eng.winner is None else 0.0))
            arr = np.array(scores, dtype=float)
            wr = float(arr.mean() * 100.0)
            se = float(arr.std(ddof=1) / np.sqrt(len(arr)) * 100.0)
            ms_str = f"{np.mean(ms_a):.1f}ms/move" if ms_a else "0-ply"
            print(f"[{mode_a} vs {mode_b}] {len(arr)} games: {wr:.2f}% ± {se:.2f}% | {ms_str} | wall={time.perf_counter()-t0:.1f}s")

        _play_match("ismcts-ppo", "0ply", args.games, 70000)
        _play_match("ismcts-spec", "0ply", args.games, 70000)
        _play_match("ismcts-ppo", "hard", args.games, 80000)
        _play_match("ismcts-spec", "hard", args.games, 80000)
    else:
        run_ab_benchmark(num_simulations=args.sims, max_depth=args.depth, games=args.games)
