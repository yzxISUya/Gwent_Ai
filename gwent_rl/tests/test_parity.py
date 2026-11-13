"""
Cross-language parity test and throughput benchmark between
`gwent_rl.engine.GwentEngine` (Python) and `gwent_web/src/engine/engine.js` (Node.js).
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gwent_rl.engine import (
    GwentEngine,
    HeuristicAI,
    PRESET_LIST,
    CORE_DECK_PAIRS,
    OBS_DIM,
    ACTION_DIM,
)
from gwent_rl.engine.rng import RngState

ROOT = Path(__file__).resolve().parents[2]


def run_parity_check(num_games: int = 100) -> None:
    preset_ids = [p["id"] for p in PRESET_LIST]
    rng = RngState(20261106)
    bot_p0 = HeuristicAI("normal", seed=101)
    bot_p1 = HeuristicAI("hard", seed=202)

    test_cases = []
    total_steps = 0

    for g in range(num_games):
        seed = 1000 + g
        p0 = preset_ids[g % len(preset_ids)]
        p1 = preset_ids[(g * 3 + 1) % len(preset_ids)]
        eng = GwentEngine(seed=seed, preset0=p0, preset1=p1)

        actions = []
        snapshots = []

        step_idx = 0
        while not eng.done and step_idx < 300:
            mask = eng.get_legal_mask()
            # Record snapshot on key steps (start, middle, medic/decoy, end)
            if step_idx in (0, 3, 10, 20) or eng.sub_phase in ("MEDIC", "DECOY"):
                obs = eng.encode_observation()
                snapshots.append(
                    {
                        "step": step_idx,
                        "rngState": eng.rng.state,
                        "round": eng.round,
                        "turn": eng.turn,
                        "subPhase": eng.sub_phase,
                        "scores": list(eng.scores),
                        "gems": list(eng.gems),
                        "mask": mask,
                        "obsSum": float(np.sum(obs)),
                        "obsMax": float(np.max(obs)),
                        "obs": [float(x) for x in obs],
                    }
                )

            # Mix heuristic and random actions to cover all edge cases
            if g % 2 == 0:
                legal = [i for i, ok in enumerate(mask) if ok]
                action = legal[rng.rand_int(len(legal))]
            else:
                action = bot_p0.select_action(eng) if eng.turn == 0 else bot_p1.select_action(eng)

            actions.append(action)
            eng.step(action)
            step_idx += 1
            total_steps += 1

        test_cases.append(
            {
                "seed": seed,
                "preset0": p0,
                "preset1": p1,
                "actions": actions,
                "finalScores": list(eng.scores),
                "finalGems": list(eng.gems),
                "finalWinner": eng.winner,
                "finalRng": eng.rng.state,
                "roundHistory": eng.round_history,
                "snapshots": snapshots,
            }
        )

    node_script = """
import fs from 'node:fs';
import { GwentEngine, HeuristicAI } from './gwent_web/src/engine/engine.js';

const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
let checkedSteps = 0;
let checkedSnapshots = 0;
const jsBotP0 = new HeuristicAI('normal', 101);
const jsBotP1 = new HeuristicAI('hard', 202);

for (let cIdx = 0; cIdx < cases.length; cIdx++) {
  const tc = cases[cIdx];
  const eng = new GwentEngine(tc.seed, tc.preset0, tc.preset1);
  const snapMap = new Map(tc.snapshots.map(s => [s.step, s]));

  for (let s = 0; s < tc.actions.length; s++) {
    const snap = snapMap.get(s);
    if (snap) {
      checkedSnapshots++;
      const mask = eng.getLegalMask();
      const obs = eng.encodeObservation();
      if (eng.rng.state !== snap.rngState) {
        throw new Error(`Case ${cIdx} step ${s}: rngState mismatch JS=${eng.rng.state} PY=${snap.rngState}`);
      }
      if (eng.round !== snap.round || eng.turn !== snap.turn || eng.sub_phase !== snap.subPhase) {
        throw new Error(`Case ${cIdx} step ${s}: phase/turn mismatch`);
      }
      if (eng.scores[0] !== snap.scores[0] || eng.scores[1] !== snap.scores[1]) {
        throw new Error(`Case ${cIdx} step ${s}: scores mismatch JS=${eng.scores} PY=${snap.scores}`);
      }
      for (let i = 0; i < mask.length; i++) {
        if (mask[i] !== snap.mask[i]) {
          throw new Error(`Case ${cIdx} step ${s}: mask[${i}] mismatch JS=${mask[i]} PY=${snap.mask[i]}`);
        }
      }
      let obsSum = 0;
      for (let i = 0; i < obs.length; i++) obsSum += obs[i];
      if (Math.abs(obsSum - snap.obsSum) > 1e-3) {
        throw new Error(`Case ${cIdx} step ${s}: obsSum mismatch JS=${obsSum} PY=${snap.obsSum}`);
      }
      if (obs.length !== snap.obs.length) {
        throw new Error(`Case ${cIdx} step ${s}: obs length mismatch JS=${obs.length} PY=${snap.obs.length}`);
      }
      for (let i = 0; i < snap.obs.length; i++) {
        if (Math.abs(obs[i] - snap.obs[i]) > 1e-5) {
          throw new Error(`Case ${cIdx} step ${s}: obs[${i}] mismatch JS=${obs[i]} PY=${snap.obs[i]}`);
        }
      }
    }
    if (cIdx % 2 === 1) {
      const jsAct = eng.turn === 0 ? jsBotP0.selectAction(eng) : jsBotP1.selectAction(eng);
      if (jsAct !== tc.actions[s]) {
        throw new Error(`Case ${cIdx} step ${s}: HeuristicAI action mismatch JS=${jsAct} PY=${tc.actions[s]}`);
      }
    }
    eng.step(tc.actions[s]);
    checkedSteps++;
  }

  if (eng.winner !== tc.finalWinner || eng.rng.state !== tc.finalRng) {
    throw new Error(`Case ${cIdx} final mismatch: JS winner=${eng.winner} PY=${tc.finalWinner}`);
  }
}

console.log(`PARITY_OK: ${cases.length} games, ${checkedSteps} steps, ${checkedSnapshots} deep snapshots 100% matched!`);
"""
    tmp_json = ROOT / "gwent_rl" / "tests" / "_parity_cases.json"
    tmp_mjs = ROOT / "_run_parity_check.mjs"
    try:
        tmp_json.write_text(json.dumps(test_cases), encoding="utf-8")
        tmp_mjs.write_text(node_script, encoding="utf-8")
        res = subprocess.run(
            ["node", str(tmp_mjs), str(tmp_json)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        print(res.stdout.strip())
    finally:
        if tmp_json.exists():
            tmp_json.unlink()
        if tmp_mjs.exists():
            tmp_mjs.unlink()


def run_throughput_and_ai_benchmark() -> None:
    preset_ids = [p["id"] for p in PRESET_LIST]
    rng = RngState(999)
    eng = GwentEngine(seed=1, preset0=preset_ids[0], preset1=preset_ids[1])

    t0 = time.perf_counter()
    steps = 0
    games = 0
    while steps < 20000:
        eng.reset(
            seed=games + 1,
            preset0=preset_ids[games % len(preset_ids)],
            preset1=preset_ids[(games + 3) % len(preset_ids)],
        )
        while not eng.done:
            legal = eng.get_legal_actions()
            a = legal[rng.rand_int(len(legal))]
            eng.step(a)
            _ = eng.encode_observation()
            steps += 1
        games += 1
    dt = time.perf_counter() - t0
    print(f"Single-core Python throughput (with {OBS_DIM}-dim obs encoding): {steps / dt:,.0f} steps/sec ({games / dt:,.1f} games/sec)")

    # Benchmark Hard AI vs Normal AI vs Easy AI using Duplicate Paired-Seed Round-Robin
    # on Stage-1 Core 2x2 Deck Pool (nr-temeria & ng-white-flame, 200 duplicate pairs = 400 games per matchup)
    import math
    pairs = [("normal", "easy"), ("hard", "normal"), ("hard", "easy")]
    pair_scores = {}
    num_dup_pairs = 200  # 400 games per matchup
    for m_idx, (d_a, d_b) in enumerate(pairs):
        wins_a, wins_b, draws = 0, 0, 0
        dup_means = []
        for pair_idx in range(num_dup_pairs):
            p0, p1 = CORE_DECK_PAIRS[pair_idx % len(CORE_DECK_PAIRS)]
            game_seed = 50000 + m_idx * 10000 + pair_idx
            pair_pts = 0.0
            for side_a in (0, 1):
                bot_a = HeuristicAI(d_a, seed=1010 + pair_idx)
                bot_b = HeuristicAI(d_b, seed=2020 + pair_idx)
                eng.reset(seed=game_seed, preset0=p0, preset1=p1)
                while not eng.done:
                    if eng.turn == side_a:
                        a = bot_a.select_action(eng)
                    else:
                        a = bot_b.select_action(eng)
                    eng.step(a)
                if eng.winner == side_a:
                    wins_a += 1
                    pair_pts += 1.0
                elif eng.winner is None:
                    draws += 1
                    pair_pts += 0.5
                else:
                    wins_b += 1
            dup_means.append(pair_pts / 2.0)

        n_games = num_dup_pairs * 2
        wr_a = float(np.mean(dup_means))
        dup_se = float(np.std(dup_means, ddof=1) / math.sqrt(num_dup_pairs))
        elo_diff = -400.0 * math.log10(max(1e-4, (1.0 / max(1e-4, wr_a)) - 1.0))
        pair_scores[(d_a, d_b)] = (wins_a, wins_b, draws, wr_a, elo_diff, dup_se)
        print(
            f"Duplicate Matchup {d_a.upper():6s} vs {d_b.upper():6s} ({n_games} games / {num_dup_pairs} pairs): "
            f"{wins_a}W - {wins_b}L - {draws}D | Score={wr_a*100:.1f}% ± {dup_se*100:.1f}% (SE) | Empirical Elo Δ = +{elo_diff:.0f}"
        )

    elo_normal = 1200.0
    elo_easy = round(elo_normal - pair_scores[("normal", "easy")][4])
    elo_hard = round(elo_normal + pair_scores[("hard", "normal")][4])
    print(f"Calibrated 2x2 Core Pool Elos (anchored at Normal=1200): Easy={elo_easy}, Normal=1200, Hard={elo_hard}")


if __name__ == "__main__":
    print(f"OBS_DIM = {OBS_DIM}, ACTION_DIM = {ACTION_DIM}")
    run_parity_check(num_games=120)
    run_throughput_and_ai_benchmark()
