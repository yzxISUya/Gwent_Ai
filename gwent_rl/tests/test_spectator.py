"""
Unit & verification tests for `spectator_v1.onnx`:
  1. Strict antisymmetric zero-sum property: P(P0) + P(P1) == 1 across all phases & matchups.
  2. Cross-language [obs(0), obs(1)] parity between Python and JS `GwentEngine`.
  3. Full-game curve smoothness check on `seed=20261106` (nr-temeria vs ng-white-flame).
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gwent_rl.engine import ALL_DECK_PAIRS, GwentEngine, OBS_DIM
from gwent_rl.agents.ismcts import OnnxEvaluator, SpectatorOnnxEvaluator


def _default_spectator_path() -> Path:
    web_path = ROOT / "gwent_web" / "public" / "models" / "spectator_v1.onnx"
    if web_path.exists():
        return web_path
    return ROOT / "gwent_rl" / "models" / "spectator_v1.onnx"


def test_zero_sum_antisymmetry() -> None:
    spec_path = _default_spectator_path()
    assert spec_path.exists(), f"Missing spectator model: {spec_path}"
    spec = SpectatorOnnxEvaluator(str(spec_path))

    max_err = 0.0
    init_probs = []
    for idx, (p0, p1) in enumerate(ALL_DECK_PAIRS):
        eng = GwentEngine(seed=10000 + idx, preset0=p0, preset1=p1)
        for step_i in range(6):
            if eng.done:
                break
            o0 = eng.encode_observation(0)
            o1 = eng.encode_observation(1)
            prob0 = spec(o0, o1)
            prob1 = spec(o1, o0)
            err = abs(prob0 + prob1 - 1.0)
            if err > max_err:
                max_err = err
            assert 0.0 <= prob0 <= 1.0 and 0.0 <= prob1 <= 1.0
            if step_i == 0:
                init_probs.append(prob0)
            legal = eng.get_legal_actions()
            eng.step(legal[step_i % len(legal)])

    assert max_err < 1e-4, f"Zero-sum antisymmetry violation: max_err={max_err}"
    print(
        f"[OK] test_zero_sum_antisymmetry passed | max |P0+P1-1| = {max_err:.2e} | "
        f"64-pair st=0 mean P0={np.mean(init_probs)*100:.2f}%"
    )


def test_obs_pair_js_parity() -> None:
    eng = GwentEngine(seed=20261106, preset0="nr-temeria", preset1="ng-white-flame")
    steps_data = []
    actions = []
    for st in range(16):
        if eng.done:
            break
        o0 = eng.encode_observation(0)
        o1 = eng.encode_observation(1)
        steps_data.append(
            {
                "step": st,
                "turn": eng.turn,
                "pair": np.concatenate([o0, o1], axis=0).tolist(),
            }
        )
        legal = eng.get_legal_actions()
        act = legal[st % len(legal)]
        actions.append(act)
        eng.step(act)

    node_code = """
import fs from 'node:fs';
import { GwentEngine } from './gwent_web/src/engine/engine.js';

const { actions, steps } = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const eng = new GwentEngine(20261106, 'nr-temeria', 'ng-white-flame');
let maxDiff = 0;
for (let s = 0; s < steps.length; s++) {
  const o0 = eng.encodeObservation(0);
  const o1 = eng.encodeObservation(1);
  const pair = new Float32Array(o0.length + o1.length);
  pair.set(o0, 0);
  pair.set(o1, o0.length);
  const pyPair = steps[s].pair;
  if (pair.length !== pyPair.length) {
    throw new Error(`Length mismatch at step ${s}: ${pair.length} != ${pyPair.length}`);
  }
  for (let i = 0; i < pair.length; i++) {
    const d = Math.abs(pair[i] - pyPair[i]);
    if (d > maxDiff) maxDiff = d;
    if (d > 1e-5) {
      throw new Error(`ObsPair mismatch at step ${s} idx ${i}: JS=${pair[i]} PY=${pyPair[i]}`);
    }
  }
  eng.step(actions[s]);
}
console.log(`OBS_PAIR_PARITY_OK: ${steps.length} steps, maxDiff=${maxDiff.toExponential(2)}`);
"""
    tmp_json = ROOT / "gwent_rl" / "tests" / "_spec_parity.json"
    tmp_mjs = ROOT / "_run_spec_parity.mjs"
    try:
        tmp_json.write_text(json.dumps({"actions": actions, "steps": steps_data}), encoding="utf-8")
        tmp_mjs.write_text(node_code, encoding="utf-8")
        res = subprocess.run(
            ["node", str(tmp_mjs), str(tmp_json)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"[OK] {res.stdout.strip()}")
    finally:
        if tmp_json.exists():
            tmp_json.unlink()
        if tmp_mjs.exists():
            tmp_mjs.unlink()


def test_seed_20261106_curve() -> None:
    spec_path = _default_spectator_path()
    ppo_path = ROOT / "gwent_web" / "public" / "models" / "ppo_v1.onnx"
    spec = SpectatorOnnxEvaluator(str(spec_path))
    ppo = OnnxEvaluator(str(ppo_path))

    eng = GwentEngine(seed=20261106, preset0="nr-temeria", preset1="ng-white-flame")
    ppo_wrs = []
    spec_wrs = []
    turns = []
    max_sum_err = 0.0

    print("\n--- Seed=20261106 (nr-temeria vs ng-white-flame) Curve Comparison ---")
    for st in range(120):
        if eng.done:
            break
        pov = eng.turn
        o0 = eng.encode_observation(0)
        o1 = eng.encode_observation(1)
        mask = np.array(eng.get_legal_mask(), dtype=bool)
        probs, val = ppo(o0 if pov == 0 else o1, mask)
        ppo_wr = ((val if pov == 0 else -val) + 1.0) * 50.0
        p0_spec = spec(o0, o1) * 100.0
        p1_spec = spec(o1, o0) * 100.0
        sum_err = abs(p0_spec + p1_spec - 100.0)
        if sum_err > max_sum_err:
            max_sum_err = sum_err

        ppo_wrs.append(ppo_wr)
        spec_wrs.append(p0_spec)
        turns.append(pov)
        if st < 16 or eng.round >= 2:
            print(
                f"  st={st:2d} R{eng.round} {eng.sub_phase:8s} P{pov} | "
                f"PPO={ppo_wr:5.1f}% | Spec P0={p0_spec:5.1f}% P1={p1_spec:5.1f}% (sum={p0_spec+p1_spec:.4f}%)"
            )

        legal = eng.get_legal_actions()
        act = max(legal, key=lambda a: float(probs[a]))
        eng.step(act)

    ppo_alt_deltas = [abs(ppo_wrs[i + 1] - ppo_wrs[i]) for i in range(len(turns) - 1) if turns[i] != turns[i + 1]]
    spec_alt_deltas = [abs(spec_wrs[i + 1] - spec_wrs[i]) for i in range(len(turns) - 1) if turns[i] != turns[i + 1]]
    print(
        f"[Curve Stats] winner=P{eng.winner} | max|P0+P1-100%|={max_sum_err:.2e} | "
        f"Turn-Switch Mean |ΔWR|: PPO={np.mean(ppo_alt_deltas):.2f}% vs Spectator={np.mean(spec_alt_deltas):.2f}%"
    )
    assert max_sum_err < 1e-3
    assert np.mean(spec_alt_deltas) < np.mean(ppo_alt_deltas)


if __name__ == "__main__":
    test_obs_pair_js_parity()
    if _default_spectator_path().exists():
        test_zero_sum_antisymmetry()
        test_seed_20261106_curve()
    else:
        print("[SKIP] spectator_v1.onnx not yet exported; ran JS/PY obs_pair parity only.")
