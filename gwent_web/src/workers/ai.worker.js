/**
 * WebWorker for Asynchronous Gwent AI Inference & X-Ray Analysis.
 * Supports:
 *  1. ONNX Runtime Web Masked PPO models (`ppo-v1`)
 *  2. Built-in Heuristic AI (`heuristic-hard`, `heuristic-normal`, `heuristic-easy`)
 */
import * as ort from 'onnxruntime-web';
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm?url';
import {
  GwentEngine,
  HeuristicAI,
  OBS_DIM,
  ACTION_DIM,
} from '../engine/engine.js';
import { GwentISMCTS } from './ismcts.js';

// Use Vite-resolved local WASM URL alongside bundled JS glue (works in both dev and preview/build)
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = { wasm: ortWasmUrl };

const sessionCache = new Map();

async function runOnnxInference(sess, obsArr, maskBool) {
  const obsTensor = new ort.Tensor('float32', obsArr, [1, OBS_DIM]);
  const maskTensor = new ort.Tensor('bool', Uint8Array.from(maskBool, (b) => (b ? 1 : 0)), [1, ACTION_DIM]);
  let out = null;
  try {
    out = await sess.run({ obs: obsTensor, mask: maskTensor });
    const probs = new Float32Array(out.probs.data);
    const value = Number(out.value.data[0]);
    return { probs, value };
  } finally {
    if (obsTensor && typeof obsTensor.dispose === 'function') obsTensor.dispose();
    if (maskTensor && typeof maskTensor.dispose === 'function') maskTensor.dispose();
    if (out?.probs && typeof out.probs.dispose === 'function') out.probs.dispose();
    if (out?.value && typeof out.value.dispose === 'function') out.value.dispose();
  }
}

async function getOnnxSession(modelUrl) {
  if (sessionCache.has(modelUrl)) return sessionCache.get(modelUrl);
  const sess = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
  });
  sessionCache.set(modelUrl, sess);
  return sess;
}

let cachedEngine = null;
let cachedSeed = null;
let cachedPreset0 = null;
let cachedPreset1 = null;
let cachedHistory = [];

function reconstructEngine(payload) {
  const history = payload.actionHistory || [];
  const canReuse =
    cachedEngine !== null &&
    cachedSeed === payload.seed &&
    cachedPreset0 === payload.preset0 &&
    cachedPreset1 === payload.preset1 &&
    history.length >= cachedHistory.length &&
    cachedHistory.every((a, idx) => a === history[idx]);

  if (canReuse) {
    // Incrementally step only newly appended actions (O(1) per move instead of O(N^2) total replay)
    for (let i = cachedHistory.length; i < history.length; i++) {
      if (cachedEngine.done) break;
      cachedEngine.step(history[i]);
    }
  } else {
    // New game, reset, or branch: initialize fresh engine
    cachedEngine = new GwentEngine(payload.seed, payload.preset0, payload.preset1);
    cachedSeed = payload.seed;
    cachedPreset0 = payload.preset0;
    cachedPreset1 = payload.preset1;
    for (const act of history) {
      if (cachedEngine.done) break;
      cachedEngine.step(act);
    }
  }
  cachedHistory = history.slice();
  return cachedEngine;
}

function evaluateHeuristicDistribution(eng, difficulty = 'hard') {
  const bot = new HeuristicAI(difficulty, eng.rng.state ^ 0x13572468);
  const p = eng.turn;
  const legal = eng.getLegalActions();
  const scores = new Map();

  let bestAction = legal[0];
  if (legal.length === 1) {
    scores.set(legal[0], 10.0);
  } else {
    bestAction = bot.selectAction(eng);
    for (const a of legal) {
      const sim = eng.clone();
      sim.step(a);
      let v = bot.evaluateState(sim, p);
      if (a === bestAction) v += 18.0;
      scores.set(a, v);
    }
  }

  // Softmax over scores
  const vals = legal.map((a) => scores.get(a));
  const maxV = Math.max(...vals);
  const exps = vals.map((v) => Math.exp((v - maxV) / 12.0));
  const sumExp = exps.reduce((s, x) => s + x, 0) || 1.0;

  const probs = new Float32Array(ACTION_DIM);
  legal.forEach((a, i) => {
    probs[a] = exps[i] / sumExp;
  });

  const rawStateVal = bot.evaluateState(eng, p);
  const value = Math.tanh(rawStateVal / 45.0);
  return { bestAction, probs, value };
}

async function evaluateSpectatorP0WinRate(eng, spectatorUrl) {
  const sess = await getOnnxSession(spectatorUrl);
  const obs0 = eng.encodeObservation(0);
  const obs1 = eng.encodeObservation(1);
  const pairArr = new Float32Array(OBS_DIM * 2);
  pairArr.set(obs0, 0);
  pairArr.set(obs1, OBS_DIM);
  const pairTensor = new ort.Tensor('float32', pairArr, [1, OBS_DIM * 2]);
  let out = null;
  try {
    out = await sess.run({ obs_pair: pairTensor });
    const p0WinProb = Math.max(0.0, Math.min(1.0, Number(out.p0_win_prob.data[0])));
    return {
      p0WinProb,
      p0WinRate: p0WinProb * 100.0,
      p0Value: 2.0 * p0WinProb - 1.0,
    };
  } finally {
    if (pairTensor && typeof pairTensor.dispose === 'function') pairTensor.dispose();
    if (out?.p0_win_prob && typeof out.p0_win_prob.dispose === 'function') out.p0_win_prob.dispose();
  }
}

self.onmessage = async (e) => {
  const { reqId, modelType, modelUrl, enableSpectator = false, spectatorUrl = null, statePayload } = e.data;
  try {
    const eng = reconstructEngine(statePayload);
    if (eng.done) {
      self.postMessage({ reqId, done: true });
      return;
    }

    const pov = eng.turn;
    const legal = eng.getLegalActions();
    let bestAction = legal[0];
    let probs = new Float32Array(ACTION_DIM);
    let value = 0.0;
    let engineUsed = modelType;

    if ((modelType.startsWith('onnx:') || modelType.startsWith('onnx-ismcts:')) && modelUrl) {
      try {
        const sess = await getOnnxSession(modelUrl);
        if (modelType.startsWith('onnx-ismcts:')) {
          const mcts = new GwentISMCTS(
            (obsArr, maskBool) => runOnnxInference(sess, obsArr, maskBool),
            {
              numSimulations: 96,
              maxDepth: 6,
              cPuct: 1.0,
              maxChildren: 5,
              fpuReduction: 0.15,
              skipProbThreshold: 0.96,
              seed: (2026 + cachedHistory.length * 131 + eng.rng.state) >>> 0,
            },
          );
          const searchRes = await mcts.search(eng, 450);
          bestAction = searchRes.bestAction;
          probs = searchRes.probs;
          value = searchRes.value;
          engineUsed =
            searchRes.stats.sims > 0
              ? `${modelType} (${searchRes.stats.sims} sims / ${searchRes.stats.elapsedMs.toFixed(0)}ms)`
              : `${modelType} (0-ply forced)`;
        } else {
          const inf = await runOnnxInference(sess, eng.encodeObservation(pov), eng.getLegalMask());
          probs = inf.probs;
          value = inf.value;
          let maxP = -1;
          for (const a of legal) {
            if (probs[a] > maxP) {
              maxP = probs[a];
              bestAction = a;
            }
          }
        }
      } catch (onnxErr) {
        console.warn('[ONNX Inference Fallback to Heuristic]', onnxErr);
        const fallback = evaluateHeuristicDistribution(eng, 'hard');
        bestAction = fallback.bestAction;
        probs = fallback.probs;
        value = fallback.value;
        engineUsed = `${modelType} (启发式兜底)`;
      }
    } else {
      const diff = modelType.replace('heuristic-', '') || 'hard';
      const res = evaluateHeuristicDistribution(eng, diff);
      bestAction = res.bestAction;
      probs = res.probs;
      value = res.value;
    }

    // Build Top-K candidate actions list for AI X-Ray Panel
    const candidates = legal.map((a) => ({
      action: a,
      label: eng.actionLabel(a, 'zh'),
      prob: Number(probs[a] || 0),
    }));
    candidates.sort((x, y) => y.prob - x.prob);

    // Isolated Spectator Win-Rate Evaluation (strictly zero effect on bestAction / probs / MCTS)
    let p0WinRate = null;
    let spectatorValue = null;
    let spectatorActive = false;
    let spectatorError = null;
    if (enableSpectator && spectatorUrl) {
      try {
        const specRes = await evaluateSpectatorP0WinRate(eng, spectatorUrl);
        p0WinRate = specRes.p0WinRate;
        spectatorValue = specRes.p0Value;
        spectatorActive = true;
      } catch (specErr) {
        spectatorError = String(specErr?.message || specErr);
        console.warn('[Spectator ONNX Error]', specErr);
      }
    }

    self.postMessage({
      reqId,
      pov,
      bestAction,
      bestLabel: eng.actionLabel(bestAction, 'zh'),
      value: Number(value),
      p0WinRate: p0WinRate === null ? null : Number(p0WinRate),
      spectatorValue: spectatorValue === null ? null : Number(spectatorValue),
      spectatorActive,
      spectatorError,
      topActions: candidates.slice(0, 6),
      engineUsed,
    });
  } catch (err) {
    self.postMessage({ reqId, error: String(err?.message || err) });
  }
};
