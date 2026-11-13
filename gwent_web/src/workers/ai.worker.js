/**
 * WebWorker for Asynchronous Gwent AI Inference & X-Ray Analysis.
 * Supports:
 *  1. ONNX Runtime Web Masked PPO models (`ppo-v1`)
 *  2. Built-in Heuristic AI (`heuristic-hard`, `heuristic-normal`, `heuristic-easy`)
 */
import * as ort from 'onnxruntime-web';
import {
  GwentEngine,
  HeuristicAI,
  OBS_DIM,
  ACTION_DIM,
} from '../engine/engine.js';

// Configure ONNX Runtime Web WASM paths via CDN fallback
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

const sessionCache = new Map();

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

  if (legal.length === 1) {
    scores.set(legal[0], 10.0);
  } else {
    const chosen = bot.selectAction(eng);
    for (const a of legal) {
      const sim = eng.clone();
      sim.step(a);
      let v = bot.evaluateState(sim, p);
      if (a === chosen) v += 18.0;
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
  const bestAction = bot.selectAction(eng);
  return { bestAction, probs, value };
}

self.onmessage = async (e) => {
  const { reqId, modelType, modelUrl, statePayload } = e.data;
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

    if (modelType.startsWith('onnx:') && modelUrl) {
      try {
        const sess = await getOnnxSession(modelUrl);
        const obsArr = eng.encodeObservation(pov);
        const maskBool = eng.getLegalMask();
        const obsTensor = new ort.Tensor('float32', obsArr, [1, OBS_DIM]);
        const maskTensor = new ort.Tensor('bool', Uint8Array.from(maskBool, (b) => (b ? 1 : 0)), [1, ACTION_DIM]);
        let out = null;
        try {
          out = await sess.run({ obs: obsTensor, mask: maskTensor });
          probs = new Float32Array(out.probs.data);
          value = Number(out.value.data[0]);
        } finally {
          // Dispose tensors if method exists
          if (obsTensor && typeof obsTensor.dispose === 'function') obsTensor.dispose();
          if (maskTensor && typeof maskTensor.dispose === 'function') maskTensor.dispose();
        }

        let maxP = -1;
        for (const a of legal) {
          if (probs[a] > maxP) {
            maxP = probs[a];
            bestAction = a;
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

    // Value in [-1, 1] from current player `pov`'s perspective -> P0 win rate in [0, 100]
    const p0Value = pov === 0 ? value : -value;
    const p0WinRate = Math.max(0, Math.min(100, (p0Value + 1.0) * 50.0));

    self.postMessage({
      reqId,
      pov,
      bestAction,
      bestLabel: eng.actionLabel(bestAction, 'zh'),
      value: Number(value),
      p0WinRate: Number(p0WinRate),
      topActions: candidates.slice(0, 6),
      engineUsed,
    });
  } catch (err) {
    self.postMessage({ reqId, error: String(err?.message || err) });
  }
};
