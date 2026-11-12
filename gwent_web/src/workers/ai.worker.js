/**
 * WebWorker for Asynchronous Gwent AI Inference & X-Ray Analysis.
 * Supports:
 *  1. ONNX Runtime Web models (`ppo-v1`, `dmc-v1`, etc.)
 *  2. Client-side ISMCTS (Information-Set Monte-Carlo Tree Search)
 *  3. Built-in Heuristic AI (`heuristic-hard`, `heuristic-normal`, `heuristic-easy`)
 */
import * as ort from 'onnxruntime-web';
import {
  GwentEngine,
  HeuristicAI,
  RngState,
  OBS_DIM,
  ACTION_DIM,
} from '../engine/engine.js';

// Configure ONNX Runtime Web WASM paths via CDN fallback if needed
ort.env.wasm.numThreads = 1;

const sessionCache = new Map();

async function getOnnxSession(modelUrl) {
  if (sessionCache.has(modelUrl)) return sessionCache.get(modelUrl);
  const sess = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
  });
  sessionCache.set(modelUrl, sess);
  return sess;
}

function reconstructEngine(payload) {
  const eng = new GwentEngine(payload.seed, payload.preset0, payload.preset1);
  for (const act of payload.actionHistory || []) {
    if (eng.done) break;
    eng.step(act);
  }
  return eng;
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

function runISMCTS(eng, simulations = 48) {
  const p = eng.turn;
  const legalRoot = eng.getLegalActions();
  // Use fast 0-ply 'normal' heuristic for MCTS rollouts to prevent 2800+ nested engine.clone() calls
  const bot = new HeuristicAI('normal', eng.rng.state ^ 0xabcdef);
  if (legalRoot.length <= 1) {
    return evaluateHeuristicDistribution(eng, 'hard');
  }

  const rng = new RngState(eng.rng.state ^ 0x98765432);
  const visits = new Map();
  const valSums = new Map();
  for (const a of legalRoot) {
    visits.set(a, 0);
    valSums.set(a, 0.0);
  }

  for (let simIdx = 0; simIdx < simulations; simIdx++) {
    // Determinize opponent hand & deck, preserving cards during mulligan
    const sim = eng.clone();
    sim.rng = new RngState(rng.randInt(0x7fffffff));
    const opp = sim.players[1 - p];
    const oppHandLen = opp.hand.length;
    const pool = rng.shuffle([...opp.hand, ...opp.deck, ...(opp.mulligan_discards || [])]);
    opp.hand = pool.slice(0, oppHandLen);
    opp.deck = pool.slice(oppHandLen);
    opp.mulligan_discards = [];
    sim.players[p].deck = rng.shuffle(sim.players[p].deck);

    // UCB1 selection at root
    const totalN = Math.max(1, simIdx);
    let bestA = legalRoot[0];
    let bestUcb = -1e9;
    for (const a of legalRoot) {
      const n = visits.get(a);
      const q = n > 0 ? valSums.get(a) / n : 0.0;
      const u = 1.4 * Math.sqrt(Math.log(totalN + 1) / (n + 1));
      if (q + u > bestUcb) {
        bestUcb = q + u;
        bestA = a;
      }
    }

    sim.step(bestA);
    let rolloutDepth = 0;
    while (!sim.done && rolloutDepth < 6) {
      const rAct = bot.selectAction(sim);
      sim.step(rAct);
      rolloutDepth++;
    }

    const leafVal = Math.tanh(bot.evaluateState(sim, p) / 45.0);
    visits.set(bestA, visits.get(bestA) + 1);
    valSums.set(bestA, valSums.get(bestA) + leafVal);
  }

  const probs = new Float32Array(ACTION_DIM);
  let bestAction = legalRoot[0];
  let maxVisits = -1;
  for (const a of legalRoot) {
    const v = visits.get(a);
    probs[a] = v / simulations;
    if (v > maxVisits) {
      maxVisits = v;
      bestAction = a;
    }
  }
  const value = maxVisits > 0 ? valSums.get(bestAction) / maxVisits : 0.0;
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
          obsTensor.dispose?.();
          maskTensor.dispose?.();
          out?.probs?.dispose?.();
          out?.value?.dispose?.();
        }

        let maxP = -1;
        for (const a of legal) {
          if (probs[a] > maxP) {
            maxP = probs[a];
            bestAction = a;
          }
        }
      } catch (onnxErr) {
        // Fallback to ISMCTS or Heuristic if .onnx file fails to load or inference fails
        try {
          const fallback = runISMCTS(eng, 48);
          bestAction = fallback.bestAction;
          probs = fallback.probs;
          value = fallback.value;
          engineUsed = `${modelType} (ISMCTS Fallback)`;
        } catch (ismctsErr) {
          const fallback = evaluateHeuristicDistribution(eng, 'hard');
          bestAction = fallback.bestAction;
          probs = fallback.probs;
          value = fallback.value;
          engineUsed = `${modelType} (Heuristic Fallback)`;
        }
      }
    } else if (modelType === 'ismcts') {
      const res = runISMCTS(eng, 56);
      bestAction = res.bestAction;
      probs = res.probs;
      value = res.value;
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
