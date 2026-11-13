/**
 * Single-Observer Information Set Monte Carlo Tree Search (SO-IS-MCTS) for Gwent Web Worker.
 * Mirrors gwent_rl/agents/ismcts.py: strict non-cheating determinization from `pov` observation,
 * PUCT selection with availability-adjusted exploration, FPU, unseen-opp union masking, and
 * Masked PPO policy + value evaluation.
 */
import {
  ACTION_DIM,
  CARDS_BY_IDX,
  HORN_ROW_OFFSET,
  NUM_CARDS,
  PASS_ACTION,
  ROW_KEYS,
} from '../engine/data.js';
import { RngState } from '../engine/rng.js';

/**
 * Deduce the exact multiset of cards currently hidden in the opponent's
 * (hand + deck + mulligan_discards) strictly from public zones and POV's own zones.
 */
export function computeOppUnseenCounts(engine, pov) {
  const me = engine.players[pov];
  const opp = engine.players[1 - pov];

  const visCounts = new Int32Array(NUM_CARDS);
  for (const p of engine.players) {
    for (const r of ROW_KEYS) {
      for (const inst of p.rows[r]) {
        visCounts[inst.card_idx] += 1;
      }
      if (p.row_horn[r] !== null && p.row_horn[r].card_idx >= 0) {
        visCounts[p.row_horn[r].card_idx] += 1;
      }
    }
    for (const inst of p.graveyard) {
      visCounts[inst.card_idx] += 1;
    }
  }
  for (const wInst of engine.weather_cards) {
    visCounts[wInst.card_idx] += 1;
  }

  const myUnseen = new Int32Array(NUM_CARDS);
  for (const inst of me.hand) myUnseen[inst.card_idx] += 1;
  for (const inst of me.deck) myUnseen[inst.card_idx] += 1;
  for (const inst of me.mulligan_discards) myUnseen[inst.card_idx] += 1;

  const oppUnseen = new Int32Array(NUM_CARDS);
  for (let c = 0; c < NUM_CARDS; c++) {
    const rem = me.initial_counts[c] + opp.initial_counts[c] - visCounts[c] - myUnseen[c];
    if (rem > 0) oppUnseen[c] = rem;
  }
  return oppUnseen;
}

/**
 * Clone `engine` and resample all hidden opponent/deck info strictly consistent
 * with `pov`'s observation, with an isolated simulation RNG.
 */
export function determinizeEngine(engine, pov, rng) {
  const sim = engine.clone();
  const oppIdx = 1 - pov;
  const opp = sim.players[oppIdx];
  const me = sim.players[pov];

  const hOpp = opp.hand.length;
  const dOpp = opp.deck.length + opp.mulligan_discards.length;
  const oppUnseenCounts = computeOppUnseenCounts(sim, pov);

  let pool = [];
  for (let cidx = 0; cidx < NUM_CARDS; cidx++) {
    const cnt = oppUnseenCounts[cidx];
    if (cnt <= 0) continue;
    const cid = CARDS_BY_IDX[cidx].id;
    for (let k = 0; k < cnt; k++) {
      sim.uid_seq += 1;
      pool.push({
        uid: sim.uid_seq,
        card_id: cid,
        card_idx: cidx,
        owner: oppIdx,
      });
    }
  }

  pool = rng.shuffle(pool);
  opp.hand = pool.slice(0, hOpp);
  opp.deck = pool.slice(hOpp, hOpp + dOpp);
  opp.mulligan_discards = [];

  if (me.deck.length > 0) {
    me.deck = rng.shuffle(me.deck);
  }

  sim.rng = new RngState(rng.randInt(0x7fffffff) + 1);
  return sim;
}

function createEdge(prior) {
  return {
    visits: 0,
    availVisits: 0,
    valueSum: 0.0,
    prior,
  };
}

function createNode(turn) {
  return {
    turn,
    expanded: false,
    valueEst: 0.0,
    priors: null,
    edges: new Map(),
    children: new Map(),
  };
}

export class GwentISMCTS {
  constructor(
    evalFn,
    {
      numSimulations = 96,
      maxDepth = 6,
      cPuct = 1.0,
      maxChildren = 5,
      fpuReduction = 0.15,
      skipProbThreshold = 0.96,
      seed = 42,
    } = {},
  ) {
    this.evalFn = evalFn;
    this.numSimulations = numSimulations;
    this.maxDepth = maxDepth;
    this.cPuct = cPuct;
    this.maxChildren = maxChildren;
    this.fpuReduction = fpuReduction;
    this.skipProbThreshold = skipProbThreshold;
    this.rng = new RngState(seed);
  }

  _terminalValue(eng, rootPov) {
    if (eng.winner === rootPov) return 1.0;
    if (eng.winner === 1 - rootPov) return -1.0;
    return 0.0;
  }

  async _evaluateNode(eng, rootPov, useUnseenOppUnion = false) {
    const turn = eng.turn;
    const obs = eng.encodeObservation(turn);
    const mask = eng.getLegalMask();
    if (useUnseenOppUnion && turn !== rootPov && eng.sub_phase === 'PLAY') {
      const unseen = computeOppUnseenCounts(eng, rootPov);
      for (let cidx = 0; cidx < NUM_CARDS; cidx++) {
        if (unseen[cidx] <= 0) continue;
        const c = CARDS_BY_IDX[cidx];
        if (c.id === 'commander-s-horn') {
          const pl = eng.players[turn];
          for (let rI = 0; rI < ROW_KEYS.length; rI++) {
            if (pl.row_horn[ROW_KEYS[rI]] === null) {
              mask[HORN_ROW_OFFSET + rI] = true;
            }
          }
        } else if (c.id === 'decoy') {
          if (eng._hasDecoyTarget(turn)) {
            mask[cidx] = true;
          }
        } else {
          mask[cidx] = true;
        }
      }
    }
    const { probs, value } = await this.evalFn(obs, mask);
    const valRoot = turn === rootPov ? Number(value) : -Number(value);
    return { probs, valRoot };
  }

  async search(engine, timeBudgetMs = 450) {
    const rootPov = engine.turn;
    const legal = engine.getLegalActions();
    if (!legal || legal.length === 0) {
      return {
        bestAction: PASS_ACTION,
        probs: new Float32Array(ACTION_DIM),
        value: 0.0,
        stats: { sims: 0, nnEvals: 0, elapsedMs: 0.0 },
      };
    }

    const { probs: rootProbs, valRoot: rootVal } = await this._evaluateNode(engine, rootPov, false);
    let nnEvals = 1;

    if (legal.length === 1 || this.numSimulations <= 0) {
      const probsOut = new Float32Array(ACTION_DIM);
      probsOut[legal[0]] = 1.0;
      return {
        bestAction: legal[0],
        probs: probsOut,
        value: rootVal,
        stats: { sims: 0, nnEvals, elapsedMs: 0.0 },
      };
    }

    const sortedLegal = legal.slice().sort((a, b) => rootProbs[b] - rootProbs[a]);
    if (rootProbs[sortedLegal[0]] >= this.skipProbThreshold) {
      const probsOut = new Float32Array(ACTION_DIM);
      for (const a of legal) probsOut[a] = rootProbs[a];
      return {
        bestAction: sortedLegal[0],
        probs: probsOut,
        value: rootVal,
        stats: { sims: 0, nnEvals, elapsedMs: 0.0 },
      };
    }

    const root = createNode(rootPov);
    root.expanded = true;
    root.valueEst = rootVal;
    root.priors = rootProbs;

    const rootCands = sortedLegal.slice(0, this.maxChildren);
    if (legal.includes(PASS_ACTION) && !rootCands.includes(PASS_ACTION) && engine.sub_phase === 'PLAY') {
      rootCands.push(PASS_ACTION);
    }
    for (const a of rootCands) {
      root.edges.set(a, createEdge(Number(rootProbs[a])));
    }

    const tStart = performance.now();
    let simsDone = 0;

    for (let simIdx = 0; simIdx < this.numSimulations; simIdx++) {
      if (timeBudgetMs !== null && simIdx > 0) {
        const elapsedMs = performance.now() - tStart;
        if (elapsedMs >= timeBudgetMs) break;
      }

      const simEng = determinizeEngine(engine, rootPov, this.rng);
      let node = root;
      const path = [];
      let depth = 0;
      let leafVal = 0.0;

      while (true) {
        if (simEng.done) {
          leafVal = this._terminalValue(simEng, rootPov);
          break;
        }
        if (depth >= this.maxDepth) {
          const evRes = await this._evaluateNode(simEng, rootPov, false);
          leafVal = evRes.valRoot;
          nnEvals += 1;
          break;
        }

        const simLegal = simEng.getLegalActions();
        if (!simLegal || simLegal.length === 0) {
          leafVal = this._terminalValue(simEng, rootPov);
          break;
        }

        if (!node.expanded) {
          const evRes = await this._evaluateNode(simEng, rootPov, true);
          leafVal = evRes.valRoot;
          nnEvals += 1;
          node.expanded = true;
          node.valueEst = leafVal;
          node.priors = evRes.probs;

          const candActions = simLegal
            .slice()
            .sort((a, b) => evRes.probs[b] - evRes.probs[a])
            .slice(0, this.maxChildren);
          if (simLegal.includes(PASS_ACTION) && !candActions.includes(PASS_ACTION) && simEng.sub_phase === 'PLAY') {
            candActions.push(PASS_ACTION);
          }
          for (const a of candActions) {
            node.edges.set(a, createEdge(Number(evRes.probs[a])));
          }
          break;
        }

        let availActions;
        if (node === root) {
          availActions = rootCands;
        } else {
          const edgeLimit = Math.max(this.maxChildren * 2, 16);
          for (const a of simLegal) {
            if (!node.edges.has(a) && node.edges.size < edgeLimit) {
              const pr = node.priors !== null ? Number(node.priors[a]) : 1.0 / Math.max(1, simLegal.length);
              node.edges.set(a, createEdge(Math.max(pr, 1e-3)));
            }
          }
          availActions = simLegal.filter((a) => node.edges.has(a));
          if (availActions.length === 0) {
            availActions = simLegal.slice(0, this.maxChildren);
            for (const a of availActions) {
              node.edges.set(a, createEdge(1.0 / Math.max(1, availActions.length)));
            }
          }
        }

        const isRootPlayer = simEng.turn === rootPov;
        for (const a of availActions) {
          node.edges.get(a).availVisits += 1;
        }

        let priorSum = 0.0;
        for (const a of availActions) {
          priorSum += node.edges.get(a).prior;
        }
        const useUniformP = priorSum <= 1e-8;
        if (useUniformP) priorSum = availActions.length;

        const parentQTurn = isRootPlayer ? node.valueEst : -node.valueEst;
        const fpuVal = parentQTurn - this.fpuReduction;

        let bestA = availActions[0];
        let bestScore = -1e18;
        for (const a of availActions) {
          const edge = node.edges.get(a);
          let qEst;
          if (edge.visits === 0) {
            qEst = fpuVal;
          } else {
            const qRoot = edge.valueSum / edge.visits;
            qEst = isRootPlayer ? qRoot : -qRoot;
          }
          const localP = useUniformP ? 1.0 / priorSum : edge.prior / priorSum;
          const uEst = (this.cPuct * localP * Math.sqrt(Math.max(1, edge.availVisits))) / (1.0 + edge.visits);
          const sc = qEst + uEst;
          if (sc > bestScore) {
            bestScore = sc;
            bestA = a;
          }
        }

        path.push([node, bestA]);
        simEng.step(bestA);
        depth += 1;

        let nextNode = node.children.get(bestA);
        if (!nextNode) {
          nextNode = createNode(simEng.turn);
          node.children.set(bestA, nextNode);
        }
        node = nextNode;
      }

      for (const [pNode, pAct] of path) {
        const e = pNode.edges.get(pAct);
        e.visits += 1;
        e.valueSum += leafVal;
      }
      simsDone += 1;
    }

    const visitProbs = new Float32Array(ACTION_DIM);
    let totalV = 0;
    for (const e of root.edges.values()) totalV += e.visits;

    let bestAction;
    let searchVal;
    if (totalV > 0) {
      bestAction = rootCands[0];
      let bestTuple = [-1, -1e9, -1];
      for (const [a, e] of root.edges.entries()) {
        visitProbs[a] = e.visits / totalV;
        const qAvg = e.visits > 0 ? e.valueSum / e.visits : -1e9;
        if (
          e.visits > bestTuple[0] ||
          (e.visits === bestTuple[0] && (qAvg > bestTuple[1] || (qAvg === bestTuple[1] && e.prior > bestTuple[2])))
        ) {
          bestTuple = [e.visits, qAvg, e.prior];
          bestAction = a;
        }
      }
      const bestEdge = root.edges.get(bestAction);
      searchVal = bestEdge.visits > 0 ? bestEdge.valueSum / bestEdge.visits : rootVal;
    } else {
      for (const a of legal) visitProbs[a] = rootProbs[a];
      bestAction = sortedLegal[0];
      searchVal = rootVal;
    }

    const elapsedMs = performance.now() - tStart;
    return {
      bestAction,
      probs: visitProbs,
      value: searchVal,
      stats: { sims: simsDone, nnEvals, elapsedMs },
    };
  }
}
