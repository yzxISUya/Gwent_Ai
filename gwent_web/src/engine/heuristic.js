/**
 * Built-in Heuristic AI (Easy / Normal / Hard) for Gwent Web.
 */

import { RngState } from './rng.js';
import {
  CARDS_BY_IDX,
  NUM_CARDS,
  ROW_KEYS,
  HORN_ROW_OFFSET,
  USE_LEADER_ACTION,
  PASS_ACTION,
} from './data.js';

export class HeuristicAI {
  constructor(difficulty = 'normal', seed = 42) {
    this.difficulty = ['easy', 'normal', 'hard'].includes(difficulty) ? difficulty : 'normal';
    this.rng = new RngState(seed);
  }

  selectAction(engine) {
    const legal = engine.getLegalActions();
    if (!legal.length) return PASS_ACTION;
    if (legal.length === 1) return legal[0];

    const p = engine.turn;
    const me = engine.players[p];

    // Tier 1: Easy (Random / Novice)
    if (this.difficulty === 'easy') {
      if (engine.sub_phase === 'MULLIGAN') return PASS_ACTION;
      if (legal.length > 1 && legal.includes(PASS_ACTION)) {
        if (engine.players[1 - p].passed && engine.scores[p] > engine.scores[1 - p]) {
          if (this.rng.nextFloat() < 0.55) return PASS_ACTION;
        }
        if (engine.scores[p] <= engine.scores[1 - p] && me.hand.length > 4) {
          const nonPass = legal.filter((a) => a !== PASS_ACTION);
          return nonPass[this.rng.randInt(nonPass.length)];
        }
      }
      return legal[this.rng.randInt(legal.length)];
    }

    // Tier 2: Normal (0-ply Static Rule-Based Heuristic, No Forward Clone)
    if (this.difficulty === 'normal') {
      if (engine.sub_phase === 'MULLIGAN') return this._chooseMulliganNormal(engine, p, legal);
      if (engine.sub_phase === 'MEDIC') return this._chooseMedicNormal(engine, p, legal);
      if (engine.sub_phase === 'DECOY') return this._chooseDecoyTargetNormal(engine, p, legal);
      return this._choosePlayNormal(engine, p, legal);
    }

    // Tier 3: Hard (1-ply Forward Simulation via engine.clone() + Search)
    if (engine.sub_phase === 'MULLIGAN') return this._chooseMulliganHard(engine, p, legal);
    if (engine.sub_phase === 'MEDIC') return this._chooseMedicHard(engine, p, legal);
    if (engine.sub_phase === 'DECOY') return this._chooseDecoyTargetHard(engine, p, legal);
    return this._choosePlayHard(engine, p, legal);
  }

  // ── NORMAL TIER (0-ply Static Heuristics + Stochastic Noise) ──

  _chooseMulliganNormal(engine, p, legal) {
    for (const a of legal) {
      if (a === PASS_ACTION) continue;
      const c = CARDS_BY_IDX[a];
      if (c._is_unit && !c._is_spy && !c._is_medic && c.strength <= 1) return a;
    }
    return PASS_ACTION;
  }

  _chooseMedicNormal(engine, p, legal) {
    let bestA = legal[0];
    let bestVal = -999;
    for (const a of legal) {
      const c = CARDS_BY_IDX[a];
      const val = c.strength + this.rng.nextFloat() * 4.0;
      if (val > bestVal) {
        bestVal = val;
        bestA = a;
      }
    }
    return bestA;
  }

  _chooseDecoyTargetNormal(engine, p, legal) {
    return legal[this.rng.randInt(legal.length)];
  }

  _choosePlayNormal(engine, p, legal) {
    const me = engine.players[p];
    const opp = engine.players[1 - p];
    const myScore = engine.scores[p];
    const oppScore = engine.scores[1 - p];
    const winning =
      myScore > oppScore ||
      (myScore === oppScore && me.faction === 'nilfgaard' && opp.faction !== 'nilfgaard');

    if (opp.passed && winning) return PASS_ACTION;

    if (!opp.passed && engine.gems[p] === 2) {
      if (winning && myScore - oppScore >= 5 && this.rng.nextFloat() < 0.42) {
        return PASS_ACTION;
      }
      if (oppScore - myScore >= 38) {
        return PASS_ACTION;
      }
    }

    let bestAction = PASS_ACTION;
    let bestVal = -1e9;

    for (const a of legal) {
      const noise = this.rng.nextFloat() * 6.0;
      let val = 0;
      if (a === PASS_ACTION) {
        if (!winning && engine.gems[p] === 1 && legal.length > 1) continue;
        val = -4.0 + noise;
      } else if (a === USE_LEADER_ACTION) {
        val = 6.5 + noise;
      } else if (a >= HORN_ROW_OFFSET && a < HORN_ROW_OFFSET + 3) {
        const rKey = ROW_KEYS[a - HORN_ROW_OFFSET];
        const nUnits = me.rows[rKey].length;
        val = (nUnits >= 2 ? 8.0 : 3.0) + noise;
      } else {
        const c = CARDS_BY_IDX[a];
        if (c.id === 'decoy') {
          val = 2.5 + noise;
        } else if (c._is_spy) {
          val = (engine.gems[p] === 2 ? 5.0 - 0.5 * c.strength : 0.5) + noise;
        } else if (c.type === 'special') {
          val = (oppScore >= myScore ? 7.5 : 3.5) + noise;
        } else if (c.type === 'weather') {
          val = (oppScore > myScore ? 5.5 : 3.0) + noise;
        } else {
          val =
            c.strength +
            (c._is_medic ? 1.0 : 0) +
            (c._is_bond || c._is_morale || c._is_muster ? 1.0 : 0) +
            noise;
        }
      }
      if (val > bestVal) {
        bestVal = val;
        bestAction = a;
      }
    }
    return bestAction;
  }

  // ── HARD TIER (1-ply Forward Simulation via engine.clone() + Search) ──

  _chooseMulliganHard(engine, p, legal) {
    const pl = engine.players[p];
    const handCounts = {};
    for (const inst of pl.hand) {
      handCounts[inst.card_idx] = (handCounts[inst.card_idx] || 0) + 1;
    }
    let worstAction = PASS_ACTION;
    let worstScore = 0;
    for (const a of legal) {
      if (a === PASS_ACTION) continue;
      const c = CARDS_BY_IDX[a];
      if (c._is_muster && (handCounts[a] || 0) > 1) return a;
      if (c.type === 'weather' && (handCounts[a] || 0) > 1) return a;
      if (c._is_unit && !c._is_spy && !c._is_medic && c.strength <= 2) {
        const sc = 3 - c.strength;
        if (sc > worstScore) {
          worstScore = sc;
          worstAction = a;
        }
      }
    }
    return worstAction;
  }

  _chooseMedicHard(engine, p, legal) {
    let bestA = legal[0];
    let bestVal = -999;
    for (const a of legal) {
      const c = CARDS_BY_IDX[a];
      let val = 0;
      if (c._is_medic) val = 30 + c.strength;
      else if (c._is_spy) val = engine.gems[p] > 1 ? 16 - c.strength : -10;
      else val = c.strength + (c._is_bond || c._is_morale || c._is_muster ? 4.5 : 0);
      if (val > bestVal) {
        bestVal = val;
        bestA = a;
      }
    }
    return bestA;
  }

  _chooseDecoyTargetHard(engine, p, legal) {
    let bestA = legal[0];
    let bestVal = -999;
    for (const a of legal) {
      const c = CARDS_BY_IDX[a];
      let val = 0;
      if (c._is_spy) val = 50 + c.strength;
      else if (c._is_medic) val = 40;
      else val = 10 - c.strength;
      if (val > bestVal) {
        bestVal = val;
        bestA = a;
      }
    }
    return bestA;
  }

  evaluateState(engine, p) {
    return this._evaluateStateHard(engine, p);
  }

  _evaluateState(engine, p) {
    return this._evaluateStateHard(engine, p);
  }

  _evaluateStateHard(engine, p) {
    if (engine.done) {
      if (engine.winner === p) return 500;
      if (engine.winner === 1 - p) return -500;
      return 0;
    }
    const me = engine.players[p];
    const opp = engine.players[1 - p];
    const myScore = engine.scores[p];
    const oppScore = engine.scores[1 - p];
    const diff = myScore - oppScore;
    const gemDiff = (engine.gems[p] - engine.gems[1 - p]) * 38.0;
    const handDiff = (me.hand.length - opp.hand.length) * 10.5;
    let leadBonus = 0;
    if (
      myScore > oppScore ||
      (myScore === oppScore && me.faction === 'nilfgaard' && opp.faction !== 'nilfgaard')
    ) {
      leadBonus = opp.passed ? 14.0 : 5.0;
    }
    return diff + gemDiff + handDiff + leadBonus;
  }

  _choosePlayHard(engine, p, legal) {
    const me = engine.players[p];
    const opp = engine.players[1 - p];
    const myScore = engine.scores[p];
    const oppScore = engine.scores[1 - p];
    const winning =
      myScore > oppScore ||
      (myScore === oppScore && me.faction === 'nilfgaard' && opp.faction !== 'nilfgaard');

    if (opp.passed && winning) return PASS_ACTION;

    if (
      !opp.passed &&
      engine.gems[p] === 2 &&
      oppScore - myScore >= 20 &&
      me.hand.length <= opp.hand.length + 1
    ) {
      for (const a of legal) {
        if (a < NUM_CARDS && CARDS_BY_IDX[a]._is_spy) return a;
      }
      return PASS_ACTION;
    }

    for (const a of legal) {
      if (a < NUM_CARDS && CARDS_BY_IDX[a]._is_spy) {
        const cSpy = CARDS_BY_IDX[a];
        if (!opp.passed) return a;
        if (engine.gems[p] === 2 && (oppScore - myScore >= 15 || engine.gems[1 - p] === 1)) return a;
        if (myScore > oppScore + cSpy.strength) return a;
      }
    }

    for (const a of legal) {
      if (a < NUM_CARDS && CARDS_BY_IDX[a].id === 'decoy') {
        for (const r of ROW_KEYS) {
          for (const inst of me.rows[r]) {
            const c = CARDS_BY_IDX[inst.card_idx];
            if (c._is_spy || (c._is_medic && engine._hasRevivableGrave(p))) return a;
          }
        }
      }
    }

    let bestAction = PASS_ACTION;
    let bestVal = -1e9;
    const baseDiff = myScore - oppScore;

    for (const a of legal) {
      const jitter = this.rng.nextFloat() * 0.05;
      let val = 0;
      if (a === PASS_ACTION) {
        if (!winning && engine.gems[p] === 1 && legal.length > 1) continue;
        val = this._evaluateStateHard(engine, p) + jitter;
        if (winning && !opp.passed && myScore - oppScore >= 15 && engine.gems[p] === 2) {
          val += 9.0;
        } else {
          val -= 6.0;
        }
      } else {
        if (a < NUM_CARDS && CARDS_BY_IDX[a].id === 'decoy') continue;
        const sim = engine.clone();
        sim.step(a);
        let guard = 0;
        while (
          !sim.done &&
          sim.turn === p &&
          (sim.sub_phase === 'MEDIC' || sim.sub_phase === 'DECOY') &&
          guard < 5
        ) {
          const subLegal = sim.getLegalActions();
          if (!subLegal.length) break;
          const subA =
            sim.sub_phase === 'MEDIC'
              ? this._chooseMedicHard(sim, p, subLegal)
              : this._chooseDecoyTargetHard(sim, p, subLegal);
          sim.step(subA);
          guard++;
        }
        val = this._evaluateStateHard(sim, p) + jitter;

        if (a < NUM_CARDS && (CARDS_BY_IDX[a].type === 'special' || CARDS_BY_IDX[a].type === 'weather')) {
          const newDiff = sim.scores[p] - sim.scores[1 - p];
          if (newDiff <= baseDiff) val -= 25.0;
        }

        if (opp.passed && sim.scores[p] > sim.scores[1 - p]) {
          val += 28.0 - 0.35 * (sim.scores[p] - sim.scores[1 - p]);
          if (a < NUM_CARDS && CARDS_BY_IDX[a]._is_hero && engine.gems[1 - p] > 1) {
            val -= 8.0;
          }
        } else if (a < NUM_CARDS && CARDS_BY_IDX[a]._is_hero && engine.round === 1) {
          val -= 5.0;
        }
      }
      if (val > bestVal) {
        bestVal = val;
        bestAction = a;
      }
    }
    return bestAction;
  }
}
