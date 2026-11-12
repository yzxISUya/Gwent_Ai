/**
 * Isomorphic Gwent (The Witcher 3) Core Engine & RL Encoder in ES Modules.
 * Bit-for-bit identical in state transitions, PRNG, 91-dim Legal Action Mask,
 * and 689-dim Float32 Observation Vector to `gwent_rl/engine/engine.py`.
 */

import cardsJson from '../../../shared/data/cards.json' with { type: 'json' };
import leadersJson from '../../../shared/data/leaders.json' with { type: 'json' };
import factionsJson from '../../../shared/data/factions.json' with { type: 'json' };
import presetsJson from '../../../shared/data/presets.json' with { type: 'json' };
import schemaJson from '../../../shared/spec/schema.json' with { type: 'json' };

export const ROW_KEYS = ['close', 'ranged', 'siege'];
export const WEATHER_FOR_ROW = { close: 'frost', ranged: 'fog', siege: 'rain' };
export const STARTING_HAND = 10;
export const MULLIGAN_MAX = 2;
export const STARTING_GEMS = 2;
export const MAX_ROUNDS = 3;

export const SCHEMA = schemaJson;
export const NUM_CARDS = SCHEMA.numCards;
export const NUM_LEADERS = SCHEMA.numLeaders;
export const ACTION_DIM = SCHEMA.actionSpace.totalDim;
export const HORN_ROW_OFFSET = SCHEMA.actionSpace.hornRowOffset;
export const USE_LEADER_ACTION = SCHEMA.actionSpace.useLeaderAction;
export const PASS_ACTION = SCHEMA.actionSpace.passOrFinishAction;
export const OBS_DIM = 51 + 36 + 7 * NUM_CARDS;

export const FACTION_TO_IDX = {};
SCHEMA.factions.forEach((fid, i) => {
  FACTION_TO_IDX[fid] = i;
});

export const CARDS_BY_ID = {};
export const CARDS_BY_IDX = new Array(NUM_CARDS);
for (const [cid, raw] of Object.entries(cardsJson)) {
  const c = structuredClone(raw);
  const abMap = {};
  for (const a of c.abilities || []) abMap[a.id] = a;
  c._ab_map = abMap;
  c._is_unit = c.type === 'unit';
  c._is_hero = !!c.hero;
  c._is_spy = 'spy' in abMap;
  c._is_medic = 'medic' in abMap;
  c._is_bond = 'bond' in abMap;
  c._is_morale = 'morale' in abMap;
  c._is_horn_unit = c._is_unit && ('horn' in abMap);
  c._is_muster = 'muster' in abMap;
  c._muster_targets = new Set(abMap.muster?.targets || []);
  c._muster_from_hand = (abMap.muster?.from || ['deck']).includes('hand');
  c._primary_row = c.rows?.[0] || null;
  c._weather_kind = abMap.weather?.weather || null;
  CARDS_BY_ID[cid] = c;
  CARDS_BY_IDX[c.idx] = c;
}

export const LEADERS_BY_ID = {};
export const LEADERS_BY_IDX = new Array(NUM_LEADERS);
for (const [lid, raw] of Object.entries(leadersJson)) {
  const l = structuredClone(raw);
  const ab = l.ability[0];
  l._ab_id = ab.id;
  l._ab = ab;
  l._is_passive = !!ab.passive || ab.id === 'cancelLeader';
  LEADERS_BY_ID[lid] = l;
  LEADERS_BY_IDX[l.idx] = l;
}

export const FACTIONS_BY_ID = factionsJson;
export const PRESET_LIST = presetsJson;
export const PRESETS_BY_ID = {};
for (const p of PRESET_LIST) {
  PRESETS_BY_ID[p.id] = p;
}

export function expandPresetDeck(presetId) {
  const preset = PRESETS_BY_ID[presetId];
  const out = [];
  for (const entry of preset.cards) {
    for (let i = 0; i < (entry.count | 0); i++) out.push(entry.id);
  }
  return out;
}

// ────────────────────────────────────────────────────────────────
// Deterministic 32-bit PRNG (Mulberry32)
// ────────────────────────────────────────────────────────────────

export function hashSeed(seed) {
  if (typeof seed === 'number' && Number.isFinite(seed)) return seed >>> 0;
  const str = String(seed ?? 'gwent');
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

export class RngState {
  constructor(seed = 1) {
    this.state = hashSeed(seed);
  }

  clone() {
    const r = new RngState(0);
    r.state = this.state;
    return r;
  }

  nextFloat() {
    let t = (this.state = (this.state + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
    t = (t ^ (t + (Math.imul(t ^ (t >>> 7), t | 61) >>> 0))) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296.0;
  }

  randInt(n) {
    if (n <= 0) return 0;
    return Math.floor(this.nextFloat() * n);
  }

  shuffle(arr) {
    const out = arr.slice();
    for (let i = out.length - 1; i > 0; i--) {
      const j = this.randInt(i + 1);
      const tmp = out[i];
      out[i] = out[j];
      out[j] = tmp;
    }
    return out;
  }

  pickRandom(arr) {
    if (!arr || !arr.length) return null;
    return arr[this.randInt(arr.length)];
  }
}

// ────────────────────────────────────────────────────────────────
// Row Power Computation
// ────────────────────────────────────────────────────────────────

export function computeRowPowers(rowInsts, hasRowHorn, weatherActive, doubleSpy) {
  if (!rowInsts.length) {
    return { powers: [], total: 0, hornActive: hasRowHorn };
  }

  let hornSources = hasRowHorn ? 1 : 0;
  let moraleCount = 0;
  const bondCounts = new Map();

  for (const inst of rowInsts) {
    const c = CARDS_BY_IDX[inst.card_idx];
    if (c._is_horn_unit) hornSources += 1;
    if (c._is_morale) moraleCount += 1;
    if (c._is_bond) {
      bondCounts.set(inst.card_idx, (bondCounts.get(inst.card_idx) || 0) + 1);
    }
  }

  const hornActive = hornSources > 0;
  const powers = [];
  let total = 0;

  for (const inst of rowInsts) {
    const c = CARDS_BY_IDX[inst.card_idx];
    let power = c.strength;
    if (!c._is_hero) {
      if (weatherActive && power > 1) power = 1;
      if (doubleSpy && c._is_spy) power *= 2;
      if (c._is_bond) {
        const n = bondCounts.get(inst.card_idx) || 1;
        if (n > 1) power *= n;
      }
      const morale = moraleCount - (c._is_morale ? 1 : 0);
      if (morale > 0) power += morale;
      if (hornActive) {
        if (!c._is_horn_unit || hornSources > 1) power *= 2;
      }
    }
    powers.push(power);
    total += power;
  }

  return { powers, total, hornActive };
}

// ────────────────────────────────────────────────────────────────
// Core Gwent Engine
// ────────────────────────────────────────────────────────────────

export class GwentEngine {
  constructor(seed = 1, preset0 = 'nr-temeria', preset1 = 'ng-white-flame', _empty = false) {
    if (_empty) return;
    this.reset(seed, preset0, preset1);
  }

  clone() {
    const eng = new GwentEngine(1, 'nr-temeria', 'ng-white-flame', true);
    eng.rng = this.rng.clone();
    eng.uid_seq = this.uid_seq;
    eng.round = this.round;
    eng.turn = this.turn;
    eng.first_player = this.first_player;
    eng.sub_phase = this.sub_phase;
    eng.gems = [this.gems[0], this.gems[1]];
    eng.weather = { ...this.weather };
    eng.weather_cards = this.weather_cards.map((i) => ({ ...i }));
    eng.double_spy = this.double_spy;
    eng.players = [this._clonePlayer(this.players[0]), this._clonePlayer(this.players[1])];
    eng.scores = [this.scores[0], this.scores[1]];
    eng.row_scores = [{ ...this.row_scores[0] }, { ...this.row_scores[1] }];
    eng.round_history = this.round_history.map((h) => ({ ...h, scores: [...h.scores], gems: [...h.gems] }));
    eng.done = this.done;
    eng.winner = this.winner;
    return eng;
  }

  _clonePlayer(p) {
    return {
      index: p.index,
      faction: p.faction,
      leader_id: p.leader_id,
      leader_idx: p.leader_idx,
      leader_used: p.leader_used,
      leader_canceled: p.leader_canceled,
      draw_extra_start: p.draw_extra_start,
      hand: p.hand.map((i) => ({ ...i })),
      deck: p.deck.map((i) => ({ ...i })),
      graveyard: p.graveyard.map((i) => ({ ...i })),
      mulligan_discards: p.mulligan_discards.map((i) => ({ ...i })),
      mulligan_left: p.mulligan_left,
      rows: {
        close: p.rows.close.map((i) => ({ ...i })),
        ranged: p.rows.ranged.map((i) => ({ ...i })),
        siege: p.rows.siege.map((i) => ({ ...i })),
      },
      row_horn: {
        close: p.row_horn.close ? { ...p.row_horn.close } : null,
        ranged: p.row_horn.ranged ? { ...p.row_horn.ranged } : null,
        siege: p.row_horn.siege ? { ...p.row_horn.siege } : null,
      },
      passed: p.passed,
      pending_draws: p.pending_draws,
      keep_uid: p.keep_uid,
      initial_counts: p.initial_counts,
    };
  }

  reset(seed = 1, preset0 = 'nr-temeria', preset1 = 'ng-white-flame') {
    this.rng = new RngState(seed);
    this.uid_seq = 0;
    this.round = 1;
    this.gems = [STARTING_GEMS, STARTING_GEMS];
    this.weather = { frost: false, fog: false, rain: false };
    this.weather_cards = [];
    this.double_spy = false;
    this.round_history = [];
    this.done = false;
    this.winner = null;

    const presets = [PRESETS_BY_ID[preset0], PRESETS_BY_ID[preset1]];
    this.players = [];

    for (const pIdx of [0, 1]) {
      const pr = presets[pIdx];
      const cardIds = expandPresetDeck(pr.id);
      const initCounts = new Array(NUM_CARDS).fill(0);
      const insts = [];
      for (const cid of cardIds) {
        const cdata = CARDS_BY_ID[cid];
        const cidx = cdata.idx;
        initCounts[cidx] += 1;
        this.uid_seq += 1;
        insts.push({
          uid: this.uid_seq,
          card_id: cid,
          card_idx: cidx,
          owner: pIdx,
        });
      }
      const shuffledDeck = this.rng.shuffle(insts);
      this.players.push({
        index: pIdx,
        faction: pr.faction,
        leader_id: pr.leaderId,
        leader_idx: LEADERS_BY_ID[pr.leaderId].idx,
        leader_used: false,
        leader_canceled: false,
        draw_extra_start: false,
        hand: [],
        deck: shuffledDeck,
        graveyard: [],
        mulligan_discards: [],
        mulligan_left: MULLIGAN_MAX,
        rows: { close: [], ranged: [], siege: [] },
        row_horn: { close: null, ranged: null, siege: null },
        passed: false,
        pending_draws: 0,
        keep_uid: null,
        initial_counts: initCounts,
      });
    }

    this._applyLeaderPassives();

    for (let i = 0; i < STARTING_HAND; i++) {
      this._draw(0, 1);
      this._draw(1, 1);
    }

    for (const p of this.players) {
      if (p.draw_extra_start) this._draw(p.index, 1);
    }

    const st0 = this.players[0].faction === 'scoiatael';
    const st1 = this.players[1].faction === 'scoiatael';
    if (st0 !== st1) {
      this.first_player = st0 ? 0 : 1;
    } else {
      this.first_player = this.rng.randInt(2);
    }

    this.sub_phase = 'MULLIGAN';
    this.turn = 0;
    this.scores = [0, 0];
    this.row_scores = [
      { close: 0, ranged: 0, siege: 0 },
      { close: 0, ranged: 0, siege: 0 },
    ];
  }

  _applyLeaderPassives() {
    for (const p of this.players) {
      const ldata = LEADERS_BY_IDX[p.leader_idx];
      if (ldata._ab_id === 'cancelLeader') {
        p.leader_used = true;
        const opp = this.players[1 - p.index];
        opp.leader_canceled = true;
      }
    }
    for (const p of this.players) {
      if (p.leader_canceled) continue;
      const ldata = LEADERS_BY_IDX[p.leader_idx];
      const abId = ldata._ab_id;
      if (abId === 'doubleSpy') {
        this.double_spy = true;
        p.leader_used = true;
      } else if (abId === 'drawExtraStart') {
        p.draw_extra_start = true;
        p.leader_used = true;
      }
    }
  }

  _draw(pIdx, count) {
    const pl = this.players[pIdx];
    for (let i = 0; i < count; i++) {
      if (!pl.deck.length) break;
      pl.hand.push(pl.deck.shift());
    }
  }

  _recomputeScores() {
    for (const pIdx of [0, 1]) {
      const pl = this.players[pIdx];
      let total = 0;
      const rScores = {};
      for (const r of ROW_KEYS) {
        const hasHorn = pl.row_horn[r] !== null;
        const wActive = this.weather[WEATHER_FOR_ROW[r]];
        const { total: rTot } = computeRowPowers(pl.rows[r], hasHorn, wActive, this.double_spy);
        rScores[r] = rTot;
        total += rTot;
      }
      this.scores[pIdx] = total;
      this.row_scores[pIdx] = rScores;
    }
  }

  isLeaderAvailable(pIdx) {
    const pl = this.players[pIdx];
    if (pl.passed || pl.leader_used || pl.leader_canceled) return false;
    const ldata = LEADERS_BY_IDX[pl.leader_idx];
    if (ldata._is_passive) return false;
    const abId = ldata._ab_id;
    const ab = ldata._ab;
    if (abId === 'clearWeather') {
      return this.weather.frost || this.weather.fog || this.weather.rain;
    }
    if (abId === 'leaderHorn') {
      return pl.row_horn[ab.row] === null;
    }
    if (abId === 'playWeatherFromDeck') {
      const wKind = ab.weather;
      if (this.weather[wKind]) return false;
      return pl.deck.some((inst) => CARDS_BY_IDX[inst.card_idx]._weather_kind === wKind);
    }
    return false;
  }

  _hasDecoyTarget(pIdx) {
    const pl = this.players[pIdx];
    for (const r of ROW_KEYS) {
      for (const inst of pl.rows[r]) {
        if (!CARDS_BY_IDX[inst.card_idx]._is_hero) return true;
      }
    }
    return false;
  }

  _hasRevivableGrave(pIdx) {
    const pl = this.players[pIdx];
    for (const inst of pl.graveyard) {
      const c = CARDS_BY_IDX[inst.card_idx];
      if (c._is_unit && !c._is_hero) return true;
    }
    return false;
  }

  getLegalMask() {
    const mask = new Array(ACTION_DIM).fill(false);
    if (this.done || this.sub_phase === 'GAME_OVER') return mask;

    const pIdx = this.turn;
    const pl = this.players[pIdx];

    if (this.sub_phase === 'MULLIGAN') {
      for (const inst of pl.hand) mask[inst.card_idx] = true;
      mask[PASS_ACTION] = true;
      return mask;
    }

    if (this.sub_phase === 'MEDIC') {
      for (const inst of pl.graveyard) {
        const c = CARDS_BY_IDX[inst.card_idx];
        if (c._is_unit && !c._is_hero) mask[inst.card_idx] = true;
      }
      return mask;
    }

    if (this.sub_phase === 'DECOY') {
      for (const r of ROW_KEYS) {
        for (const inst of pl.rows[r]) {
          const c = CARDS_BY_IDX[inst.card_idx];
          if (c._is_unit && !c._is_hero) mask[inst.card_idx] = true;
        }
      }
      return mask;
    }

    // sub_phase === 'PLAY'
    mask[PASS_ACTION] = true;
    if (this.isLeaderAvailable(pIdx)) mask[USE_LEADER_ACTION] = true;

    let hasDecoyTgt = null;
    for (const inst of pl.hand) {
      const cidx = inst.card_idx;
      const c = CARDS_BY_IDX[cidx];
      const cid = c.id;
      if (cid === 'commander-s-horn') {
        ROW_KEYS.forEach((r, rI) => {
          if (pl.row_horn[r] === null) mask[HORN_ROW_OFFSET + rI] = true;
        });
      } else if (cid === 'decoy') {
        if (hasDecoyTgt === null) hasDecoyTgt = this._hasDecoyTarget(pIdx);
        if (hasDecoyTgt) mask[cidx] = true;
      } else {
        mask[cidx] = true;
      }
    }
    return mask;
  }

  getLegalActions() {
    const mask = this.getLegalMask();
    const out = [];
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) out.push(i);
    }
    return out;
  }

  step(action) {
    if (this.done) return { done: true, winner: this.winner };

    const pIdx = this.turn;
    const pl = this.players[pIdx];

    if (this.sub_phase === 'MULLIGAN') {
      if (action === PASS_ACTION) {
        this._finishPlayerMulligan(pIdx);
      } else {
        const idxInHand = pl.hand.findIndex((inst) => inst.card_idx === action);
        if (idxInHand < 0) throw new Error(`Illegal mulligan card_idx ${action}`);
        const [discarded] = pl.hand.splice(idxInHand, 1);
        pl.mulligan_discards.push(discarded);
        this._draw(pIdx, 1);
        pl.mulligan_left -= 1;
        if (pl.mulligan_left <= 0 || !pl.hand.length) {
          this._finishPlayerMulligan(pIdx);
        }
      }
      return { done: this.done, winner: this.winner };
    }

    if (this.sub_phase === 'MEDIC') {
      const idxInGrave = pl.graveyard.findIndex(
        (inst) =>
          inst.card_idx === action &&
          CARDS_BY_IDX[inst.card_idx]._is_unit &&
          !CARDS_BY_IDX[inst.card_idx]._is_hero
      );
      if (idxInGrave < 0) throw new Error(`Illegal medic target card_idx ${action}`);
      const [revived] = pl.graveyard.splice(idxInGrave, 1);
      this._playUnit(pIdx, revived);
      return { done: this.done, winner: this.winner };
    }

    if (this.sub_phase === 'DECOY') {
      let targetInst = null;
      for (const r of ROW_KEYS) {
        const i = pl.rows[r].findIndex(
          (inst) => inst.card_idx === action && !CARDS_BY_IDX[action]._is_hero
        );
        if (i >= 0) {
          [targetInst] = pl.rows[r].splice(i, 1);
          break;
        }
      }
      if (!targetInst) throw new Error(`Illegal decoy target card_idx ${action}`);
      targetInst.owner = pIdx;
      pl.hand.push(targetInst);
      this.sub_phase = 'PLAY';
      this._advanceTurn();
      return { done: this.done, winner: this.winner };
    }

    // sub_phase === 'PLAY'
    if (action === PASS_ACTION) {
      pl.passed = true;
      this._advanceTurn();
      return { done: this.done, winner: this.winner };
    }

    if (action === USE_LEADER_ACTION) {
      if (!this.isLeaderAvailable(pIdx)) throw new Error('Leader ability not available');
      pl.leader_used = true;
      const ldata = LEADERS_BY_IDX[pl.leader_idx];
      const abId = ldata._ab_id;
      const ab = ldata._ab;
      if (abId === 'clearWeather') {
        this._clearWeather();
      } else if (abId === 'leaderHorn') {
        const row = ab.row;
        if (pl.row_horn[row] === null) {
          pl.row_horn[row] = { uid: -1, card_id: null, card_idx: -1, owner: pIdx, leader: true };
        }
      } else if (abId === 'playWeatherFromDeck') {
        const wKind = ab.weather;
        const deckI = pl.deck.findIndex(
          (inst) => CARDS_BY_IDX[inst.card_idx]._weather_kind === wKind
        );
        if (deckI >= 0) {
          const [wInst] = pl.deck.splice(deckI, 1);
          this._applyWeatherCard(pIdx, wInst, wKind);
        }
      }
      this._advanceTurn();
      return { done: this.done, winner: this.winner };
    }

    if (action >= HORN_ROW_OFFSET && action < HORN_ROW_OFFSET + 3) {
      const rIdx = action - HORN_ROW_OFFSET;
      const row = ROW_KEYS[rIdx];
      if (pl.row_horn[row] !== null) throw new Error(`Row ${row} already has a horn`);
      const hornI = pl.hand.findIndex(
        (inst) => CARDS_BY_IDX[inst.card_idx].id === 'commander-s-horn'
      );
      if (hornI < 0) throw new Error('No commander-s-horn in hand');
      const [hornInst] = pl.hand.splice(hornI, 1);
      pl.row_horn[row] = hornInst;
      this._advanceTurn();
      return { done: this.done, winner: this.winner };
    }

    const handI = pl.hand.findIndex((inst) => inst.card_idx === action);
    if (handI < 0) throw new Error(`Card idx ${action} not in hand`);
    const [inst] = pl.hand.splice(handI, 1);
    const cdata = CARDS_BY_IDX[action];
    const cid = cdata.id;

    if (cid === 'decoy') {
      pl.graveyard.push(inst);
      this.sub_phase = 'DECOY';
      this._recomputeScores();
      return { done: this.done, winner: this.winner };
    }

    if (cdata.type === 'weather') {
      this._applyWeatherCard(pIdx, inst, cdata._weather_kind);
      this._advanceTurn();
      return { done: this.done, winner: this.winner };
    }

    if (cid === 'scorch') {
      pl.graveyard.push(inst);
      this._applyScorch();
      this._advanceTurn();
      return { done: this.done, winner: this.winner };
    }

    this._playUnit(pIdx, inst);
    return { done: this.done, winner: this.winner };
  }

  _finishPlayerMulligan(pIdx) {
    const pl = this.players[pIdx];
    if (pl.mulligan_discards.length) {
      pl.deck.push(...pl.mulligan_discards);
      pl.mulligan_discards = [];
      pl.deck = this.rng.shuffle(pl.deck);
    }
    pl.mulligan_left = 0;
    if (pIdx === 0) {
      this.turn = 1;
    } else {
      this._startRound(1);
    }
  }

  _playUnit(pIdx, inst) {
    const cdata = CARDS_BY_IDX[inst.card_idx];
    const row = cdata._primary_row;
    const boardOwner = cdata._is_spy ? 1 - pIdx : pIdx;
    inst.owner = boardOwner;
    this.players[boardOwner].rows[row].push(inst);

    if (cdata._is_spy) {
      this._draw(pIdx, 2);
    }

    if (cdata._is_muster) {
      const targets = cdata._muster_targets;
      const pl = this.players[pIdx];
      const summoned = [];

      const newDeck = [];
      for (const dInst of pl.deck) {
        if (targets.has(dInst.card_id)) summoned.push(dInst);
        else newDeck.push(dInst);
      }
      pl.deck = newDeck;

      if (cdata._muster_from_hand) {
        const newHand = [];
        for (const hInst of pl.hand) {
          if (targets.has(hInst.card_id)) summoned.push(hInst);
          else newHand.push(hInst);
        }
        pl.hand = newHand;
      }

      for (const sInst of summoned) {
        const sCard = CARDS_BY_IDX[sInst.card_idx];
        const sRow = sCard._primary_row;
        sInst.owner = pIdx;
        pl.rows[sRow].push(sInst);
      }
    }

    if (cdata._is_medic && this._hasRevivableGrave(pIdx)) {
      this.sub_phase = 'MEDIC';
      this._recomputeScores();
      return;
    }

    this.sub_phase = 'PLAY';
    this._advanceTurn();
  }

  _applyWeatherCard(pIdx, inst, wKind) {
    if (wKind === 'clear') {
      this.players[pIdx].graveyard.push(inst);
      this._clearWeather();
      return;
    }
    if (this.weather[wKind]) {
      this.players[pIdx].graveyard.push(inst);
      return;
    }
    this.weather[wKind] = true;
    inst.owner = pIdx;
    this.weather_cards.push(inst);
  }

  _clearWeather() {
    for (const wInst of this.weather_cards) {
      this.players[wInst.owner].graveyard.push(wInst);
    }
    this.weather_cards = [];
    this.weather.frost = false;
    this.weather.fog = false;
    this.weather.rain = false;
  }

  _applyScorch() {
    let maxPower = 0;
    const rowPowersMap = {};
    for (const pIdx of [0, 1]) {
      const pl = this.players[pIdx];
      for (const r of ROW_KEYS) {
        const { powers } = computeRowPowers(
          pl.rows[r],
          pl.row_horn[r] !== null,
          this.weather[WEATHER_FOR_ROW[r]],
          this.double_spy
        );
        rowPowersMap[`${pIdx}:${r}`] = powers;
        for (let i = 0; i < pl.rows[r].length; i++) {
          const inst = pl.rows[r][i];
          const pw = powers[i];
          if (!CARDS_BY_IDX[inst.card_idx]._is_hero && pw > maxPower) {
            maxPower = pw;
          }
        }
      }
    }

    if (maxPower <= 0) return;

    for (const pIdx of [0, 1]) {
      const pl = this.players[pIdx];
      for (const r of ROW_KEYS) {
        const powers = rowPowersMap[`${pIdx}:${r}`];
        const kept = [];
        for (let i = 0; i < pl.rows[r].length; i++) {
          const inst = pl.rows[r][i];
          const pw = powers[i];
          if (!CARDS_BY_IDX[inst.card_idx]._is_hero && pw === maxPower) {
            pl.graveyard.push(inst);
          } else {
            kept.push(inst);
          }
        }
        pl.rows[r] = kept;
      }
    }
  }

  _canAct(pIdx) {
    const pl = this.players[pIdx];
    if (pl.passed) return false;
    if (pl.hand.length > 0) return true;
    return this.isLeaderAvailable(pIdx);
  }

  _advanceTurn() {
    this._recomputeScores();
    const cur = this.turn;
    const opp = 1 - cur;
    if (!this.players[opp].passed) {
      this.turn = opp;
    }

    for (let guard = 0; guard < 4; guard++) {
      const c = this.turn;
      if (!this.players[c].passed && !this._canAct(c)) {
        this.players[c].passed = true;
      }
      if (this.players[0].passed && this.players[1].passed) {
        this._endRound();
        return;
      }
      if (this.players[c].passed) {
        this.turn = 1 - c;
        continue;
      }
      break;
    }
  }

  _startRound(roundNo) {
    this.round = roundNo;
    this.sub_phase = 'PLAY';
    this.turn = this.first_player;
    for (const p of this.players) {
      p.passed = false;
      p.keep_uid = null;
      if (p.pending_draws > 0) {
        this._draw(p.index, p.pending_draws);
        p.pending_draws = 0;
      }
    }

    this._recomputeScores();
    for (let guard = 0; guard < 4; guard++) {
      const c = this.turn;
      if (!this.players[c].passed && !this._canAct(c)) {
        this.players[c].passed = true;
      }
      if (this.players[0].passed && this.players[1].passed) {
        this._endRound();
        return;
      }
      if (this.players[c].passed) {
        this.turn = 1 - c;
        continue;
      }
      break;
    }
  }

  _endRound() {
    this._recomputeScores();
    const [s0, s1] = this.scores;
    const starter = this.first_player;
    let winner = null;
    let tie = false;
    let nilfTie = false;

    if (s0 === s1) {
      tie = true;
      const n0 = this.players[0].faction === 'nilfgaard';
      const n1 = this.players[1].faction === 'nilfgaard';
      if (n0 !== n1) {
        winner = n0 ? 0 : 1;
        nilfTie = true;
        this.gems[1 - winner] -= 1;
      } else {
        this.gems[0] -= 1;
        this.gems[1] -= 1;
      }
    } else {
      winner = s0 > s1 ? 0 : 1;
      this.gems[1 - winner] -= 1;
    }

    this.round_history.push({
      round: this.round,
      scores: [s0, s1],
      winner,
      tie,
      nilfTie,
      gems: [this.gems[0], this.gems[1]],
    });

    for (const p of this.players) {
      if (p.faction !== 'monsters') continue;
      const units = [];
      for (const r of ROW_KEYS) {
        for (const inst of p.rows[r]) {
          if (!CARDS_BY_IDX[inst.card_idx]._is_hero) {
            units.push(inst);
          }
        }
      }
      const kept = this.rng.pickRandom(units);
      if (kept) p.keep_uid = kept.uid;
    }

    for (const p of this.players) {
      for (const r of ROW_KEYS) {
        const stay = [];
        for (const inst of p.rows[r]) {
          if (p.keep_uid !== null && inst.uid === p.keep_uid) {
            stay.push(inst);
          } else {
            p.graveyard.push(inst);
          }
        }
        p.rows[r] = stay;
        if (p.row_horn[r] !== null) {
          if (!p.row_horn[r].leader) {
            p.graveyard.push(p.row_horn[r]);
          }
          p.row_horn[r] = null;
        }
      }
    }

    this._clearWeather();
    this._recomputeScores();

    if (winner !== null && this.players[winner].faction === 'northern-realms') {
      this.players[winner].pending_draws += 1;
    }

    if (this.gems[0] <= 0 || this.gems[1] <= 0 || this.round >= MAX_ROUNDS) {
      if (this.gems[0] <= 0 && this.gems[1] <= 0) this.winner = null;
      else if (this.gems[0] <= 0) this.winner = 1;
      else if (this.gems[1] <= 0) this.winner = 0;
      else this.winner = this.gems[0] === this.gems[1] ? null : this.gems[0] > this.gems[1] ? 0 : 1;
      this.done = true;
      this.sub_phase = 'GAME_OVER';
      return;
    }

    this.first_player = winner !== null ? winner : starter;
    this._startRound(this.round + 1);
  }

  encodeObservation(pov = null) {
    if (pov === null) pov = this.turn;
    const me = this.players[pov];
    const opp = this.players[1 - pov];
    const obs = new Float32Array(OBS_DIM);

    // Block 0: Global & Meta State (51 dims)
    if (this.round >= 1 && this.round <= 3) obs[this.round - 1] = 1.0;
    const subMap = { MULLIGAN: 0, PLAY: 1, MEDIC: 2, DECOY: 3 };
    if (this.sub_phase in subMap) obs[3 + subMap[this.sub_phase]] = 1.0;

    obs[7] = this.gems[pov] / 2.0;
    obs[8] = this.gems[1 - pov] / 2.0;
    obs[9] = me.passed ? 1.0 : 0.0;
    obs[10] = opp.passed ? 1.0 : 0.0;

    const sMe = this.scores[pov];
    const sOpp = this.scores[1 - pov];
    obs[11] = sMe / 100.0;
    obs[12] = sOpp / 100.0;
    obs[13] = Math.tanh((sMe - sOpp) / 30.0);

    obs[14] = this.weather.frost ? 1.0 : 0.0;
    obs[15] = this.weather.fog ? 1.0 : 0.0;
    obs[16] = this.weather.rain ? 1.0 : 0.0;
    obs[17] = this.double_spy ? 1.0 : 0.0;

    obs[18] = me.hand.length / 15.0;
    obs[19] = (me.deck.length + me.mulligan_discards.length) / 30.0;
    obs[20] = me.graveyard.length / 30.0;
    obs[21] = opp.hand.length / 15.0;
    obs[22] = (opp.deck.length + opp.mulligan_discards.length) / 30.0;
    obs[23] = opp.graveyard.length / 30.0;

    obs[24] = this.isLeaderAvailable(pov) ? 1.0 : 0.0;
    obs[25] = this.isLeaderAvailable(1 - pov) ? 1.0 : 0.0;

    obs[26 + FACTION_TO_IDX[me.faction]] = 1.0;
    obs[30 + FACTION_TO_IDX[opp.faction]] = 1.0;
    obs[34 + me.leader_idx] = 1.0;
    obs[42 + opp.leader_idx] = 1.0;
    obs[50] = me.mulligan_left / 2.0;

    // Block 1: 6 Board Rows Summary (36 dims, offset 51..86)
    let offset = 51;
    for (const pl of [me, opp]) {
      for (const r of ROW_KEYS) {
        const hasHorn = pl.row_horn[r] !== null;
        const wActive = this.weather[WEATHER_FOR_ROW[r]];
        const { powers, total: rTot, hornActive } = computeRowPowers(
          pl.rows[r],
          hasHorn,
          wActive,
          this.double_spy
        );
        let maxNonHero = 0;
        let heroSum = 0;
        for (let i = 0; i < pl.rows[r].length; i++) {
          const inst = pl.rows[r][i];
          const pw = powers[i];
          if (CARDS_BY_IDX[inst.card_idx]._is_hero) heroSum += pw;
          else if (pw > maxNonHero) maxNonHero = pw;
        }
        obs[offset] = rTot / 60.0;
        obs[offset + 1] = pl.rows[r].length / 10.0;
        obs[offset + 2] = hornActive ? 1.0 : 0.0;
        obs[offset + 3] = wActive ? 1.0 : 0.0;
        obs[offset + 4] = maxNonHero / 20.0;
        obs[offset + 5] = heroSum / 30.0;
        offset += 6;
      }
    }

    // Block 2: Card Multiset Counts (7 * NUM_CARDS = 602 dims, offset 87..688)
    const b0 = 87;
    const b1 = b0 + NUM_CARDS;
    const b2 = b1 + NUM_CARDS;
    const b3 = b2 + NUM_CARDS;
    const b4 = b3 + NUM_CARDS;
    const b5 = b4 + NUM_CARDS;
    const b6 = b5 + NUM_CARDS;
    const step = 1.0 / 3.0;

    for (const inst of me.hand) obs[b0 + inst.card_idx] += step;
    for (const inst of me.deck) obs[b1 + inst.card_idx] += step;
    for (const inst of me.mulligan_discards) obs[b1 + inst.card_idx] += step;
    for (const r of ROW_KEYS) {
      for (const inst of me.rows[r]) obs[b2 + inst.card_idx] += step;
      for (const inst of opp.rows[r]) obs[b3 + inst.card_idx] += step;
    }
    for (const inst of me.graveyard) obs[b4 + inst.card_idx] += step;
    for (const inst of opp.graveyard) obs[b5 + inst.card_idx] += step;

    // Opponent unseen deduced strictly from public board/graveyard and known initial deck multiset:
    // opp_unseen = (me.initial_counts + opp.initial_counts) - total_visible - my_unseen
    // Strictly adheres to non-cheating POV encapsulation without reading opp.hand or opp.deck!
    const visCounts = new Int8Array(NUM_CARDS);
    for (const p of this.players) {
      for (const r of ROW_KEYS) {
        for (const inst of p.rows[r]) visCounts[inst.card_idx]++;
        if (p.row_horn[r] && p.row_horn[r].card_idx >= 0) visCounts[p.row_horn[r].card_idx]++;
      }
      for (const inst of p.graveyard) visCounts[inst.card_idx]++;
    }
    for (const wInst of this.weather_cards) visCounts[wInst.card_idx]++;

    const myUnseen = new Int8Array(NUM_CARDS);
    for (const inst of me.hand) myUnseen[inst.card_idx]++;
    for (const inst of me.deck) myUnseen[inst.card_idx]++;
    for (const inst of me.mulligan_discards) myUnseen[inst.card_idx]++;

    for (let c = 0; c < NUM_CARDS; c++) {
      const unseenC = (me.initial_counts[c] + opp.initial_counts[c]) - visCounts[c] - myUnseen[c];
      if (unseenC > 0) obs[b6 + c] = unseenC * step;
    }

    return obs;
  }

  /** Human-readable action label for UI & AI X-Ray panel */
  actionLabel(action, lang = 'zh') {
    if (action === PASS_ACTION) {
      if (this.sub_phase === 'MULLIGAN') return lang === 'zh' ? '结束换牌' : 'Finish Mulligan';
      return lang === 'zh' ? '放弃本局 (Pass)' : 'Pass Round';
    }
    if (action === USE_LEADER_ACTION) {
      const pl = this.players[this.turn];
      const l = LEADERS_BY_IDX[pl.leader_idx];
      return lang === 'zh' ? `领袖技能：${l.nameZh}` : `Leader: ${l.nameEn}`;
    }
    if (action >= HORN_ROW_OFFSET && action < HORN_ROW_OFFSET + 3) {
      const rowZh = ['近战行', '远程行', '攻城行'][action - HORN_ROW_OFFSET];
      const rowEn = ['Close', 'Ranged', 'Siege'][action - HORN_ROW_OFFSET];
      return lang === 'zh' ? `指挥官号角 → ${rowZh}` : `Commander's Horn → ${rowEn}`;
    }
    const c = CARDS_BY_IDX[action];
    if (!c) return `Action #${action}`;
    const name = lang === 'zh' ? c.nameZh : c.nameEn;
    if (this.sub_phase === 'MULLIGAN') return lang === 'zh' ? `换掉：${name}` : `Mulligan: ${name}`;
    if (this.sub_phase === 'MEDIC') return lang === 'zh' ? `医师复活：${name}` : `Revive: ${name}`;
    if (this.sub_phase === 'DECOY') return lang === 'zh' ? `诱饵回收：${name}` : `Decoy: ${name}`;
    return lang === 'zh' ? `打出：${name}` : `Play: ${name}`;
  }
}

// ────────────────────────────────────────────────────────────────
// Built-in Heuristic AI (Easy / Normal / Hard) for Web & Arena
// ────────────────────────────────────────────────────────────────

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

