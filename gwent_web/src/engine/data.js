/**
 * Canonical metadata and card definitions shared across Gwent Web.
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
export const OBS_DIM = 57 + 36 + 7 * NUM_CARDS;

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
