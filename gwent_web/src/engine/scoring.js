/**
 * Pure, fast row power and combat strength calculations for Gwent Web.
 */

import { CARDS_BY_IDX } from './data.js';

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
