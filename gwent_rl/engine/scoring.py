"""
Pure, fast row and board power computation for Gwent.
Canonical order:
  Hero immunity -> Weather (min(1, base)) -> Double Spy (x2) -> Tight Bond (xN)
  -> Morale (+M) -> Commander's Horn (x2)
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any
from .data import CARDS_BY_IDX, ROW_KEYS

WEATHER_FOR_ROW = {
    "close": "frost",
    "ranged": "fog",
    "siege": "rain",
}


def compute_row_powers(
    row_insts: List[Dict[str, Any]],
    has_row_horn: bool,
    weather_active: bool,
    double_spy: bool,
) -> Tuple[List[int], int, bool]:
    """
    Returns (unit_powers, row_total, horn_active) for a single row.
    """
    if not row_insts:
        return [], 0, has_row_horn

    horn_sources = 1 if has_row_horn else 0
    morale_count = 0
    bond_counts: Dict[int, int] = {}

    for inst in row_insts:
        c = CARDS_BY_IDX[inst["card_idx"]]
        if c["_is_horn_unit"]:
            horn_sources += 1
        if c["_is_morale"]:
            morale_count += 1
        if c["_is_bond"]:
            cidx = inst["card_idx"]
            bond_counts[cidx] = bond_counts.get(cidx, 0) + 1

    horn_active = horn_sources > 0
    powers: List[int] = []
    total = 0

    for inst in row_insts:
        c = CARDS_BY_IDX[inst["card_idx"]]
        power = c["strength"]
        if not c["_is_hero"]:
            if weather_active:
                if power > 1:
                    power = 1
            if double_spy and c["_is_spy"]:
                power *= 2
            if c["_is_bond"]:
                n = bond_counts.get(inst["card_idx"], 1)
                if n > 1:
                    power *= n
            morale = morale_count - (1 if c["_is_morale"] else 0)
            if morale > 0:
                power += morale
            if horn_active:
                if (not c["_is_horn_unit"]) or (horn_sources > 1):
                    power *= 2
        powers.append(power)
        total += power

    return powers, total, horn_active
