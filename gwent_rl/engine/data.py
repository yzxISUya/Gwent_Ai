"""
Loads the shared JSON data contract (cards, leaders, factions, presets, schema)
and exposes fast integer-indexed and string-keyed lookup tables.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any, Set

ROOT_DIR = Path(__file__).resolve().parents[2]
SHARED_DATA_DIR = ROOT_DIR / "shared" / "data"
SHARED_SPEC_DIR = ROOT_DIR / "shared" / "spec"

ROW_KEYS = ("close", "ranged", "siege")
ROW_TO_IDX = {"close": 0, "ranged": 1, "siege": 2}

with open(SHARED_DATA_DIR / "cards.json", "r", encoding="utf-8") as f:
    CARDS_BY_ID: Dict[str, Dict[str, Any]] = json.load(f)

with open(SHARED_DATA_DIR / "leaders.json", "r", encoding="utf-8") as f:
    LEADERS_BY_ID: Dict[str, Dict[str, Any]] = json.load(f)

with open(SHARED_DATA_DIR / "factions.json", "r", encoding="utf-8") as f:
    FACTIONS_BY_ID: Dict[str, Dict[str, Any]] = json.load(f)

with open(SHARED_DATA_DIR / "presets.json", "r", encoding="utf-8") as f:
    PRESET_LIST: List[Dict[str, Any]] = json.load(f)

with open(SHARED_SPEC_DIR / "schema.json", "r", encoding="utf-8") as f:
    SCHEMA: Dict[str, Any] = json.load(f)

NUM_CARDS: int = SCHEMA["numCards"]
NUM_LEADERS: int = SCHEMA["numLeaders"]
ACTION_DIM: int = SCHEMA["actionSpace"]["totalDim"]
HORN_ROW_OFFSET: int = SCHEMA["actionSpace"]["hornRowOffset"]
USE_LEADER_ACTION: int = SCHEMA["actionSpace"]["useLeaderAction"]
PASS_ACTION: int = SCHEMA["actionSpace"]["passOrFinishAction"]

FACTION_IDS: List[str] = SCHEMA["factions"]
FACTION_TO_IDX: Dict[str, int] = {fid: i for i, fid in enumerate(FACTION_IDS)}

CARDS_BY_IDX: List[Dict[str, Any]] = [None] * NUM_CARDS  # type: ignore
for cid, cdata in CARDS_BY_ID.items():
    idx = cdata["idx"]
    # Precompute fast boolean flags & ability dicts on card dict
    ab_map = {a["id"]: a for a in cdata.get("abilities", [])}
    cdata["_ab_map"] = ab_map
    cdata["_is_unit"] = cdata["type"] == "unit"
    cdata["_is_hero"] = bool(cdata.get("hero", False))
    cdata["_is_spy"] = "spy" in ab_map
    cdata["_is_medic"] = "medic" in ab_map
    cdata["_is_bond"] = "bond" in ab_map
    cdata["_is_morale"] = "morale" in ab_map
    cdata["_is_horn_unit"] = cdata["_is_unit"] and ("horn" in ab_map)
    cdata["_is_muster"] = "muster" in ab_map
    cdata["_muster_targets"] = set(ab_map["muster"].get("targets", [])) if "muster" in ab_map else set()
    cdata["_muster_from_hand"] = "hand" in ab_map["muster"].get("from", ["deck"]) if "muster" in ab_map else False
    cdata["_primary_row"] = cdata["rows"][0] if cdata.get("rows") else None
    cdata["_weather_kind"] = ab_map["weather"]["weather"] if "weather" in ab_map else None
    CARDS_BY_IDX[idx] = cdata

LEADERS_BY_IDX: List[Dict[str, Any]] = [None] * NUM_LEADERS  # type: ignore
for lid, ldata in LEADERS_BY_ID.items():
    idx = ldata["idx"]
    ab = ldata["ability"][0]
    ldata["_ab_id"] = ab["id"]
    ldata["_ab"] = ab
    ldata["_is_passive"] = bool(ab.get("passive", False)) or ab["id"] == "cancelLeader"
    LEADERS_BY_IDX[idx] = ldata

PRESETS_BY_ID: Dict[str, Dict[str, Any]] = {p["id"]: p for p in PRESET_LIST}

# Stage-1 Core Symmetric Benchmark Pool: Northern Realms (Foltest Siege) & Nilfgaard (Emhyr Spy)
CORE_PRESET_IDS: tuple[str, ...] = ("nr-temeria", "ng-white-flame")
CORE_DECK_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (p0, p1) for p0 in CORE_PRESET_IDS for p1 in CORE_PRESET_IDS
)

# Full 8-Preset Pool across all 4 factions & 8 leaders (8x8 = 64 ordered matchups)
ALL_PRESET_IDS: tuple[str, ...] = tuple(p["id"] for p in PRESET_LIST)
ALL_DECK_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (p0, p1) for p0 in ALL_PRESET_IDS for p1 in ALL_PRESET_IDS
)


def expand_preset_deck(preset_id: str) -> List[str]:
    preset = PRESETS_BY_ID[preset_id]
    out: List[str] = []
    for entry in preset["cards"]:
        out.extend([entry["id"]] * int(entry["count"]))
    return out
