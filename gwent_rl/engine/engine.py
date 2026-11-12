"""
Core minimalist Gwent (The Witcher 3) rules engine in pure Python.
Designed for high-throughput RL self-play and 100% bit-level parity with
the Web JS engine.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .rng import RngState
from .data import (
    CARDS_BY_ID,
    CARDS_BY_IDX,
    LEADERS_BY_ID,
    LEADERS_BY_IDX,
    PRESETS_BY_ID,
    PRESET_LIST,
    ROW_KEYS,
    ROW_TO_IDX,
    FACTION_TO_IDX,
    NUM_CARDS,
    NUM_LEADERS,
    ACTION_DIM,
    HORN_ROW_OFFSET,
    USE_LEADER_ACTION,
    PASS_ACTION,
    expand_preset_deck,
)
from .scoring import compute_row_powers, WEATHER_FOR_ROW

STARTING_HAND = 10
MULLIGAN_MAX = 2
STARTING_GEMS = 2
MAX_ROUNDS = 3
OBS_DIM = 51 + 36 + 7 * NUM_CARDS  # 689 dims for NUM_CARDS=86


class PlayerState:
    __slots__ = (
        "index",
        "faction",
        "leader_id",
        "leader_idx",
        "leader_used",
        "leader_canceled",
        "draw_extra_start",
        "hand",
        "deck",
        "graveyard",
        "mulligan_discards",
        "mulligan_left",
        "rows",
        "row_horn",
        "passed",
        "pending_draws",
        "keep_uid",
        "initial_counts",
    )

    def __init__(self, index: int, faction: str, leader_id: str, deck: List[Dict[str, Any]], initial_counts: List[int]):
        self.index: int = index
        self.faction: str = faction
        self.leader_id: str = leader_id
        self.leader_idx: int = LEADERS_BY_ID[leader_id]["idx"]
        self.leader_used: bool = False
        self.leader_canceled: bool = False
        self.draw_extra_start: bool = False
        self.hand: List[Dict[str, Any]] = []
        self.deck: List[Dict[str, Any]] = deck
        self.graveyard: List[Dict[str, Any]] = []
        self.mulligan_discards: List[Dict[str, Any]] = []
        self.mulligan_left: int = MULLIGAN_MAX
        self.rows: Dict[str, List[Dict[str, Any]]] = {"close": [], "ranged": [], "siege": []}
        self.row_horn: Dict[str, Optional[Dict[str, Any]]] = {"close": None, "ranged": None, "siege": None}
        self.passed: bool = False
        self.pending_draws: int = 0
        self.keep_uid: Optional[int] = None
        self.initial_counts: List[int] = initial_counts

    def clone(self) -> "PlayerState":
        p = PlayerState.__new__(PlayerState)
        p.index = self.index
        p.faction = self.faction
        p.leader_id = self.leader_id
        p.leader_idx = self.leader_idx
        p.leader_used = self.leader_used
        p.leader_canceled = self.leader_canceled
        p.draw_extra_start = self.draw_extra_start
        p.hand = [inst.copy() for inst in self.hand]
        p.deck = [inst.copy() for inst in self.deck]
        p.graveyard = [inst.copy() for inst in self.graveyard]
        p.mulligan_discards = [inst.copy() for inst in self.mulligan_discards]
        p.mulligan_left = self.mulligan_left
        p.rows = {
            "close": [inst.copy() for inst in self.rows["close"]],
            "ranged": [inst.copy() for inst in self.rows["ranged"]],
            "siege": [inst.copy() for inst in self.rows["siege"]],
        }
        p.row_horn = {
            "close": self.row_horn["close"].copy() if self.row_horn["close"] else None,
            "ranged": self.row_horn["ranged"].copy() if self.row_horn["ranged"] else None,
            "siege": self.row_horn["siege"].copy() if self.row_horn["siege"] else None,
        }
        p.passed = self.passed
        p.pending_draws = self.pending_draws
        p.keep_uid = self.keep_uid
        p.initial_counts = self.initial_counts  # immutable after reset
        return p


class GwentEngine:
    """
    Deterministic micro-step Gwent state machine.
    Sub-phases:
      - 'MULLIGAN': Player `turn` chooses a card_idx in hand to redraw (up to 2 times), or PASS_ACTION to finish.
      - 'PLAY': Normal turn. Player `turn` can play a card_idx, place a Horn on a row, use Leader, or Pass.
      - 'MEDIC': Player `turn` chooses a non-hero unit card_idx from their graveyard to revive immediately.
      - 'DECOY': Player `turn` chooses a non-hero unit card_idx on their board to return to hand.
      - 'GAME_OVER': Terminal state.
    """

    __slots__ = (
        "rng",
        "uid_seq",
        "round",
        "turn",
        "first_player",
        "sub_phase",
        "gems",
        "weather",
        "weather_cards",
        "double_spy",
        "players",
        "scores",
        "row_scores",
        "round_history",
        "done",
        "winner",
    )

    def __init__(
        self,
        seed: int | str = 1,
        preset0: str = "nr-temeria",
        preset1: str = "ng-white-flame",
        _empty: bool = False,
    ):
        if _empty:
            return
        self.reset(seed=seed, preset0=preset0, preset1=preset1)

    def clone(self) -> "GwentEngine":
        eng = GwentEngine(_empty=True)
        eng.rng = self.rng.clone()
        eng.uid_seq = self.uid_seq
        eng.round = self.round
        eng.turn = self.turn
        eng.first_player = self.first_player
        eng.sub_phase = self.sub_phase
        eng.gems = [self.gems[0], self.gems[1]]
        eng.weather = self.weather.copy()
        eng.weather_cards = [inst.copy() for inst in self.weather_cards]
        eng.double_spy = self.double_spy
        eng.players = [self.players[0].clone(), self.players[1].clone()]
        eng.scores = [self.scores[0], self.scores[1]]
        eng.row_scores = [self.row_scores[0].copy(), self.row_scores[1].copy()]
        eng.round_history = [h.copy() for h in self.round_history]
        eng.done = self.done
        eng.winner = self.winner
        return eng

    def reset(
        self,
        seed: int | str = 1,
        preset0: str = "nr-temeria",
        preset1: str = "ng-white-flame",
    ) -> None:
        self.rng = RngState(seed)
        self.uid_seq = 0
        self.round = 1
        self.gems = [STARTING_GEMS, STARTING_GEMS]
        self.weather = {"frost": False, "fog": False, "rain": False}
        self.weather_cards: List[Dict[str, Any]] = []
        self.double_spy = False
        self.round_history: List[Dict[str, Any]] = []
        self.done = False
        self.winner: Optional[int] = None

        presets = [PRESETS_BY_ID[preset0], PRESETS_BY_ID[preset1]]
        self.players: List[PlayerState] = []

        for p_idx in (0, 1):
            pr = presets[p_idx]
            card_ids = expand_preset_deck(pr["id"])
            init_counts = [0] * NUM_CARDS
            insts: List[Dict[str, Any]] = []
            for cid in card_ids:
                cdata = CARDS_BY_ID[cid]
                cidx = cdata["idx"]
                init_counts[cidx] += 1
                self.uid_seq += 1
                insts.append(
                    {
                        "uid": self.uid_seq,
                        "card_id": cid,
                        "card_idx": cidx,
                        "owner": p_idx,
                    }
                )
            shuffled_deck = self.rng.shuffle(insts)
            self.players.append(
                PlayerState(
                    index=p_idx,
                    faction=pr["faction"],
                    leader_id=pr["leaderId"],
                    deck=shuffled_deck,
                    initial_counts=init_counts,
                )
            )

        self._apply_leader_passives()

        # Draw initial 10 cards alternating P0, P1 (matches original engine order)
        for _ in range(STARTING_HAND):
            self._draw(0, 1)
            self._draw(1, 1)

        for p in self.players:
            if p.draw_extra_start:
                self._draw(p.index, 1)

        # Determine Round 1 starter: Scoia'tael passive or coin flip
        st0 = self.players[0].faction == "scoiatael"
        st1 = self.players[1].faction == "scoiatael"
        if st0 != st1:
            self.first_player = 0 if st0 else 1
        else:
            self.first_player = self.rng.rand_int(2)

        # Begin Mulligan with Player 0
        self.sub_phase = "MULLIGAN"
        self.turn = 0
        self.scores = [0, 0]
        self.row_scores = [
            {"close": 0, "ranged": 0, "siege": 0},
            {"close": 0, "ranged": 0, "siege": 0},
        ]

    def _apply_leader_passives(self) -> None:
        # Check White Flame (cancelLeader) first
        for p in self.players:
            ldata = LEADERS_BY_IDX[p.leader_idx]
            if ldata["_ab_id"] == "cancelLeader":
                p.leader_used = True
                opp = self.players[1 - p.index]
                opp.leader_canceled = True

        for p in self.players:
            if p.leader_canceled:
                continue
            ldata = LEADERS_BY_IDX[p.leader_idx]
            ab_id = ldata["_ab_id"]
            if ab_id == "doubleSpy":
                self.double_spy = True
                p.leader_used = True
            elif ab_id == "drawExtraStart":
                p.draw_extra_start = True
                p.leader_used = True

    def _draw(self, p_idx: int, count: int) -> None:
        pl = self.players[p_idx]
        for _ in range(count):
            if not pl.deck:
                break
            pl.hand.append(pl.deck.pop(0))

    def _recompute_scores(self) -> None:
        for p_idx in (0, 1):
            pl = self.players[p_idx]
            total = 0
            r_scores = {}
            for r in ROW_KEYS:
                has_horn = pl.row_horn[r] is not None
                w_active = self.weather[WEATHER_FOR_ROW[r]]
                _, r_tot, _ = compute_row_powers(pl.rows[r], has_horn, w_active, self.double_spy)
                r_scores[r] = r_tot
                total += r_tot
            self.scores[p_idx] = total
            self.row_scores[p_idx] = r_scores

    def is_leader_available(self, p_idx: int) -> bool:
        pl = self.players[p_idx]
        if pl.passed or pl.leader_used or pl.leader_canceled:
            return False
        ldata = LEADERS_BY_IDX[pl.leader_idx]
        if ldata["_is_passive"]:
            return False
        ab_id = ldata["_ab_id"]
        ab = ldata["_ab"]
        if ab_id == "clearWeather":
            return any(self.weather.values())
        if ab_id == "leaderHorn":
            row = ab["row"]
            return pl.row_horn[row] is None
        if ab_id == "playWeatherFromDeck":
            w_kind = ab["weather"]
            if self.weather.get(w_kind, False):
                return False
            return any(CARDS_BY_IDX[inst["card_idx"]]["_weather_kind"] == w_kind for inst in pl.deck)
        return False

    def _has_decoy_target(self, p_idx: int) -> bool:
        pl = self.players[p_idx]
        for r in ROW_KEYS:
            for inst in pl.rows[r]:
                if not CARDS_BY_IDX[inst["card_idx"]]["_is_hero"]:
                    return True
        return False

    def _revivable_grave_Check(self, p_idx: int) -> bool:
        pl = self.players[p_idx]
        for inst in pl.graveyard:
            c = CARDS_BY_IDX[inst["card_idx"]]
            if c["_is_unit"] and not c["_is_hero"]:
                return True
        return False

    def get_legal_mask(self) -> List[bool]:
        mask = [False] * ACTION_DIM
        if self.done or self.sub_phase == "GAME_OVER":
            return mask

        p_idx = self.turn
        pl = self.players[p_idx]

        if self.sub_phase == "MULLIGAN":
            for inst in pl.hand:
                mask[inst["card_idx"]] = True
            mask[PASS_ACTION] = True
            return mask

        if self.sub_phase == "MEDIC":
            for inst in pl.graveyard:
                c = CARDS_BY_IDX[inst["card_idx"]]
                if c["_is_unit"] and not c["_is_hero"]:
                    mask[inst["card_idx"]] = True
            return mask

        if self.sub_phase == "DECOY":
            for r in ROW_KEYS:
                for inst in pl.rows[r]:
                    c = CARDS_BY_IDX[inst["card_idx"]]
                    if c["_is_unit"] and not c["_is_hero"]:
                        mask[inst["card_idx"]] = True
            return mask

        # sub_phase == 'PLAY'
        mask[PASS_ACTION] = True
        if self.is_leader_available(p_idx):
            mask[USE_LEADER_ACTION] = True

        has_decoy_tgt = None
        for inst in pl.hand:
            cidx = inst["card_idx"]
            c = CARDS_BY_IDX[cidx]
            cid = c["id"]
            if cid == "commander-s-horn":
                for r_i, r in enumerate(ROW_KEYS):
                    if pl.row_horn[r] is None:
                        mask[HORN_ROW_OFFSET + r_i] = True
            elif cid == "decoy":
                if has_decoy_tgt is None:
                    has_decoy_tgt = self._has_decoy_target(p_idx)
                if has_decoy_tgt:
                    mask[cidx] = True
            else:
                mask[cidx] = True

        return mask

    def get_legal_actions(self) -> List[int]:
        mask = self.get_legal_mask()
        return [i for i, ok in enumerate(mask) if ok]

    def step(self, action: int) -> Tuple[bool, Optional[int]]:
        if self.done:
            return True, self.winner

        p_idx = self.turn
        pl = self.players[p_idx]

        if self.sub_phase == "MULLIGAN":
            if action == PASS_ACTION:
                self._finish_player_mulligan(p_idx)
            else:
                # Redraw chosen card_idx
                idx_in_hand = next(
                    (i for i, inst in enumerate(pl.hand) if inst["card_idx"] == action),
                    -1,
                )
                if idx_in_hand < 0:
                    raise ValueError(f"Illegal mulligan card_idx {action}")
                discarded = pl.hand.pop(idx_in_hand)
                pl.mulligan_discards.append(discarded)
                self._draw(p_idx, 1)
                pl.mulligan_left -= 1
                if pl.mulligan_left <= 0 or not pl.hand:
                    self._finish_player_mulligan(p_idx)
            return self.done, self.winner

        if self.sub_phase == "MEDIC":
            idx_in_grave = next(
                (
                    i
                    for i, inst in enumerate(pl.graveyard)
                    if inst["card_idx"] == action
                    and CARDS_BY_IDX[inst["card_idx"]]["_is_unit"]
                    and not CARDS_BY_IDX[inst["card_idx"]]["_is_hero"]
                ),
                -1,
            )
            if idx_in_grave < 0:
                raise ValueError(f"Illegal medic target card_idx {action}")
            revived = pl.graveyard.pop(idx_in_grave)
            self._play_unit(p_idx, revived)
            return self.done, self.winner

        if self.sub_phase == "DECOY":
            target_inst = None
            for r in ROW_KEYS:
                for i, inst in enumerate(pl.rows[r]):
                    if inst["card_idx"] == action and not CARDS_BY_IDX[action]["_is_hero"]:
                        target_inst = pl.rows[r].pop(i)
                        break
                if target_inst is not None:
                    break
            if target_inst is None:
                raise ValueError(f"Illegal decoy target card_idx {action}")
            target_inst["owner"] = p_idx
            pl.hand.append(target_inst)
            self.sub_phase = "PLAY"
            self._advance_turn()
            return self.done, self.winner

        # sub_phase == 'PLAY'
        if action == PASS_ACTION:
            pl.passed = True
            self._advance_turn()
            return self.done, self.winner

        if action == USE_LEADER_ACTION:
            if not self.is_leader_available(p_idx):
                raise ValueError("Leader ability not available")
            pl.leader_used = True
            ldata = LEADERS_BY_IDX[pl.leader_idx]
            ab_id = ldata["_ab_id"]
            ab = ldata["_ab"]
            if ab_id == "clearWeather":
                self._clear_weather()
            elif ab_id == "leaderHorn":
                row = ab["row"]
                if pl.row_horn[row] is None:
                    pl.row_horn[row] = {"uid": -1, "card_id": None, "card_idx": -1, "owner": p_idx, "leader": True}
            elif ab_id == "playWeatherFromDeck":
                w_kind = ab["weather"]
                deck_i = next(
                    (
                        i
                        for i, inst in enumerate(pl.deck)
                        if CARDS_BY_IDX[inst["card_idx"]]["_weather_kind"] == w_kind
                    ),
                    -1,
                )
                if deck_i >= 0:
                    w_inst = pl.deck.pop(deck_i)
                    self._apply_weather_card(p_idx, w_inst, w_kind)
            self._advance_turn()
            return self.done, self.winner

        if HORN_ROW_OFFSET <= action < HORN_ROW_OFFSET + 3:
            r_idx = action - HORN_ROW_OFFSET
            row = ROW_KEYS[r_idx]
            if pl.row_horn[row] is not None:
                raise ValueError(f"Row {row} already has a horn")
            horn_i = next(
                (i for i, inst in enumerate(pl.hand) if CARDS_BY_IDX[inst["card_idx"]]["id"] == "commander-s-horn"),
                -1,
            )
            if horn_i < 0:
                raise ValueError("No commander-s-horn in hand")
            horn_inst = pl.hand.pop(horn_i)
            pl.row_horn[row] = horn_inst
            self._advance_turn()
            return self.done, self.winner

        # Action in [0..NUM_CARDS-1]
        hand_i = next((i for i, inst in enumerate(pl.hand) if inst["card_idx"] == action), -1)
        if hand_i < 0:
            raise ValueError(f"Card idx {action} not in hand")
        inst = pl.hand.pop(hand_i)
        cdata = CARDS_BY_IDX[action]
        cid = cdata["id"]

        if cid == "decoy":
            pl.graveyard.append(inst)
            self.sub_phase = "DECOY"
            self._recompute_scores()
            return self.done, self.winner

        if cdata["type"] == "weather":
            self._apply_weather_card(p_idx, inst, cdata["_weather_kind"])
            self._advance_turn()
            return self.done, self.winner

        if cid == "scorch":
            pl.graveyard.append(inst)
            self._apply_scorch()
            self._advance_turn()
            return self.done, self.winner

        # Unit card
        self._play_unit(p_idx, inst)
        return self.done, self.winner

    def _finish_player_mulligan(self, p_idx: int) -> None:
        pl = self.players[p_idx]
        if pl.mulligan_discards:
            pl.deck.extend(pl.mulligan_discards)
            pl.mulligan_discards.clear()
            pl.deck = self.rng.shuffle(pl.deck)
        pl.mulligan_left = 0
        if p_idx == 0:
            self.turn = 1
        else:
            self._start_round(1)

    def _play_unit(self, p_idx: int, inst: Dict[str, Any]) -> None:
        cdata = CARDS_BY_IDX[inst["card_idx"]]
        row = cdata["_primary_row"]
        board_owner = (1 - p_idx) if cdata["_is_spy"] else p_idx
        inst["owner"] = board_owner
        self.players[board_owner].rows[row].append(inst)

        # 1. Spy draw
        if cdata["_is_spy"]:
            self._draw(p_idx, 2)

        # 2. Muster
        if cdata["_is_muster"]:
            targets = cdata["_muster_targets"]
            pl = self.players[p_idx]
            summoned: List[Dict[str, Any]] = []

            new_deck = []
            for d_inst in pl.deck:
                if d_inst["card_id"] in targets:
                    summoned.append(d_inst)
                else:
                    new_deck.append(d_inst)
            pl.deck = new_deck

            if cdata["_muster_from_hand"]:
                new_hand = []
                for h_inst in pl.hand:
                    if h_inst["card_id"] in targets:
                        summoned.append(h_inst)
                    else:
                        new_hand.append(h_inst)
                pl.hand = new_hand

            for s_inst in summoned:
                s_card = CARDS_BY_IDX[s_inst["card_idx"]]
                s_row = s_card["_primary_row"]
                s_inst["owner"] = p_idx
                pl.rows[s_row].append(s_inst)

        # 3. Medic
        if cdata["_is_medic"] and self._revivable_grave_Check(p_idx):
            self.sub_phase = "MEDIC"
            self._recompute_scores()
            return

        self.sub_phase = "PLAY"
        self._advance_turn()

    def _apply_weather_card(self, p_idx: int, inst: Dict[str, Any], w_kind: str) -> None:
        if w_kind == "clear":
            self.players[p_idx].graveyard.append(inst)
            self._clear_weather()
            return
        if self.weather.get(w_kind, False):
            self.players[p_idx].graveyard.append(inst)
            return
        self.weather[w_kind] = True
        inst["owner"] = p_idx
        self.weather_cards.append(inst)

    def _clear_weather(self) -> None:
        for w_inst in self.weather_cards:
            self.players[w_inst["owner"]].graveyard.append(w_inst)
        self.weather_cards.clear()
        self.weather["frost"] = False
        self.weather["fog"] = False
        self.weather["rain"] = False

    def _apply_scorch(self) -> None:
        # Find max effective power among all non-hero units on board
        max_power = 0
        row_powers_map: Dict[Tuple[int, str], List[int]] = {}
        for p_idx in (0, 1):
            pl = self.players[p_idx]
            for r in ROW_KEYS:
                powers, _, _ = compute_row_powers(
                    pl.rows[r],
                    pl.row_horn[r] is not None,
                    self.weather[WEATHER_FOR_ROW[r]],
                    self.double_spy,
                )
                row_powers_map[(p_idx, r)] = powers
                for inst, pw in zip(pl.rows[r], powers):
                    if not CARDS_BY_IDX[inst["card_idx"]]["_is_hero"]:
                        if pw > max_power:
                            max_power = pw

        if max_power <= 0:
            return

        for p_idx in (0, 1):
            pl = self.players[p_idx]
            for r in ROW_KEYS:
                powers = row_powers_map[(p_idx, r)]
                kept = []
                for inst, pw in zip(pl.rows[r], powers):
                    if (not CARDS_BY_IDX[inst["card_idx"]]["_is_hero"]) and pw == max_power:
                        pl.graveyard.append(inst)
                    else:
                        kept.append(inst)
                pl.rows[r] = kept

    def _can_act(self, p_idx: int) -> bool:
        pl = self.players[p_idx]
        if pl.passed:
            return False
        if pl.hand:
            # Wait: what if player only has Decoy in hand and no decoy targets on board, and no leader?
            # In PLAY, PASS_ACTION is always legal so player can always at least Pass;
            # mirrors original engine _canAct: hand.length > 0 or leader available.
            return True
        return self.is_leader_available(p_idx)

    def _advance_turn(self) -> None:
        self._recompute_scores()
        cur = self.turn
        opp = 1 - cur
        if not self.players[opp].passed:
            self.turn = opp

        for _ in range(4):
            cur = self.turn
            if (not self.players[cur].passed) and (not self._can_act(cur)):
                self.players[cur].passed = True
            if self.players[0].passed and self.players[1].passed:
                self._end_round()
                return
            if self.players[cur].passed:
                self.turn = 1 - cur
                continue
            break

    def _start_round(self, round_no: int) -> None:
        self.round = round_no
        self.sub_phase = "PLAY"
        self.turn = self.first_player
        for p in self.players:
            p.passed = False
            p.keep_uid = None
            if p.pending_draws > 0:
                self._draw(p.index, p.pending_draws)
                p.pending_draws = 0

        self._recompute_scores()
        # Handle edge case if starter has no cards and no leader at round start
        for _ in range(4):
            cur = self.turn
            if (not self.players[cur].passed) and (not self._can_act(cur)):
                self.players[cur].passed = True
            if self.players[0].passed and self.players[1].passed:
                self._end_round()
                return
            if self.players[cur].passed:
                self.turn = 1 - cur
                continue
            break

    def _end_round(self) -> None:
        self._recompute_scores()
        s0, s1 = self.scores[0], self.scores[1]
        starter = self.first_player
        winner: Optional[int] = None
        tie = False
        nilf_tie = False

        if s0 == s1:
            tie = True
            n0 = self.players[0].faction == "nilfgaard"
            n1 = self.players[1].faction == "nilfgaard"
            if n0 != n1:
                winner = 0 if n0 else 1
                nilf_tie = True
                self.gems[1 - winner] -= 1
            else:
                self.gems[0] -= 1
                self.gems[1] -= 1
        else:
            winner = 0 if s0 > s1 else 1
            self.gems[1 - winner] -= 1

        self.round_history.append(
            {
                "round": self.round,
                "scores": [s0, s1],
                "winner": winner,
                "tie": tie,
                "nilfTie": nilf_tie,
                "gems": [self.gems[0], self.gems[1]],
            }
        )

        # Monsters faction passive: keep 1 random unit on board
        for p in self.players:
            if p.faction != "monsters":
                continue
            units: List[Dict[str, Any]] = []
            for r in ROW_KEYS:
                units.extend(p.rows[r])
            kept = self.rng.pick_random(units)
            if kept is not None:
                p.keep_uid = kept["uid"]

        # Clear boards to graveyard (except kept monster unit)
        for p in self.players:
            for r in ROW_KEYS:
                stay = []
                for inst in p.rows[r]:
                    if p.keep_uid is not None and inst["uid"] == p.keep_uid:
                        stay.append(inst)
                    else:
                        p.graveyard.append(inst)
                p.rows[r] = stay
                if p.row_horn[r] is not None:
                    if not p.row_horn[r].get("leader", False):
                        p.graveyard.append(p.row_horn[r])
                    p.row_horn[r] = None

        self._clear_weather()
        self._recompute_scores()

        # Northern Realms faction passive: draw 1 card on round win
        if winner is not None and self.players[winner].faction == "northern-realms":
            self.players[winner].pending_draws += 1

        if self.gems[0] <= 0 or self.gems[1] <= 0 or self.round >= MAX_ROUNDS:
            if self.gems[0] <= 0 and self.gems[1] <= 0:
                self.winner = None
            elif self.gems[0] <= 0:
                self.winner = 1
            elif self.gems[1] <= 0:
                self.winner = 0
            else:
                self.winner = None if self.gems[0] == self.gems[1] else (0 if self.gems[0] > self.gems[1] else 1)
            self.done = True
            self.sub_phase = "GAME_OVER"
            return

        self.first_player = winner if winner is not None else starter
        self._start_round(self.round + 1)

    def encode_observation(self, pov: Optional[int] = None) -> np.ndarray:
        """
        Deterministic 689-dim float32 observation vector from player `pov`'s perspective.
        """
        if pov is None:
            pov = self.turn
        me = self.players[pov]
        opp = self.players[1 - pov]
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # --- Block 0: Global & Meta State (51 dims) ---
        if 1 <= self.round <= 3:
            obs[self.round - 1] = 1.0
        sub_map = {"MULLIGAN": 0, "PLAY": 1, "MEDIC": 2, "DECOY": 3}
        if self.sub_phase in sub_map:
            obs[3 + sub_map[self.sub_phase]] = 1.0

        obs[7] = self.gems[pov] / 2.0
        obs[8] = self.gems[1 - pov] / 2.0
        obs[9] = 1.0 if me.passed else 0.0
        obs[10] = 1.0 if opp.passed else 0.0

        s_me = float(self.scores[pov])
        s_opp = float(self.scores[1 - pov])
        obs[11] = s_me / 100.0
        obs[12] = s_opp / 100.0
        obs[13] = math.tanh((s_me - s_opp) / 30.0)

        obs[14] = 1.0 if self.weather["frost"] else 0.0
        obs[15] = 1.0 if self.weather["fog"] else 0.0
        obs[16] = 1.0 if self.weather["rain"] else 0.0
        obs[17] = 1.0 if self.double_spy else 0.0

        obs[18] = len(me.hand) / 15.0
        obs[19] = (len(me.deck) + len(me.mulligan_discards)) / 30.0
        obs[20] = len(me.graveyard) / 30.0
        obs[21] = len(opp.hand) / 15.0
        obs[22] = (len(opp.deck) + len(opp.mulligan_discards)) / 30.0
        obs[23] = len(opp.graveyard) / 30.0

        obs[24] = 1.0 if self.is_leader_available(pov) else 0.0
        obs[25] = 1.0 if self.is_leader_available(1 - pov) else 0.0

        obs[26 + FACTION_TO_IDX[me.faction]] = 1.0
        obs[30 + FACTION_TO_IDX[opp.faction]] = 1.0
        obs[34 + me.leader_idx] = 1.0
        obs[42 + opp.leader_idx] = 1.0
        obs[50] = me.mulligan_left / 2.0

        # --- Block 1: 6 Board Rows Summary (36 dims, offset 51..86) ---
        offset = 51
        for pl in (me, opp):
            for r in ROW_KEYS:
                has_horn = pl.row_horn[r] is not None
                w_active = self.weather[WEATHER_FOR_ROW[r]]
                powers, r_tot, horn_active = compute_row_powers(
                    pl.rows[r], has_horn, w_active, self.double_spy
                )
                max_non_hero = 0
                hero_sum = 0
                for inst, pw in zip(pl.rows[r], powers):
                    if CARDS_BY_IDX[inst["card_idx"]]["_is_hero"]:
                        hero_sum += pw
                    elif pw > max_non_hero:
                        max_non_hero = pw
                obs[offset] = r_tot / 60.0
                obs[offset + 1] = len(pl.rows[r]) / 10.0
                obs[offset + 2] = 1.0 if horn_active else 0.0
                obs[offset + 3] = 1.0 if w_active else 0.0
                obs[offset + 4] = max_non_hero / 20.0
                obs[offset + 5] = hero_sum / 30.0
                offset += 6

        # --- Block 2: Card Multiset Counts (7 * NUM_CARDS = 602 dims, offset 87..688) ---
        # 0: my_hand, 1: my_deck, 2: my_board, 3: opp_board, 4: my_grave, 5: opp_grave, 6: opp_unseen
        b0 = 87
        b1 = b0 + NUM_CARDS
        b2 = b1 + NUM_CARDS
        b3 = b2 + NUM_CARDS
        b4 = b3 + NUM_CARDS
        b5 = b4 + NUM_CARDS
        b6 = b5 + NUM_CARDS

        for inst in me.hand:
            obs[b0 + inst["card_idx"]] += 1.0 / 3.0
        for inst in me.deck:
            obs[b1 + inst["card_idx"]] += 1.0 / 3.0
        for inst in me.mulligan_discards:
            obs[b1 + inst["card_idx"]] += 1.0 / 3.0
        for r in ROW_KEYS:
            for inst in me.rows[r]:
                obs[b2 + inst["card_idx"]] += 1.0 / 3.0
            for inst in opp.rows[r]:
                obs[b3 + inst["card_idx"]] += 1.0 / 3.0
        for inst in me.graveyard:
            obs[b4 + inst["card_idx"]] += 1.0 / 3.0
        for inst in opp.graveyard:
            obs[b5 + inst["card_idx"]] += 1.0 / 3.0

        # Opponent unseen (hand + deck + mulligan_discards) — exact multiset without leaking hand vs deck!
        for inst in opp.hand:
            obs[b6 + inst["card_idx"]] += 1.0 / 3.0
        for inst in opp.deck:
            obs[b6 + inst["card_idx"]] += 1.0 / 3.0
        for inst in opp.mulligan_discards:
            obs[b6 + inst["card_idx"]] += 1.0 / 3.0

        return obs
