"""
Heuristic AI bots (Easy, Normal, Hard) for benchmarking RL agents and
providing fast baseline opponents in both Python and Web.

Difficulty Tiers:
- easy:   Random legal action (skips mulligan, avoids premature pass when behind with >5 cards).
- normal: 0-ply static rule-based heuristic (no `engine.clone()` forward simulation, evaluates
          cards by static base strength + simple tag bonuses + stochastic noise from `self.rng`).
- hard:   1-ply forward lookahead search via `engine.clone()` with full Medic/Decoy sub-phase
          rollout, Scorch self-burn prevention, Muster/Bond/Weather exact swing calculation,
          card-advantage tempo passing, and minimum-card win commitment.
"""
from __future__ import annotations
from .rng import RngState
from .data import (
    CARDS_BY_IDX,
    ROW_KEYS,
    NUM_CARDS,
    HORN_ROW_OFFSET,
    USE_LEADER_ACTION,
    PASS_ACTION,
)
from .engine import GwentEngine


class HeuristicAI:
    def __init__(self, difficulty: str = "normal", seed: int = 42):
        self.difficulty = difficulty if difficulty in ("easy", "normal", "hard") else "normal"
        self.rng = RngState(seed)

    def select_action(self, engine: GwentEngine) -> int:
        legal = engine.get_legal_actions()
        if not legal:
            return PASS_ACTION
        if len(legal) == 1:
            return legal[0]

        p = engine.turn
        me = engine.players[p]

        # ── Tier 1: Easy (Random / Novice) ──
        if self.difficulty == "easy":
            if engine.sub_phase == "MULLIGAN":
                return PASS_ACTION
            if len(legal) > 1 and PASS_ACTION in legal:
                if engine.players[1 - p].passed and engine.scores[p] > engine.scores[1 - p]:
                    if self.rng.next_float() < 0.55:
                        return PASS_ACTION
                if engine.scores[p] <= engine.scores[1 - p] and len(me.hand) > 4:
                    non_pass = [a for a in legal if a != PASS_ACTION]
                    return non_pass[self.rng.rand_int(len(non_pass))]
            return legal[self.rng.rand_int(len(legal))]

        # ── Tier 2: Normal (0-ply Static Rule-Based Heuristic, No Forward Clone) ──
        if self.difficulty == "normal":
            if engine.sub_phase == "MULLIGAN":
                return self._choose_mulligan_normal(engine, p, legal)
            if engine.sub_phase == "MEDIC":
                return self._choose_medic_normal(engine, p, legal)
            if engine.sub_phase == "DECOY":
                return self._choose_decoy_target_normal(engine, p, legal)
            return self._choose_play_normal(engine, p, legal)

        # ── Tier 3: Hard (1-ply Forward Simulation via engine.clone() + Tactical Search) ──
        if engine.sub_phase == "MULLIGAN":
            return self._choose_mulligan_hard(engine, p, legal)
        if engine.sub_phase == "MEDIC":
            return self._choose_medic_hard(engine, p, legal)
        if engine.sub_phase == "DECOY":
            return self._choose_decoy_target_hard(engine, p, legal)

        return self._choose_play_hard(engine, p, legal)

    # ════════════════════════════════════════════════════════════════
    # NORMAL TIER (0-ply Static Heuristics + Stochastic Noise)
    # ════════════════════════════════════════════════════════════════

    def _choose_mulligan_normal(self, engine: GwentEngine, p: int, legal: list[int]) -> int:
        for a in legal:
            if a == PASS_ACTION:
                continue
            c = CARDS_BY_IDX[a]
            if c["_is_unit"] and not c["_is_spy"] and not c["_is_medic"] and c["strength"] <= 1:
                return a
        return PASS_ACTION

    def _choose_medic_normal(self, engine: GwentEngine, p: int, legal: list[int]) -> int:
        best_a = legal[0]
        best_val = -999.0
        for a in legal:
            c = CARDS_BY_IDX[a]
            val = float(c["strength"]) + self.rng.next_float() * 4.0
            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    def _choose_decoy_target_normal(self, engine: GwentEngine, p: int, legal: list[int]) -> int:
        return legal[self.rng.rand_int(len(legal))]

    def _choose_play_normal(self, engine: GwentEngine, p: int, legal: list[int]) -> int:
        me = engine.players[p]
        opp = engine.players[1 - p]
        my_score = engine.scores[p]
        opp_score = engine.scores[1 - p]
        winning = my_score > opp_score or (
            my_score == opp_score and me.faction == "nilfgaard" and opp.faction != "nilfgaard"
        )

        # Pass if opponent already passed and we are ahead
        if opp.passed and winning:
            return PASS_ACTION

        # Casual player often passes prematurely with a modest lead in Round 1/2, or fights too long when behind
        if (not opp.passed) and engine.gems[p] == 2:
            if winning and (my_score - opp_score >= 5) and self.rng.next_float() < 0.42:
                return PASS_ACTION
            if opp_score - my_score >= 38:
                return PASS_ACTION

        best_action = PASS_ACTION
        best_val = -1e9

        for a in legal:
            noise = self.rng.next_float() * 6.0
            if a == PASS_ACTION:
                if not winning and engine.gems[p] == 1 and len(legal) > 1:
                    continue
                val = -4.0 + noise
            elif a == USE_LEADER_ACTION:
                val = 6.5 + noise
            elif HORN_ROW_OFFSET <= a < HORN_ROW_OFFSET + 3:
                r_key = ROW_KEYS[a - HORN_ROW_OFFSET]
                n_units = len(me.rows[r_key])
                val = (8.0 if n_units >= 2 else 3.0) + noise
            else:
                c = CARDS_BY_IDX[a]
                if c["id"] == "decoy":
                    val = 2.5 + noise
                elif c["_is_spy"]:
                    # Casual 0-ply player undervalues giving points to opponent
                    val = (5.0 - 0.5 * c["strength"] if engine.gems[p] == 2 else 0.5) + noise
                elif c["type"] == "special":
                    val = (7.5 if opp_score >= my_score else 3.5) + noise
                elif c["type"] == "weather":
                    val = (5.5 if opp_score > my_score else 3.0) + noise
                else:
                    # Greedy: plays highest strength (including heroes) first!
                    val = (
                        float(c["strength"])
                        + (1.0 if c["_is_medic"] else 0.0)
                        + (1.0 if c["_is_bond"] or c["_is_morale"] or c["_is_muster"] else 0.0)
                        + noise
                    )

            if val > best_val:
                best_val = val
                best_action = a

        return best_action

    # ════════════════════════════════════════════════════════════════
    # HARD TIER (1-ply Forward Simulation via engine.clone() + Search)
    # ════════════════════════════════════════════════════════════════

    def _choose_mulligan_hard(self, engine: GwentEngine, p: int, legal: list[int]) -> int:
        pl = engine.players[p]
        hand_counts = {}
        for inst in pl.hand:
            cidx = inst["card_idx"]
            hand_counts[cidx] = hand_counts.get(cidx, 0) + 1

        worst_action = PASS_ACTION
        worst_score = 0.0
        for a in legal:
            if a == PASS_ACTION:
                continue
            c = CARDS_BY_IDX[a]
            if c["_is_muster"] and hand_counts.get(a, 0) > 1:
                return a
            if c["type"] == "weather" and hand_counts.get(a, 0) > 1:
                return a
            if c["_is_unit"] and not c["_is_spy"] and not c["_is_medic"] and c["strength"] <= 2:
                score = 3.0 - c["strength"]
                if score > worst_score:
                    worst_score = score
                    worst_action = a
        return worst_action

    def _choose_medic_hard(self, engine: GwentEngine, p: int, legal: list[int]) -> int:
        best_a = legal[0]
        best_val = -999.0
        for a in legal:
            c = CARDS_BY_IDX[a]
            if c["_is_medic"]:
                val = 30.0 + c["strength"]
            elif c["_is_spy"]:
                val = 16.0 - c["strength"] if engine.gems[p] > 1 else -10.0
            else:
                val = float(c["strength"]) + (4.5 if c["_is_bond"] or c["_is_morale"] or c["_is_muster"] else 0.0)
            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    def _choose_decoy_target_hard(self, engine: GwentEngine, p: int, legal: list[int]) -> int:
        best_a = legal[0]
        best_val = -999.0
        for a in legal:
            c = CARDS_BY_IDX[a]
            if c["_is_spy"]:
                val = 50.0 + c["strength"]
            elif c["_is_medic"]:
                val = 40.0
            else:
                val = 10.0 - c["strength"]
            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    def evaluate_state(self, engine: GwentEngine, p: int) -> float:
        return self._evaluate_state_hard(engine, p)

    def _evaluate_state(self, engine: GwentEngine, p: int) -> float:
        return self._evaluate_state_hard(engine, p)

    def _evaluate_state_hard(self, engine: GwentEngine, p: int) -> float:
        if engine.done:
            if engine.winner == p:
                return 500.0
            if engine.winner == (1 - p):
                return -500.0
            return 0.0

        me = engine.players[p]
        opp = engine.players[1 - p]
        my_score = engine.scores[p]
        opp_score = engine.scores[1 - p]
        diff = float(my_score - opp_score)

        gem_diff = (engine.gems[p] - engine.gems[1 - p]) * 38.0
        hand_diff = (len(me.hand) - len(opp.hand)) * 10.5

        lead_bonus = 0.0
        if my_score > opp_score or (my_score == opp_score and me.faction == "nilfgaard" and opp.faction != "nilfgaard"):
            lead_bonus = 14.0 if opp.passed else 5.0

        return diff + gem_diff + hand_diff + lead_bonus

    def _choose_play_hard(self, engine: GwentEngine, p: int, legal: list[int]) -> int:
        me = engine.players[p]
        opp = engine.players[1 - p]
        my_score = engine.scores[p]
        opp_score = engine.scores[1 - p]
        winning = my_score > opp_score or (
            my_score == opp_score and me.faction == "nilfgaard" and opp.faction != "nilfgaard"
        )

        # 1. If opponent already passed and we are winning, PASS immediately!
        if opp.passed and winning:
            return PASS_ACTION

        # 2. Strategic Pass in Round 1/2 if opponent is way ahead (> 20 pts) and we have 2 gems
        if (not opp.passed) and engine.gems[p] == 2 and (opp_score - my_score >= 20) and len(me.hand) <= len(opp.hand) + 1:
            for a in legal:
                if a < NUM_CARDS and CARDS_BY_IDX[a]["_is_spy"]:
                    return a
            return PASS_ACTION

        # 3. Always prioritize Spies early when opponent has not passed (or when bailing out on 2 gems)
        for a in legal:
            if a < NUM_CARDS and CARDS_BY_IDX[a]["_is_spy"]:
                c_spy = CARDS_BY_IDX[a]
                if not opp.passed:
                    return a
                if engine.gems[p] == 2 and (opp_score - my_score >= 15 or engine.gems[1 - p] == 1):
                    return a
                if my_score > opp_score + c_spy["strength"]:
                    return a

        # 4. Decoy a Spy or Medic on our board if available
        for a in legal:
            if a < NUM_CARDS and CARDS_BY_IDX[a]["id"] == "decoy":
                for r in ROW_KEYS:
                    for inst in me.rows[r]:
                        c = CARDS_BY_IDX[inst["card_idx"]]
                        if c["_is_spy"] or (c["_is_medic"] and engine._revivable_grave_Check(p)):
                            return a

        # 5. Evaluate all candidate actions via 1-ply simulation
        best_action = PASS_ACTION
        best_val = -1e9
        base_diff = my_score - opp_score

        for a in legal:
            jitter = self.rng.next_float() * 0.05
            if a == PASS_ACTION:
                if not winning and engine.gems[p] == 1 and len(legal) > 1:
                    continue
                val = self._evaluate_state_hard(engine, p) + jitter
                if winning and not opp.passed and my_score - opp_score >= 15 and engine.gems[p] == 2:
                    val += 9.0
                else:
                    val -= 6.0
            else:
                if a < NUM_CARDS and CARDS_BY_IDX[a]["id"] == "decoy":
                    continue
                sim = engine.clone()
                sim.step(a)
                guard = 0
                while not sim.done and sim.turn == p and sim.sub_phase in ("MEDIC", "DECOY") and guard < 5:
                    sub_legal = sim.get_legal_actions()
                    if not sub_legal:
                        break
                    sub_a = (
                        self._choose_medic_hard(sim, p, sub_legal)
                        if sim.sub_phase == "MEDIC"
                        else self._choose_decoy_target_hard(sim, p, sub_legal)
                    )
                    sim.step(sub_a)
                    guard += 1
                val = self._evaluate_state_hard(sim, p) + jitter

                # Prevent self-harming Scorch or zero-effect special/weather cards
                if a < NUM_CARDS and CARDS_BY_IDX[a]["type"] in ("special", "weather"):
                    new_diff = sim.scores[p] - sim.scores[1 - p]
                    if new_diff <= base_diff:
                        val -= 25.0

                # Minimum-card commitment when opponent already passed
                if opp.passed and sim.scores[p] > sim.scores[1 - p]:
                    val += 28.0 - 0.35 * (sim.scores[p] - sim.scores[1 - p])
                    if a < NUM_CARDS and CARDS_BY_IDX[a]["_is_hero"] and engine.gems[1 - p] > 1:
                        val -= 8.0
                elif a < NUM_CARDS and CARDS_BY_IDX[a]["_is_hero"] and engine.round == 1:
                    val -= 5.0

            if val > best_val:
                best_val = val
                best_action = a

        return best_action
