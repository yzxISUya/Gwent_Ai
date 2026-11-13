"""
Single-Observer Information Set Monte Carlo Tree Search (SO-IS-MCTS) for Gwent.
Combines strict non-cheating determinization from the active player's POV
with Masked PPO policy priors and calibrated win-value leaf evaluation.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from gwent_rl.engine import ACTION_DIM, OBS_DIM, GwentEngine
from gwent_rl.engine.data import (
    CARDS_BY_IDX,
    HORN_ROW_OFFSET,
    NUM_CARDS,
    PASS_ACTION,
    ROW_KEYS,
)
from gwent_rl.engine.rng import RngState


EvalFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]]


def compute_opp_unseen_counts(engine: GwentEngine, pov: int) -> List[int]:
    """
    Deduce exact multiset of cards currently hidden in opponent's (hand + deck + mulligan_discards)
    strictly from public zones and the acting player's own zones (zero hidden-info leakage).
    """
    me = engine.players[pov]
    opp = engine.players[1 - pov]

    vis_counts = [0] * NUM_CARDS
    for p in engine.players:
        for r in ROW_KEYS:
            for inst in p.rows[r]:
                vis_counts[inst["card_idx"]] += 1
            if p.row_horn[r] is not None and p.row_horn[r].get("card_idx", -1) >= 0:
                vis_counts[p.row_horn[r]["card_idx"]] += 1
        for inst in p.graveyard:
            vis_counts[inst["card_idx"]] += 1
    for w_inst in engine.weather_cards:
        vis_counts[w_inst["card_idx"]] += 1

    my_unseen = [0] * NUM_CARDS
    for inst in me.hand:
        my_unseen[inst["card_idx"]] += 1
    for inst in me.deck:
        my_unseen[inst["card_idx"]] += 1
    for inst in me.mulligan_discards:
        my_unseen[inst["card_idx"]] += 1

    opp_unseen = [0] * NUM_CARDS
    for c in range(NUM_CARDS):
        rem = (me.initial_counts[c] + opp.initial_counts[c]) - vis_counts[c] - my_unseen[c]
        if rem > 0:
            opp_unseen[c] = rem
    return opp_unseen


def determinize_engine(engine: GwentEngine, pov: int, rng: RngState) -> GwentEngine:
    """
    Create a cloned simulation world consistent with player `pov`'s information set:
    - Opponent hand & deck are re-sampled uniformly from `opp_unseen` preserving sizes.
    - Own deck order is re-shuffled (keeping mulligan_discards intact during MULLIGAN).
    - Simulation RNG is re-seeded independently so future randomness cannot leak.
    """
    sim = engine.clone()
    opp_idx = 1 - pov
    opp = sim.players[opp_idx]
    me = sim.players[pov]

    h_opp = len(opp.hand)
    d_opp = len(opp.deck) + len(opp.mulligan_discards)
    opp_unseen_counts = compute_opp_unseen_counts(sim, pov)

    pool: List[Dict] = []
    for cidx, cnt in enumerate(opp_unseen_counts):
        if cnt <= 0:
            continue
        cid = CARDS_BY_IDX[cidx]["id"]
        for _ in range(cnt):
            sim.uid_seq += 1
            pool.append({
                "uid": sim.uid_seq,
                "card_id": cid,
                "card_idx": cidx,
                "owner": opp_idx,
            })

    pool = rng.shuffle(pool)
    opp.hand = pool[:h_opp]
    opp.deck = pool[h_opp : h_opp + d_opp]
    opp.mulligan_discards = []

    if me.deck:
        me.deck = rng.shuffle(me.deck)

    sim.rng = RngState(rng.rand_int(2**31 - 1) + 1)
    return sim


@dataclass
class ISEdge:
    visits: int = 0
    avail_visits: int = 0
    value_sum: float = 0.0
    prior: float = 0.0


@dataclass
class ISNode:
    turn: int
    expanded: bool = False
    value_est: float = 0.0
    priors: Optional[np.ndarray] = None
    edges: Dict[int, ISEdge] = field(default_factory=dict)
    children: Dict[int, "ISNode"] = field(default_factory=dict)


class GwentISMCTS:
    """
    Single-Observer IS-MCTS with PUCT selection, availability-adjusted exploration,
    FPU, and neural Policy+Value leaf evaluation.
    """

    def __init__(
        self,
        eval_fn: EvalFn,
        num_simulations: int = 96,
        max_depth: int = 6,
        c_puct: float = 1.0,
        max_children: int = 5,
        fpu_reduction: float = 0.15,
        skip_prob_threshold: float = 0.96,
        seed: int = 42,
    ):
        self.eval_fn = eval_fn
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.max_children = max_children
        self.fpu_reduction = fpu_reduction
        self.skip_prob_threshold = skip_prob_threshold
        self.rng = RngState(seed)

    def _terminal_value(self, eng: GwentEngine, root_pov: int) -> float:
        if eng.winner == root_pov:
            return 1.0
        if eng.winner == (1 - root_pov):
            return -1.0
        return 0.0

    def _evaluate_node(
        self,
        eng: GwentEngine,
        root_pov: int,
        use_unseen_opp_union: bool = False,
    ) -> Tuple[np.ndarray, float]:
        turn = eng.turn
        obs = eng.encode_observation(turn)
        mask = np.array(eng.get_legal_mask(), dtype=bool)
        if use_unseen_opp_union and turn != root_pov and eng.sub_phase == "PLAY":
            unseen = compute_opp_unseen_counts(eng, root_pov)
            for cidx, cnt in enumerate(unseen):
                if cnt > 0:
                    c = CARDS_BY_IDX[cidx]
                    if c["id"] == "commander-s-horn":
                        pl = eng.players[turn]
                        for r_i, r_k in enumerate(ROW_KEYS):
                            if pl.row_horn[r_k] is None:
                                mask[HORN_ROW_OFFSET + r_i] = True
                    elif c["id"] == "decoy":
                        if eng._has_decoy_target(turn):
                            mask[cidx] = True
                    else:
                        mask[cidx] = True
        probs, val_turn = self.eval_fn(obs, mask)
        val_root = float(val_turn) if turn == root_pov else -float(val_turn)
        return probs, val_root

    def search(
        self,
        engine: GwentEngine,
        time_budget_ms: Optional[float] = None,
    ) -> Tuple[int, np.ndarray, float, Dict[str, float]]:
        root_pov = engine.turn
        legal = engine.get_legal_actions()
        if not legal:
            return PASS_ACTION, np.zeros(ACTION_DIM, dtype=np.float32), 0.0, {"sims": 0, "nn_evals": 0, "elapsed_ms": 0.0}

        # Root evaluation (always 1 NN call to establish calibrated root prior & baseline value)
        root_probs, root_val = self._evaluate_node(engine, root_pov, use_unseen_opp_union=False)
        nn_evals = 1

        if len(legal) == 1 or self.num_simulations <= 0:
            probs_out = np.zeros(ACTION_DIM, dtype=np.float32)
            probs_out[legal[0]] = 1.0
            return legal[0], probs_out, root_val, {"sims": 0, "nn_evals": nn_evals, "elapsed_ms": 0.0}

        sorted_legal = sorted(legal, key=lambda a: float(root_probs[a]), reverse=True)
        if float(root_probs[sorted_legal[0]]) >= self.skip_prob_threshold:
            probs_out = np.zeros(ACTION_DIM, dtype=np.float32)
            for a in legal:
                probs_out[a] = float(root_probs[a])
            return sorted_legal[0], probs_out, root_val, {"sims": 0, "nn_evals": nn_evals, "elapsed_ms": 0.0}

        root = ISNode(turn=root_pov, expanded=True, value_est=root_val, priors=root_probs)
        root_cands = sorted_legal[: self.max_children]
        if PASS_ACTION in legal and PASS_ACTION not in root_cands and engine.sub_phase == "PLAY":
            root_cands.append(PASS_ACTION)

        for a in root_cands:
            root.edges[a] = ISEdge(prior=float(root_probs[a]))

        t_start = time.perf_counter()
        sims_done = 0

        for sim_idx in range(self.num_simulations):
            if time_budget_ms is not None and sim_idx > 0:
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                if elapsed_ms >= time_budget_ms:
                    break

            sim_eng = determinize_engine(engine, root_pov, self.rng)
            node = root
            path: List[Tuple[ISNode, int]] = []
            depth = 0
            leaf_val = 0.0

            while True:
                if sim_eng.done:
                    leaf_val = self._terminal_value(sim_eng, root_pov)
                    break

                if depth >= self.max_depth:
                    _, leaf_val = self._evaluate_node(sim_eng, root_pov, use_unseen_opp_union=False)
                    nn_evals += 1
                    break

                sim_legal = sim_eng.get_legal_actions()
                if not sim_legal:
                    leaf_val = self._terminal_value(sim_eng, root_pov)
                    break

                if not node.expanded:
                    probs, leaf_val = self._evaluate_node(sim_eng, root_pov, use_unseen_opp_union=True)
                    nn_evals += 1
                    node.expanded = True
                    node.value_est = leaf_val
                    node.priors = probs
                    cand_actions = sorted(sim_legal, key=lambda a: float(probs[a]), reverse=True)[: self.max_children]
                    if PASS_ACTION in sim_legal and PASS_ACTION not in cand_actions and sim_eng.sub_phase == "PLAY":
                        cand_actions.append(PASS_ACTION)
                    for a in cand_actions:
                        node.edges[a] = ISEdge(prior=float(probs[a]))
                    break

                if node is root:
                    avail_actions = root_cands
                else:
                    # Dynamically include legal actions for the current determinization using cached union priors
                    for a in sim_legal:
                        if a not in node.edges and len(node.edges) < max(self.max_children * 2, 16):
                            pr = float(node.priors[a]) if node.priors is not None else 1.0 / max(1, len(sim_legal))
                            node.edges[a] = ISEdge(prior=max(pr, 1e-3))
                    avail_actions = [a for a in sim_legal if a in node.edges]
                    if not avail_actions:
                        avail_actions = sim_legal[: self.max_children]
                        for a in avail_actions:
                            node.edges[a] = ISEdge(prior=1.0 / max(1, len(avail_actions)))

                is_root_player = (sim_eng.turn == root_pov)
                for a in avail_actions:
                    node.edges[a].avail_visits += 1

                # Re-normalize local prior over currently available actions in this determinization
                prior_sum = sum(node.edges[a].prior for a in avail_actions)
                if prior_sum <= 1e-8:
                    prior_sum = float(len(avail_actions))
                    use_uniform_p = True
                else:
                    use_uniform_p = False

                parent_q_turn = node.value_est if is_root_player else -node.value_est
                fpu_val = parent_q_turn - self.fpu_reduction

                best_a = avail_actions[0]
                best_score = -1e18
                for a in avail_actions:
                    edge = node.edges[a]
                    if edge.visits == 0:
                        q_est = fpu_val
                    else:
                        q_root = edge.value_sum / edge.visits
                        q_est = q_root if is_root_player else -q_root
                    local_p = (1.0 / prior_sum) if use_uniform_p else (edge.prior / prior_sum)
                    u_est = self.c_puct * local_p * math.sqrt(max(1, edge.avail_visits)) / (1.0 + edge.visits)
                    sc = q_est + u_est
                    if sc > best_score:
                        best_score = sc
                        best_a = a

                path.append((node, best_a))
                sim_eng.step(best_a)
                depth += 1

                if best_a not in node.children:
                    node.children[best_a] = ISNode(turn=sim_eng.turn)
                node = node.children[best_a]

            for p_node, p_act in path:
                e = p_node.edges[p_act]
                e.visits += 1
                e.value_sum += leaf_val
            sims_done += 1

        # Build visit-count policy distribution and choose best root action
        visit_probs = np.zeros(ACTION_DIM, dtype=np.float32)
        total_v = sum(e.visits for e in root.edges.values())
        if total_v > 0:
            for a, e in root.edges.items():
                visit_probs[a] = e.visits / float(total_v)
            best_action = max(
                root.edges.keys(),
                key=lambda a: (root.edges[a].visits, root.edges[a].value_sum / max(1, root.edges[a].visits), root.edges[a].prior),
            )
            best_edge = root.edges[best_action]
            search_val = (best_edge.value_sum / best_edge.visits) if best_edge.visits > 0 else root_val
        else:
            for a in legal:
                visit_probs[a] = float(root_probs[a])
            best_action = max(legal, key=lambda a: float(root_probs[a]))
            search_val = root_val

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        stats = {
            "sims": sims_done,
            "nn_evals": nn_evals,
            "elapsed_ms": elapsed_ms,
        }
        return int(best_action), visit_probs, float(search_val), stats


class OnnxEvaluator:
    """Callable `(obs, mask) -> (probs, value)` backed by onnxruntime CPU session."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])

    def __call__(self, obs: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        probs, val = self.sess.run(
            ["probs", "value"],
            {"obs": obs[None, :].astype(np.float32, copy=False), "mask": mask[None, :].astype(bool, copy=False)},
        )
        return probs[0], float(val.reshape(-1)[0])
