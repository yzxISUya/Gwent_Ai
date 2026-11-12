"""
Information-Set Monte-Carlo Tree Search (ISMCTS) for Gwent.
Handles imperfect information (opponent's hidden hand vs deck order) by sampling
consistent determinizations (`opp.hand` U `opp.deck` reshuffled and split to
preserve `len(opp.hand)` and `len(opp.deck)`) and using PUCT guided by a
Policy-Value Evaluator (PyTorch, ONNX Runtime, or Heuristic AI).
"""
from __future__ import annotations
import math
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from gwent_rl.engine import GwentEngine, HeuristicAI, ACTION_DIM
from gwent_rl.engine.rng import RngState


EvaluatorFn = Callable[[GwentEngine, int], Tuple[np.ndarray, float]]


class ISMCTSNode:
    __slots__ = ("visits", "value_sum", "prior", "children")

    def __init__(self, prior: float = 1.0):
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.children: Dict[int, "ISMCTSNode"] = {}

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0


def determinize_engine(engine: GwentEngine, root_player: int, rng: RngState) -> GwentEngine:
    """
    Clones `engine` and randomizes all hidden information from `root_player`'s POV:
    - Opponent's `hand` and `deck` are pooled, shuffled, and re-dealt to match exact counts.
    - Root player's `deck` order is shuffled.
    - Engine's internal `rng` is re-seeded so random game events (e.g. Monster keep) vary.
    """
    sim = engine.clone()
    sim.rng = RngState(rng.rand_int(2**31 - 1))

    me = sim.players[root_player]
    opp = sim.players[1 - root_player]

    me.deck = rng.shuffle(me.deck)

    opp_hand_len = len(opp.hand)
    unseen_pool = rng.shuffle(opp.hand + opp.deck + opp.mulligan_discards)
    opp.hand = unseen_pool[:opp_hand_len]
    opp.deck = unseen_pool[opp_hand_len:]
    opp.mulligan_discards.clear()
    return sim


class ISMCTSAgent:
    def __init__(
        self,
        evaluator: Optional[EvaluatorFn] = None,
        num_simulations: int = 64,
        c_puct: float = 1.5,
        seed: int = 42,
    ):
        self.rng = RngState(seed)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.heuristic = HeuristicAI("hard", seed=seed + 100)
        self.evaluator = evaluator or self._default_evaluator

    def _default_evaluator(self, engine: GwentEngine, pov: int) -> Tuple[np.ndarray, float]:
        legal = engine.get_legal_actions()
        probs = np.zeros(ACTION_DIM, dtype=np.float32)
        if legal:
            best_a = self.heuristic.select_action(engine)
            for a in legal:
                probs[a] = 0.6 if a == best_a else 0.4 / max(1, len(legal) - 1)
            probs /= max(1e-8, probs.sum())
        raw_val = self.heuristic.evaluate_state(engine, pov)
        value = math.tanh(raw_val / 45.0)
        return probs, value

    def search(self, engine: GwentEngine) -> Tuple[int, np.ndarray, float]:
        """
        Runs ISMCTS from the current state and returns:
          (best_action, visit_probs[91], root_value)
        """
        legal_root = engine.get_legal_actions()
        if len(legal_root) == 1:
            probs = np.zeros(ACTION_DIM, dtype=np.float32)
            probs[legal_root[0]] = 1.0
            _, val = self.evaluator(engine, engine.turn)
            return legal_root[0], probs, val

        root_player = engine.turn
        root = ISMCTSNode(prior=1.0)
        init_probs, root_val = self.evaluator(engine, root_player)
        for a in legal_root:
            root.children[a] = ISMCTSNode(prior=float(init_probs[a]))

        for _ in range(self.num_simulations):
            sim = determinize_engine(engine, root_player, self.rng)
            node = root
            path: List[Tuple[ISMCTSNode, int]] = []

            # Selection + Expansion along determinized trajectory
            depth = 0
            while not sim.done and depth < 12:
                cur_p = sim.turn
                legal = sim.get_legal_actions()
                if not legal:
                    break

                # Ensure all currently legal actions exist in node.children
                if not any(a in node.children for a in legal):
                    priors, leaf_val = self.evaluator(sim, cur_p)
                    for a in legal:
                        if a not in node.children:
                            node.children[a] = ISMCTSNode(prior=float(priors[a]))
                    # Evaluate leaf from root_player perspective
                    val_for_root = leaf_val if cur_p == root_player else -leaf_val
                    for parent_node, player_at_node in path:
                        parent_node.visits += 1
                        parent_node.value_sum += val_for_root if player_at_node == root_player else -val_for_root
                    break

                total_sqrt = math.sqrt(max(1, sum(node.children[a].visits for a in legal if a in node.children)))
                best_a = legal[0]
                best_ucb = -1e9
                for a in legal:
                    if a not in node.children:
                        node.children[a] = ISMCTSNode(prior=1.0 / len(legal))
                    child = node.children[a]
                    u = self.c_puct * child.prior * total_sqrt / (1 + child.visits)
                    ucb = child.q_value + u
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_a = a

                child_node = node.children[best_a]
                path.append((child_node, cur_p))
                sim.step(best_a)
                node = child_node
                depth += 1
            else:
                if sim.done:
                    if sim.winner == root_player:
                        val_for_root = 1.0
                    elif sim.winner == (1 - root_player):
                        val_for_root = -1.0
                    else:
                        val_for_root = 0.0
                else:
                    _, v = self.evaluator(sim, root_player)
                    val_for_root = v
                for parent_node, player_at_node in path:
                    parent_node.visits += 1
                    parent_node.value_sum += val_for_root if player_at_node == root_player else -val_for_root

        visit_counts = np.zeros(ACTION_DIM, dtype=np.float32)
        for a in legal_root:
            visit_counts[a] = float(root.children[a].visits)
        if visit_counts.sum() > 0:
            visit_probs = visit_counts / visit_counts.sum()
        else:
            visit_probs = init_probs

        best_action = int(np.argmax(visit_probs))
        best_q = root.children[best_action].q_value if root.children[best_action].visits > 0 else root_val
        return best_action, visit_probs, float(best_q)

    def select_action(self, engine: GwentEngine) -> int:
        action, _, _ = self.search(engine)
        return action
