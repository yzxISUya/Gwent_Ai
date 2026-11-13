"""
Unified Masked PPO + PFSP League Self-Play Trainer & Evaluator for Gwent.
Features:
  - Full 8-preset training & 64-pair Duplicate Paired-Seed evaluation (`ALL_DECK_PAIRS`)
  - Calibrated Win-Probability Critic trained via BCEWithLogitsLoss (2 * sigmoid(z) - 1)
  - Batched GPU inference for PFSP neural snapshot opponents
  - Permanent milestone anchors + sliding-window snapshots in PFSP League
  - True Stage-2 Critic-only warmup + KL early stopping (`target_kl`)
"""
from __future__ import annotations
import copy
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gwent_rl.engine import (
    GwentEngine,
    HeuristicAI,
    ALL_PRESET_IDS,
    ALL_DECK_PAIRS,
    CORE_PRESET_IDS,
    CORE_DECK_PAIRS,
    OBS_DIM,
    ACTION_DIM,
)
from gwent_rl.env import GwentSingleAgentEnv
from gwent_rl.league import LeagueMember, PFSPLeague
from .networks import GwentPolicyValueNet


def make_torch_policy_fn(model: nn.Module, device: torch.device, deterministic: bool = False):
    model_copy = copy.deepcopy(model).to("cpu").eval()

    @torch.no_grad()
    def _policy_fn(engine: GwentEngine) -> int:
        obs = torch.from_numpy(engine.encode_observation()).unsqueeze(0)
        mask = torch.tensor([engine.get_legal_mask()], dtype=torch.bool)
        probs, _ = model_copy(obs, mask)
        p0 = probs[0]
        if torch.isnan(p0).any() or p0.sum() <= 0:
            legal = engine.get_legal_actions()
            return legal[0] if legal else 90
        if deterministic:
            return int(torch.argmax(p0).item())
        dist = torch.distributions.Categorical(probs=p0)
        return int(dist.sample().item())

    return _policy_fn


def snapshot_cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def composite_eval_score(ev: Dict[str, float]) -> float:
    return ev["easy"] * 0.20 + ev["normal"] * 0.35 + ev["hard"] * 0.45


def evaluate_against_heuristics(
    model: nn.Module,
    device: torch.device,
    games_per_diff: int = 128,
    base_seed: int = 80000,
    deck_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict[str, float]:
    """
    Vectorized Duplicate / Paired-Seed Evaluation.
    Defaults to `ALL_DECK_PAIRS` (all 8x8 = 64 ordered deck matchups across 4 factions & 8 leaders).
    With `games_per_diff = 128`, every single one of the 64 matchups is played twice with seats swapped.
    """
    pairs_pool = list(deck_pairs) if deck_pairs is not None else list(ALL_DECK_PAIRS)
    model.eval()
    num_pairs = max(1, games_per_diff // 2)
    results: Dict[str, float] = {}

    for d_idx, diff in enumerate(("easy", "normal", "hard")):
        engines: List[GwentEngine] = []
        bots: List[HeuristicAI] = []
        sides: List[int] = []

        for pair_idx in range(num_pairs):
            p0, p1 = pairs_pool[pair_idx % len(pairs_pool)]
            game_seed = base_seed + d_idx * 10000 + pair_idx
            for agent_side in (0, 1):
                bot = HeuristicAI(difficulty=diff, seed=1010 + d_idx * 500 + pair_idx)
                eng = GwentEngine(seed=game_seed, preset0=p0, preset1=p1)
                while not eng.done and eng.turn != agent_side:
                    eng.step(bot.select_action(eng))
                engines.append(eng)
                bots.append(bot)
                sides.append(agent_side)

        while True:
            active = [i for i, e in enumerate(engines) if not e.done]
            if not active:
                break
            batch_obs = np.array(
                [engines[i].encode_observation(pov=sides[i]) for i in active],
                dtype=np.float32,
            )
            batch_mask = np.array(
                [engines[i].get_legal_mask() for i in active],
                dtype=bool,
            )
            with torch.no_grad():
                obs_t = torch.from_numpy(batch_obs).to(device)
                mask_t = torch.from_numpy(batch_mask).to(device)
                probs, _ = model(obs_t, mask_t)
                actions = torch.argmax(probs, dim=1).cpu().numpy()

            for b_pos, idx in enumerate(active):
                eng = engines[idx]
                side = sides[idx]
                bot = bots[idx]
                eng.step(int(actions[b_pos]))
                while not eng.done and eng.turn != side:
                    eng.step(bot.select_action(eng))

        pair_scores: List[float] = []
        for pair_idx in range(num_pairs):
            pair_pts = 0.0
            for offset in (0, 1):
                idx = pair_idx * 2 + offset
                eng = engines[idx]
                side = sides[idx]
                if eng.winner == side:
                    pair_pts += 1.0
                elif eng.winner is None:
                    pair_pts += 0.5
            pair_scores.append(pair_pts / 2.0)

        results[diff] = float(np.mean(pair_scores))
        results[f"{diff}_se"] = float(np.std(pair_scores, ddof=1) / np.sqrt(num_pairs)) if num_pairs > 1 else 0.0

    model.train()
    return results


def select_best_from_top_k(
    candidates: List[Tuple[float, int, nn.Module]],
    device: torch.device,
    audit_games_per_diff: int = 384,
    audit_seed: int = 900_000,
    deck_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    tag: str = "PPO",
) -> Tuple[nn.Module, Dict[str, Any]]:
    print(f"\n[{tag} Top-{len(candidates)} Final Audit] Running {audit_games_per_diff} Duplicate games/difficulty (out-of-sample seeds)...")
    best_model = candidates[0][2]
    best_audit_score = -1.0
    best_audit_info: Dict[str, Any] = {}

    for rank_idx, (train_score, step_id, cand_model) in enumerate(candidates):
        ev = evaluate_against_heuristics(
            cand_model,
            device,
            games_per_diff=audit_games_per_diff,
            base_seed=audit_seed,
            deck_pairs=deck_pairs,
        )
        audit_score = composite_eval_score(ev)
        print(
            f"  Candidate #{rank_idx+1} (from update {step_id}, in-train={train_score*100:.1f}%) -> "
            f"{audit_games_per_diff}-game Audit: Easy={ev['easy']*100:.1f}%±{ev['easy_se']*100:.1f}% | "
            f"Normal={ev['normal']*100:.1f}%±{ev['normal_se']*100:.1f}% | "
            f"Hard={ev['hard']*100:.1f}%±{ev['hard_se']*100:.1f}% (Composite={audit_score*100:.1f}%)"
        )
        if audit_score > best_audit_score:
            best_audit_score = audit_score
            best_model = cand_model
            best_audit_info = {
                "selected_from": step_id,
                "games_per_diff": audit_games_per_diff,
                "wr_easy": float(ev["easy"] * 100),
                "se_easy": float(ev["easy_se"] * 100),
                "wr_normal": float(ev["normal"] * 100),
                "se_normal": float(ev["normal_se"] * 100),
                "wr_hard": float(ev["hard"] * 100),
                "se_hard": float(ev["hard_se"] * 100),
                "composite": float(audit_score * 100),
            }

    print(f"[{tag} Final Selected] Update {best_audit_info['selected_from']} (Composite={best_audit_info['composite']:.1f}%)\n")
    return best_model, best_audit_info


def _set_actor_trainable(model: GwentPolicyValueNet, trainable: bool) -> None:
    for mod in (model.input_proj, model.blocks, model.policy_head):
        for p in mod.parameters():
            p.requires_grad = trainable
    for p in model.value_head.parameters():
        p.requires_grad = True


def _step_opponents_batched(
    envs: List[GwentSingleAgentEnv],
    opp_members: List[LeagueMember],
    opp_model_cache: Dict[str, Tuple[GwentPolicyValueNet, bool]],
    hidden_dim: int,
    num_blocks: int,
    device: torch.device,
    rng_seed_base: int,
) -> None:
    """
    Advance all environments until each is either `done` or back on `agent_player`'s turn.
    Neural snapshot opponents are batched by model name on `device` for high GPU throughput.
    """
    # Prune stale cached opponent models evicted from league
    active_names = {m.name for m in opp_members if m.state_dict is not None}
    for cached_name in list(opp_model_cache.keys()):
        if cached_name not in active_names:
            del opp_model_cache[cached_name]

    while True:
        neural_groups: Dict[str, List[int]] = defaultdict(list)
        any_progressed = False

        for i, env in enumerate(envs):
            while not env.engine.done and env.engine.turn != env.agent_player:
                member = opp_members[i]
                if member.state_dict is None:
                    assert member.policy_fn is not None
                    env.engine.step(member.policy_fn(env.engine))
                    any_progressed = True
                else:
                    neural_groups[member.name].append(i)
                    break

        if not neural_groups:
            if not any_progressed:
                break
            continue

        for m_name, env_indices in neural_groups.items():
            if m_name not in opp_model_cache:
                # Find state_dict from the first matching member
                sample_member = opp_members[env_indices[0]]
                net = GwentPolicyValueNet(OBS_DIM, ACTION_DIM, hidden_dim, num_blocks).to(device)
                net.load_state_dict(sample_member.state_dict)
                net.eval()
                opp_model_cache[m_name] = (net, sample_member.deterministic)

            net, is_det = opp_model_cache[m_name]
            batch_obs = np.array(
                [envs[i].engine.encode_observation(pov=envs[i].engine.turn) for i in env_indices],
                dtype=np.float32,
            )
            batch_mask = np.array(
                [envs[i].engine.get_legal_mask() for i in env_indices],
                dtype=bool,
            )
            with torch.no_grad():
                obs_t = torch.from_numpy(batch_obs).to(device)
                mask_t = torch.from_numpy(batch_mask).to(device)
                probs, _ = net(obs_t, mask_t)
                if is_det:
                    actions = torch.argmax(probs, dim=-1).cpu().numpy()
                else:
                    # Safe multinomial sampling over valid masked probabilities
                    clean_probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                    row_sums = clean_probs.sum(dim=-1, keepdim=True)
                    safe_probs = torch.where(row_sums > 0, clean_probs / torch.clamp(row_sums, min=1e-12), mask_t.float())
                    actions = torch.multinomial(safe_probs, num_samples=1).squeeze(-1).cpu().numpy()

            for b_pos, idx in enumerate(env_indices):
                envs[idx].engine.step(int(actions[b_pos]))


def train_masked_ppo(
    total_timesteps: int = 2_457_600,
    num_envs: int = 32,
    num_steps: int = 256,
    hidden_dim: int = 512,
    num_blocks: int = 3,
    lr: float = 2.0e-4,
    min_lr: float = 1.5e-5,
    lr_schedule: str = "cosine",
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef_start: float = 0.020,
    ent_coef_end: float = 0.004,
    vf_coef: float = 0.75,
    target_kl: Optional[float] = 0.03,
    update_epochs: int = 4,
    minibatch_size: int = 512,
    shaping_start: float = 0.15,
    preset_ids: Optional[Sequence[str]] = None,
    eval_deck_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    eval_games_per_diff: int = 128,
    audit_games_per_diff: int = 384,
    eval_every: Optional[int] = None,
    snapshot_every: Optional[int] = None,
    warm_start_path: Optional[str | Path] = None,
    critic_warmup_updates: int = 5,
    prev_history: Optional[List[Dict[str, Any]]] = None,
    save_dir: str | Path = "models",
    device_str: str = "auto",
    seed: int = 42,
    on_eval_callback: Optional[Callable[[GwentPolicyValueNet, List[Dict[str, Any]]], None]] = None,
) -> Tuple[GwentPolicyValueNet, List[Dict[str, Any]], Dict[str, Any]]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    torch.manual_seed(seed)
    np.random.seed(seed)

    active_presets = tuple(preset_ids) if preset_ids is not None else ALL_PRESET_IDS
    eval_pairs = list(eval_deck_pairs) if eval_deck_pairs is not None else list(ALL_DECK_PAIRS)

    model = GwentPolicyValueNet(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    ).to(device)

    warm_started = False
    if warm_start_path is not None and Path(warm_start_path).exists():
        try:
            state = torch.load(warm_start_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            warm_started = True
            print(f"[WarmStart] Loaded compatible checkpoint from {Path(warm_start_path).resolve()}")
        except Exception as e:
            print(f"[WarmStart] Skipped {warm_start_path} (architecture/obs mismatch or error): {e}")

    league = PFSPLeague(seed=seed + (100 if warm_started else 0), max_snapshots=12, max_anchors=4)
    history: List[Dict[str, Any]] = list(prev_history) if (warm_started and prev_history) else []

    if warm_started and history:
        last_rec = history[-1]
        start_update = int(last_rec.get("update", 0))
        start_steps = int(last_rec.get("steps", 0))
        episodes_completed = int(last_rec.get("episodes", 0))
        ckpt_idx = int(last_rec.get("checkpoint", len(history) - 1))
        league.learner_elo = float(last_rec.get("elo", 1200.0))
        s1_sd = snapshot_cpu_state_dict(model)
        league.add_snapshot("snap_s1_greedy", state_dict=s1_sd, deterministic=True, is_anchor=True)
        league.add_snapshot("snap_s1_stoch", state_dict=s1_sd, deterministic=False, is_anchor=False)
    else:
        start_update = 0
        start_steps = 0
        episodes_completed = 0
        ckpt_idx = 0

    envs = [
        GwentSingleAgentEnv(
            opponent="normal",
            seed=seed * 10 + i,
            shaping_weight=0.0 if warm_started else shaping_start,
            gamma=gamma,
            preset_ids=list(active_presets),
        )
        for i in range(num_envs)
    ]
    opp_members: List[LeagueMember] = []
    opp_model_cache: Dict[str, Tuple[GwentPolicyValueNet, bool]] = {}

    batch_size = num_envs * num_steps
    num_updates = max(1, total_timesteps // batch_size)
    eval_interval = eval_every if eval_every is not None else max(1, num_updates // 20)
    snap_interval = snapshot_every if snapshot_every is not None else max(1, num_updates // 12)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    obs_buf = torch.zeros((num_steps, num_envs, OBS_DIM), dtype=torch.float32, device=device)
    mask_buf = torch.zeros((num_steps, num_envs, ACTION_DIM), dtype=torch.bool, device=device)
    actions_buf = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    for i, env in enumerate(envs):
        member = league.sample_opponent()
        opp_members.append(member)
        env.engine.reset(
            seed=env.rng.rand_int(2**31 - 1),
            preset0=env.preset_ids[env.rng.rand_int(len(env.preset_ids))],
            preset1=env.preset_ids[env.rng.rand_int(len(env.preset_ids))],
        )
        env.agent_player = env.rng.rand_int(2)
        env.episode_Count += 1

    _step_opponents_batched(envs, opp_members, opp_model_cache, hidden_dim, num_blocks, device, seed)
    next_obs = torch.tensor(np.array([e._get_obs()["obs"] for e in envs]), dtype=torch.float32, device=device)
    next_mask = torch.tensor(np.array([e._get_obs()["action_mask"] for e in envs]), dtype=torch.bool, device=device)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)

    top_candidates: List[Tuple[float, int, nn.Module]] = []
    eval0 = evaluate_against_heuristics(
        model, device, games_per_diff=eval_games_per_diff, deck_pairs=eval_pairs
    )
    init_score = composite_eval_score(eval0)
    top_candidates.append((init_score, start_update, copy.deepcopy(model)))
    if not history:
        history.append(
            {
                "checkpoint": 0,
                "update": 0,
                "steps": 0,
                "episodes": 0,
                "elo": float(league.learner_elo),
                "wr_easy": float(eval0["easy"] * 100),
                "se_easy": float(eval0["easy_se"] * 100),
                "wr_normal": float(eval0["normal"] * 100),
                "se_normal": float(eval0["normal_se"] * 100),
                "wr_hard": float(eval0["hard"] * 100),
                "se_hard": float(eval0["hard_se"] * 100),
                "pg_loss": 0.0,
                "v_loss": 0.693,
                "entropy": 2.4,
                "approx_kl": 0.0,
            }
        )

    t0 = time.perf_counter()
    print(
        f"[Masked PPO] Device={device} | Presets={len(active_presets)} | Updates={num_updates} "
        f"({total_timesteps:,} steps) | WarmStart={warm_started}"
    )
    print(
        f"[Masked PPO] Init Eval ({eval_games_per_diff} games/diff): "
        f"Easy={eval0['easy']*100:.1f}% | Normal={eval0['normal']*100:.1f}% | Hard={eval0['hard']*100:.1f}%"
    )

    anchor_count = 0
    for update in range(1, num_updates + 1):
        global_update = start_update + update
        progress = (update - 1.0) / max(1, num_updates)

        is_critic_warmup = warm_started and (update <= critic_warmup_updates)
        if warm_started and update == 1 and critic_warmup_updates > 0:
            _set_actor_trainable(model, False)
            print(f"[WarmStart] Freezing Actor for {critic_warmup_updates} Critic-only warmup updates...")
        elif warm_started and update == critic_warmup_updates + 1:
            _set_actor_trainable(model, True)
            print("[WarmStart] Critic warmup complete; unfreezing full Actor-Critic network.")

        shaping_w = 0.0 if warm_started else max(0.0, shaping_start * (1.0 - progress * 1.25))
        for env in envs:
            env.shaping_weight = shaping_w

        ent_coef = ent_coef_start * (1.0 - progress) + ent_coef_end * progress

        if lr_schedule == "cosine":
            if warm_started and update <= 3:
                cur_lr = min_lr + (lr - min_lr) * (update / 3.0)
            else:
                offset_u = 3.0 if warm_started else 0.0
                cos_p = (update - offset_u) / max(1.0, num_updates - offset_u)
                cur_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + np.cos(np.pi * cos_p))
        else:
            cur_lr = max(min_lr, lr * (1.0 - progress))

        for pg in optimizer.param_groups:
            pg["lr"] = float(cur_lr)

        # Rollout Collection
        for step in range(num_steps):
            obs_buf[step] = next_obs
            mask_buf[step] = next_mask
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_obs, next_mask)
                values_buf[step] = value
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            actions_cpu = action.cpu().numpy()
            prev_pots = [envs[i].step_agent_only(int(actions_cpu[i])) for i in range(num_envs)]

            _step_opponents_batched(envs, opp_members, opp_model_cache, hidden_dim, num_blocks, device, seed + step)

            obs_np, mask_np, rew_np, done_np = [], [], [], []
            need_post_reset_opp = False
            for i, env in enumerate(envs):
                d, r, done, _, info = env.finalize_step(prev_pots[i])
                if done:
                    episodes_completed += 1
                    w = info["winner"]
                    learner_won = None if w is None else (w == env.agent_player)
                    league.record_result(opp_members[i].name, learner_won, k_factor=6.0 if warm_started else 8.0)
                    opp_members[i] = league.sample_opponent()
                    env.engine.reset(
                        seed=env.rng.rand_int(2**31 - 1),
                        preset0=env.preset_ids[env.rng.rand_int(len(env.preset_ids))],
                        preset1=env.preset_ids[env.rng.rand_int(len(env.preset_ids))],
                    )
                    env.agent_player = env.rng.rand_int(2)
                    env.episode_Count += 1
                    need_post_reset_opp = True
                rew_np.append(r)
                done_np.append(1.0 if done else 0.0)

            if need_post_reset_opp:
                _step_opponents_batched(envs, opp_members, opp_model_cache, hidden_dim, num_blocks, device, seed + step + 9999)

            for env in envs:
                d = env._get_obs()
                obs_np.append(d["obs"])
                mask_np.append(d["action_mask"])

            rewards_buf[step] = torch.tensor(rew_np, dtype=torch.float32, device=device)
            next_obs = torch.tensor(np.array(obs_np), dtype=torch.float32, device=device)
            next_mask = torch.tensor(np.array(mask_np), dtype=torch.bool, device=device)
            next_done = torch.tensor(done_np, dtype=torch.float32, device=device)

        # Compute GAE advantages & returns
        with torch.no_grad():
            _, _, _, next_value = model.get_action_and_value(next_obs, next_mask)
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, OBS_DIM))
        b_mask = mask_buf.reshape((-1, ACTION_DIM))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = torch.clamp(returns.reshape(-1), -1.0, 1.0)
        b_target_probs = torch.clamp((b_returns + 1.0) * 0.5, 0.0, 1.0)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std(unbiased=False) + 1e-8)

        b_inds = np.arange(batch_size)
        pg_losses, v_losses, entropies, kls = [], [], [], []
        for _ in range(update_epochs):
            np.random.shuffle(b_inds)
            epoch_kls = []
            for start in range(0, batch_size, minibatch_size):
                mb = b_inds[start : start + minibatch_size]
                _, newlogprob, entropy, _, new_v_logit = model.get_action_and_value(
                    b_obs[mb], b_mask[mb], b_actions[mb], return_logit=True
                )
                v_loss = F.binary_cross_entropy_with_logits(new_v_logit, b_target_probs[mb])

                if is_critic_warmup:
                    pg_loss = torch.zeros((), device=device)
                    ent_loss = entropy.mean()
                    loss = vf_coef * v_loss
                    approx_kl = 0.0
                else:
                    logratio = torch.clamp(newlogprob - b_logprobs[mb], -20.0, 20.0)
                    ratio = logratio.exp()
                    with torch.no_grad():
                        approx_kl = float(((ratio - 1.0) - logratio).mean().item())
                    mb_adv = b_advantages[mb]
                    pg_loss1 = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    ent_loss = entropy.mean()
                    loss = pg_loss - ent_coef * ent_loss + vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropies.append(float(ent_loss.item()))
                epoch_kls.append(approx_kl)
                kls.append(approx_kl)

            if (not is_critic_warmup) and (target_kl is not None) and (np.mean(epoch_kls) > target_kl * 1.5):
                break

        if update % 2 == 0 or update == 1:
            fps_instant = int((update * batch_size) / max(1e-3, time.perf_counter() - t0))
            print(
                f"  [PPO Heartbeat] Upd {global_update:03d}/{start_update + num_updates} | "
                f"Steps={start_steps + update * batch_size:,} | Games={episodes_completed:,} ({fps_instant} sps) | "
                f"V-BCE={np.mean(v_losses):.4f} | KL={np.mean(kls):.4f} | Elo={league.learner_elo:.0f}"
            )

        if update % snap_interval == 0:
            snap_idx = global_update
            snap_seq = update // snap_interval
            is_det = (snap_seq % 2 == 0)
            is_anch = (anchor_count < 4) and (snap_seq % 3 == 0)
            if is_anch:
                anchor_count += 1
            league.add_snapshot(
                f"snap_ppo_{snap_idx}",
                state_dict=snapshot_cpu_state_dict(model),
                deterministic=is_det,
                is_anchor=is_anch,
            )

        if update % eval_interval == 0 or update == num_updates:
            ckpt_idx += 1
            ev = evaluate_against_heuristics(
                model, device, games_per_diff=eval_games_per_diff, deck_pairs=eval_pairs
            )
            cur_steps = start_steps + update * batch_size
            rec = {
                "checkpoint": ckpt_idx,
                "update": global_update,
                "steps": cur_steps,
                "episodes": episodes_completed,
                "elo": float(league.learner_elo),
                "wr_easy": float(ev["easy"] * 100),
                "se_easy": float(ev["easy_se"] * 100),
                "wr_normal": float(ev["normal"] * 100),
                "se_normal": float(ev["normal_se"] * 100),
                "wr_hard": float(ev["hard"] * 100),
                "se_hard": float(ev["hard_se"] * 100),
                "pg_loss": float(np.mean(pg_losses)),
                "v_loss": float(np.mean(v_losses)),
                "entropy": float(np.mean(entropies)),
                "approx_kl": float(np.mean(kls)),
            }
            history.append(rec)
            score = composite_eval_score(ev)
            torch.save(model.state_dict(), save_path / "ppo_latest.pt")
            if not top_candidates or score >= top_candidates[0][0]:
                torch.save(model.state_dict(), save_path / "ppo_best.pt")
            top_candidates.append((score, global_update, copy.deepcopy(model)))
            top_candidates.sort(key=lambda x: x[0], reverse=True)
            top_candidates = top_candidates[:3]

            fps = int((update * batch_size) / max(1e-3, time.perf_counter() - t0))
            print(
                f"[PPO] Upd {global_update:03d}/{start_update + num_updates} | Steps={cur_steps:,} (Games={episodes_completed:,}, {fps} sps) | "
                f"Elo={rec['elo']:.0f} | WR Easy={rec['wr_easy']:.1f}%±{rec['se_easy']:.1f}% "
                f"Normal={rec['wr_normal']:.1f}%±{rec['se_normal']:.1f}% "
                f"Hard={rec['wr_hard']:.1f}%±{rec['se_hard']:.1f}%"
            )
            if on_eval_callback is not None:
                on_eval_callback(model, history)

    _set_actor_trainable(model, True)
    best_model, audit_info = select_best_from_top_k(
        top_candidates,
        device,
        audit_games_per_diff=audit_games_per_diff,
        audit_seed=900_000,
        deck_pairs=eval_pairs,
        tag="PPO",
    )
    return best_model, history, audit_info
