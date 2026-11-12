"""
Masked PPO + PFSP League Self-Play Trainer for Gwent.
Uses vectorized `GwentSingleAgentEnv` instances and `GwentPolicyValueNet`.
"""
from __future__ import annotations
import copy
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gwent_rl.engine import GwentEngine, HeuristicAI, CORE_DECK_PAIRS, OBS_DIM, ACTION_DIM
from gwent_rl.env import GwentSingleAgentEnv
from gwent_rl.league import PFSPLeague
from .networks import GwentPolicyValueNet


def make_torch_policy_fn(model: nn.Module, device: torch.device, deterministic: bool = False):
    model_Copy = copy.deepcopy(model).to("cpu").eval()

    @torch.no_grad()
    def _policy_fn(engine: GwentEngine) -> int:
        obs = torch.from_numpy(engine.encode_observation()).unsqueeze(0)
        mask = torch.tensor([engine.get_legal_mask()], dtype=torch.bool)
        probs, _ = model_Copy(obs, mask)
        if deterministic:
            return int(torch.argmax(probs[0]).item())
        dist = torch.distributions.Categorical(probs=probs[0])
        return int(dist.sample().item())

    return _policy_fn


def evaluate_against_heuristics(
    model: nn.Module,
    device: torch.device,
    games_per_diff: int = 128,
    base_seed: int = 80000,
) -> Dict[str, float]:
    """
    Vectorized Duplicate / Paired-Seed Evaluation on the 2x2 Core Deck Pool (`nr-temeria` & `ng-white-flame`).
    Every `(game_seed, p0, p1)` deal is played twice with seats swapped (`agent_side = 0` and `agent_side = 1`),
    stepping all games for a difficulty in lockstep with batched GPU inference `(N, OBS_DIM)`.
    """
    model.eval()
    num_pairs = max(1, games_per_diff // 2)
    results: Dict[str, float] = {}

    for d_idx, diff in enumerate(("easy", "normal", "hard")):
        engines: List[GwentEngine] = []
        bots: List[HeuristicAI] = []
        sides: List[int] = []

        for pair_idx in range(num_pairs):
            p0, p1 = CORE_DECK_PAIRS[pair_idx % len(CORE_DECK_PAIRS)]
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


def train_masked_ppo(
    total_timesteps: int = 200_000,
    num_envs: int = 8,
    num_steps: int = 256,
    hidden_dim: int = 512,
    num_blocks: int = 3,
    lr: float = 3e-4,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.015,
    vf_coef: float = 0.5,
    update_epochs: int = 4,
    minibatch_size: int = 256,
    snapshot_interval: int = 15,
    eval_interval: int = 10,
    save_dir: str | Path = "checkpoints/ppo",
    device_str: str = "auto",
) -> GwentPolicyValueNet:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    league = PFSPLeague(seed=42)
    envs = [GwentSingleAgentEnv(opponent="normal", seed=100 + i) for i in range(num_envs)]
    env_opp_names = ["heuristic_normal"] * num_envs

    model = GwentPolicyValueNet(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    batch_size = num_envs * num_steps
    obs_buf = torch.zeros((num_steps, num_envs, OBS_DIM), dtype=torch.float32, device=device)
    mask_buf = torch.zeros((num_steps, num_envs, ACTION_DIM), dtype=torch.bool, device=device)
    actions_buf = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    next_obs_list = []
    next_mask_list = []
    for i, env in enumerate(envs):
        member = league.sample_opponent()
        env.set_opponent(member.policy_fn)
        env_opp_names[i] = member.name
        d, _ = env.reset()
        next_obs_list.append(d["obs"])
        next_mask_list.append(d["action_mask"])

    next_obs = torch.tensor(np.array(next_obs_list), dtype=torch.float32, device=device)
    next_mask = torch.tensor(np.array(next_mask_list), dtype=torch.bool, device=device)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)

    num_updates = max(1, total_timesteps // batch_size)
    best_hard_wr = 0.0
    t_start = time.perf_counter()

    print(f"[Masked PPO] Starting on {device} | Total steps={total_timesteps:,} | Updates={num_updates}")

    for update in range(1, num_updates + 1):
        # Anneal reward shaping weight toward 0 over second half of training
        progress = (update - 1.0) / max(1, num_updates)
        shaping_w = max(0.0, 0.15 * (1.0 - progress * 1.2))
        for env in envs:
            env.shaping_weight = shaping_w

        for step in range(num_steps):
            obs_buf[step] = next_obs
            mask_buf[step] = next_mask
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = model.get_action_and_value(next_obs, next_mask)
                values_buf[step] = value
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            obs_np = []
            mask_np = []
            rew_np = []
            done_np = []

            actions_cpu = action.cpu().numpy()
            for i, env in enumerate(envs):
                d, r, done, _, info = env.step(int(actions_cpu[i]))
                if done:
                    w = info["winner"]
                    learner_won = None if w is None else (w == env.agent_player)
                    league.record_result(env_opp_names[i], learner_won)
                    member = league.sample_opponent()
                    env.set_opponent(member.policy_fn)
                    env_opp_names[i] = member.name
                    d, _ = env.reset()
                obs_np.append(d["obs"])
                mask_np.append(d["action_mask"])
                rew_np.append(r)
                done_np.append(1.0 if done else 0.0)

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
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        for _ in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs[mb_inds], b_mask[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if update % snapshot_interval == 0:
            league.add_snapshot(f"snap_u{update}", make_torch_policy_fn(model, device))

        if update % eval_interval == 0 or update == num_updates:
            eval_res = evaluate_against_heuristics(model, device, games_per_diff=30)
            elapsed = time.perf_counter() - t_start
            fps = int((update * batch_size) / max(1e-3, elapsed))
            print(
                f"Update {update}/{num_updates} | Steps={update * batch_size:,} ({fps} sps) | "
                f"Elo={league.learner_elo:.0f} | "
                f"WR vs Easy={eval_res['easy']*100:.1f}% Normal={eval_res['normal']*100:.1f}% Hard={eval_res['hard']*100:.1f}%"
            )
            torch.save(model.state_dict(), save_path / "ppo_latest.pt")
            if eval_res["hard"] >= best_hard_wr:
                best_hard_wr = eval_res["hard"]
                torch.save(model.state_dict(), save_path / "ppo_best.pt")

    return model
