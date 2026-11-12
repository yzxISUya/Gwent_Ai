"""
DouZero-style Deep Monte-Carlo (DMC) Trainer for Gwent.
Directly learns Q(s, a) -> Monte-Carlo episode return G_t in [-1, 1] using
epsilon-greedy exploration over legal actions and PFSP League opponents.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gwent_rl.engine import GwentEngine, CORE_PRESET_IDS, OBS_DIM, ACTION_DIM
from gwent_rl.engine.rng import RngState
from gwent_rl.league import PFSPLeague
from .networks import GwentDMCNet
from .masked_ppo import evaluate_against_heuristics, make_torch_policy_fn


def train_dmc(
    total_episodes: int = 8_000,
    batch_episodes: int = 32,
    hidden_dim: int = 512,
    num_blocks: int = 3,
    lr: float = 3e-4,
    eps_start: float = 0.25,
    eps_end: float = 0.03,
    minibatch_size: int = 256,
    snapshot_interval: int = 15,
    eval_interval: int = 10,
    save_dir: str | Path = "checkpoints/dmc",
    device_str: str = "auto",
) -> GwentDMCNet:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model = GwentDMCNet(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    league = PFSPLeague(seed=777)
    rng = RngState(12345)
    preset_ids = list(CORE_PRESET_IDS)

    num_iterations = max(1, total_episodes // batch_episodes)
    best_hard_wr = 0.0
    total_steps = 0
    t_start = time.perf_counter()

    print(f"[DMC] Starting on {device} | Total episodes={total_episodes:,} | Iterations={num_iterations}")

    for it in range(1, num_iterations + 1):
        progress = (it - 1.0) / max(1, num_iterations)
        eps = eps_start + (eps_end - eps_start) * progress
        cur_lr = lr * (1.0 - 0.9 * progress)
        for pg in optimizer.param_groups:
            pg["lr"] = max(1e-5, cur_lr)

        obs_list = []
        act_list = []
        ret_list = []

        model.eval()
        for ep in range(batch_episodes):
            member = league.sample_opponent()
            p0 = preset_ids[rng.rand_int(len(preset_ids))]
            p1 = preset_ids[rng.rand_int(len(preset_ids))]
            eng = GwentEngine(seed=rng.rand_int(2**31 - 1), preset0=p0, preset1=p1)
            learner_side = ep % 2

            ep_obs = []
            ep_acts = []

            while not eng.done:
                if eng.turn == learner_side:
                    obs = eng.encode_observation(pov=learner_side)
                    legal = eng.get_legal_actions()
                    if rng.next_float() < eps:
                        action = legal[rng.rand_int(len(legal))]
                    else:
                        with torch.no_grad():
                            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                            mask_t = torch.tensor([eng.get_legal_mask()], dtype=torch.bool, device=device)
                            probs, _ = model(obs_t, mask_t)
                            action = int(torch.argmax(probs[0]).item())
                    ep_obs.append(obs)
                    ep_acts.append(action)
                    eng.step(action)
                else:
                    opp_action = member.policy_fn(eng)
                    eng.step(opp_action)

            if eng.winner == learner_side:
                g_return = 1.0
                learner_won = True
            elif eng.winner == (1 - learner_side):
                g_return = -1.0
                learner_won = False
            else:
                g_return = 0.0
                learner_won = None

            league.record_result(member.name, learner_won)
            for o, a in zip(ep_obs, ep_acts):
                obs_list.append(o)
                act_list.append(a)
                ret_list.append(g_return)
            total_steps += len(ep_obs)

        # Train Q-network on collected Monte-Carlo transitions
        model.train()
        b_obs = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)
        b_acts = torch.tensor(act_list, dtype=torch.long, device=device)
        b_rets = torch.tensor(ret_list, dtype=torch.float32, device=device)

        n_samples = len(obs_list)
        inds = np.arange(n_samples)
        for _ in range(2):
            np.random.shuffle(inds)
            for start in range(0, n_samples, minibatch_size):
                mb = inds[start : start + minibatch_size]
                q_all = model.predict_q(b_obs[mb])
                q_taken = q_all.gather(1, b_acts[mb].unsqueeze(1)).squeeze(1)
                # Robust Huber / Smooth L1 loss for Monte-Carlo return regression
                loss = nn.functional.smooth_l1_loss(q_taken, b_rets[mb])

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if it % snapshot_interval == 0:
            league.add_snapshot(f"snap_dmc_{it}", make_torch_policy_fn(model, device, deterministic=True))

        if it % eval_interval == 0 or it == num_iterations:
            eval_res = evaluate_against_heuristics(model, device, games_per_diff=128)
            elapsed = time.perf_counter() - t_start
            print(
                f"Iter {it}/{num_iterations} | Episodes={it * batch_episodes:,} | Steps={total_steps:,} | "
                f"Elo={league.learner_elo:.0f} | "
                f"WR vs Easy={eval_res['easy']*100:.1f}% Normal={eval_res['normal']*100:.1f}% Hard={eval_res['hard']*100:.1f}%"
            )
            torch.save(model.state_dict(), save_path / "dmc_latest.pt")
            if eval_res["hard"] >= best_hard_wr:
                best_hard_wr = eval_res["hard"]
                torch.save(model.state_dict(), save_path / "dmc_best.pt")

    return model
