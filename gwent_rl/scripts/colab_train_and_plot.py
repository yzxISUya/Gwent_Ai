"""
Self-contained Google Colab GPU training, Stage-2 continuation warm-start,
benchmarking, ONNX export, Google Drive persistence, and multi-chart plotting
script for Gwent-RL (Masked PPO + PFSP League Self-Play).

Experimental Protocol:
- Core Deck Pool: `nr-temeria` (Northern Realms Foltest Siege) & `ng-white-flame` (Nilfgaard Emhyr Spy)
  covering all 2x2 = 4 ordered matchups (NR mirror, NG mirror, NR vs NG, NG vs NR).
- Masked PPO + PFSP League:
  Up to 2,457,600 environment steps (300 updates, ~130,000 games) across vectorized environments
  with Stage-2 Critic warmup, Cosine LR decay (1.2e-4 -> 1.5e-5), tight clip (0.15), and PFSP champion seeding.
- Duplicate Paired-Seed Evaluation:
  - In-training checkpoints: 64 duplicate pairs = 128 games per difficulty.
  - Final Top-3 Audit: 200 fresh out-of-sample duplicate pairs = 400 games per difficulty to prevent Winner's Curse.
- Automatic Google Drive Backup:
  - Saves checkpoints, ONNX models, metrics JSON, charts, and notebook to `/content/drive/MyDrive/Gwent_RL` when mounted.
"""
from __future__ import annotations
import copy
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class TeeLogger:
    def __init__(self, filepath: str = "/content/training.log"):
        self.terminal = sys.stdout
        self.log_path = Path(filepath)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(self.log_path, "a", encoding="utf-8")
        except Exception:
            self.log_file = None

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        if self.log_file:
            try:
                self.log_file.write(message)
                self.log_file.flush()
            except Exception:
                pass

    def flush(self):
        self.terminal.flush()
        if self.log_file:
            try:
                self.log_file.flush()
            except Exception:
                pass

if Path("/content").exists():
    sys.stdout = TeeLogger()

from gwent_rl.engine import (
    GwentEngine,
    HeuristicAI,
    CORE_PRESET_IDS,
    CORE_DECK_PAIRS,
    OBS_DIM,
    ACTION_DIM,
)
from gwent_rl.engine.rng import RngState
from gwent_rl.env import GwentSingleAgentEnv
from gwent_rl.league import PFSPLeague
from gwent_rl.agents.networks import GwentPolicyValueNet
from gwent_rl.agents.masked_ppo import evaluate_against_heuristics, make_torch_policy_fn

OUT_DIR = Path("results")
MODELS_DIR = Path("models")
DRIVE_DIR = Path("/content/drive/MyDrive/Gwent_RL")


# ---------------------------------------------------------------------------
# Pure-Python ONNX Initializer Extractor (Fallback when only .onnx is present)
# ---------------------------------------------------------------------------
def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _iter_fields(buf: bytes):
    pos = 0
    n = len(buf)
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:
            val, pos = _read_varint(buf, pos)
            yield field_num, wire_type, val
        elif wire_type == 1:
            yield field_num, wire_type, buf[pos : pos + 8]
            pos += 8
        elif wire_type == 2:
            length, pos = _read_varint(buf, pos)
            yield field_num, wire_type, buf[pos : pos + length]
            pos += length
        elif wire_type == 5:
            yield field_num, wire_type, buf[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire_type {wire_type}")


def _parse_tensor_proto(buf: bytes) -> Tuple[str, torch.Tensor]:
    dims: List[int] = []
    name = ""
    raw_data: Optional[bytes] = None
    float_data: List[float] = []
    for fnum, wtype, val in _iter_fields(buf):
        if fnum == 1:  # dims
            if wtype == 0:
                dims.append(val)
            elif wtype == 2:
                p = 0
                while p < len(val):
                    d, p = _read_varint(val, p)
                    dims.append(d)
        elif fnum == 4:  # float_data
            if wtype == 2:
                float_data.extend(np.frombuffer(val, dtype="<f4").tolist())
            elif wtype == 5:
                float_data.append(float(np.frombuffer(val, dtype="<f4")[0]))
        elif fnum == 8 and wtype == 2:  # name
            name = val.decode("utf-8", errors="ignore")
        elif fnum == 9 and wtype == 2:  # raw_data
            raw_data = val
    if raw_data is not None:
        arr = np.frombuffer(raw_data, dtype="<f4").copy()
    else:
        arr = np.array(float_data, dtype="<f4")
    if dims:
        arr = arr.reshape(dims)
    return name, torch.from_numpy(arr)


def extract_state_dict_from_onnx(onnx_path: Path, model: nn.Module) -> Dict[str, torch.Tensor]:
    buf = onnx_path.read_bytes()
    initializers: Dict[str, torch.Tensor] = {}
    for fnum, wtype, val in _iter_fields(buf):
        if fnum == 7 and wtype == 2:
            for g_fnum, g_wtype, g_val in _iter_fields(val):
                if g_fnum == 5 and g_wtype == 2:
                    tname, tensor = _parse_tensor_proto(g_val)
                    if tname:
                        initializers[tname] = tensor

    state_dict: Dict[str, torch.Tensor] = {}
    target_names = list(model.state_dict().keys())
    for name in target_names:
        if name in initializers:
            state_dict[name] = initializers[name]
        else:
            for init_name, init_val in initializers.items():
                if init_name.endswith(name) or name.endswith(init_name):
                    state_dict[name] = init_val
                    break
    return state_dict


def load_pretrained_weights(model: nn.Module, device: torch.device) -> bool:
    pt_candidates = [
        MODELS_DIR / "ppo_best.pt",
        Path("gwent_rl/models/ppo_best.pt"),
    ]
    for pt in pt_candidates:
        if pt.exists():
            try:
                state = torch.load(pt, map_location=device, weights_only=True)
                model.load_state_dict(state)
                print(f"[WarmStart] Loaded PyTorch checkpoint from {pt.resolve()}")
                return True
            except Exception as e:
                print(f"[WarmStart] Failed loading {pt}: {e}")

    onnx_candidates = [
        MODELS_DIR / "ppo_v1.onnx",
        Path("gwent_web/public/models/ppo_v1.onnx"),
    ]
    for onnx_p in onnx_candidates:
        if onnx_p.exists():
            try:
                sd = extract_state_dict_from_onnx(onnx_p, model)
                if len(sd) >= len(model.state_dict()) * 0.8:
                    model.load_state_dict(sd, strict=False)
                    print(f"[WarmStart] Loaded {len(sd)} weights from ONNX fallback {onnx_p.resolve()}")
                    return True
            except Exception as e:
                print(f"[WarmStart] Failed extracting ONNX weights from {onnx_p}: {e}")
    return False


def load_stage1_metrics() -> Dict[str, Any]:
    metric_paths = [
        OUT_DIR / "training_metrics.json",
        Path("gwent_rl/results/training_metrics.json"),
    ]
    for p in metric_paths:
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                print(f"[Metrics] Loaded previous Stage-1 metrics from {p.resolve()}")
                return data
            except Exception as e:
                print(f"[Metrics] Failed reading {p}: {e}")
    return {}


def sync_to_google_drive() -> None:
    if not DRIVE_DIR.exists():
        return
    try:
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)
        for f in MODELS_DIR.glob("*.*"):
            shutil.copy2(f, DRIVE_DIR / f.name)
        for f in OUT_DIR.glob("*.*"):
            shutil.copy2(f, DRIVE_DIR / f.name)
        log_f = Path("/content/training.log")
        if log_f.exists():
            shutil.copy2(log_f, DRIVE_DIR / "training.log")
        print(f"[Drive Sync] Mirrored checkpoints and metrics to {DRIVE_DIR.resolve()}")
    except Exception as e:
        print(f"[Drive Sync Error] {e}")


def export_model_onnx(model: nn.Module, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    dummy_obs = torch.zeros((1, OBS_DIM), dtype=torch.float32)
    dummy_mask = torch.ones((1, ACTION_DIM), dtype=torch.bool)
    torch.onnx.export(
        model_cpu,
        (dummy_obs, dummy_mask),
        str(out_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["obs", "mask"],
        output_names=["probs", "value"],
        dynamic_axes={
            "obs": {0: "batch"},
            "mask": {0: "batch"},
            "probs": {0: "batch"},
            "value": {0: "batch"},
        },
    )
    print(f"[ONNX Exported] {out_path.resolve()} ({out_path.stat().st_size / 1024:.1f} KB)")


def _composite_score(ev: Dict[str, float]) -> float:
    return ev["easy"] * 0.20 + ev["normal"] * 0.35 + ev["hard"] * 0.45


def select_best_from_top_k(
    candidates: List[Tuple[float, int, nn.Module]],
    device: torch.device,
    audit_games_per_diff: int = 400,
    audit_seed: int = 900_000,
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
        )
        audit_score = _composite_score(ev)
        print(
            f"  Candidate #{rank_idx+1} (from step/iter {step_id}, in-train={train_score*100:.1f}%) -> "
            f"400-game Audit: Easy={ev['easy']*100:.1f}%±{ev['easy_se']*100:.1f}% | "
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

    print(f"[{tag} Final Selected] Step/Iter {best_audit_info['selected_from']} (Composite={best_audit_info['composite']:.1f}%)\n")
    return best_model, best_audit_info


def run_ppo_experiment(
    device: torch.device,
    total_timesteps: int = 1_228_800,
    num_envs: int = 32,
    num_steps: int = 256,
    hidden_dim: int = 512,
    num_blocks: int = 3,
    lr: float = 1.2e-4,
    eval_games_per_diff: int = 128,
    prev_history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[GwentPolicyValueNet, List[Dict[str, Any]], Dict[str, Any]]:
    batch_size = num_envs * num_steps  # 8192 steps per update -> 150 updates
    num_updates = total_timesteps // batch_size
    eval_every = max(1, num_updates // 20)
    snap_every = max(1, num_updates // 12)

    model = GwentPolicyValueNet(OBS_DIM, ACTION_DIM, hidden_dim, num_blocks).to(device)
    warm_started = load_pretrained_weights(model, device)

    league = PFSPLeague(seed=142 if warm_started else 42)
    history: List[Dict[str, Any]] = list(prev_history) if (warm_started and prev_history) else []

    if warm_started and history:
        last_rec = history[-1]
        start_update = int(last_rec.get("update", 150))
        start_steps = int(last_rec.get("steps", 1_228_800))
        episodes_completed = int(last_rec.get("episodes", 64892))
        ckpt_idx = int(last_rec.get("checkpoint", len(history) - 1))
        league.learner_elo = float(last_rec.get("elo", 1456.7))
        # Seed League with Stage-1 Champion snapshots (greedy & stochastic)
        league.add_snapshot("snap_ppo_s1_greedy", make_torch_policy_fn(model, device, deterministic=True))
        league.add_snapshot("snap_ppo_s1_stoch", make_torch_policy_fn(model, device, deterministic=False))
    else:
        start_update = 0
        start_steps = 0
        episodes_completed = 0
        ckpt_idx = 0

    envs = [
        GwentSingleAgentEnv(opponent="normal", seed=500 + i, preset_ids=CORE_PRESET_IDS)
        for i in range(num_envs)
    ]
    env_opp_names = ["heuristic_normal"] * num_envs

    top_candidates: List[Tuple[float, int, nn.Module]] = []
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    obs_buf = torch.zeros((num_steps, num_envs, OBS_DIM), dtype=torch.float32, device=device)
    mask_buf = torch.zeros((num_steps, num_envs, ACTION_DIM), dtype=torch.bool, device=device)
    actions_buf = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    next_obs_list, next_mask_list = [], []
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

    t0 = time.perf_counter()

    # Evaluate initial / warm-start state & seed top_candidates so Stage 2 never regresses
    eval0 = evaluate_against_heuristics(model, device, games_per_diff=eval_games_per_diff)
    init_score = _composite_score(eval0)
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
                "v_loss": 0.5,
                "entropy": 2.4,
            }
        )
    print(
        f"[PPO Stage-2 WarmStart={warm_started}] Start Upd={start_update} Steps={start_steps:,} Elo={league.learner_elo:.0f} | "
        f"WR Easy={eval0['easy']*100:.1f}%±{eval0['easy_se']*100:.1f}% "
        f"Normal={eval0['normal']*100:.1f}%±{eval0['normal_se']*100:.1f}% "
        f"Hard={eval0['hard']*100:.1f}%±{eval0['hard_se']*100:.1f}% (Comp={init_score*100:.1f}%)"
    )

    clip_coef = 0.15 if warm_started else 0.20

    for update in range(1, num_updates + 1):
        global_update = start_update + update
        progress = (update - 1.0) / max(1, num_updates)

        if warm_started:
            shaping_w = 0.0
            ent_coef = 0.005 * (1.0 - progress) + 0.0015 * progress
            # 3-update linear LR warmup (2e-5 -> lr), then Cosine decay to 1.5e-5
            if update <= 3:
                cur_lr = 2e-5 + (lr - 2e-5) * (update / 3.0)
            else:
                cos_p = (update - 3.0) / max(1, num_updates - 3)
                cur_lr = 1.5e-5 + 0.5 * (lr - 1.5e-5) * (1.0 + np.cos(np.pi * cos_p))
            for pg in optimizer.param_groups:
                pg["lr"] = float(cur_lr)
        else:
            shaping_w = max(0.0, 0.16 * (1.0 - progress * 1.25))
            ent_coef = 0.020 * (1.0 - progress) + 0.004 * progress
            cos_p = progress
            cur_lr = 5e-5 + 0.5 * (lr - 5e-5) * (1.0 + np.cos(np.pi * cos_p))
            for pg in optimizer.param_groups:
                pg["lr"] = float(cur_lr)

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

            obs_np, mask_np, rew_np, done_np = [], [], [], []
            actions_cpu = action.cpu().numpy()
            for i, env in enumerate(envs):
                d, r, done, _, info = env.step(int(actions_cpu[i]))
                if done:
                    episodes_completed += 1
                    w = info["winner"]
                    learner_won = None if w is None else (w == env.agent_player)
                    league.record_result(env_opp_names[i], learner_won, k_factor=6.0 if warm_started else 8.0)
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
                delta = rewards_buf[t] + 0.995 * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + 0.995 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1, OBS_DIM))
        b_mask = mask_buf.reshape((-1, ACTION_DIM))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        # Value target bounded in [-1.0, 1.0] to match Tanh value head
        b_returns = torch.clamp(returns.reshape(-1), -1.0, 1.0)
        # Global advantage normalization across the entire rollout batch
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        b_inds = np.arange(batch_size)
        pg_losses, v_losses, entropies = [], [], []
        for _ in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, 512):
                mb = b_inds[start : start + 512]
                _, newlogprob, entropy, newvalue = model.get_action_and_value(
                    b_obs[mb], b_mask[mb], b_actions[mb]
                )
                logratio = torch.clamp(newlogprob - b_logprobs[mb], -20.0, 20.0)
                ratio = logratio.exp()
                mb_adv = b_advantages[mb]

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss - ent_coef * ent_loss + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

                pg_losses.append(float(pg_loss.item()))
                v_losses.append(float(v_loss.item()))
                entropies.append(float(ent_loss.item()))

        if update % 2 == 0 or update == 1:
            fps_instant = int((update * batch_size) / max(1e-3, time.perf_counter() - t0))
            print(
                f"  [PPO Heartbeat] Upd {global_update:03d}/{start_update + num_updates} | "
                f"Steps={start_steps + update * batch_size:,} | Games={episodes_completed:,} ({fps_instant} sps) | "
                f"V-Loss={np.mean(v_losses):.4f} | Elo={league.learner_elo:.0f}"
            )

        if update % snap_every == 0:
            league.add_snapshot(f"snap_ppo_{global_update}", make_torch_policy_fn(model, device))

        if update % eval_every == 0 or update == num_updates:
            ckpt_idx += 1
            ev = evaluate_against_heuristics(model, device, games_per_diff=eval_games_per_diff)
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
            }
            history.append(rec)
            score = _composite_score(ev)
            if not top_candidates or score >= top_candidates[0][0]:
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), MODELS_DIR / "ppo_best.pt")
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

    best_model, audit_info = select_best_from_top_k(
        top_candidates, device, audit_games_per_diff=400, audit_seed=900_000, tag="PPO"
    )
    return best_model, history, audit_info


def _plot_with_se_band(ax, x, hist, key_wr, key_se, fmt, color, label):
    y = np.array([r[key_wr] for r in hist], dtype=float)
    se = np.array([r.get(key_se, 0.0) for r in hist], dtype=float)
    ax.plot(x, y, fmt, markersize=3.8, linewidth=2.0, color=color, label=label)
    ax.fill_between(x, np.clip(y - se, 0, 100), np.clip(y + se, 0, 100), color=color, alpha=0.14)


def render_training_charts(
    ppo_hist: List[Dict[str, Any]],
    ppo_audit: Dict[str, Any] | None = None,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")

    ppo_x = [r["steps"] / 1000.0 for r in ppo_hist]

    # 1. Chart 1: Elo Rating Progression
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    ax.plot(ppo_x, [r["elo"] for r in ppo_hist], "-o", markersize=3.8, linewidth=2.2, color="#0284c7", label=f"Masked PPO ({ppo_hist[-1]['steps']/1e6:.2f}M Steps / {ppo_hist[-1]['episodes']:,} Games)")
    ax.axhline(1328, color="#ef4444", linestyle="--", linewidth=1.3, label="Heuristic Hard Baseline (1328)")
    ax.axhline(1200, color="#64748b", linestyle=":", linewidth=1.2, label="Heuristic Normal Baseline (1200)")
    ax.axhline(932, color="#94a3b8", linestyle="-.", linewidth=1.1, label="Heuristic Easy Baseline (932)")
    ax.set_title("Gwent RL Self-Play League — Elo Progression (NR & NG 2x2 Pool)", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Environment Steps (Thousands)", fontsize=11)
    ax.set_ylabel("League Elo Rating", fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart1_elo_progression.png")
    plt.close(fig)

    # 2. Chart 2: Masked PPO Win Rates vs Easy / Normal / Hard
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=160)
    _plot_with_se_band(ax, ppo_x, ppo_hist, "wr_easy", "se_easy", "-o", "#22c55e", "vs Heuristic Easy (932)")
    _plot_with_se_band(ax, ppo_x, ppo_hist, "wr_normal", "se_normal", "-^", "#0284c7", "vs Heuristic Normal (0-ply, 1200)")
    _plot_with_se_band(ax, ppo_x, ppo_hist, "wr_hard", "se_hard", "-s", "#ef4444", "vs Heuristic Hard (1-ply Search, 1328)")
    if ppo_audit:
        ax.scatter([ppo_x[-1]], [ppo_audit["wr_hard"]], marker="*", s=160, color="#b91c1c", zorder=6, label=f"400-Game Final Audit Hard: {ppo_audit['wr_hard']:.1f}%±{ppo_audit['se_hard']:.1f}%")
    ax.axhline(50.0, color="#64748b", linestyle="--", linewidth=1.2, label="50% Parity Threshold")
    ax.set_ylim(0, 105)
    ax.set_title("Masked PPO — Duplicate Paired Win Rate vs Baselines (±1 SE)", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Environment Steps (Thousands)", fontsize=11)
    ax.set_ylabel("Win Rate (%)", fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=9.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart2_ppo_winrates.png")
    plt.close(fig)

    # 3. Chart 3: Loss & Policy Entropy Convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=160)
    ax1.plot(ppo_x[1:], [r["v_loss"] for r in ppo_hist[1:]], "-o", markersize=3.2, color="#0284c7", label="Value Loss (MSE)")
    ax1.plot(ppo_x[1:], [r["pg_loss"] for r in ppo_hist[1:]], "-^", markersize=3.2, color="#f59e0b", label="Policy Clip Loss")
    ax1.set_title("Masked PPO — Loss Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Environment Steps (k)")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(ppo_x[1:], [r["entropy"] for r in ppo_hist[1:]], "-s", markersize=3.2, color="#10b981", label="Policy Shannon Entropy")
    ax2.set_title("Masked PPO — Policy Entropy (Exploration)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Environment Steps (k)")
    ax2.set_ylabel("Entropy (nats)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart3_loss_and_entropy.png")
    plt.close(fig)

    # 4. Combined 2x2 Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), dpi=160)
    # Top-Left: Elo
    axes[0, 0].plot(ppo_x, [r["elo"] for r in ppo_hist], "-o", markersize=3.2, color="#0284c7", label="Masked PPO")
    axes[0, 0].axhline(1328, color="#ef4444", linestyle="--", label="Heuristic Hard (1328)")
    axes[0, 0].axhline(1200, color="#64748b", linestyle=":", label="Heuristic Normal (1200)")
    axes[0, 0].axhline(932, color="#94a3b8", linestyle="-.", label="Heuristic Easy (932)")
    axes[0, 0].set_title("1. League Elo Progression", fontweight="bold")
    axes[0, 0].set_xlabel("Steps (k)")
    axes[0, 0].set_ylabel("Elo")
    axes[0, 0].legend(fontsize=8.5)

    # Top-Right: Win Rates
    _plot_with_se_band(axes[0, 1], ppo_x, ppo_hist, "wr_easy", "se_easy", "-o", "#22c55e", "vs Easy")
    _plot_with_se_band(axes[0, 1], ppo_x, ppo_hist, "wr_normal", "se_normal", "-^", "#0284c7", "vs Normal (0-ply)")
    _plot_with_se_band(axes[0, 1], ppo_x, ppo_hist, "wr_hard", "se_hard", "-s", "#ef4444", "vs Hard (1-ply)")
    axes[0, 1].axhline(50.0, color="#64748b", linestyle="--")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].set_title("2. Duplicate Paired Win Rates (±1 SE)", fontweight="bold")
    axes[0, 1].set_xlabel("Steps (k)")
    axes[0, 1].set_ylabel("Win Rate (%)")
    axes[0, 1].legend(fontsize=8.5)

    # Bottom-Left: Value Loss & Policy Loss
    axes[1, 0].plot(ppo_x[1:], [r["v_loss"] for r in ppo_hist[1:]], "-o", markersize=3.2, color="#0284c7", label="Value Loss")
    axes[1, 0].plot(ppo_x[1:], [r["pg_loss"] for r in ppo_hist[1:]], "-^", markersize=3.2, color="#f59e0b", label="Policy Loss")
    axes[1, 0].set_title("3. PPO Optimization Losses", fontweight="bold")
    axes[1, 0].set_xlabel("Steps (k)")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend(fontsize=8.5)

    # Bottom-Right: Policy Entropy
    axes[1, 1].plot(ppo_x[1:], [r["entropy"] for r in ppo_hist[1:]], "-s", markersize=3.2, color="#10b981", label="Policy Entropy")
    axes[1, 1].set_title("4. Policy Shannon Entropy", fontweight="bold")
    axes[1, 1].set_xlabel("Steps (k)")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].legend(fontsize=8.5)

    fig.suptitle("Gwent-RL Self-Play Training Report (Masked PPO + PFSP League)", fontsize=14.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_dashboard.png")
    plt.close(fig)
    print(f"[Charts] Saved high-res figures to {OUT_DIR.resolve()}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Gwent-RL Masked PPO Colab Training Pipeline")
    parser.add_argument("--steps", type=int, default=None, help="Total PPO timesteps")
    parser.add_argument("--audit-games", type=int, default=400, help="Audit games per difficulty")
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Gwent-RL Colab Masked PPO + PFSP League Pipeline | Device: {device} | Core Pool: {CORE_PRESET_IDS} ===")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    prev_metrics = load_stage1_metrics()
    prev_ppo_hist = prev_metrics.get("ppo_history", [])

    is_ppo_warm = ((MODELS_DIR / "ppo_best.pt").exists() or Path("gwent_rl/models/ppo_best.pt").exists()) and bool(prev_ppo_hist)
    ppo_steps = args.steps if args.steps is not None else (1_228_800 if is_ppo_warm else 2_457_600)

    print(f"[Plan] Masked PPO Steps: {ppo_steps:,} ({ppo_steps // 8192} updates, WarmStart={is_ppo_warm})")

    ppo_model, ppo_hist, ppo_audit = run_ppo_experiment(
        device=device,
        total_timesteps=ppo_steps,
        prev_history=prev_ppo_hist if is_ppo_warm else None,
    )
    torch.save(ppo_model.state_dict(), MODELS_DIR / "ppo_best.pt")
    export_model_onnx(ppo_model, MODELS_DIR / "ppo_v1.onnx")
    sync_to_google_drive()

    metrics = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "algo": "Masked PPO + PFSP League",
        "core_preset_ids": list(CORE_PRESET_IDS),
        "eval_protocol": "Duplicate Paired-Seed (128 games/diff in-train, 400 games/diff Top-3 Final Audit)",
        "calibrated_baseline_elos": {"easy": 932, "normal": 1200, "hard": 1328},
        "stage1_ppo_final_audit": prev_metrics.get("ppo_final_audit"),
        "ppo_final_audit": ppo_audit,
        "ppo_history": ppo_hist,
    }
    (OUT_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    render_training_charts(ppo_hist, ppo_audit)
    sync_to_google_drive()
    print("=== Training, Top-3 Audit, ONNX Export, Chart Rendering & Google Drive Sync Complete! ===")


if __name__ == "__main__":
    main()
