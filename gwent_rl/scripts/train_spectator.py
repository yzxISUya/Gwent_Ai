"""
Decoupled Route-A Training for the Antisymmetric Zero-Sum Spectator Value Network (`spectator_v1`).

Workflow:
  1. Freeze `ppo_best.pt` (`GwentPolicyValueNet`) and run batched mirror self-play across
     all 64 deck matchups (`ALL_DECK_PAIRS`) to collect unbiased state pairs `[obs(0), obs(1)]`
     and terminal match outcomes `y in {0.0, 0.5, 1.0}`.
  2. Train `GwentSpectatorValueNet` with antisymmetric logit:
       logit(s) = 0.5 * (f_theta(obs0, obs1) - f_theta(obs1, obs0))
       p0_win_prob = sigmoid(logit(s))
     using BCEWithLogitsLoss, AdamW, and CosineAnnealingLR.
  3. Evaluate LogLoss, Brier Score, 10-bin Calibration / ECE, and zero-sum symmetry error
     on an independent validation set, plus trace `seed=20261106` (`nr-temeria` vs `ng-white-flame`).
  4. Save `spectator_best.pt` and export `spectator_v1.onnx` (when `onnx` is installed).
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gwent_rl.engine import ALL_DECK_PAIRS, ACTION_DIM, OBS_DIM, GwentEngine
from gwent_rl.agents.networks import GwentPolicyValueNet, GwentSpectatorValueNet
from gwent_rl.scripts.export_onnx import export_spectator_to_onnx

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "gwent_rl" / "models"
WEB_MODELS_DIR = ROOT / "gwent_web" / "public" / "models"
RESULTS_DIR = ROOT / "gwent_rl" / "results"


@torch.no_grad()
def collect_mirror_selfplay(
    ppo_model: GwentPolicyValueNet,
    device: torch.device,
    num_games: int,
    base_seed: int,
    num_envs: int = 256,
    greedy_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run batched mirror self-play (`ppo_best.pt` vs `ppo_best.pt`) and collect
    `(obs0, obs1, y_p0, turn)` for every decision step from `st=0` to terminal.
    Stored as float16 for memory efficiency (~1.4 GB for 500k steps).
    """
    ppo_model.eval()
    engines: List[GwentEngine] = []
    ep_obs0: List[List[np.ndarray]] = [[] for _ in range(num_envs)]
    ep_obs1: List[List[np.ndarray]] = [[] for _ in range(num_envs)]
    ep_turn: List[List[int]] = [[] for _ in range(num_envs)]
    ep_det: List[bool] = [False] * num_envs

    next_game_idx = 0

    def _init_game(slot: int) -> bool:
        nonlocal next_game_idx
        if next_game_idx >= num_games:
            return False
        g_idx = next_game_idx
        next_game_idx += 1
        p0, p1 = ALL_DECK_PAIRS[g_idx % len(ALL_DECK_PAIRS)]
        seed = base_seed + g_idx * 17
        if slot < len(engines):
            engines[slot].reset(seed=seed, preset0=p0, preset1=p1)
        else:
            engines.append(GwentEngine(seed=seed, preset0=p0, preset1=p1))
        ep_obs0[slot] = []
        ep_obs1[slot] = []
        ep_turn[slot] = []
        ep_det[slot] = ((g_idx // len(ALL_DECK_PAIRS)) % 10) < int(round(greedy_ratio * 10))
        return True

    active = []
    for slot in range(min(num_envs, num_games)):
        if _init_game(slot):
            active.append(slot)

    all_obs0: List[np.ndarray] = []
    all_obs1: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_turns: List[np.ndarray] = []
    games_done = 0
    t0 = time.perf_counter()

    while active:
        batch_obs_act = np.empty((len(active), OBS_DIM), dtype=np.float32)
        batch_mask = np.empty((len(active), ACTION_DIM), dtype=bool)
        for b_i, slot in enumerate(active):
            eng = engines[slot]
            o0 = eng.encode_observation(0)
            o1 = eng.encode_observation(1)
            ep_obs0[slot].append(o0.astype(np.float16))
            ep_obs1[slot].append(o1.astype(np.float16))
            ep_turn[slot].append(eng.turn)
            batch_obs_act[b_i] = o0 if eng.turn == 0 else o1
            batch_mask[b_i] = eng.get_legal_mask()

        obs_t = torch.from_numpy(batch_obs_act).to(device)
        mask_t = torch.from_numpy(batch_mask).to(device)
        probs, _ = ppo_model(obs_t, mask_t)
        probs_np = probs.cpu().numpy()

        # Vectorized sampling for stochastic slots, argmax for greedy slots
        greedy_actions = np.argmax(probs_np, axis=1)
        # Safe sampling on CPU/GPU
        sampled_actions = torch.multinomial(probs.clamp(min=0.0), num_samples=1).squeeze(1).cpu().numpy()

        still_active: List[int] = []
        for b_i, slot in enumerate(active):
            eng = engines[slot]
            act = int(greedy_actions[b_i] if ep_det[slot] else sampled_actions[b_i])
            eng.step(act)

            if eng.done or len(ep_obs0[slot]) >= 250:
                if eng.done:
                    y_val = 1.0 if eng.winner == 0 else (0.5 if eng.winner is None else 0.0)
                else:
                    y_val = 0.5
                n_steps = len(ep_obs0[slot])
                if n_steps > 0:
                    all_obs0.append(np.stack(ep_obs0[slot], axis=0))
                    all_obs1.append(np.stack(ep_obs1[slot], axis=0))
                    all_y.append(np.full((n_steps,), y_val, dtype=np.float32))
                    all_turns.append(np.array(ep_turn[slot], dtype=np.int8))
                games_done += 1
                if games_done % 2000 == 0 or games_done == num_games:
                    dt = time.perf_counter() - t0
                    print(
                        f"  [Self-Play] {games_done:,}/{num_games:,} games "
                        f"({games_done / max(dt, 1e-6):.1f} games/s)"
                    )
                if _init_game(slot):
                    still_active.append(slot)
            else:
                still_active.append(slot)
        active = still_active

    obs0_arr = np.concatenate(all_obs0, axis=0)
    obs1_arr = np.concatenate(all_obs1, axis=0)
    y_arr = np.concatenate(all_y, axis=0)
    turn_arr = np.concatenate(all_turns, axis=0)
    return obs0_arr, obs1_arr, y_arr, turn_arr


def compute_calibration_metrics(probs: np.ndarray, targets: np.ndarray, num_bins: int = 10) -> Dict[str, Any]:
    eps = 1e-6
    p = np.clip(probs.astype(np.float64), eps, 1.0 - eps)
    y = targets.astype(np.float64)
    bce = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    brier = float(np.mean((p - y) ** 2))

    bins = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    cal_bins = []
    n = len(p)
    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == num_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            mean_pred = float(np.mean(p[mask]))
            mean_emp = float(np.mean(y[mask]))
            ece += (cnt / n) * abs(mean_pred - mean_emp)
        else:
            mean_pred = float(0.5 * (lo + hi))
            mean_emp = None
        cal_bins.append(
            {
                "bin": f"[{lo:.1f}, {hi:.1f}]",
                "count": cnt,
                "mean_pred": round(mean_pred, 4),
                "empirical_wr": round(mean_emp, 4) if mean_emp is not None else None,
            }
        )

    return {
        "logloss": round(bce, 5),
        "brier": round(brier, 5),
        "ece": round(float(ece), 5),
        "mean_pred": round(float(np.mean(p)), 5),
        "mean_target": round(float(np.mean(y)), 5),
        "bins": cal_bins,
    }


@torch.no_grad()
def evaluate_ppo_baseline(
    ppo_model: GwentPolicyValueNet,
    obs0: np.ndarray,
    obs1: np.ndarray,
    turns: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> Dict[str, Any]:
    ppo_model.eval()
    n = len(y)
    preds = np.empty(n, dtype=np.float32)
    dummy_mask = torch.ones((batch_size, ACTION_DIM), dtype=torch.bool, device=device)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        bs = end - start
        t_slice = turns[start:end]
        o_act = np.where(t_slice[:, None] == 0, obs0[start:end], obs1[start:end]).astype(np.float32)
        obs_t = torch.from_numpy(o_act).to(device)
        _, val_t = ppo_model(obs_t, dummy_mask[:bs])
        val_np = val_t.squeeze(-1).cpu().numpy()
        p0_val = np.where(t_slice == 0, val_np, -val_np)
        preds[start:end] = np.clip((p0_val + 1.0) * 0.5, 0.0, 1.0)

    return compute_calibration_metrics(preds, y)


@torch.no_grad()
def evaluate_spectator(
    spec_model: GwentSpectatorValueNet,
    obs0: np.ndarray,
    obs1: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> Dict[str, Any]:
    spec_model.eval()
    n = len(y)
    preds = np.empty(n, dtype=np.float32)
    max_sym_err = 0.0

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        o0 = torch.from_numpy(obs0[start:end].astype(np.float32)).to(device)
        o1 = torch.from_numpy(obs1[start:end].astype(np.float32)).to(device)
        p01 = spec_model(torch.cat([o0, o1], dim=1)).squeeze(-1)
        p10 = spec_model(torch.cat([o1, o0], dim=1)).squeeze(-1)
        sym_err = float(torch.max(torch.abs(p01 + p10 - 1.0)).item())
        if sym_err > max_sym_err:
            max_sym_err = sym_err
        preds[start:end] = p01.cpu().numpy()

    metrics = compute_calibration_metrics(preds, y)
    metrics["zero_sum_max_err"] = float(max_sym_err)
    return metrics


@torch.no_grad()
def trace_seed_20261106(
    ppo_model: GwentPolicyValueNet,
    spec_model: GwentSpectatorValueNet,
    device: torch.device,
) -> List[Dict[str, Any]]:
    ppo_model.eval()
    spec_model.eval()
    eng = GwentEngine(seed=20261106, preset0="nr-temeria", preset1="ng-white-flame")
    trace = []
    for st in range(120):
        if eng.done:
            break
        pov = eng.turn
        o0 = eng.encode_observation(0)
        o1 = eng.encode_observation(1)
        mask = np.array(eng.get_legal_mask(), dtype=bool)

        o_act = o0 if pov == 0 else o1
        probs, val = ppo_model(
            torch.from_numpy(o_act).unsqueeze(0).to(device),
            torch.from_numpy(mask).unsqueeze(0).to(device),
        )
        v_ppo = float(val.item())
        ppo_p0_wr = ((v_ppo if pov == 0 else -v_ppo) + 1.0) * 50.0

        pair = np.concatenate([o0, o1], axis=0)
        p0_spec = float(spec_model(torch.from_numpy(pair).unsqueeze(0).to(device)).item()) * 100.0
        p1_spec = float(
            spec_model(torch.from_numpy(np.concatenate([o1, o0], axis=0)).unsqueeze(0).to(device)).item()
        ) * 100.0

        trace.append(
            {
                "step": st,
                "round": eng.round,
                "sub_phase": eng.sub_phase,
                "turn": pov,
                "ppo_p0_wr": round(ppo_p0_wr, 2),
                "spec_p0_wr": round(p0_spec, 2),
                "spec_p1_wr": round(p1_spec, 2),
                "spec_sum": round(p0_spec + p1_spec, 6),
            }
        )
        legal = eng.get_legal_actions()
        probs_np = probs[0].cpu().numpy()
        act = max(legal, key=lambda a: float(probs_np[a]))
        eng.step(act)
    return trace


def train_spectator(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Spectator Value Net Training | Device: {device} ===")

    ckpt_path = Path(args.ppo_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {ckpt_path}")

    ppo_model = GwentPolicyValueNet(obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=512, num_blocks=3).to(device)
    ppo_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    ppo_model.eval()
    for p in ppo_model.parameters():
        p.requires_grad = False

    print(f"[1/4] Collecting {args.train_games:,} training games via frozen ppo_best mirror self-play...")
    tr_obs0, tr_obs1, tr_y, _ = collect_mirror_selfplay(
        ppo_model=ppo_model,
        device=device,
        num_games=args.train_games,
        base_seed=100_000,
        num_envs=args.num_envs,
        greedy_ratio=args.greedy_ratio,
    )
    print(f"  Train dataset: {len(tr_y):,} state pairs | mean(y)={float(np.mean(tr_y)):.4f}")

    print(f"[2/4] Collecting {args.val_games:,} independent validation games...")
    va_obs0, va_obs1, va_y, va_turns = collect_mirror_selfplay(
        ppo_model=ppo_model,
        device=device,
        num_games=args.val_games,
        base_seed=5_000_000,
        num_envs=args.num_envs,
        greedy_ratio=args.greedy_ratio,
    )
    print(f"  Val dataset: {len(va_y):,} state pairs | mean(y)={float(np.mean(va_y)):.4f}")

    ppo_baseline = evaluate_ppo_baseline(ppo_model, va_obs0, va_obs1, va_turns, va_y, device)
    print(
        f"  [Baseline PPO Critic on Val] LogLoss={ppo_baseline['logloss']:.5f} | "
        f"Brier={ppo_baseline['brier']:.5f} | ECE={ppo_baseline['ece']:.5f}"
    )

    spec_model = GwentSpectatorValueNet(
        obs_dim=OBS_DIM,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(spec_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    n_train = len(tr_y)
    rng = np.random.default_rng(2026)
    best_val_bce = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0

    print(f"[3/4] Training GwentSpectatorValueNet (hidden={args.hidden_dim}, blocks={args.num_blocks}, epochs={args.epochs})...")
    for epoch in range(1, args.epochs + 1):
        spec_model.train()
        perm = rng.permutation(n_train)
        total_loss = 0.0
        total_cnt = 0
        t_ep = time.perf_counter()

        for start in range(0, n_train, args.batch_size):
            idx = perm[start : start + args.batch_size]
            o0 = torch.from_numpy(tr_obs0[idx].astype(np.float32)).to(device)
            o1 = torch.from_numpy(tr_obs1[idx].astype(np.float32)).to(device)
            yb = torch.from_numpy(tr_y[idx]).unsqueeze(1).to(device)

            logits = spec_model.logit(torch.cat([o0, o1], dim=1))
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(spec_model.parameters(), 1.0)
            optimizer.step()

            bs = len(idx)
            total_loss += float(loss.item()) * bs
            total_cnt += bs

        scheduler.step()
        train_bce = total_loss / max(1, total_cnt)
        val_metrics = evaluate_spectator(spec_model, va_obs0, va_obs1, va_y, device)
        dt_ep = time.perf_counter() - t_ep

        improved = val_metrics["logloss"] < best_val_bce - 1e-5
        if improved:
            best_val_bce = val_metrics["logloss"]
            best_state = {k: v.detach().cpu().clone() for k, v in spec_model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        print(
            f"  Epoch {epoch:02d}/{args.epochs} ({dt_ep:.1f}s) | "
            f"train_bce={train_bce:.5f} | val_bce={val_metrics['logloss']:.5f} | "
            f"brier={val_metrics['brier']:.5f} | ece={val_metrics['ece']:.5f} | "
            f"sym_err={val_metrics['zero_sum_max_err']:.2e}"
            + (" *" if improved else "")
        )
        if no_improve >= args.patience:
            print(f"  Early stopping at epoch {epoch} (best epoch={best_epoch}, best val_bce={best_val_bce:.5f})")
            break

    assert best_state is not None
    spec_model.load_state_dict(best_state)
    spec_model.to(device).eval()

    final_val = evaluate_spectator(spec_model, va_obs0, va_obs1, va_y, device)
    seed_trace = trace_seed_20261106(ppo_model, spec_model, device)

    print("\n[4/4] Final Validation & Seed=20261106 Trace:")
    print(f"  PPO Critic Baseline : LogLoss={ppo_baseline['logloss']:.5f}, Brier={ppo_baseline['brier']:.5f}, ECE={ppo_baseline['ece']:.5f}")
    print(
        f"  Spectator Net (Best): LogLoss={final_val['logloss']:.5f}, Brier={final_val['brier']:.5f}, "
        f"ECE={final_val['ece']:.5f}, zero_sum_max_err={final_val['zero_sum_max_err']:.2e}"
    )
    print("  Calibration Bins:")
    for b in final_val["bins"]:
        print(f"    {b['bin']} n={b['count']:6d} | pred={b['mean_pred']:.4f} | emp={b['empirical_wr']}")

    print("\n  Seed=20261106 (nr-temeria vs ng-white-flame) first 14 steps:")
    for row in seed_trace[:14]:
        print(
            f"    st={row['step']:2d} R{row['round']} {row['sub_phase']:8s} P{row['turn']} | "
            f"PPO={row['ppo_p0_wr']:5.1f}% | Spec P0={row['spec_p0_wr']:5.1f}% P1={row['spec_p1_wr']:5.1f}%"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pt_path = MODELS_DIR / "spectator_best.pt"
    torch.save(
        {
            "state_dict": best_state,
            "obs_dim": OBS_DIM,
            "hidden_dim": args.hidden_dim,
            "num_blocks": args.num_blocks,
            "best_epoch": best_epoch,
            "val_metrics": final_val,
        },
        pt_path,
    )
    print(f"[Saved Checkpoint] {pt_path}")

    metrics_out = {
        "train_games": args.train_games,
        "train_steps": int(len(tr_y)),
        "val_games": args.val_games,
        "val_steps": int(len(va_y)),
        "hidden_dim": args.hidden_dim,
        "num_blocks": args.num_blocks,
        "best_epoch": best_epoch,
        "ppo_baseline_val": ppo_baseline,
        "spectator_val": final_val,
        "seed_20261106_trace": seed_trace,
    }
    (RESULTS_DIR / "spectator_metrics.json").write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    try:
        export_spectator_to_onnx(spec_model, MODELS_DIR / "spectator_v1.onnx")
        if WEB_MODELS_DIR.exists():
            export_spectator_to_onnx(spec_model, WEB_MODELS_DIR / "spectator_v1.onnx")
    except Exception as e:
        print(f"[ONNX Export Deferred] Could not export ONNX in current environment: {e}")


def export_only(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    hidden_dim = ckpt.get("hidden_dim", args.hidden_dim) if isinstance(ckpt, dict) else args.hidden_dim
    num_blocks = ckpt.get("num_blocks", args.num_blocks) if isinstance(ckpt, dict) else args.num_blocks
    model = GwentSpectatorValueNet(
        obs_dim=OBS_DIM,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=0.0,
    )
    model.load_state_dict(state)
    export_spectator_to_onnx(model, MODELS_DIR / "spectator_v1.onnx")
    if WEB_MODELS_DIR.exists():
        export_spectator_to_onnx(model, WEB_MODELS_DIR / "spectator_v1.onnx")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export Gwent Spectator Value Net")
    parser.add_argument("--ppo-ckpt", type=str, default=str(MODELS_DIR / "ppo_best.pt"))
    parser.add_argument("--train-games", type=int, default=24000)
    parser.add_argument("--val-games", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--greedy-ratio", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--ckpt", type=str, default=str(MODELS_DIR / "spectator_best.pt"))
    args = parser.parse_args()

    if args.export_only:
        export_only(args)
    else:
        train_spectator(args)


if __name__ == "__main__":
    main()
