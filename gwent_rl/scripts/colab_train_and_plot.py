"""
Google Colab GPU training, benchmarking, ONNX export, Google Drive persistence,
and pure Masked-PPO multi-chart plotting runner.

Delegates core training, batched GPU opponent inference, PFSP league, and
Duplicate Paired-Seed evaluation to `gwent_rl.agents.masked_ppo` (zero duplicated PPO loop).
"""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
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

from gwent_rl.engine import ALL_PRESET_IDS, OBS_DIM, ACTION_DIM
from gwent_rl.agents.masked_ppo import train_masked_ppo
from gwent_rl.scripts.export_onnx import export_to_onnx

OUT_DIR = Path("gwent_rl/results") if Path("gwent_rl").exists() else Path("results")
MODELS_DIR = Path("gwent_rl/models") if Path("gwent_rl").exists() else Path("models")
WEB_MODELS_DIR = Path("gwent_web/public/models")
DRIVE_DIR = Path("/content/drive/MyDrive/Gwent_RL")


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
    ax.plot(
        ppo_x,
        [r["elo"] for r in ppo_hist],
        "-o",
        markersize=3.8,
        linewidth=2.2,
        color="#0284c7",
        label=f"Masked PPO ({ppo_hist[-1]['steps']/1e6:.2f}M Steps / {ppo_hist[-1]['episodes']:,} Games)",
    )
    ax.axhline(1328, color="#ef4444", linestyle="--", linewidth=1.3, label="Heuristic Hard Baseline (1328)")
    ax.axhline(1200, color="#64748b", linestyle=":", linewidth=1.2, label="Heuristic Normal Baseline (1200)")
    ax.axhline(932, color="#94a3b8", linestyle="-.", linewidth=1.1, label="Heuristic Easy Baseline (932)")
    ax.set_title("Gwent RL Self-Play League — Elo Progression (All 8 Decks / 4 Factions)", fontsize=12.5, fontweight="bold")
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
        ax.scatter(
            [ppo_x[-1]],
            [ppo_audit["wr_hard"]],
            marker="*",
            s=160,
            color="#b91c1c",
            zorder=6,
            label=f"{ppo_audit['games_per_diff']}-Game Final Audit Hard: {ppo_audit['wr_hard']:.1f}%±{ppo_audit['se_hard']:.1f}%",
        )
    ax.axhline(50.0, color="#64748b", linestyle="--", linewidth=1.2, label="50% Parity Threshold")
    ax.set_ylim(0, 105)
    ax.set_title("Masked PPO — Duplicate Paired Win Rate Across All 64 Matchups (±1 SE)", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("Environment Steps (Thousands)", fontsize=11)
    ax.set_ylabel("Win Rate (%)", fontsize=11)
    ax.legend(loc="lower right", frameon=True, fontsize=9.0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "chart2_ppo_winrates.png")
    plt.close(fig)

    # 3. Chart 3: Loss & Policy Entropy Convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=160)
    ax1.plot(ppo_x[1:], [r["v_loss"] for r in ppo_hist[1:]], "-o", markersize=3.2, color="#0284c7", label="Value Loss (BCE)")
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

    # 4. Combined 2x2 Dashboard (Pure Masked PPO)
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.5), dpi=160)
    axes[0, 0].plot(ppo_x, [r["elo"] for r in ppo_hist], "-o", markersize=3.2, color="#0284c7", label="Masked PPO")
    axes[0, 0].axhline(1328, color="#ef4444", linestyle="--", label="Heuristic Hard (1328)")
    axes[0, 0].axhline(1200, color="#64748b", linestyle=":", label="Heuristic Normal (1200)")
    axes[0, 0].axhline(932, color="#94a3b8", linestyle="-.", label="Heuristic Easy (932)")
    axes[0, 0].set_title("1. League Elo Progression", fontweight="bold")
    axes[0, 0].set_xlabel("Steps (k)")
    axes[0, 0].set_ylabel("Elo")
    axes[0, 0].legend(fontsize=8.5)

    _plot_with_se_band(axes[0, 1], ppo_x, ppo_hist, "wr_easy", "se_easy", "-o", "#22c55e", "vs Easy")
    _plot_with_se_band(axes[0, 1], ppo_x, ppo_hist, "wr_normal", "se_normal", "-^", "#0284c7", "vs Normal (0-ply)")
    _plot_with_se_band(axes[0, 1], ppo_x, ppo_hist, "wr_hard", "se_hard", "-s", "#ef4444", "vs Hard (1-ply)")
    axes[0, 1].axhline(50.0, color="#64748b", linestyle="--")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].set_title("2. Duplicate Paired Win Rates (All 64 Matchups, ±1 SE)", fontweight="bold")
    axes[0, 1].set_xlabel("Steps (k)")
    axes[0, 1].set_ylabel("Win Rate (%)")
    axes[0, 1].legend(fontsize=8.5)

    axes[1, 0].plot(ppo_x[1:], [r["v_loss"] for r in ppo_hist[1:]], "-o", markersize=3.2, color="#0284c7", label="Value Loss (BCE)")
    axes[1, 0].plot(ppo_x[1:], [r["pg_loss"] for r in ppo_hist[1:]], "-^", markersize=3.2, color="#f59e0b", label="Policy Loss")
    axes[1, 0].set_title("3. PPO Optimization Losses", fontweight="bold")
    axes[1, 0].set_xlabel("Steps (k)")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend(fontsize=8.5)

    axes[1, 1].plot(ppo_x[1:], [r["entropy"] for r in ppo_hist[1:]], "-s", markersize=3.2, color="#10b981", label="Policy Entropy")
    axes[1, 1].set_title("4. Policy Shannon Entropy", fontweight="bold")
    axes[1, 1].set_xlabel("Steps (k)")
    axes[1, 1].set_ylabel("Entropy")
    axes[1, 1].legend(fontsize=8.5)

    fig.suptitle("Gwent-RL Self-Play Training Report (Masked PPO + PFSP League, 8 Decks)", fontsize=14.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "training_dashboard.png")
    plt.close(fig)
    print(f"[Charts] Saved high-res figures to {OUT_DIR.resolve()}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Gwent-RL Colab Masked PPO Training Pipeline")
    parser.add_argument("--steps", type=int, default=2_457_600, help="Total PPO timesteps (default 2.46M = 300 updates)")
    parser.add_argument("--num-envs", type=int, default=32, help="Vectorized environments")
    parser.add_argument("--lr", type=float, default=2.0e-4, help="Peak learning rate")
    parser.add_argument("--audit-games", type=int, default=384, help="Out-of-sample audit games per difficulty")
    parser.add_argument("--warm-start", type=str, default=None, help="Optional .pt checkpoint for Stage-2 continuation")
    parser.add_argument("--critic-warmup", type=int, default=5, help="Critic-only warmup updates when warm-starting")
    args, _ = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Gwent-RL Colab Masked PPO + PFSP League | Device: {device} | Presets: {len(ALL_PRESET_IDS)} ===")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    prev_history = None
    if args.warm_start:
        metrics_file = OUT_DIR / "training_metrics.json"
        if metrics_file.exists():
            try:
                prev_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                if prev_metrics.get("obs_dim") == OBS_DIM:
                    prev_history = prev_metrics.get("ppo_history", [])
            except Exception:
                pass

    def _on_eval(_model, hist):
        (OUT_DIR / "training_metrics.json").write_text(
            json.dumps(
                {
                    "device": str(device),
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                    "obs_dim": OBS_DIM,
                    "action_dim": ACTION_DIM,
                    "algo": "Masked PPO + PFSP League (BCE Win-Prob Critic)",
                    "preset_ids": list(ALL_PRESET_IDS),
                    "ppo_history": hist,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        sync_to_google_drive()

    ppo_model, ppo_hist, ppo_audit = train_masked_ppo(
        total_timesteps=args.steps,
        num_envs=args.num_envs,
        num_steps=256,
        lr=args.lr,
        min_lr=1.5e-5,
        lr_schedule="cosine",
        clip_coef=0.18 if args.warm_start else 0.20,
        ent_coef_start=0.008 if args.warm_start else 0.020,
        ent_coef_end=0.002 if args.warm_start else 0.004,
        vf_coef=0.75,
        target_kl=0.03,
        minibatch_size=512,
        preset_ids=ALL_PRESET_IDS,
        eval_games_per_diff=128,
        audit_games_per_diff=args.audit_games,
        warm_start_path=args.warm_start,
        critic_warmup_updates=args.critic_warmup,
        prev_history=prev_history,
        save_dir=MODELS_DIR,
        device_str=str(device),
        on_eval_callback=_on_eval,
    )

    torch.save(ppo_model.state_dict(), MODELS_DIR / "ppo_best.pt")
    try:
        export_to_onnx(
            model=ppo_model,
            output_path=MODELS_DIR / "ppo_v1.onnx",
            model_id="ppo-v1",
            model_name="RL · Masked PPO (Self-Play)",
            algo="ppo",
            description="Masked PPO + PFSP 全阵营 8 卡组自对弈模型（浏览器端 ONNX 实时推理）",
        )
        if WEB_MODELS_DIR.exists():
            export_to_onnx(
                model=ppo_model,
                output_path=WEB_MODELS_DIR / "ppo_v1.onnx",
                model_id="ppo-v1",
                model_name="RL · Masked PPO (Self-Play)",
                algo="ppo",
                description="Masked PPO + PFSP 全阵营 8 卡组自对弈模型（浏览器端 ONNX 实时推理）",
            )
    except Exception as e:
        print(f"[ONNX Export Skipped] Could not export ONNX (install onnx/onnxscript if needed): {e}")

    metrics = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "algo": "Masked PPO + PFSP League (BCE Win-Prob Critic)",
        "preset_ids": list(ALL_PRESET_IDS),
        "eval_protocol": f"Duplicate Paired-Seed (128 games/diff across all 64 pairs in-train, {args.audit_games} games/diff Top-3 Final Audit)",
        "calibrated_baseline_elos": {"easy": 932, "normal": 1200, "hard": 1328},
        "ppo_final_audit": ppo_audit,
        "ppo_history": ppo_hist,
    }
    (OUT_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    render_training_charts(ppo_hist, ppo_audit)
    sync_to_google_drive()
    print("=== Training, Top-3 Audit, ONNX Export, Chart Rendering & Drive Sync Complete! ===")


if __name__ == "__main__":
    main()
