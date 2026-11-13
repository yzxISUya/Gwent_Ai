"""
Unified CLI training script for Gwent RL (Masked PPO + PFSP League Self-Play).
Can be run locally or on Google Colab GPU via `colab-operator`, and automatically
exports the best audited checkpoint to `.onnx` in `gwent_web/public/models/`.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from gwent_rl.agents.masked_ppo import train_masked_ppo
from gwent_rl.scripts.export_onnx import export_to_onnx, DEFAULT_MODELS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gwent RL Agent (Masked PPO + PFSP League)")
    parser.add_argument("--steps", type=int, default=655_360, help="Total environment steps")
    parser.add_argument("--num-envs", type=int, default=16, help="Number of parallel environments")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension of ResMLP")
    parser.add_argument("--num-blocks", type=int, default=3, help="Number of ResMLP blocks")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, or auto)")
    parser.add_argument("--warm-start", type=str, default=None, help="Optional .pt checkpoint to warm-start")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR), help="Output models directory")
    args = parser.parse_args()

    out_dir = Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ppo_model, _, _ = train_masked_ppo(
        total_timesteps=args.steps,
        num_envs=args.num_envs,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        lr=args.lr,
        warm_start_path=args.warm_start,
        device_str=args.device,
    )
    export_to_onnx(
        model=ppo_model,
        output_path=out_dir / "ppo_v1.onnx",
        model_id="ppo-v1",
        model_name="RL · Masked PPO (Self-Play)",
        algo="ppo",
        description="Masked PPO + PFSP 全阵营 8 卡组自对弈模型（浏览器端 ONNX 实时推理）",
    )


if __name__ == "__main__":
    main()
