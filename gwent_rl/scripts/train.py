"""
Unified CLI training script for Gwent RL (Masked PPO & DouZero-style DMC).
Can be run locally or on Google Colab GPU via `colab-operator`, and automatically
exports the best and latest checkpoints to `.onnx` in `gwent_web/public/models/`.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from gwent_rl.agents.masked_ppo import train_masked_ppo
from gwent_rl.agents.dmc import train_dmc
from gwent_rl.scripts.export_onnx import export_to_onnx, DEFAULT_MODELS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Gwent RL Agent (PPO or DMC)")
    parser.add_argument("--algo", type=str, choices=["ppo", "dmc", "both"], default="both")
    parser.add_argument("--ppo-steps", type=int, default=300_000)
    parser.add_argument("--dmc-episodes", type=int, default=10_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--models-dir", type=str, default=str(DEFAULT_MODELS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.algo in ("ppo", "both"):
        ppo_model = train_masked_ppo(
            total_timesteps=args.ppo_steps,
            num_envs=args.num_envs,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            lr=args.lr,
            device_str=args.device,
        )
        export_to_onnx(
            model=ppo_model,
            output_path=out_dir / "ppo_v1.onnx",
            model_id="ppo-v1",
            model_name="Masked PPO + PFSP League",
            algo="ppo",
            description="Trained with Masked PPO + PFSP Self-Play League",
        )

    if args.algo in ("dmc", "both"):
        dmc_model = train_dmc(
            total_episodes=args.dmc_episodes,
            hidden_dim=args.hidden_dim,
            num_blocks=args.num_blocks,
            lr=args.lr,
            device_str=args.device,
        )
        export_to_onnx(
            model=dmc_model,
            output_path=out_dir / "dmc_v1.onnx",
            model_id="dmc-v1",
            model_name="DouZero DMC (Deep Monte-Carlo)",
            algo="dmc",
            description="Trained with DouZero-style Deep Monte-Carlo Q-learning",
        )


if __name__ == "__main__":
    main()
