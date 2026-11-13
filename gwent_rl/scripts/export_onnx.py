"""
Exports a trained Gwent PyTorch model (GwentPolicyValueNet) to ONNX format
for in-browser inference with `onnxruntime-web`, and validates numerical parity
with `onnxruntime`.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from gwent_rl.engine import OBS_DIM, ACTION_DIM
from gwent_rl.agents.networks import GwentPolicyValueNet

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = ROOT / "gwent_web" / "public" / "models"


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    model_id: str = "ppo-v1",
    model_name: str = "RL · Masked PPO (Self-Play)",
    algo: str = "ppo",
    description: str = "Masked PPO + PFSP 种群自对弈模型（浏览器端 ONNX 实时推理）",
) -> Path:
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    ext_data_file = Path(str(out_file) + ".data")
    if ext_data_file.exists():
        ext_data_file.unlink()
    model_cpu = model.to("cpu").eval()

    dummy_obs = torch.zeros((1, OBS_DIM), dtype=torch.float32)
    dummy_mask = torch.ones((1, ACTION_DIM), dtype=torch.bool)

    torch.onnx.export(
        model_cpu,
        (dummy_obs, dummy_mask),
        str(out_file),
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

    # Collapse any sidecar .onnx.data created by PyTorch dynamo exporter into a single browser-loadable .onnx file
    if ext_data_file.exists():
        try:
            import onnx

            onnx_model = onnx.load(str(out_file), load_external_data=True)
            onnx.save_model(onnx_model, str(out_file), save_as_external_data=False)
            ext_data_file.unlink()
        except Exception as e:
            print(f"[ONNX Warning] Failed to inline {ext_data_file.name}: {e}")

    # Verify with onnxruntime if available
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            pt_probs, pt_val = model_cpu(dummy_obs, dummy_mask)
        ort_probs, ort_val = sess.run(
            ["probs", "value"],
            {"obs": dummy_obs.numpy(), "mask": dummy_mask.numpy()},
        )
        max_diff_p = float(np.max(np.abs(pt_probs.numpy() - ort_probs)))
        max_diff_v = float(np.max(np.abs(pt_val.numpy() - ort_val)))
        print(f"[ONNX Verified] {out_file.name} | max_diff_probs={max_diff_p:.2e}, max_diff_val={max_diff_v:.2e}")
    except ImportError:
        print(f"[ONNX Exported] {out_file.name} (onnxruntime not installed for verification)")

    # Update manifest.json in the same directory
    manifest_path = out_file.parent / "manifest.json"
    manifest = {"models": []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    entry = {
        "id": model_id,
        "name": model_name,
        "algo": algo,
        "file": f"/models/{out_file.name}",
        "obsDim": OBS_DIM,
        "actionDim": ACTION_DIM,
        "description": description,
    }
    models_list = [m for m in manifest.get("models", []) if m.get("id") != model_id]
    models_list.append(entry)
    manifest["models"] = models_list
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Gwent PyTorch PPO checkpoint to ONNX")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt state_dict")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--id", type=str, default="ppo-v1")
    parser.add_argument("--name", type=str, default="RL · Masked PPO (Self-Play)")
    parser.add_argument("--out", type=str, default=str(DEFAULT_MODELS_DIR / "ppo_v1.onnx"))
    args = parser.parse_args()

    model = GwentPolicyValueNet(hidden_dim=args.hidden_dim, num_blocks=args.num_blocks)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    export_to_onnx(
        model=model,
        output_path=args.out,
        model_id=args.id,
        model_name=args.name,
        algo="ppo",
    )


if __name__ == "__main__":
    main()
