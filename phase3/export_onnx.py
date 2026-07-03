from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_CHECKPOINT = Path("phase3/checkpoints/best_model.pt")
DEFAULT_OUTPUT = Path("phase3/onnx/emotion_baseline.onnx")


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for ONNX export. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc
    return torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the Phase 3 MLP baseline to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify", action="store_true", help="Compare PyTorch and ONNX Runtime outputs when onnxruntime is installed")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    torch = require_torch()
    from phase3.model import EmotionMLP

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    model = EmotionMLP(
        sequence_length=int(checkpoint["sequence_length"]),
        input_features=int(checkpoint["input_features"]),
        hidden_1=int(config.get("hidden_1", 512)),
        hidden_2=int(config.get("hidden_2", 128)),
        num_classes=len(checkpoint["class_names"]),
        dropout_1=float(config.get("dropout_1", 0.3)),
        dropout_2=float(config.get("dropout_2", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, int(checkpoint["sequence_length"]), int(checkpoint["input_features"]), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"Saved ONNX model: {args.output}")

    try:
        import onnx

        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("ONNX checker: passed")
    except ModuleNotFoundError:
        print("ONNX checker skipped: install onnx to enable model validation")

    if args.verify:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError:
            print("ONNX Runtime verification skipped: install onnxruntime to compare outputs")
            return 0

        with torch.no_grad():
            torch_logits = model(dummy).numpy()
        session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        onnx_logits = session.run(["logits"], {"landmarks": dummy.numpy().astype(np.float32)})[0]
        max_abs_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
        print(f"ONNX Runtime max abs diff: {max_abs_diff:.8f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())