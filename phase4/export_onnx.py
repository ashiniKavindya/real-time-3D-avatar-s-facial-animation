from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_CHECKPOINT = Path("phase4/checkpoints/best_model.pt")
DEFAULT_OUTPUT_ROOT = Path("phase4")


def find_latest_run_dir(output_root: Path) -> Path:
    runs_root = output_root / "runs"
    if not runs_root.exists():
        raise SystemExit(f"No Phase 4 runs found in {runs_root}")

    candidates = [path for path in runs_root.iterdir() if path.is_dir()]
    if not candidates:
        raise SystemExit(f"No Phase 4 run folders found in {runs_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_checkpoint_path(checkpoint: Path | None, run_dir: Path | None, output_root: Path) -> tuple[Path, Path]:
    if checkpoint is not None:
        checkpoint_path = checkpoint
        if run_dir is None:
            run_dir = checkpoint_path.parent.parent.parent
    else:
        if run_dir is None:
            run_dir = find_latest_run_dir(output_root)
        checkpoint_path = run_dir / "model" / "checkpoints" / "best_model.pt"

    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path, run_dir


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
    parser = argparse.ArgumentParser(description="Export the Phase 4 transformer to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None, help="Explicit Phase 4 run directory")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Phase 4 output root")
    parser.add_argument("--output", type=Path, default=None, help="Override the ONNX output path")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare PyTorch and ONNX Runtime outputs when onnxruntime is installed",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    torch = require_torch()
    from phase4.model import EmotionTransformer

    checkpoint_path, run_dir = resolve_checkpoint_path(args.checkpoint, args.run_dir, args.output_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    model = EmotionTransformer(
        sequence_length=int(checkpoint["sequence_length"]),
        input_features=int(checkpoint["input_features"]),
        model_dim=int(config.get("model_dim", 256)),
        num_heads=int(config.get("num_heads", 8)),
        num_layers=int(config.get("num_layers", 2)),
        feedforward_dim=int(config.get("feedforward_dim", 512)),
        dropout=float(config.get("dropout", 0.1)),
        num_classes=len(checkpoint["class_names"]),
        max_length=max(int(checkpoint["sequence_length"]), 512),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, int(checkpoint["sequence_length"]), int(checkpoint["input_features"]), dtype=torch.float32)
    if args.output is not None:
        output_path = args.output
    else:
        output_path = run_dir / "onnx" / "emotion_transformer.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={"landmarks": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"Saved ONNX model: {output_path}")

    try:
        import onnx

        onnx_model = onnx.load(output_path)
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
        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        onnx_logits = session.run(["logits"], {"landmarks": dummy.numpy().astype(np.float32)})[0]
        max_abs_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
        print(f"ONNX Runtime max abs diff: {max_abs_diff:.8f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
