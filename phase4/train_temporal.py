from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase4.metrics import classification_report_frame, confusion_matrix, one_vs_rest_roc_auc
from phase4.plots import save_confusion_matrix_heatmap, save_roc_curves, save_training_curves


DEFAULT_DATA_DIR = Path("phase2/output")
DEFAULT_OUTPUT_DIR = Path("phase4")


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for Phase 4 training. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc
    return torch


def load_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def class_names_from_manifest(manifest: dict[str, Any]) -> list[str]:
    label_map = manifest["label_map"]
    return [name for name, _class_id in sorted(label_map.items(), key=lambda item: item[1])]


def find_shards(data_dir: Path, max_shards: int | None) -> list[Path]:
    shards = sorted(data_dir.glob("windows_*.npz"))
    if max_shards is not None:
        shards = shards[:max_shards]
    if not shards:
        raise SystemExit(f"No window shards found in {data_dir}")
    return shards


def split_indices(length: int, val_split: float, seed: int, shard_index: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + shard_index)
    indices = np.arange(length)
    rng.shuffle(indices)
    val_count = max(1, int(round(length * val_split))) if length > 1 else 0
    val_indices = np.sort(indices[:val_count])
    train_indices = indices[val_count:]
    return train_indices, val_indices


def iter_batches(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None,
):
    working = indices.copy()
    if rng is not None:
        rng.shuffle(working)
    for start in range(0, len(working), batch_size):
        batch_indices = working[start : start + batch_size]
        yield np.ascontiguousarray(X[batch_indices]), np.ascontiguousarray(y[batch_indices])


def load_shard(shard_path: Path, max_windows_per_shard: int | None) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(shard_path, allow_pickle=False)
    X = data["X"]
    y = data["y"]
    if max_windows_per_shard is not None:
        X = X[:max_windows_per_shard]
        y = y[:max_windows_per_shard]
    return X.astype(np.float32, copy=False), y.astype(np.int64, copy=False)


def build_class_weights(shards: list[Path], max_windows_per_shard: int | None) -> np.ndarray:
    counts: np.ndarray | None = None
    for shard_path in shards:
        _X, y = load_shard(shard_path, max_windows_per_shard)
        shard_counts = np.bincount(y, minlength=6).astype(np.float64)
        counts = shard_counts if counts is None else counts + shard_counts

    if counts is None or counts.sum() == 0:
        return np.ones(6, dtype=np.float32)

    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def train_epoch(torch, model, criterion, optimizer, shards, args, device, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    rng = np.random.default_rng(args.seed + epoch)

    for shard_index, shard_path in enumerate(shards):
        X, y = load_shard(shard_path, args.max_windows_per_shard)
        train_indices, _val_indices = split_indices(len(y), args.val_split, args.seed, shard_index)
        for batch_X, batch_y in iter_batches(X, y, train_indices, args.batch_size, rng):
            inputs = torch.from_numpy(batch_X).to(device=device, dtype=torch.float32)
            targets = torch.from_numpy(batch_y).to(device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(batch_y)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += len(batch_y)

    return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1)


def evaluate(torch, model, criterion, shards, args, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []

    with torch.no_grad():
        for shard_index, shard_path in enumerate(shards):
            X, y = load_shard(shard_path, args.max_windows_per_shard)
            _train_indices, val_indices = split_indices(len(y), args.val_split, args.seed, shard_index)
            for batch_X, batch_y in iter_batches(X, y, val_indices, args.batch_size, None):
                inputs = torch.from_numpy(batch_X).to(device=device, dtype=torch.float32)
                targets = torch.from_numpy(batch_y).to(device=device, dtype=torch.long)
                logits = model(inputs)
                loss = criterion(logits, targets)
                probabilities = torch.softmax(logits, dim=1)
                predictions = logits.argmax(dim=1)

                total_loss += float(loss.item()) * len(batch_y)
                total_correct += int((predictions == targets).sum().item())
                total_examples += len(batch_y)
                all_true.append(batch_y.copy())
                all_pred.append(predictions.cpu().numpy())
                all_prob.append(probabilities.cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.empty((0,), dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.empty((0,), dtype=np.int64)
    probabilities = np.concatenate(all_prob) if all_prob else np.empty((0, model.num_classes), dtype=np.float32)
    return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1), y_true, y_pred, probabilities


def write_reports(output_dir, class_names, history, y_true, y_pred, probabilities):
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report_frame(y_true, y_pred, class_names)
    report.to_csv(reports_dir / "classification_report.csv", index=False)
    (reports_dir / "classification_report.txt").write_text(report.to_string(index=False), encoding="utf-8")

    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(reports_dir / "confusion_matrix.csv")
    save_confusion_matrix_heatmap(matrix, class_names, reports_dir / "confusion_matrix.png")

    pd.DataFrame(history).to_csv(reports_dir / "training_history.csv", index=False)
    save_training_curves(history, reports_dir / "training_curves.png")

    roc_curves = one_vs_rest_roc_auc(y_true, probabilities, class_names)
    save_roc_curves(roc_curves, reports_dir / "roc_curves.png")

    auc_by_class = {class_name: float(curve["auc"]) for class_name, curve in roc_curves.items()}
    final_metrics = {
        "final_train_loss": history[-1]["train_loss"],
        "final_train_accuracy": history[-1]["train_accuracy"],
        "final_val_loss": history[-1]["val_loss"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "macro_f1": float(report.loc[report["Class"] == "macro avg", "F1-Score"].iloc[0]),
        "auc_by_class": auc_by_class,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    return final_metrics


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be > 0")
    if args.model_dim < 1 or args.num_heads < 1 or args.num_layers < 1 or args.feedforward_dim < 1:
        raise SystemExit("Transformer size parameters must be >= 1")
    if args.model_dim % args.num_heads != 0:
        raise SystemExit("--model-dim must be divisible by --num-heads")
    if not 0 <= args.dropout < 1:
        raise SystemExit("--dropout must be in [0, 1)")
    if not 0 < args.val_split < 1:
        raise SystemExit("--val-split must be between 0 and 1")
    if args.max_shards is not None and args.max_shards < 1:
        raise SystemExit("--max-shards must be >= 1 when provided")
    if args.max_windows_per_shard is not None and args.max_windows_per_shard < 1:
        raise SystemExit("--max-windows-per-shard must be >= 1 when provided")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 4 transformer trainer")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Phase 2 output directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Phase 4 output directory")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-shards", type=int, default=None, help="Optional smoke-test cap")
    parser.add_argument("--max-windows-per-shard", type=int, default=None, help="Optional smoke-test cap")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    torch = require_torch()
    from phase4.model import EmotionTransformer

    validate_args(args)

    manifest = load_manifest(args.data_dir)
    class_names = class_names_from_manifest(manifest)
    feature_shape = manifest["feature_shape"]
    sequence_length, input_features = int(feature_shape[0]), int(feature_shape[1])
    shards = find_shards(args.data_dir, args.max_shards)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    torch.manual_seed(args.seed)
    model = EmotionTransformer(
        sequence_length=sequence_length,
        input_features=input_features,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout,
        num_classes=len(class_names),
        max_length=max(sequence_length, 512),
    ).to(device)

    class_weights = build_class_weights(shards, args.max_windows_per_shard)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    checkpoints_dir = args.output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architecture": "Temporal transformer",
        "sequence_length": sequence_length,
        "input_features": input_features,
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "loss_function": "CrossEntropyLoss(weighted)",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "model_dim": args.model_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "feedforward_dim": args.feedforward_dim,
        "dropout": args.dropout,
        "class_names": class_names,
        "class_weights": class_weights.tolist(),
        "data_dir": str(args.data_dir),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Training Phase 4 transformer on {len(shards)} shard(s)")
    print(f"Device: {device}")
    print(f"Input shape: ({sequence_length}, {input_features})")
    print(f"Class weights: {class_weights.tolist()}")

    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_checkpoint = checkpoints_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(torch, model, criterion, optimizer, shards, args, device, epoch)
        val_loss, val_accuracy, y_true, y_pred, probabilities = evaluate(torch, model, criterion, shards, args, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} - "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "class_names": class_names,
                    "sequence_length": sequence_length,
                    "input_features": input_features,
                },
                best_checkpoint,
            )

    best_state = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(best_state["model_state_dict"])
    val_loss, val_accuracy, y_true, y_pred, probabilities = evaluate(torch, model, criterion, shards, args, device)
    history[-1]["val_loss"] = val_loss
    history[-1]["val_accuracy"] = val_accuracy

    metrics = write_reports(args.output_dir, class_names, history, y_true, y_pred, probabilities)
    print(f"Saved best checkpoint: {best_checkpoint}")
    print(f"Saved reports: {args.output_dir / 'reports'}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
