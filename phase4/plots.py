from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    frame = pd.DataFrame(history)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(frame["epoch"], frame["train_accuracy"], label="Train Accuracy", marker="o")
    axes[0].plot(frame["epoch"], frame["val_accuracy"], label="Validation Accuracy", marker="o")
    axes[0].set_title("Accuracy by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(frame["epoch"], frame["train_loss"], label="Train Loss", marker="o")
    axes[1].plot(frame["epoch"], frame["val_loss"], label="Validation Loss", marker="o")
    axes[1].set_title("Loss by Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix_heatmap(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2 if matrix.size and matrix.max() else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > threshold else "black"
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_roc_curves(curves: dict[str, dict[str, np.ndarray | float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    for class_name, curve in curves.items():
        auc = curve["auc"]
        label = f"{class_name} (AUC={auc:.3f})" if np.isfinite(auc) else f"{class_name} (AUC=N/A)"
        ax.plot(curve["fpr"], curve["tpr"], label=label)

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_title("Multi-Class ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
