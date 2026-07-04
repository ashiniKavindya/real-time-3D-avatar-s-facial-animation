# Phase 3 Process and File Guide

This document explains every file currently present in the Phase 3 folder.

## Goal of Phase 3

Phase 3 trains a baseline emotion classifier from Phase 2 windowed features, evaluates model quality, generates reports/plots, and exports the trained model to ONNX.

## File-by-File Description

### __init__.py
- Package marker for Phase 3.
- Contains a short module docstring describing the package purpose.

### model.py
- Defines the baseline neural network: `EmotionMLP`.
- Architecture:
  - Flattens each input window `(sequence_length, input_features)`.
  - Two fully connected hidden layers with ReLU and dropout.
  - Final linear layer outputs class logits.
- Used by both training (`train_baseline.py`) and export (`export_onnx.py`).

### train_baseline.py
- Main training and evaluation entry point for Phase 3.
- Responsibilities:
  - Loads manifest and shard files from Phase 2 output.
  - Splits each shard into train/validation subsets.
  - Trains `EmotionMLP` across epochs.
  - Tracks best validation accuracy and saves checkpoint.
  - Re-evaluates using the best checkpoint before final reporting.
  - Writes training config and report artifacts.
- Typical outputs it creates under `phase3/`:
  - `training_config.json`
  - `checkpoints/best_model.pt`
  - `reports/*` (CSV/TXT/PNG/JSON report files)

### metrics.py
- Implements core evaluation utilities:
  - Confusion matrix generation.
  - Classification report table (precision, recall, F1, support).
  - One-vs-rest ROC curve points and AUC.
- Includes input validation checks for shape/length consistency and safe handling of edge cases.

### plots.py
- Saves visualization images used in reports:
  - Training accuracy/loss curves by epoch.
  - Confusion matrix heatmap.
  - Multi-class ROC curves.
- Uses non-interactive Matplotlib backend (`Agg`) for script-based saving.

### export_onnx.py
- Exports trained Phase 3 checkpoint to ONNX format.
- Steps:
  - Loads `EmotionMLP` with checkpoint config/state.
  - Exports ONNX model with dynamic batch axis.
  - Optionally validates ONNX graph (if `onnx` is installed).
  - Optionally compares ONNX Runtime vs PyTorch outputs (if `--verify` is used and `onnxruntime` is installed).

### training_config.json
- Generated configuration snapshot from the latest training run.
- Records key hyperparameters and data settings such as:
  - model architecture and dimensions
  - optimizer/loss details
  - batch size, epochs, dropout
  - class names and source data directory
- Useful for reproducibility and experiment tracking.

## Recommended Execution Order

1. Run training:
   - `python -m phase3.train_baseline`
2. Review generated report files under `phase3/reports/`.
3. Export the best model:
   - `python -m phase3.export_onnx --verify`
