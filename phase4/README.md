# Phase 4: Temporal Transformer

Phase 4 replaces the Phase 3 MLP baseline with a temporal transformer that learns from sliding windows of facial landmark sequences.

## Files

### `__init__.py`
Marks `phase4` as a Python package and documents the package purpose.

### `model.py`
Defines `EmotionTransformer`, a sequence classifier built from:
- a linear input projection from 1,434 landmark features into transformer space
- sinusoidal positional encoding
- a stack of `nn.TransformerEncoder` layers
- mean pooling across the time axis
- a classification head that outputs emotion logits

### `metrics.py`
Contains evaluation helpers for:
- confusion matrices
- classification reports with precision, recall, F1, and support
- one-vs-rest ROC curves and AUC

### `plots.py`
Generates saved report figures for:
- training curves
- confusion matrix heatmaps
- ROC curves

### `train_temporal.py`
Trains the transformer on the window shards produced by Phase 2.

Key behavior:
- loads `phase2/output/manifest.json`
- discovers `windows_*.npz` shards
- splits each shard into train/validation windows
- computes class weights from the shard labels
- trains with weighted cross entropy
- saves the best checkpoint and writes reports to `phase4/reports`

### `export_onnx.py`
Exports the best transformer checkpoint to ONNX and optionally verifies it with ONNX Runtime.

## Running Phase 4

Train:

```powershell
python -m phase4.train_temporal --epochs 15 --device auto
```

Smoke test:

```powershell
python -m phase4.train_temporal --epochs 1 --max-shards 1 --max-windows-per-shard 32 --batch-size 8 --device cpu
```

Export ONNX:

```powershell
python -m phase4.export_onnx --verify
```

## Output Artifacts

Phase 4 writes these folders and files under `phase4/`:
- `training_config.json`
- `checkpoints/best_model.pt`
- `reports/classification_report.csv`
- `reports/classification_report.txt`
- `reports/confusion_matrix.csv`
- `reports/confusion_matrix.png`
- `reports/training_history.csv`
- `reports/training_curves.png`
- `reports/roc_curves.png`
- `reports/metrics.json`
- `onnx/emotion_transformer.onnx`
