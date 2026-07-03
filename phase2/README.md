# Phase 2: Data Preprocessing & Sequence Engineering

This phase turns the raw landmark frame table into fixed-size sequence windows for the Phase 3 baseline and later temporal models.

The preprocessor:
- loads the full dataset from `data\emotion_landmark_dataset.csv` by default
- groups frames by `video_filename` and sorts each clip by `frame_num`
- normalizes every frame with the Phase 0 face-anchor/eye-width transform
- emits overlapping windows shaped `(window_size, 1434)`
- writes compressed `.npz` shards plus a `manifest.json`
- prints progress as chunks are processed

## Full Dataset Run

From the project root, run:

```powershell
python -m phase2.preprocess_sequences
```

That default command is equivalent to:

```powershell
python -m phase2.preprocess_sequences data\emotion_landmark_dataset.csv --output-dir phase2\output --window-size 30 --stride 5 --chunksize 50000 --shard-size 10000
```

The full CSV is about 16 GB, so this can take a while. The script streams CSV rows in chunks instead of loading the whole file at once, reads landmark columns as `float32` to reduce memory use, and writes output shards under `phase2\output`.

## Quick Test Run

Use a small row cap when you want to verify changes without processing the full dataset:

```powershell
python -m phase2.preprocess_sequences --output-dir phase2\output_sample --max-rows 5000
```

## Output Files

Each `.npz` shard contains:

- `X`: float32 sequence tensor, shaped `(num_windows, window_size, 1434)`
- `y`: int64 encoded emotion labels
- `metadata`: per-window clip name, start frame, end frame, and original label

`manifest.json` stores the label map, feature columns, window shape, preprocessing config, and preprocessing summary.

The default label map is stable across sample and full runs:

```text
Angry=0, Disgust=1, Fear=2, Happy=3, Neutral=4, Sad=5
```

## Useful Options

- `dataset_path`: optional input path; default is `data\emotion_landmark_dataset.csv`
- `--output-dir`: output folder; default is `phase2\output`
- `--window-size`: frames per sequence window; default is `30`
- `--stride`: frame step between windows; default is `5`
- `--chunksize`: CSV rows loaded per chunk; default is `50000`
- `--shard-size`: approximate windows per `.npz` shard; default is `10000`
- `--max-rows`: optional row cap for testing; omit it for the full dataset
- `--progress-every-chunks`: print progress every N chunks; default is `1`

If you convert the dataset to Parquet later, pass the `.parquet` path explicitly. Parquet currently loads in one frame, which is convenient for optimized local subsets; use CSV streaming for the large raw export.