from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from phase0.landmark_normalization import landmark_block_to_array, summarize_normalization_drift


DATASET_EXTENSIONS = {".csv", ".parquet", ".pq"}
LABEL_CANDIDATES = ("label", "emotion", "emotion_label", "class")
GROUP_CANDIDATES = ("video_filename", "frame_num")


def _read_sample(dataset_path: Path, rows: int) -> pd.DataFrame:
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path, nrows=rows)
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(dataset_path)
        return frame.head(rows)
    raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")


def _pick_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def inspect_dataset(dataset_path: Path, rows: int) -> int:
    sample = _read_sample(dataset_path, rows)
    print(f"Loaded sample shape: {sample.shape}")
    print("Columns:")
    print(", ".join(sample.columns[:20]))

    label_column = _pick_column(sample.columns, LABEL_CANDIDATES)
    if label_column:
        label_counts = sample[label_column].value_counts(dropna=False).to_dict()
        print(f"Label column: {label_column}")
        print(f"Label counts: {label_counts}")
    else:
        print("Label column: not found")

    group_hits = [column for column in GROUP_CANDIDATES if column in sample.columns]
    print(f"Grouping columns present: {group_hits or 'none'}")

    landmark_row = sample.iloc[0]
    landmark_array = landmark_block_to_array(landmark_row)
    drift = summarize_normalization_drift(landmark_array)
    print("Normalization probe:")
    print(drift)

    expected_feature_count = landmark_array.shape[0] * 3
    observed_feature_columns = [column for column in sample.columns if column.startswith(("x_", "y_", "z_"))]
    print(f"Landmark count: {landmark_array.shape[0]}")
    print(f"Observed landmark feature columns: {len(observed_feature_columns)}")
    print(f"Expected landmark feature columns from row probe: {expected_feature_count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 0 dataset sanity check")
    parser.add_argument("dataset_path", type=Path, help="Path to the dataset CSV or Parquet file")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows to sample")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.dataset_path.suffix.lower() not in DATASET_EXTENSIONS:
        raise SystemExit("Please provide a CSV, Parquet, or PQ file")

    return inspect_dataset(args.dataset_path, args.rows)


if __name__ == "__main__":
    raise SystemExit(main())
