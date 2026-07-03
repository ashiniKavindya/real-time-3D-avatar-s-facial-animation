from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from phase0.landmark_normalization import normalize_landmarks


DATASET_EXTENSIONS = {".csv", ".parquet", ".pq"}
DEFAULT_DATASET_PATH = Path("data/emotion_landmark_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("phase2/output")
DEFAULT_LABEL_COLUMN = "emotion"
DEFAULT_GROUP_COLUMN = "video_filename"
DEFAULT_FRAME_COLUMN = "frame_num"
DEFAULT_LABEL_ORDER = ("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad")
METADATA_COLUMNS = (DEFAULT_GROUP_COLUMN, DEFAULT_FRAME_COLUMN, DEFAULT_LABEL_COLUMN)


@dataclass(frozen=True)
class PreprocessConfig:
    dataset_path: str
    output_dir: str
    window_size: int
    stride: int
    max_rows: int | None
    chunksize: int
    shard_size: int
    label_column: str
    group_column: str
    frame_column: str


@dataclass
class PreprocessSummary:
    frames_seen: int = 0
    groups_seen: int = 0
    windows_written: int = 0
    shards_written: int = 0
    skipped_short_groups: int = 0


def discover_landmark_columns(columns: Iterable[str]) -> list[str]:
    """Return x/y/z landmark columns ordered as x_0,y_0,z_0,..."""

    available = set(columns)
    landmark_columns: list[str] = []
    index = 0
    while True:
        triplet = [f"x_{index}", f"y_{index}", f"z_{index}"]
        if not all(column in available for column in triplet):
            break
        landmark_columns.extend(triplet)
        index += 1

    if not landmark_columns:
        raise ValueError("No landmark columns found with the expected x_N/y_N/z_N pattern")

    return landmark_columns


def frame_to_normalized_vector(values: np.ndarray) -> np.ndarray:
    landmarks = values.reshape(-1, 3).astype(np.float32, copy=False)
    normalized, _stats = normalize_landmarks(landmarks)
    return normalized.reshape(-1).astype(np.float32, copy=False)


def normalize_frame_matrix(frame: pd.DataFrame, landmark_columns: list[str]) -> np.ndarray:
    raw = frame.loc[:, landmark_columns].to_numpy(dtype=np.float32, copy=True)
    normalized = np.empty_like(raw, dtype=np.float32)
    for row_index, row in enumerate(raw):
        normalized[row_index] = frame_to_normalized_vector(row)
    return normalized


def make_label_map(labels: Iterable[str]) -> dict[str, int]:
    observed = {str(label) for label in labels}
    if observed.issubset(DEFAULT_LABEL_ORDER):
        ordered = list(DEFAULT_LABEL_ORDER)
    else:
        ordered = [label for label in DEFAULT_LABEL_ORDER if label in observed]
        ordered.extend(sorted(observed.difference(ordered)))
    return {label: index for index, label in enumerate(ordered)}


def window_group(
    group: pd.DataFrame,
    landmark_columns: list[str],
    label_map: dict[str, int],
    label_column: str,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    group = group.sort_values(DEFAULT_FRAME_COLUMN)
    if len(group) < window_size:
        feature_count = len(landmark_columns)
        return (
            np.empty((0, window_size, feature_count), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )

    normalized = normalize_frame_matrix(group, landmark_columns)
    labels = group[label_column].astype(str).to_numpy()
    frame_nums = group[DEFAULT_FRAME_COLUMN].to_numpy()
    group_name = str(group[DEFAULT_GROUP_COLUMN].iloc[0])

    windows: list[np.ndarray] = []
    y: list[int] = []
    metadata: list[dict[str, object]] = []

    for start in range(0, len(group) - window_size + 1, stride):
        end = start + window_size
        window_labels = labels[start:end]
        label = str(pd.Series(window_labels).mode(dropna=False).iloc[0])
        windows.append(normalized[start:end])
        y.append(label_map[label])
        metadata.append(
            {
                "video_filename": group_name,
                "start_frame": int(frame_nums[start]),
                "end_frame": int(frame_nums[end - 1]),
                "label": label,
            }
        )

    return np.stack(windows).astype(np.float32), np.asarray(y, dtype=np.int64), metadata


def write_shard(
    output_dir: Path,
    shard_index: int,
    windows: list[np.ndarray],
    labels: list[np.ndarray],
    metadata: list[dict[str, object]],
) -> Path | None:
    if not windows:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / f"windows_{shard_index:04d}.npz"
    np.savez_compressed(
        shard_path,
        X=np.concatenate(windows, axis=0),
        y=np.concatenate(labels, axis=0),
        metadata=np.asarray(metadata, dtype=object),
    )
    return shard_path


def csv_dtype_map(dataset_path: Path) -> dict[str, str]:
    header = pd.read_csv(dataset_path, nrows=0)
    dtypes: dict[str, str] = {}
    for column in header.columns:
        if column.startswith(("x_", "y_", "z_")):
            dtypes[column] = "float32"
        elif column == DEFAULT_FRAME_COLUMN:
            dtypes[column] = "int32"
    return dtypes


def iter_input_frames(dataset_path: Path, chunksize: int, max_rows: int | None) -> Iterable[pd.DataFrame]:
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        yield from pd.read_csv(dataset_path, chunksize=chunksize, nrows=max_rows, dtype=csv_dtype_map(dataset_path))
        return
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(dataset_path)
        if max_rows is not None:
            frame = frame.head(max_rows)
        yield frame
        return
    raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")


def print_progress(summary: PreprocessSummary) -> None:
    print(
        "Progress: "
        f"{summary.frames_seen:,} frames, "
        f"{summary.groups_seen:,} clips, "
        f"{summary.windows_written:,} windows, "
        f"{summary.shards_written:,} shards"
    )


def process_dataset(
    dataset_path: Path,
    output_dir: Path,
    window_size: int,
    stride: int,
    max_rows: int | None,
    chunksize: int,
    label_column: str,
    group_column: str,
    frame_column: str,
    shard_size: int,
    progress_every_chunks: int,
) -> PreprocessSummary:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    summary = PreprocessSummary()
    landmark_columns: list[str] | None = None
    label_map: dict[str, int] | None = None
    pending = pd.DataFrame()
    shard_windows: list[np.ndarray] = []
    shard_labels: list[np.ndarray] = []
    shard_metadata: list[dict[str, object]] = []
    shard_index = 0
    chunk_index = 0

    for chunk in iter_input_frames(dataset_path, chunksize=chunksize, max_rows=max_rows):
        chunk_index += 1
        summary.frames_seen += len(chunk)
        required = {group_column, frame_column, label_column}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if landmark_columns is None:
            landmark_columns = discover_landmark_columns(chunk.columns)
            label_map = make_label_map(chunk[label_column].dropna())
        else:
            for label in sorted(set(chunk[label_column].astype(str)) - set(label_map)):
                label_map[label] = len(label_map)

        chunk = chunk.rename(
            columns={group_column: DEFAULT_GROUP_COLUMN, frame_column: DEFAULT_FRAME_COLUMN, label_column: DEFAULT_LABEL_COLUMN}
        )
        working = pd.concat([pending, chunk], ignore_index=True) if not pending.empty else chunk
        working = working.sort_values([DEFAULT_GROUP_COLUMN, DEFAULT_FRAME_COLUMN], kind="mergesort")

        last_group = working[DEFAULT_GROUP_COLUMN].iloc[-1]
        ready = working[working[DEFAULT_GROUP_COLUMN] != last_group]
        pending = working[working[DEFAULT_GROUP_COLUMN] == last_group].copy()

        for _group_name, group in ready.groupby(DEFAULT_GROUP_COLUMN, sort=False):
            summary.groups_seen += 1
            windows, labels, metadata = window_group(
                group, landmark_columns, label_map, DEFAULT_LABEL_COLUMN, window_size, stride
            )
            if len(windows) == 0:
                summary.skipped_short_groups += 1
                continue
            shard_windows.append(windows)
            shard_labels.append(labels)
            shard_metadata.extend(metadata)
            summary.windows_written += len(windows)

            if summary.windows_written >= (summary.shards_written + 1) * shard_size:
                shard_path = write_shard(output_dir, shard_index, shard_windows, shard_labels, shard_metadata)
                if shard_path:
                    summary.shards_written += 1
                    shard_index += 1
                    shard_windows.clear()
                    shard_labels.clear()
                    shard_metadata.clear()

        if progress_every_chunks > 0 and chunk_index % progress_every_chunks == 0:
            print_progress(summary)

    if landmark_columns is None or label_map is None:
        raise ValueError("No input rows were loaded")

    if not pending.empty:
        for _group_name, group in pending.groupby(DEFAULT_GROUP_COLUMN, sort=False):
            summary.groups_seen += 1
            windows, labels, metadata = window_group(
                group, landmark_columns, label_map, DEFAULT_LABEL_COLUMN, window_size, stride
            )
            if len(windows) == 0:
                summary.skipped_short_groups += 1
                continue
            shard_windows.append(windows)
            shard_labels.append(labels)
            shard_metadata.extend(metadata)
            summary.windows_written += len(windows)

    shard_path = write_shard(output_dir, shard_index, shard_windows, shard_labels, shard_metadata)
    if shard_path:
        summary.shards_written += 1

    config = PreprocessConfig(
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        window_size=window_size,
        stride=stride,
        max_rows=max_rows,
        chunksize=chunksize,
        shard_size=shard_size,
        label_column=label_column,
        group_column=group_column,
        frame_column=frame_column,
    )
    manifest = {
        "config": asdict(config),
        "summary": asdict(summary),
        "label_map": label_map,
        "feature_columns": landmark_columns,
        "feature_shape": [window_size, len(landmark_columns)],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2 landmark sequence preprocessor")
    parser.add_argument(
        "dataset_path",
        type=Path,
        nargs="?",
        default=DEFAULT_DATASET_PATH,
        help=f"CSV, Parquet, or PQ dataset path (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for NPZ shards")
    parser.add_argument("--window-size", type=int, default=30, help="Frames per sequence window")
    parser.add_argument("--stride", type=int, default=5, help="Frame step between windows")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional development row cap")
    parser.add_argument("--chunksize", type=int, default=50_000, help="CSV rows loaded per chunk")
    parser.add_argument("--shard-size", type=int, default=10_000, help="Approximate windows per output shard")
    parser.add_argument("--progress-every-chunks", type=int, default=1, help="Print progress every N chunks; use 0 to disable")
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--group-column", default=DEFAULT_GROUP_COLUMN)
    parser.add_argument("--frame-column", default=DEFAULT_FRAME_COLUMN)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.dataset_path.exists():
        raise SystemExit(f"Dataset not found: {args.dataset_path}")
    if args.dataset_path.suffix.lower() not in DATASET_EXTENSIONS:
        raise SystemExit("Please provide a CSV, Parquet, or PQ file")

    print(f"Input: {args.dataset_path}")
    print(f"Output: {args.output_dir}")
    if args.max_rows is None:
        print("Mode: full dataset")
    else:
        print(f"Mode: capped sample ({args.max_rows:,} rows)")

    summary = process_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        max_rows=args.max_rows,
        chunksize=args.chunksize,
        label_column=args.label_column,
        group_column=args.group_column,
        frame_column=args.frame_column,
        shard_size=args.shard_size,
        progress_every_chunks=args.progress_every_chunks,
    )
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())