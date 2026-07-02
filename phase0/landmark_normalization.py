from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


LANDMARK_AXES = ("x", "y", "z")
DEFAULT_ANCHOR_INDICES = (1, 6, 168)
DEFAULT_SCALE_INDICES = (33, 263)


@dataclass(frozen=True)
class NormalizationStats:
    anchor: np.ndarray
    scale: float


def _as_landmark_array(landmarks: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(landmarks, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected landmark array shaped (n, 3), got {array.shape}")
    return array


def normalize_landmarks(
    landmarks: np.ndarray | Sequence[Sequence[float]],
    anchor_indices: Iterable[int] = DEFAULT_ANCHOR_INDICES,
    scale_indices: Iterable[int] = DEFAULT_SCALE_INDICES,
) -> tuple[np.ndarray, NormalizationStats]:
    """Center landmarks on a rigid facial anchor and divide by eye width."""

    array = _as_landmark_array(landmarks)
    anchor_points = array[list(anchor_indices)]
    anchor = anchor_points.mean(axis=0)

    scale_points = array[list(scale_indices)]
    if len(scale_points) < 2:
        raise ValueError("Scale estimation needs at least two landmark indices")

    scale = float(np.linalg.norm(scale_points[0] - scale_points[1]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    normalized = (array - anchor) / scale
    return normalized, NormalizationStats(anchor=anchor, scale=scale)


def landmark_block_to_array(frame_row, prefix: str = "") -> np.ndarray:
    """Convert a dataset row containing x/y/z landmark columns into an array."""

    if hasattr(frame_row, "to_dict"):
        values = frame_row.to_dict()
    elif isinstance(frame_row, dict):
        values = frame_row
    else:
        raise TypeError("frame_row must be a mapping-like object")

    x_values = []
    y_values = []
    z_values = []

    index = 0
    while True:
        x_key = f"{prefix}x_{index}"
        y_key = f"{prefix}y_{index}"
        z_key = f"{prefix}z_{index}"
        if x_key not in values or y_key not in values or z_key not in values:
            break
        x_values.append(values[x_key])
        y_values.append(values[y_key])
        z_values.append(values[z_key])
        index += 1

    if not x_values:
        raise ValueError("No landmark columns found with the expected x_N/y_N/z_N pattern")

    return np.column_stack([x_values, y_values, z_values]).astype(np.float32)


def summarize_normalization_drift(
    landmarks: np.ndarray | Sequence[Sequence[float]],
    anchor_indices: Iterable[int] = DEFAULT_ANCHOR_INDICES,
    scale_indices: Iterable[int] = DEFAULT_SCALE_INDICES,
) -> dict[str, float]:
    """Measure how concentrated the normalized face is around the origin."""

    normalized, stats = normalize_landmarks(landmarks, anchor_indices, scale_indices)
    centroid = normalized.mean(axis=0)
    spread = normalized.std(axis=0)

    return {
        "scale": stats.scale,
        "centroid_norm": float(np.linalg.norm(centroid)),
        "spread_mean": float(spread.mean()),
    }
