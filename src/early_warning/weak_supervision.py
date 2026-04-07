from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .onset_detection import OnsetResult


@dataclass
class WeakLabels:
    tile_ids: List[str]
    group_ids: np.ndarray
    X_seq: np.ndarray
    X_feat: np.ndarray
    y: np.ndarray
    confidence_relaxed: bool



def curve_to_features(curve: np.ndarray) -> np.ndarray:
    t = np.arange(curve.size, dtype=np.float32)
    slope = np.polyfit(t, curve, 1)[0]
    amp = float(np.max(curve) - np.min(curve))
    mean = float(np.mean(curve))
    std = float(np.std(curve))
    p10 = float(np.percentile(curve, 10))
    p90 = float(np.percentile(curve, 90))
    seasonal_strength = p90 - p10
    return np.array([mean, std, amp, slope, seasonal_strength], dtype=np.float32)


def _zscore_window(window: np.ndarray) -> np.ndarray:
    mu = float(np.mean(window))
    sigma = float(np.std(window))
    if sigma < 1e-6:
        sigma = 1.0
    return ((window - mu) / sigma).astype(np.float32)


def _window_curve(curve: np.ndarray, window_length: int, window_stride: int) -> List[np.ndarray]:
    if curve.size <= window_length:
        return [curve.astype(np.float32)]

    windows: List[np.ndarray] = []
    last_start = curve.size - window_length
    for start in range(0, last_start + 1, max(window_stride, 1)):
        end = start + window_length
        windows.append(curve[start:end].astype(np.float32))

    if (last_start % max(window_stride, 1)) != 0:
        windows.append(curve[-window_length:].astype(np.float32))

    return windows



def build_weak_labels(
    tile_ids: List[str],
    curves: Dict[str, np.ndarray],
    cluster_map: Dict[str, int],
    abandonment_cluster: int,
    onset_results: Dict[str, OnsetResult],
    confidence_threshold: float,
    window_length: int,
    window_stride: int,
) -> WeakLabels:
    def collect_samples(use_confidence_gate: bool) -> tuple[list[str], list[str], list[np.ndarray], list[np.ndarray], list[int]]:
        x_seq = []
        x_feat = []
        y = []
        out_tile_ids = []
        group_ids = []

        for tile_id in tile_ids:
            curve = curves[tile_id]
            cluster_id = cluster_map[tile_id]
            onset = onset_results.get(tile_id)

            is_abandonment = cluster_id == abandonment_cluster
            label = 1 if is_abandonment else 0

            if use_confidence_gate and is_abandonment and onset is not None and onset.confidence < confidence_threshold:
                continue

            for segment in _window_curve(curve, window_length=window_length, window_stride=window_stride):
                normalized_segment = _zscore_window(segment)
                out_tile_ids.append(tile_id)
                group_ids.append(tile_id)
                x_seq.append(normalized_segment)
                x_feat.append(curve_to_features(normalized_segment))
                y.append(label)

        return out_tile_ids, group_ids, x_seq, x_feat, y

    out_tile_ids, group_ids, x_seq, x_feat, y = collect_samples(use_confidence_gate=True)
    confidence_relaxed = False

    # Fallback: relax confidence gate when strict filtering collapses weak labels to one class.
    if y and np.unique(np.array(y, dtype=np.int64)).size < 2:
        out_tile_ids, group_ids, x_seq, x_feat, y = collect_samples(use_confidence_gate=False)
        confidence_relaxed = True

    if not out_tile_ids:
        raise RuntimeError(
            "No weak-label samples available after confidence filtering. "
            "Lower confidence_threshold or inspect onset detection outputs."
        )

    classes = np.unique(np.array(y, dtype=np.int64))
    if classes.size < 2:
        raise RuntimeError(
            "Weak-label generation produced a single class. "
            "Adjust clustering/onset settings or relax confidence filtering."
        )

    return WeakLabels(
        tile_ids=out_tile_ids,
        group_ids=np.array(group_ids),
        X_seq=np.vstack(x_seq).astype(np.float32),
        X_feat=np.vstack(x_feat).astype(np.float32),
        y=np.array(y, dtype=np.int64),
        confidence_relaxed=confidence_relaxed,
    )
