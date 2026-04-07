"""Improved weak supervision with multi-signal heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy import signal

from .onset_detection import OnsetResult


@dataclass
class WeakLabels:
    tile_ids: List[str]
    group_ids: np.ndarray
    X_seq: np.ndarray
    X_feat: np.ndarray
    y: np.ndarray
    confidence_relaxed: bool
    label_confidence: np.ndarray  # Confidence score for each label


def curve_to_features_v2(curve: np.ndarray) -> np.ndarray:
    """Extract enhanced features from a curve."""
    n = len(curve)
    t = np.arange(n, dtype=np.float32)

    # Basic features
    slope = np.polyfit(t, curve, 1)[0] if n > 1 else 0
    amp = float(np.max(curve) - np.min(curve))
    mean_val = float(np.mean(curve))
    std_val = float(np.std(curve))
    p10 = float(np.percentile(curve, 10))
    p90 = float(np.percentile(curve, 90))
    seasonal_range = p90 - p10

    # Trend features
    diff = np.diff(curve)
    mean_diff = np.mean(diff)
    max_diff = np.max(np.abs(diff))

    # Curvature (second derivative)
    if n > 2:
        diff2 = np.diff(diff)
        curvature = np.std(diff2)
    else:
        curvature = 0

    # Peak characteristics
    peaks, properties = signal.find_peaks(curve, distance=max(3, n//6))
    n_peaks = len(peaks)
    peak_heights = curve[peaks] if len(peaks) > 0 else np.array([mean_val])

    # Trough characteristics
    troughs, _ = signal.find_peaks(-curve, distance=max(3, n//6))
    n_troughs = len(troughs)

    # Decline ratio: how much of the curve is in decline
    declining_points = np.sum(diff < 0)
    decline_ratio = declining_points / (len(diff) + 1e-8)

    # Recovery index
    min_idx = np.argmin(curve)
    max_idx = np.argmax(curve)
    if max_idx < n - 1 and min_idx > 0:
        # Check if there's recovery after a drop
        post_min_recovery = (curve[-1] - curve[min_idx]) / (amp + 1e-8)
    else:
        post_min_recovery = 0

    # Early vs late decline
    half = n // 2
    first_half_slope = np.polyfit(t[:half], curve[:half], 1)[0] if half > 1 else 0
    second_half_slope = np.polyfit(t[half:], curve[half:], 1)[0] if n - half > 1 else 0
    slope_change = second_half_slope - first_half_slope

    return np.array([
        mean_val, std_val, amp, slope, seasonal_range,
        mean_diff, max_diff, curvature,
        n_peaks / (n / 12), n_troughs / (n / 12),
        decline_ratio, post_min_recovery, slope_change,
        float(np.median(curve)),
        float(np.percentile(curve, 25)),
        float(np.percentile(curve, 75)),
    ], dtype=np.float32)


def labeling_function_1_decline(curve: np.ndarray, threshold: float = -0.0003) -> Tuple[int, float]:
    """LF1: Strong negative trend indicates abandonment."""
    t = np.arange(len(curve), dtype=np.float32)
    slope = np.polyfit(t, curve, 1)[0] if len(curve) > 1 else 0

    if slope < threshold * 2:
        return 1, 0.8  # Strong decline
    elif slope < threshold:
        return 1, 0.6  # Moderate decline
    elif slope > 0.0005:
        return 0, 0.7  # Strong growth
    else:
        return 0, 0.4  # Stable or weak trend


def labeling_function_2_amplitude(curve: np.ndarray, low_amp: float = 0.08) -> Tuple[int, float]:
    """LF2: Low amplitude (loss of seasonality) indicates abandonment."""
    amp = np.max(curve) - np.min(curve)

    if amp < low_amp * 0.5:
        return 1, 0.7  # Very low amplitude
    elif amp < low_amp:
        return 1, 0.5  # Low amplitude
    elif amp > 0.3:
        return 0, 0.6  # High amplitude (healthy vegetation)
    else:
        return 0, 0.4


def labeling_function_3_decline_ratio(curve: np.ndarray) -> Tuple[int, float]:
    """LF3: High proportion of declining periods."""
    diff = np.diff(curve)
    decline_ratio = np.sum(diff < 0) / (len(diff) + 1e-8)

    if decline_ratio > 0.7:
        return 1, 0.75  # Mostly declining
    elif decline_ratio > 0.6:
        return 1, 0.55
    elif decline_ratio < 0.3:
        return 0, 0.65  # Mostly growing
    else:
        return 0, 0.4


def labeling_function_4_peak_attenuation(curve: np.ndarray) -> Tuple[int, float]:
    """LF4: Decreasing peak heights over time."""
    peaks, _ = signal.find_peaks(curve, distance=max(3, len(curve)//6))

    if len(peaks) >= 2:
        peak_heights = curve[peaks]
        t_peaks = np.arange(len(peak_heights), dtype=np.float32)
        peak_trend = np.polyfit(t_peaks, peak_heights, 1)[0]

        if peak_trend < -0.02:
            return 1, 0.8  # Strong peak attenuation
        elif peak_trend < -0.01:
            return 1, 0.6
        elif peak_trend > 0.02:
            return 0, 0.7  # Increasing peaks
        else:
            return 0, 0.4
    else:
        return 0, 0.3  # Not enough peaks to determine


def labeling_function_5_sudden_drop(curve: np.ndarray) -> Tuple[int, float]:
    """LF5: Sudden drop followed by sustained low values."""
    diff = np.diff(curve)
    std_diff = np.std(diff)

    if std_diff < 1e-6:
        return 0, 0.3

    # Find significant drops
    drop_threshold = -2.0 * std_diff
    drop_indices = np.where(diff < drop_threshold)[0]

    if len(drop_indices) > 0:
        # Check if values stay low after drop
        for drop_idx in drop_indices:
            post_drop = curve[drop_idx + 1:]
            pre_drop = curve[:drop_idx + 1]

            if len(post_drop) > 3:
                post_mean = np.mean(post_drop)
                pre_mean = np.mean(pre_drop)
                drop_magnitude = (pre_mean - post_mean) / (np.std(curve) + 1e-8)

                if drop_magnitude > 1.5:
                    return 1, 0.85  # Significant sudden drop
                elif drop_magnitude > 1.0:
                    return 1, 0.6

    return 0, 0.4


def labeling_function_6_onset_confidence(onset_result: OnsetResult | None) -> Tuple[int, float]:
    """LF6: Use onset detection confidence."""
    if onset_result is None:
        return 0, 0.3

    if onset_result.confidence > 0.7:
        return 1, onset_result.confidence
    elif onset_result.confidence > 0.5:
        return 1, onset_result.confidence * 0.8
    elif onset_result.confidence < 0.3:
        return 0, 1 - onset_result.confidence
    else:
        return 0, 0.4


def aggregate_labels(votes: List[Tuple[int, float]], method: str = "weighted") -> Tuple[int, float]:
    """Aggregate multiple labeling function votes."""
    if not votes:
        return 0, 0.0

    # Weighted voting
    label_weights = {0: 0.0, 1: 0.0}

    for label, confidence in votes:
        label_weights[label] += confidence

    total_weight = label_weights[0] + label_weights[1]
    if total_weight < 1e-8:
        return 0, 0.0

    # Final label
    final_label = 1 if label_weights[1] > label_weights[0] else 0
    confidence = abs(label_weights[1] - label_weights[0]) / total_weight

    return final_label, min(confidence * 2, 1.0)  # Scale confidence


def build_weak_labels_v2(
    tile_ids: List[str],
    curves: Dict[str, np.ndarray],
    cluster_map: Dict[str, int],
    abandonment_cluster: int,
    onset_results: Dict[str, OnsetResult],
    confidence_threshold: float = 0.5,
    window_length: int = 64,
    window_stride: int = 4,
) -> WeakLabels:
    """Build weak labels using multi-signal heuristics."""

    def collect_samples() -> tuple:
        x_seq = []
        x_feat = []
        y = []
        confidences = []
        out_tile_ids = []
        group_ids = []

        for tile_id in tile_ids:
            curve = curves[tile_id]
            cluster_id = cluster_map[tile_id]
            onset = onset_results.get(tile_id)

            # Get votes from all labeling functions
            votes = []

            # LF1: Decline trend
            votes.append(labeling_function_1_decline(curve))

            # LF2: Amplitude
            votes.append(labeling_function_2_amplitude(curve))

            # LF3: Decline ratio
            votes.append(labeling_function_3_decline_ratio(curve))

            # LF4: Peak attenuation
            votes.append(labeling_function_4_peak_attenuation(curve))

            # LF5: Sudden drop
            votes.append(labeling_function_5_sudden_drop(curve))

            # LF6: Onset confidence
            votes.append(labeling_function_6_onset_confidence(onset))

            # Cluster-based prior (soft)
            is_abandonment_cluster = cluster_id == abandonment_cluster
            if is_abandonment_cluster:
                votes.append((1, 0.5))  # Soft prior from clustering
            else:
                votes.append((0, 0.5))

            # Aggregate votes
            label, confidence = aggregate_labels(votes)

            # Filter by confidence threshold
            if confidence < confidence_threshold:
                continue

            # Generate windowed samples
            for start in range(0, max(1, len(curve) - window_length), window_stride):
                end = start + window_length
                segment = curve[start:end]

                if len(segment) < window_length:
                    segment = np.pad(segment, (0, window_length - len(segment)), mode='edge')

                # Normalize segment
                seg_mean = np.mean(segment)
                seg_std = np.std(segment) + 1e-8
                segment_norm = (segment - seg_mean) / seg_std

                out_tile_ids.append(tile_id)
                group_ids.append(tile_id)
                x_seq.append(segment_norm.astype(np.float32))
                x_feat.append(curve_to_features_v2(segment_norm))
                y.append(label)
                confidences.append(confidence)

        return out_tile_ids, group_ids, x_seq, x_feat, y, confidences

    out_tile_ids, group_ids, x_seq, x_feat, y, confidences = collect_samples()
    confidence_relaxed = False

    # Fallback: if too few samples, relax threshold
    if len(y) < 100 or len(np.unique(y)) < 2:
        confidence_threshold = 0.3
        out_tile_ids, group_ids, x_seq, x_feat, y, confidences = collect_samples()
        confidence_relaxed = True

    # Second fallback: if still single class, use cluster labels directly
    if len(y) == 0 or len(np.unique(y)) < 2:
        out_tile_ids = []
        group_ids = []
        x_seq = []
        x_feat = []
        y = []
        confidences = []

        for tile_id in tile_ids:
            curve = curves[tile_id]
            cluster_id = cluster_map[tile_id]
            label = 1 if cluster_id == abandonment_cluster else 0

            for start in range(0, max(1, len(curve) - window_length), window_stride):
                end = start + window_length
                segment = curve[start:end]

                if len(segment) < window_length:
                    segment = np.pad(segment, (0, window_length - len(segment)), mode='edge')

                seg_mean = np.mean(segment)
                seg_std = np.std(segment) + 1e-8
                segment_norm = (segment - seg_mean) / seg_std

                out_tile_ids.append(tile_id)
                group_ids.append(tile_id)
                x_seq.append(segment_norm.astype(np.float32))
                x_feat.append(curve_to_features_v2(segment_norm))
                y.append(label)
                confidences.append(0.5)

        confidence_relaxed = True

    if len(out_tile_ids) == 0:
        raise RuntimeError("No weak-label samples available after all fallbacks.")

    return WeakLabels(
        tile_ids=out_tile_ids,
        group_ids=np.array(group_ids),
        X_seq=np.vstack(x_seq).astype(np.float32),
        X_feat=np.vstack(x_feat).astype(np.float32),
        y=np.array(y, dtype=np.int64),
        confidence_relaxed=confidence_relaxed,
        label_confidence=np.array(confidences, dtype=np.float32),
    )
