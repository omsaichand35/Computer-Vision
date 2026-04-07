from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.stats import linregress


@dataclass
class ClusterSummary:
    cluster_id: int
    n_tiles: int
    mean_trend: float
    mean_amplitude: float
    peak_attenuation: float
    semantic_label: str


@dataclass
class InterpretationResult:
    summaries: List[ClusterSummary]
    label_map: Dict[int, str]
    abandonment_cluster: int



def _cluster_descriptors(curves: np.ndarray) -> tuple[float, float, float]:
    # Curves shape: (n_tiles, T)
    centroid = np.nanmean(curves, axis=0)
    t = np.arange(centroid.size)
    trend, *_ = linregress(t, centroid)
    amplitude = float(np.nanmax(centroid) - np.nanmin(centroid))

    half = centroid.size // 2
    first_peak = float(np.nanmax(centroid[:half])) if half > 0 else float(np.nanmax(centroid))
    second_peak = float(np.nanmax(centroid[half:])) if half > 0 else float(np.nanmax(centroid))
    attenuation = first_peak - second_peak
    return float(trend), amplitude, float(attenuation)



def interpret_clusters(normalized_curves: np.ndarray, labels: np.ndarray) -> InterpretationResult:
    unique_clusters = sorted(np.unique(labels).tolist())
    summaries: List[ClusterSummary] = []
    descriptors = {}

    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        cluster_curves = normalized_curves[mask]
        trend, amplitude, attenuation = _cluster_descriptors(cluster_curves)
        descriptors[cluster_id] = (trend, amplitude, attenuation, int(mask.sum()))

    # Data-driven semantic assignment using cluster-level descriptors.
    abandonment_cluster = sorted(
        unique_clusters,
        key=lambda c: (descriptors[c][0], -descriptors[c][2])
    )[0]

    active_cluster = sorted(
        [c for c in unique_clusters if c != abandonment_cluster],
        key=lambda c: descriptors[c][1],
        reverse=True,
    )[0] if len(unique_clusters) > 1 else abandonment_cluster

    label_map: Dict[int, str] = {}
    for cluster_id in unique_clusters:
        trend, amplitude, attenuation, n_tiles = descriptors[cluster_id]
        if cluster_id == abandonment_cluster:
            semantic = "potential_abandonment"
        elif cluster_id == active_cluster:
            semantic = "active_agriculture"
        else:
            semantic = "stable_perennial_vegetation"

        label_map[cluster_id] = semantic
        summaries.append(
            ClusterSummary(
                cluster_id=cluster_id,
                n_tiles=n_tiles,
                mean_trend=trend,
                mean_amplitude=amplitude,
                peak_attenuation=attenuation,
                semantic_label=semantic,
            )
        )

    return InterpretationResult(
        summaries=summaries,
        label_map=label_map,
        abandonment_cluster=abandonment_cluster,
    )
