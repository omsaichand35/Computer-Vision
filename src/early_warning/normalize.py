from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class NormalizationResult:
    tile_ids: List[str]
    raw_matrix: np.ndarray
    normalized_matrix: np.ndarray
    pca_matrix: np.ndarray | None
    pca_model: PCA | None



def _zscore_per_curve(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (matrix - mean) / std



def _minmax_per_curve(matrix: np.ndarray) -> np.ndarray:
    min_v = matrix.min(axis=1, keepdims=True)
    max_v = matrix.max(axis=1, keepdims=True)
    span = np.where((max_v - min_v) < 1e-8, 1.0, max_v - min_v)
    return (matrix - min_v) / span



def normalize_curves(
    curves: Dict[str, np.ndarray],
    mode: str = "zscore",
    use_pca: bool = True,
    pca_variance_ratio: float = 0.98,
) -> NormalizationResult:
    tile_ids = sorted(curves)
    raw = np.vstack([curves[tile_id] for tile_id in tile_ids]).astype(np.float32)

    if mode == "zscore":
        normalized = _zscore_per_curve(raw)
    elif mode == "minmax":
        normalized = _minmax_per_curve(raw)
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    pca_matrix = None
    pca_model = None
    if use_pca:
        pca = PCA(n_components=pca_variance_ratio, svd_solver="full", random_state=42)
        pca_matrix = pca.fit_transform(normalized)
        pca_model = pca

    return NormalizationResult(
        tile_ids=tile_ids,
        raw_matrix=raw,
        normalized_matrix=normalized.astype(np.float32),
        pca_matrix=pca_matrix.astype(np.float32) if pca_matrix is not None else None,
        pca_model=pca_model,
    )
