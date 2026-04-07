from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

try:
    from cuml.cluster import KMeans as CuMLKMeans
except Exception:  # pragma: no cover - optional dependency guard
    CuMLKMeans = None


@dataclass
class ClusterRun:
    k: int
    labels: np.ndarray
    inertia: float
    silhouette: float
    davies_bouldin: float
    backend: str


@dataclass
class ClusterSelection:
    best: ClusterRun
    candidates: List[ClusterRun]



def run_kmeans_grid(
    features: np.ndarray,
    k_values: Iterable[int],
    random_state: int = 42,
    n_init: int = 25,
    max_iter: int = 400,
    use_gpu: bool = True,
) -> ClusterSelection:
    runs: List[ClusterRun] = []

    for k in k_values:
        backend = "cpu"
        if use_gpu and CuMLKMeans is not None:
            backend = "gpu"
            model = CuMLKMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
            )
            labels = model.fit_predict(features)
            labels = np.asarray(labels)
            inertia = float(model.inertia_)
        else:
            model = KMeans(
                n_clusters=k,
                random_state=random_state,
                n_init=n_init,
                max_iter=max_iter,
            )
            labels = model.fit_predict(features)
            inertia = float(model.inertia_)

        if len(np.unique(labels)) <= 1:
            silhouette = -1.0
            davies = np.inf
        else:
            silhouette = float(silhouette_score(features, labels))
            davies = float(davies_bouldin_score(features, labels))

        runs.append(
            ClusterRun(
                k=k,
                labels=labels.astype(np.int32),
                inertia=inertia,
                silhouette=silhouette,
                davies_bouldin=davies,
                backend=backend,
            )
        )

    # Favor high silhouette, then lower DB index.
    best = sorted(runs, key=lambda r: (-r.silhouette, r.davies_bouldin))[0]
    return ClusterSelection(best=best, candidates=runs)



def labels_to_tile_map(tile_ids: List[str], labels: np.ndarray) -> Dict[str, int]:
    return {tile_id: int(label) for tile_id, label in zip(tile_ids, labels)}
