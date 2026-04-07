from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
try:
    import ruptures as rpt
except Exception:  # pragma: no cover - optional dependency guard
    rpt = None

from .config import OnsetConfig


@dataclass
class OnsetResult:
    tile_id: str
    onset_index: int | None
    confidence: float
    slope_method_index: int | None
    cp_method_index: int | None



def rolling_slopes(curve: np.ndarray, window: int) -> np.ndarray:
    if curve.size < window:
        return np.array([], dtype=np.float32)
    x = np.arange(window)
    slopes = []
    for i in range(curve.size - window + 1):
        y = curve[i : i + window]
        slope = np.polyfit(x, y, 1)[0]
        slopes.append(slope)
    return np.array(slopes, dtype=np.float32)



def detect_by_rolling_slope(curve: np.ndarray, cfg: OnsetConfig) -> int | None:
    slopes = rolling_slopes(curve, cfg.rolling_window)
    if slopes.size < cfg.persistence_windows:
        return None

    mean_slope = float(np.mean(slopes))
    std_slope = float(np.std(slopes))
    if std_slope < 1e-8:
        std_slope = 1.0

    z = (slopes - mean_slope) / std_slope
    half = max(curve.size // 2, 1)
    early_peak = float(np.max(curve[:half]))
    early_std = float(np.std(curve[:half]))

    for i in range(slopes.size - cfg.persistence_windows + 1):
        local = z[i : i + cfg.persistence_windows]
        if np.all(local < cfg.slope_z_threshold):
            check_start = i + cfg.rolling_window
            if check_start + cfg.min_post_points >= curve.size:
                continue
            post_peak = float(np.max(curve[check_start:]))
            if post_peak < (early_peak - cfg.peak_drop_std_multiplier * max(early_std, 1e-6)):
                return int(check_start)
    return None



def detect_by_changepoint(curve: np.ndarray, cfg: OnsetConfig) -> int | None:
    if rpt is None:
        # Fallback: strongest downward shift in first differences.
        diffs = np.diff(curve)
        if diffs.size < 4:
            return None
        drop_idx = int(np.argmin(diffs))
        return drop_idx + 1

    signal = curve.reshape(-1, 1)
    algo = rpt.Pelt(model="rbf").fit(signal)
    cps = algo.predict(pen=cfg.ruptures_penalty)
    cps = [cp for cp in cps if cp < curve.size]
    if not cps:
        return None

    # Keep only breakpoints that are followed by lower mean NDVI.
    valid = []
    for cp in cps:
        if cp <= 2 or cp >= curve.size - 2:
            continue
        before = float(np.mean(curve[:cp]))
        after = float(np.mean(curve[cp:]))
        if after < before:
            valid.append(cp)

    return int(min(valid)) if valid else None



def fuse_onset_indices(
    slope_index: int | None,
    cp_index: int | None,
    tolerance: int,
) -> tuple[int | None, float]:
    if slope_index is None and cp_index is None:
        return None, 0.0
    if slope_index is not None and cp_index is not None:
        delta = abs(slope_index - cp_index)
        if delta <= tolerance:
            onset = int(min(slope_index, cp_index))
            confidence = max(0.0, 1.0 - (delta / (tolerance + 1)))
            return onset, min(confidence + 0.3, 1.0)
        onset = int(min(slope_index, cp_index))
        return onset, 0.6

    onset = slope_index if slope_index is not None else cp_index
    return int(onset), 0.45



def detect_onset_for_tiles(
    tile_curves: Dict[str, np.ndarray],
    abandonment_tile_ids: List[str],
    cfg: OnsetConfig,
) -> Dict[str, OnsetResult]:
    results: Dict[str, OnsetResult] = {}
    for tile_id in abandonment_tile_ids:
        curve = tile_curves[tile_id]
        slope_idx = detect_by_rolling_slope(curve, cfg)
        cp_idx = detect_by_changepoint(curve, cfg)
        onset, conf = fuse_onset_indices(slope_idx, cp_idx, cfg.agreement_tolerance)
        results[tile_id] = OnsetResult(
            tile_id=tile_id,
            onset_index=onset,
            confidence=conf,
            slope_method_index=slope_idx,
            cp_method_index=cp_idx,
        )
    return results
