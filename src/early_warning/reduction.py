from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .io_utils import TileMetadata, load_tile_cube


@dataclass
class TileCurveRecord:
    tile_id: str
    row: int
    col: int
    curve_raw: np.ndarray
    valid_fraction: np.ndarray
    missing_fraction: float
    n_timesteps: int



def reduce_tile_cube_to_curve(meta: TileMetadata) -> TileCurveRecord:
    cube = load_tile_cube(meta.cube_path)
    valid_mask = np.isfinite(cube)

    # Spatial reduction step: one NDVI value per timestamp from all pixels.
    curve_raw = np.nanmean(cube, axis=(1, 2))
    valid_fraction = valid_mask.reshape(cube.shape[0], -1).mean(axis=1)
    missing_fraction = float(np.mean(~np.isfinite(curve_raw)))

    # Fill isolated missing values by local interpolation for downstream models.
    if np.any(~np.isfinite(curve_raw)):
        idx = np.arange(curve_raw.size)
        valid = np.isfinite(curve_raw)
        if valid.any():
            curve_raw[~valid] = np.interp(idx[~valid], idx[valid], curve_raw[valid])
        else:
            curve_raw[:] = 0.0

    return TileCurveRecord(
        tile_id=meta.tile_id,
        row=meta.row,
        col=meta.col,
        curve_raw=curve_raw.astype(np.float32),
        valid_fraction=valid_fraction.astype(np.float32),
        missing_fraction=missing_fraction,
        n_timesteps=int(cube.shape[0]),
    )



def build_curve_dataset(catalog: List[TileMetadata]) -> List[TileCurveRecord]:
    return [reduce_tile_cube_to_curve(meta) for meta in catalog]



def resample_curves(records: List[TileCurveRecord], target_length: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    target_x = np.linspace(0.0, 1.0, target_length)

    for rec in records:
        source_x = np.linspace(0.0, 1.0, rec.curve_raw.size)
        out[rec.tile_id] = np.interp(target_x, source_x, rec.curve_raw).astype(np.float32)

    return out
