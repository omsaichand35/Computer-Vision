from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import Affine

from .io_utils import TileMetadata
from .onset_detection import OnsetResult


@dataclass
class SpatialProducts:
    binary_grid: np.ndarray
    onset_grid: np.ndarray
    binary_raster_path: Path | None
    onset_raster_path: Path | None



def build_tile_grids(
    catalog: List[TileMetadata],
    abandonment_tile_ids: List[str],
    onset_results: Dict[str, OnsetResult],
) -> tuple[np.ndarray, np.ndarray]:
    max_row = max(item.row for item in catalog)
    max_col = max(item.col for item in catalog)

    binary = np.zeros((max_row, max_col), dtype=np.uint8)
    onset = np.full((max_row, max_col), fill_value=-1, dtype=np.int16)

    abandonment_set = set(abandonment_tile_ids)
    meta_by_id = {m.tile_id: m for m in catalog}

    for tile_id, meta in meta_by_id.items():
        r = meta.row - 1
        c = meta.col - 1
        if tile_id in abandonment_set:
            binary[r, c] = 1
            rec = onset_results.get(tile_id)
            if rec and rec.onset_index is not None:
                onset[r, c] = int(rec.onset_index)

    return binary, onset



def write_tile_level_geotiff(
    array: np.ndarray,
    catalog: List[TileMetadata],
    output_path: Path,
    nodata: int,
) -> Path:
    sample = next((m for m in catalog if m.transform_tuple is not None), None)
    if sample is None:
        raise RuntimeError("No geospatial metadata found to write GeoTIFF outputs.")

    transform = Affine(*sample.transform_tuple)
    crs = sample.crs_wkt

    pixel_width = transform.a
    pixel_height = abs(transform.e)

    # Build a coarse tile raster (one pixel per tile) using top-left tile origin.
    top_left = min(catalog, key=lambda m: (m.row, m.col))
    top_transform = Affine(*top_left.transform_tuple) if top_left.transform_tuple is not None else transform
    coarse_transform = Affine(
        pixel_width * 512,
        0.0,
        top_transform.c,
        0.0,
        -pixel_height * 512,
        top_transform.f,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=coarse_transform,
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)

    return output_path



def generate_spatial_products(
    catalog: List[TileMetadata],
    abandonment_tile_ids: List[str],
    onset_results: Dict[str, OnsetResult],
    maps_dir: Path,
) -> SpatialProducts:
    binary_grid, onset_grid = build_tile_grids(catalog, abandonment_tile_ids, onset_results)

    maps_dir.mkdir(parents=True, exist_ok=True)
    np.save(maps_dir / "abandonment_binary_grid.npy", binary_grid)
    np.save(maps_dir / "abandonment_onset_grid.npy", onset_grid)

    # Also export quick-look PNGs so users can inspect maps without loading numpy arrays.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    im0 = axes[0].imshow(binary_grid, cmap="YlOrRd", vmin=0, vmax=1)
    axes[0].set_title("Binary abandonment map")
    axes[0].set_xlabel("Tile column")
    axes[0].set_ylabel("Tile row")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    masked_onset = np.ma.masked_where(onset_grid < 0, onset_grid)
    im1 = axes[1].imshow(masked_onset, cmap="viridis")
    axes[1].set_title("Abandonment start index map")
    axes[1].set_xlabel("Tile column")
    axes[1].set_ylabel("Tile row")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(maps_dir / "abandonment_maps_preview.png", dpi=180)
    plt.close(fig)

    has_geo = any(item.transform_tuple is not None and item.crs_wkt is not None for item in catalog)
    if not has_geo:
        return SpatialProducts(
            binary_grid=binary_grid,
            onset_grid=onset_grid,
            binary_raster_path=None,
            onset_raster_path=None,
        )

    binary_raster = write_tile_level_geotiff(
        binary_grid.astype(np.uint8),
        catalog,
        maps_dir / "abandonment_binary_map.tif",
        nodata=255,
    )
    onset_raster = write_tile_level_geotiff(
        onset_grid.astype(np.int16),
        catalog,
        maps_dir / "abandonment_onset_map.tif",
        nodata=-1,
    )

    return SpatialProducts(
        binary_grid=binary_grid,
        onset_grid=onset_grid,
        binary_raster_path=binary_raster,
        onset_raster_path=onset_raster,
    )
