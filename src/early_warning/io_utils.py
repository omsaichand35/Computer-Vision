from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio


TILE_FILE_RE = re.compile(r"^tile_(\d{2})_(\d{2})\.npy$")
TILE_SUFFIX_RE = re.compile(r"(\d{2})_(\d{2})\.tif$")
DATE_FOLDER_RE = re.compile(r"^(\d{4})(\d{3})$")


@dataclass
class TileMetadata:
    tile_id: str
    row: int
    col: int
    cube_path: Path
    source_tif: Path | None = None
    crs_wkt: str | None = None
    transform_tuple: Tuple[float, float, float, float, float, float] | None = None



def ensure_dirs(paths: List[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)



def parse_tile_id_from_name(name: str) -> tuple[str, int, int] | None:
    match = TILE_FILE_RE.match(name)
    if not match:
        return None
    row, col = int(match.group(1)), int(match.group(2))
    tile_id = f"{row:02d}_{col:02d}"
    return tile_id, row, col



def load_tile_catalog(time_series_dir: Path) -> List[TileMetadata]:
    catalog: List[TileMetadata] = []
    for cube_path in sorted(time_series_dir.glob("tile_*.npy")):
        parsed = parse_tile_id_from_name(cube_path.name)
        if parsed is None:
            continue
        tile_id, row, col = parsed
        catalog.append(TileMetadata(tile_id=tile_id, row=row, col=col, cube_path=cube_path))
    return catalog



def parse_timestamp_folders(tiles_dir: Path) -> tuple[list[str], list[datetime] | None]:
    if not tiles_dir.exists():
        return [], None

    folders = sorted([p.name for p in tiles_dir.iterdir() if p.is_dir()])
    if not folders:
        return [], None

    parsed: List[datetime] = []
    for folder in folders:
        match = DATE_FOLDER_RE.match(folder)
        if not match:
            return folders, None
        year = int(match.group(1))
        doy = int(match.group(2))
        parsed.append(datetime(year, 1, 1) + timedelta(days=doy - 1))
    return folders, parsed



def attach_geospatial_metadata(catalog: List[TileMetadata], tiles_dir: Path) -> None:
    if not tiles_dir.exists():
        return

    date_folders = sorted([p for p in tiles_dir.iterdir() if p.is_dir()])
    if not date_folders:
        return

    first_folder = date_folders[0]
    tif_paths = list(first_folder.glob("*.tif"))
    tif_index: Dict[str, Path] = {}

    for tif_path in tif_paths:
        match = TILE_SUFFIX_RE.search(tif_path.name)
        if match:
            tile_id = f"{int(match.group(1)):02d}_{int(match.group(2)):02d}"
            tif_index[tile_id] = tif_path

    for item in catalog:
        tif_path = tif_index.get(item.tile_id)
        if tif_path is None:
            continue
        item.source_tif = tif_path
        with rasterio.open(tif_path) as src:
            item.crs_wkt = src.crs.to_wkt() if src.crs else None
            item.transform_tuple = tuple(src.transform)[:6]



def load_tile_cube(path: Path) -> np.ndarray:
    data = np.load(path)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array in {path}, got shape {data.shape}")
    return data.astype(np.float32)
