from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence


@dataclass
class ClusteringConfig:
    k_values: Sequence[int] = (3, 4)
    n_init: int = 25
    random_state: int = 42
    max_iter: int = 400
    use_pca: bool = True
    pca_variance_ratio: float = 0.98
    use_gpu: bool = True


@dataclass
class OnsetConfig:
    rolling_window: int = 12
    persistence_windows: int = 3
    min_post_points: int = 12
    slope_z_threshold: float = -0.35
    peak_drop_std_multiplier: float = 0.5
    ruptures_penalty: float = 4.0
    agreement_tolerance: int = 2


@dataclass
class WeakLabelConfig:
    confidence_threshold: float = 0.45
    random_state: int = 42
    test_size: float = 0.25
    lstm_epochs: int = 30
    lstm_lr: float = 1e-3
    batch_size: int = 32
    window_length: int = 64
    window_stride: int = 4
    use_gpu: bool = True
    # Improved model settings
    use_smote: bool = True
    use_focal_loss: bool = True
    cnn_epochs: int = 50
    cnn_lr: float = 1e-3
    gb_n_estimators: int = 200
    gb_max_depth: int = 6
    gb_learning_rate: float = 0.1
    use_advanced_features: bool = True


@dataclass
class PipelineConfig:
    project_root: Path
    time_series_dir: Path = field(init=False)
    tiles_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)
    maps_dir: Path = field(init=False)
    random_state: int = 42
    normalization: str = "zscore"
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    onset: OnsetConfig = field(default_factory=OnsetConfig)
    weak: WeakLabelConfig = field(default_factory=WeakLabelConfig)

    def __post_init__(self) -> None:
        remote_root = self.project_root / "Remote Sensing"
        self.time_series_dir = remote_root / "time_series"
        self.tiles_dir = remote_root / "tiles"
        self.output_dir = remote_root / "outputs" / "early_warning"
        self.figures_dir = self.output_dir / "figures"
        self.tables_dir = self.output_dir / "tables"
        self.maps_dir = self.output_dir / "maps"

    @property
    def required_dirs(self) -> List[Path]:
        return [self.output_dir, self.figures_dir, self.tables_dir, self.maps_dir]


DEFAULT_TILE_SHAPE = (512, 512)
