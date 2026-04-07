from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .clustering import labels_to_tile_map, run_kmeans_grid
from .config import PipelineConfig
from .interpretation import interpret_clusters
from .io_utils import attach_geospatial_metadata, ensure_dirs, load_tile_catalog, parse_timestamp_folders
from .models_rf import train_rf_baseline
from .models_seq import train_lstm_baseline
from .normalize import normalize_curves
from .onset_detection import detect_onset_for_tiles
from .reduction import build_curve_dataset, resample_curves
from .spatial_mapping import generate_spatial_products
from .visualization import (
    plot_before_after_comparison,
    plot_cluster_curves,
    plot_onset_timeline,
    plot_spatial_heatmaps,
)
from .weak_supervision import build_weak_labels



def _date_from_index(index: int | None, dates: List[pd.Timestamp]) -> str | None:
    if index is None:
        return None
    if not dates:
        return f"t_{index}"
    index = max(0, min(index, len(dates) - 1))
    return str(dates[index].date())


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    if minutes > 0:
        return f"{minutes}m {sec}s"
    return f"{sec}s"



def run_pipeline(config: PipelineConfig) -> Dict[str, str]:
    total_stages = 14
    completed_stages = 0
    run_start = time.perf_counter()
    stage_times: Dict[str, float] = {}

    def finish_stage(stage_name: str, stage_start: float) -> None:
        nonlocal completed_stages
        end = time.perf_counter()
        duration = end - stage_start
        completed_stages += 1
        stage_times[stage_name] = duration

        elapsed = end - run_start
        avg = elapsed / completed_stages
        remaining = (total_stages - completed_stages) * avg
        print(
            f"[{completed_stages}/{total_stages}] {stage_name} done in {_format_seconds(duration)} "
            f"| elapsed: {_format_seconds(elapsed)} | est remaining: {_format_seconds(remaining)}"
        )

    stage_start = time.perf_counter()
    ensure_dirs(config.required_dirs)
    finish_stage("prepare_output_dirs", stage_start)

    stage_start = time.perf_counter()
    catalog = load_tile_catalog(config.time_series_dir)
    if not catalog:
        raise RuntimeError(f"No tile time-series found in {config.time_series_dir}")
    finish_stage("load_tile_catalog", stage_start)

    stage_start = time.perf_counter()
    attach_geospatial_metadata(catalog, config.tiles_dir)
    folder_names, parsed_dates = parse_timestamp_folders(config.tiles_dir)
    date_index = [pd.Timestamp(d) for d in parsed_dates] if parsed_dates else []
    finish_stage("load_metadata_and_timestamps", stage_start)

    stage_start = time.perf_counter()
    records = build_curve_dataset(catalog)
    target_length = max(rec.n_timesteps for rec in records)
    reduced = resample_curves(records, target_length=target_length)
    finish_stage("reduce_curves", stage_start)

    stage_start = time.perf_counter()
    norm = normalize_curves(
        reduced,
        mode=config.normalization,
        use_pca=config.clustering.use_pca,
        pca_variance_ratio=config.clustering.pca_variance_ratio,
    )
    finish_stage("normalize_and_pca", stage_start)

    stage_start = time.perf_counter()
    cluster_features = norm.pca_matrix if norm.pca_matrix is not None else norm.normalized_matrix
    cluster_sel = run_kmeans_grid(
        cluster_features,
        k_values=config.clustering.k_values,
        random_state=config.clustering.random_state,
        n_init=config.clustering.n_init,
        max_iter=config.clustering.max_iter,
        use_gpu=config.clustering.use_gpu,
    )
    finish_stage("clustering", stage_start)

    stage_start = time.perf_counter()
    best_labels = cluster_sel.best.labels
    cluster_map = labels_to_tile_map(norm.tile_ids, best_labels)
    interpretation = interpret_clusters(norm.normalized_matrix, best_labels)

    abandonment_tile_ids = [
        tile_id
        for tile_id in norm.tile_ids
        if cluster_map[tile_id] == interpretation.abandonment_cluster
    ]
    finish_stage("cluster_interpretation", stage_start)

    stage_start = time.perf_counter()
    onset_results = detect_onset_for_tiles(
        tile_curves={tile_id: reduced[tile_id] for tile_id in norm.tile_ids},
        abandonment_tile_ids=abandonment_tile_ids,
        cfg=config.onset,
    )
    finish_stage("onset_detection", stage_start)

    stage_start = time.perf_counter()
    spatial = generate_spatial_products(
        catalog=catalog,
        abandonment_tile_ids=abandonment_tile_ids,
        onset_results=onset_results,
        maps_dir=config.maps_dir,
    )
    finish_stage("spatial_outputs", stage_start)

    stage_start = time.perf_counter()
    plot_cluster_curves(
        raw_curves=norm.raw_matrix,
        labels=best_labels,
        interpretation=interpretation,
        output_path=config.figures_dir / "cluster_curves.png",
    )
    plot_onset_timeline(onset_results, config.figures_dir / "abandonment_onset_timeline.png")
    plot_spatial_heatmaps(spatial.binary_grid, spatial.onset_grid, config.figures_dir / "spatial_heatmaps.png")
    plot_before_after_comparison(
        tile_curves={tile_id: reduced[tile_id] for tile_id in norm.tile_ids},
        onset_results=onset_results,
        output_path=config.figures_dir / "before_after_ndvi.png",
    )
    finish_stage("visualizations", stage_start)

    stage_start = time.perf_counter()
    weak = build_weak_labels(
        tile_ids=norm.tile_ids,
        curves={tile_id: norm.normalized_matrix[i] for i, tile_id in enumerate(norm.tile_ids)},
        cluster_map=cluster_map,
        abandonment_cluster=interpretation.abandonment_cluster,
        onset_results=onset_results,
        confidence_threshold=config.weak.confidence_threshold,
        window_length=config.weak.window_length,
        window_stride=config.weak.window_stride,
    )

    class_vals, class_counts = np.unique(weak.y, return_counts=True)
    weak_class_distribution = {str(int(k)): int(v) for k, v in zip(class_vals, class_counts)}
    finish_stage("weak_label_build", stage_start)

    stage_start = time.perf_counter()
    rf_result = train_rf_baseline(
        X=weak.X_feat,
        y=weak.y,
        random_state=config.weak.random_state,
        test_size=config.weak.test_size,
        group_ids=weak.group_ids,
    )
    finish_stage("rf_training", stage_start)

    stage_start = time.perf_counter()
    seq_result = train_lstm_baseline(
        X_seq=weak.X_seq,
        y=weak.y,
        random_state=config.weak.random_state,
        test_size=config.weak.test_size,
        epochs=config.weak.lstm_epochs,
        learning_rate=config.weak.lstm_lr,
        batch_size=config.weak.batch_size,
        use_gpu=config.weak.use_gpu,
        group_ids=weak.group_ids,
    )
    finish_stage("lstm_training", stage_start)

    stage_start = time.perf_counter()
    rows = []
    for i, tile_id in enumerate(norm.tile_ids):
        cluster_id = cluster_map[tile_id]
        semantic = interpretation.label_map[cluster_id]
        onset = onset_results.get(tile_id)
        onset_index = onset.onset_index if onset is not None else None
        onset_date = _date_from_index(onset_index, date_index)
        confidence = onset.confidence if onset is not None else 0.0

        rows.append(
            {
                "tile_id": tile_id,
                "cluster_id": cluster_id,
                "semantic_label": semantic,
                "abandonment_cluster": int(cluster_id == interpretation.abandonment_cluster),
                "onset_index": onset_index,
                "onset_date": onset_date,
                "onset_confidence": confidence,
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("tile_id")
    summary_csv = config.tables_dir / "tile_abandonment_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    cluster_metrics = pd.DataFrame(
        [
            {
                "k": run.k,
                "inertia": run.inertia,
                "silhouette": run.silhouette,
                "davies_bouldin": run.davies_bouldin,
                "backend": run.backend,
                "selected": int(run.k == cluster_sel.best.k),
            }
            for run in cluster_sel.candidates
        ]
    )
    cluster_metrics.to_csv(config.tables_dir / "cluster_metrics.csv", index=False)

    summaries_json = {
        "abandonment_cluster": interpretation.abandonment_cluster,
        "cluster_summaries": [asdict(s) for s in interpretation.summaries],
        "rf_report": rf_result.report,
        "rf_train_report": rf_result.train_report,
        "rf_test_report": rf_result.test_report,
        "rf_train_confusion_matrix": rf_result.train_confusion_matrix,
        "rf_test_confusion_matrix": rf_result.test_confusion_matrix,
        "rf_train_size": rf_result.train_size,
        "rf_test_size": rf_result.test_size,
        "lstm_accuracy": seq_result.accuracy,
        "lstm_train_report": seq_result.train_report,
        "lstm_test_report": seq_result.test_report,
        "lstm_train_confusion_matrix": seq_result.train_confusion_matrix,
        "lstm_test_confusion_matrix": seq_result.test_confusion_matrix,
        "lstm_validation_loss_curve": seq_result.validation_loss_curve,
        "lstm_train_size": seq_result.train_size,
        "lstm_test_size": seq_result.test_size,
        "lstm_device": seq_result.device,
        "clustering_backend": cluster_sel.best.backend,
        "weak_label_distribution": weak_class_distribution,
        "weak_label_samples": int(weak.y.size),
        "weak_label_unique_tiles": int(len(np.unique(weak.group_ids))),
        "weak_label_confidence_relaxed": bool(weak.confidence_relaxed),
        "weak_label_window_length": int(config.weak.window_length),
        "weak_label_window_stride": int(config.weak.window_stride),
        "timestamps_detected": folder_names,
        "binary_raster": str(spatial.binary_raster_path) if spatial.binary_raster_path else None,
        "onset_raster": str(spatial.onset_raster_path) if spatial.onset_raster_path else None,
    }

    report_json = config.output_dir / "run_report.json"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(summaries_json, f, indent=2)
    finish_stage("export_reports", stage_start)

    total_runtime = time.perf_counter() - run_start
    print(f"Total pipeline runtime: {_format_seconds(total_runtime)}")

    summaries_json["timing_seconds"] = {
        "total": total_runtime,
        **stage_times,
    }

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(summaries_json, f, indent=2)

    return {
        "summary_csv": str(summary_csv),
        "report_json": str(report_json),
        "figures_dir": str(config.figures_dir),
        "maps_dir": str(config.maps_dir),
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="Early warning agricultural abandonment pipeline")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Workspace project root containing the Remote Sensing folder",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="zscore",
        choices=["zscore", "minmax"],
        help="Per-tile temporal normalization mode",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU acceleration where available (cuML for clustering, CUDA for LSTM).",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU execution for all stages.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig(project_root=args.project_root)
    cfg.normalization = args.normalization
    if args.no_gpu:
        cfg.clustering.use_gpu = False
        cfg.weak.use_gpu = False
    elif args.use_gpu:
        cfg.clustering.use_gpu = True
        cfg.weak.use_gpu = True

    outputs = run_pipeline(cfg)
    print("Pipeline completed successfully.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
