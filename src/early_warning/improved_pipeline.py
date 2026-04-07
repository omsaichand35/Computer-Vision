"""Improved pipeline with advanced features and better models."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .clustering import labels_to_tile_map, run_kmeans_grid
from .interpretation import interpret_clusters
from .io_utils import attach_geospatial_metadata, ensure_dirs, load_tile_catalog, parse_timestamp_folders
from .normalize import normalize_curves
from .onset_detection import detect_onset_for_tiles
from .reduction import build_curve_dataset, resample_curves
from .spatial_mapping import generate_spatial_products
from .visualization import (
    plot_cluster_curves,
    plot_onset_timeline,
    plot_spatial_heatmaps,
    plot_before_after_comparison,
)
from .feature_extraction import extract_all_features, get_feature_names
from .weak_supervision_v2 import build_weak_labels_v2
from .models_improved import (
    train_gradient_boosting,
    train_xgboost,
    train_lightgbm,
    train_improved_cnn,
    train_ensemble,
    ModelResult,
)


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {sec}s"
    if minutes > 0:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _date_from_index(index: int | None, dates: List[pd.Timestamp]) -> str | None:
    if index is None:
        return None
    if not dates:
        return f"t_{index}"
    index = max(0, min(index, len(dates) - 1))
    return str(dates[index].date())


def run_improved_pipeline(config: PipelineConfig) -> Dict[str, str]:
    """Run the improved pipeline with better feature extraction and models."""

    total_stages = 12
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

    # Stage 1: Prepare directories
    stage_start = time.perf_counter()
    ensure_dirs(config.required_dirs)
    finish_stage("prepare_output_dirs", stage_start)

    # Stage 2: Load tile catalog
    stage_start = time.perf_counter()
    catalog = load_tile_catalog(config.time_series_dir)
    if not catalog:
        raise RuntimeError(f"No tile time-series found in {config.time_series_dir}")
    finish_stage("load_tile_catalog", stage_start)

    # Stage 3: Load metadata
    stage_start = time.perf_counter()
    attach_geospatial_metadata(catalog, config.tiles_dir)
    folder_names, parsed_dates = parse_timestamp_folders(config.tiles_dir)
    date_index = [pd.Timestamp(d) for d in parsed_dates] if parsed_dates else []
    finish_stage("load_metadata_and_timestamps", stage_start)

    # Stage 4: Build and reduce curves
    stage_start = time.perf_counter()
    records = build_curve_dataset(catalog)
    target_length = max(rec.n_timesteps for rec in records)
    reduced = resample_curves(records, target_length=target_length)
    finish_stage("reduce_curves", stage_start)

    # Stage 5: Normalize
    stage_start = time.perf_counter()
    norm = normalize_curves(
        reduced,
        mode=config.normalization,
        use_pca=config.clustering.use_pca,
        pca_variance_ratio=config.clustering.pca_variance_ratio,
    )
    finish_stage("normalize_and_pca", stage_start)

    # Stage 6: Clustering
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

    # Stage 7: Cluster interpretation
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

    # Stage 8: Onset detection
    stage_start = time.perf_counter()
    onset_results = detect_onset_for_tiles(
        tile_curves={tile_id: reduced[tile_id] for tile_id in norm.tile_ids},
        abandonment_tile_ids=abandonment_tile_ids,
        cfg=config.onset,
    )
    finish_stage("onset_detection", stage_start)

    # Stage 9: Spatial outputs
    stage_start = time.perf_counter()
    spatial = generate_spatial_products(
        catalog=catalog,
        abandonment_tile_ids=abandonment_tile_ids,
        onset_results=onset_results,
        maps_dir=config.maps_dir,
    )
    finish_stage("spatial_outputs", stage_start)

    # Stage 10: Visualizations
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

    # Stage 11: Build weak labels with improved method
    stage_start = time.perf_counter()
    weak = build_weak_labels_v2(
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
    print(f"\nWeak label distribution: {dict(zip(class_vals, class_counts))}")
    print(f"Label confidence - mean: {np.mean(weak.label_confidence):.3f}, std: {np.std(weak.label_confidence):.3f}")
    finish_stage("weak_label_build", stage_start)

    # Stage 12: Train improved models
    stage_start = time.perf_counter()

    # Extract advanced features
    print("\nExtracting advanced features...")
    X_advanced = np.vstack([extract_all_features(weak.X_seq[i]) for i in range(len(weak.X_seq))])
    print(f"Advanced feature shape: {X_advanced.shape}")
    print(f"Feature names: {get_feature_names()}")

    # Train multiple models and select best
    print("\nTraining Gradient Boosting...")
    gb_result = train_gradient_boosting(
        X=X_advanced,
        y=weak.y,
        random_state=config.weak.random_state,
        test_size=config.weak.test_size,
        use_smote=config.weak.use_smote,
        n_estimators=config.weak.gb_n_estimators,
        max_depth=config.weak.gb_max_depth,
        learning_rate=config.weak.gb_learning_rate,
    )

    print(f"\nGB Results - F1 Macro: {gb_result.f1_macro:.3f}, F1 Class 1: {gb_result.f1_class_1:.3f}")
    print(gb_result.test_report)

    # Try XGBoost if available
    xgb_result = None
    try:
        print("\nTraining XGBoost...")
        xgb_result = train_xgboost(
            X=X_advanced,
            y=weak.y,
            random_state=config.weak.random_state,
            test_size=config.weak.test_size,
            use_smote=config.weak.use_smote,
        )
        print(f"XGB Results - F1 Macro: {xgb_result.f1_macro:.3f}, F1 Class 1: {xgb_result.f1_class_1:.3f}")
    except Exception as e:
        print(f"XGBoost not available or failed: {e}")

    # Try LightGBM if available
    lgbm_result = None
    try:
        print("\nTraining LightGBM...")
        lgbm_result = train_lightgbm(
            X=X_advanced,
            y=weak.y,
            random_state=config.weak.random_state,
            test_size=config.weak.test_size,
            use_smote=config.weak.use_smote,
        )
        print(f"LightGBM Results - F1 Macro: {lgbm_result.f1_macro:.3f}, F1 Class 1: {lgbm_result.f1_class_1:.3f}")
    except Exception as e:
        print(f"LightGBM not available or failed: {e}")

    # Train improved CNN
    print("\nTraining Improved CNN with Focal Loss...")
    cnn_result = train_improved_cnn(
        X_seq=weak.X_seq,
        y=weak.y,
        random_state=config.weak.random_state,
        test_size=config.weak.test_size,
        epochs=config.weak.cnn_epochs,
        learning_rate=config.weak.cnn_lr,
        batch_size=config.weak.batch_size,
        use_gpu=config.weak.use_gpu,
        use_focal_loss=config.weak.use_focal_loss,
    )
    print(f"CNN Results - F1 Macro: {cnn_result.f1_macro:.3f}, F1 Class 1: {cnn_result.f1_class_1:.3f}")

    # Select best model based on F1 for class 1 (abandonment detection)
    all_results = [gb_result, xgb_result, lgbm_result, cnn_result]
    all_results = [r for r in all_results if r is not None]
    best_result = max(all_results, key=lambda r: r.f1_class_1)

    print(f"\n=== BEST MODEL: {best_result.model_name} ===")
    print(f"F1 Macro: {best_result.f1_macro:.3f}")
    print(f"F1 Class 1 (Abandonment): {best_result.f1_class_1:.3f}")
    print(f"Accuracy: {best_result.accuracy:.3f}")

    finish_stage("model_training", stage_start)

    # Generate summary report
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

    # Cluster metrics
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

    # Build report
    def _model_report_to_dict(result: ModelResult) -> dict:
        return {
            "model_name": result.model_name,
            "accuracy": result.accuracy,
            "f1_macro": result.f1_macro,
            "f1_weighted": result.f1_weighted,
            "f1_class_1": result.f1_class_1,
            "train_size": result.train_size,
            "test_size": result.test_size,
            "device": result.device,
            "train_report": result.train_report,
            "test_report": result.test_report,
            "train_confusion_matrix": result.train_confusion_matrix,
            "test_confusion_matrix": result.test_confusion_matrix,
        }

    summaries_json = {
        "abandonment_cluster": interpretation.abandonment_cluster,
        "cluster_summaries": [asdict(s) for s in interpretation.summaries],
        "best_model": _model_report_to_dict(best_result),
        "all_models": {r.model_name: _model_report_to_dict(r) for r in all_results},
        "feature_names": get_feature_names(),
        "weak_label_distribution": weak_class_distribution,
        "weak_label_samples": int(weak.y.size),
        "weak_label_unique_tiles": int(len(np.unique(weak.group_ids))),
        "weak_label_confidence_relaxed": bool(weak.confidence_relaxed),
        "weak_label_mean_confidence": float(np.mean(weak.label_confidence)),
        "weak_label_window_length": int(config.weak.window_length),
        "weak_label_window_stride": int(config.weak.window_stride),
        "timestamps_detected": folder_names,
        "binary_raster": str(spatial.binary_raster_path) if spatial.binary_raster_path else None,
        "onset_raster": str(spatial.onset_raster_path) if spatial.onset_raster_path else None,
    }

    report_json = config.output_dir / "run_report_improved.json"
    with report_json.open("w", encoding="utf-8") as f:
        json.dump(summaries_json, f, indent=2)
    finish_stage("export_reports", stage_start)

    total_runtime = time.perf_counter() - run_start
    print(f"\n=== IMPROVED PIPELINE COMPLETED ===")
    print(f"Total runtime: {_format_seconds(total_runtime)}")
    print(f"Best model: {best_result.model_name}")
    print(f"Best F1 Macro: {best_result.f1_macro:.3f}")
    print(f"Best F1 Class 1 (Abandonment): {best_result.f1_class_1:.3f}")

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
        "best_model": best_result.model_name,
        "best_f1_class_1": str(best_result.f1_class_1),
    }
