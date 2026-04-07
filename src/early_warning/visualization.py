"""Visualization utilities for the improved GIS modelling pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .interpretation import InterpretationResult
from .onset_detection import OnsetResult


def robust_limits(arr: np.ndarray, symmetric: bool = False) -> Tuple[float, float]:
    """Compute robust limits for visualization using percentiles."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return -1.0, 1.0

    if symmetric:
        q = float(np.percentile(np.abs(valid), 98))
        q = max(q, 1e-6)
        return -q, q

    lo = float(np.percentile(valid, 2))
    hi = float(np.percentile(valid, 98))
    if abs(hi - lo) < 1e-6:
        hi = lo + 1e-6
    return lo, hi


def plot_cluster_curves(
    raw_curves: np.ndarray,
    labels: np.ndarray,
    interpretation: InterpretationResult,
    output_path: Path,
) -> None:
    """Plot NDVI curves for each cluster with mean and standard deviation."""
    clusters = sorted(np.unique(labels).tolist())
    n_clusters = len(clusters)
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 3.8 * n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]

    for ax, cluster_id in zip(axes, clusters):
        cluster_curves = raw_curves[labels == cluster_id]
        for curve in cluster_curves[:12]:
            ax.plot(curve, color="#8cb4ff", alpha=0.35, linewidth=1.0)
        mean_curve = cluster_curves.mean(axis=0)
        std_curve = cluster_curves.std(axis=0)
        ax.plot(mean_curve, color="#0a3d91", linewidth=2.2)
        ax.fill_between(
            np.arange(mean_curve.size),
            mean_curve - std_curve,
            mean_curve + std_curve,
            color="#6b9cff",
            alpha=0.2,
        )
        semantic = interpretation.label_map[cluster_id]
        ax.set_title(f"Cluster {cluster_id}: {semantic} (n={cluster_curves.shape[0]})")
        ax.set_ylabel("Mean NDVI")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time index")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_onset_timeline(
    onset_results: Dict[str, OnsetResult],
    output_path: Path,
) -> None:
    """Plot timeline of detected abandonment onset indices."""
    rows = []
    for tile_id, rec in onset_results.items():
        if rec.onset_index is None:
            continue
        rows.append((tile_id, rec.onset_index, rec.confidence))

    rows = sorted(rows, key=lambda r: r[1])
    if not rows:
        return

    tile_ids = [r[0] for r in rows]
    onset_idx = [r[1] for r in rows]
    conf = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(np.arange(len(rows)), onset_idx, color="#ff8d6d")
    for b, c in zip(bars, conf):
        b.set_alpha(0.4 + 0.6 * c)

    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(tile_ids, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Estimated abandonment start index")
    ax.set_title("Abandonment onset timeline (abandonment cluster tiles)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_spatial_heatmaps(
    binary_grid: np.ndarray,
    onset_grid: np.ndarray,
    output_path: Path,
) -> None:
    """Plot spatial heatmaps of abandonment predictions."""
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_before_after_comparison(
    tile_curves: Dict[str, np.ndarray],
    onset_results: Dict[str, OnsetResult],
    output_path: Path,
    top_n: int = 8,
) -> None:
    """Plot before/after NDVI comparison for top abandonment detections."""
    ranked = [
        (tile_id, rec) for tile_id, rec in onset_results.items()
        if rec.onset_index is not None
    ]
    ranked.sort(key=lambda x: x[1].confidence, reverse=True)
    ranked = ranked[:top_n]
    if not ranked:
        return

    fig, axes = plt.subplots(len(ranked), 1, figsize=(12, 2.6 * len(ranked)), sharex=True)
    if len(ranked) == 1:
        axes = [axes]

    for ax, (tile_id, rec) in zip(axes, ranked):
        curve = tile_curves[tile_id]
        onset = int(rec.onset_index)
        ax.plot(curve, color="#2b59c3", linewidth=1.8)
        ax.axvline(onset, color="#c44536", linestyle="--", linewidth=1.4)
        ax.fill_between(
            np.arange(curve.size),
            curve.min(),
            curve.max(),
            where=np.arange(curve.size) >= onset,
            color="#f7b6a6",
            alpha=0.3,
        )
        ax.set_title(f"{tile_id} | onset={onset} | confidence={rec.confidence:.2f}")
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Time index")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_model_comparison(
    model_results: Dict[str, Dict],
    output_path: Path,
) -> None:
    """Plot comparison of F1 scores across different models."""
    models = list(model_results.keys())
    f1_macro = [model_results[m]["f1_macro"] for m in models]
    f1_class1 = [model_results[m]["f1_class_1"] for m in models]
    accuracy = [model_results[m]["accuracy"] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, f1_macro, width, label="F1 Macro", color="#3498db")
    bars2 = ax.bar(x, f1_class1, width, label="F1 Class 1 (Abandonment)", color="#e74c3c")
    bars3 = ax.bar(x + width, accuracy, width, label="Accuracy", color="#2ecc71")

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion_matrices(
    model_results: Dict[str, Dict],
    output_path: Path,
) -> None:
    """Plot confusion matrices for all trained models."""
    models = list(model_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(4 * (n_models + 1) // 2, 3.5 * 2))
    axes = axes.flatten()

    for idx, model_name in enumerate(models):
        cm = model_results[model_name].get("test_confusion_matrix", [])
        if not cm:
            continue

        cm_arr = np.array(cm)
        im = axes[idx].imshow(cm_arr, cmap="Blues", aspect="auto")
        axes[idx].set_title(f"{model_name}\nTest Confusion Matrix")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

        # Add annotations
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                axes[idx].text(
                    j, i, str(cm_arr[i, j]),
                    ha="center", va="center", fontsize=12,
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black"
                )

        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    # Hide empty subplots
    for idx in range(len(models), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Confusion Matrices - All Models", fontsize=14, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    output_path: Path,
    top_k: int = 20,
) -> None:
    """Plot top feature importances from tree-based models."""
    if len(importances) != len(feature_names):
        return

    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_k]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_features))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax.barh(y_pos, top_importances, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_k} Feature Importances")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_prediction_probability_distribution(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
) -> None:
    """Plot distribution of prediction probabilities vs actual labels."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram of probabilities by class
    class_0 = y_proba[y_true == 0]
    class_1 = y_proba[y_true == 1]

    axes[0].hist(class_0, bins=30, alpha=0.6, label="Class 0 (Non-abandonment)", color="#3498db", density=True)
    axes[0].hist(class_1, bins=30, alpha=0.6, label="Class 1 (Abandonment)", color="#e74c3c", density=True)
    axes[0].axvline(0.5, color="black", linestyle="--", linewidth=2, label="Threshold")
    axes[0].set_xlabel("Predicted Probability")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Distribution of Prediction Probabilities")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Reliability diagram
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_proba, n_bins=10)

    axes[1].plot(mean_predicted_value, fraction_of_positives, "s-", label="Model", color="#2ecc71")
    axes[1].plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    axes[1].set_xlabel("Mean Predicted Probability")
    axes[1].set_ylabel("Fraction of Positives")
    axes[1].set_title("Reliability Diagram (Calibration)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_temporal_change_heatmaps(
    summary: pd.DataFrame,
    time_series_dir: Path,
    smoothed_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    tile_size: int = 512,
) -> None:
    """Generate temporal change heatmaps with improved metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_row = int(summary["tile_id"].str.slice(0, 2).astype(int).max())
    n_col = int(summary["tile_id"].str.slice(3, 5).astype(int).max())

    # Build metric grids
    full_slope_grid = np.full((n_row, n_col), np.nan, dtype=np.float32)
    recent_slope_grid = np.full((n_row, n_col), np.nan, dtype=np.float32)
    recent_delta_grid = np.full((n_row, n_col), np.nan, dtype=np.float32)
    projected_1y_grid = np.full((n_row, n_col), np.nan, dtype=np.float32)

    for _, row in summary.iterrows():
        tile_id = str(row["tile_id"])
        curve_path = time_series_dir / f"tile_{tile_id}.npy"
        if not curve_path.exists():
            continue

        curve = np.load(curve_path)
        if curve.ndim == 3:
            curve = np.nanmean(curve, axis=(1, 2))
        curve = curve.astype(np.float32)

        if curve.size < 8:
            continue

        r = int(row["tile_id"].split("_")[0]) - 1
        c = int(row["tile_id"].split("_")[1]) - 1

        w = min(32, max(8, curve.size // 4))
        recent = curve[-w:]

        t = np.arange(curve.size, dtype=np.float32)
        full_slope = np.polyfit(t, curve, 1)[0] if curve.size > 1 else 0
        recent_slope = np.polyfit(np.arange(w, dtype=np.float32), recent, 1)[0] if w > 1 else 0

        first_w = curve[:w]
        recent_delta = float(np.mean(recent) - np.mean(first_w))
        projected_1y = recent_slope * 23.0

        full_slope_grid[r, c] = full_slope
        recent_slope_grid[r, c] = recent_slope
        recent_delta_grid[r, c] = recent_delta
        projected_1y_grid[r, c] = projected_1y

    # Save grids
    np.save(output_dir / "temporal_full_slope_grid.npy", full_slope_grid)
    np.save(output_dir / "temporal_recent_slope_grid.npy", recent_slope_grid)
    np.save(output_dir / "temporal_recent_delta_grid.npy", recent_delta_grid)
    np.save(output_dir / "temporal_projected_1y_delta_grid.npy", projected_1y_grid)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axs = axes.flatten()

    binary = summary["abandonment_cluster"].values.reshape(n_row, n_col)
    onset = summary["onset_index"].values.reshape(n_row, n_col)
    onset = np.where(np.isnan(onset), -1, onset)

    im0 = axs[0].imshow(binary, cmap="YlOrRd", vmin=0, vmax=1)
    axs[0].set_title("Binary abandonment map")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(np.ma.masked_where(onset < 0, onset), cmap="viridis")
    axs[1].set_title("Detected onset index")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(full_slope_grid, cmap="RdYlGn", vmin=-0.25, vmax=0.25)
    axs[2].set_title("Full-series slope (NDVI/step)")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    im3 = axs[3].imshow(recent_slope_grid, cmap="RdYlGn", vmin=-0.25, vmax=0.25)
    axs[3].set_title("Recent slope (last window)")
    fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    im4 = axs[4].imshow(recent_delta_grid, cmap="RdYlGn", vmin=-25, vmax=25)
    axs[4].set_title("Recent vs early mean NDVI delta")
    fig.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.04)

    im5 = axs[5].imshow(projected_1y_grid, cmap="RdYlGn", vmin=-10, vmax=10)
    axs[5].set_title("Projected 1-year NDVI delta")
    fig.colorbar(im5, ax=axs[5], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel("Tile column")
        ax.set_ylabel("Tile row")

    fig.suptitle("Spatial Temporal-Change Heatmaps from NDVI Time-Series", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    fig.savefig(output_dir / "temporal_change_heatmaps.png", dpi=220)
    plt.close(fig)

    # High-res maps
    _build_highres_maps(summary, smoothed_dir, raw_dir, output_dir, tile_size)


def _build_highres_maps(
    summary: pd.DataFrame,
    smoothed_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    tile_size: int = 512,
) -> None:
    """Build high-resolution pixel-level maps."""
    n_row = int(summary["tile_id"].str.slice(0, 2).astype(int).max())
    n_col = int(summary["tile_id"].str.slice(3, 5).astype(int).max())

    h_total = n_row * tile_size
    w_total = n_col * tile_size

    binary_hr = np.zeros((h_total, w_total), dtype=np.float32)
    onset_hr = np.full((h_total, w_total), np.nan, dtype=np.float32)
    full_slope_hr = np.full((h_total, w_total), np.nan, dtype=np.float32)
    recent_slope_hr = np.full((h_total, w_total), np.nan, dtype=np.float32)
    recent_delta_hr = np.full((h_total, w_total), np.nan, dtype=np.float32)
    projected_1y_hr = np.full((h_total, w_total), np.nan, dtype=np.float32)

    for _, row in summary.iterrows():
        tile_id = str(row["tile_id"])
        smooth_path = smoothed_dir / f"tile_{tile_id}.npy"
        raw_path = raw_dir / f"tile_{tile_id}.npy"

        if smooth_path.exists():
            cube = np.load(smooth_path).astype(np.float32)
        elif raw_path.exists():
            cube = np.load(raw_path).astype(np.float32)
        else:
            continue

        if cube.ndim != 3:
            continue

        t, h, w = cube.shape
        recent_w = min(32, max(12, t // 6))

        # Pixel-wise slope
        t_arr = np.arange(t, dtype=np.float32)
        t_centered = t_arr - np.mean(t_arr)
        denom = np.sum(t_centered ** 2)
        mean_map = np.mean(cube, axis=0)
        full_slope_px = np.sum(t_centered[:, None, None] * (cube - mean_map), axis=0) / max(denom, 1e-6)

        # Recent slope
        t_recent = np.arange(recent_w, dtype=np.float32)
        t_r_centered = t_recent - np.mean(t_recent)
        r_denom = np.sum(t_r_centered ** 2)
        recent_mean = np.mean(cube[-recent_w:], axis=0)
        recent_slope_px = np.sum(t_r_centered[:, None, None] * (cube[-recent_w:] - recent_mean), axis=0) / max(r_denom, 1e-6)

        recent_delta_px = np.mean(cube[-recent_w:], axis=0) - np.mean(cube[:recent_w], axis=0)
        projected_1y_px = recent_slope_px * 23.0

        rr, cc = int(tile_id.split("_")[0]) - 1, int(tile_id.split("_")[1]) - 1
        r0, r1 = rr * tile_size, rr * tile_size + h
        c0, c1 = cc * tile_size, cc * tile_size + w

        full_slope_hr[r0:r1, c0:c1] = full_slope_px
        recent_slope_hr[r0:r1, c0:c1] = recent_slope_px
        recent_delta_hr[r0:r1, c0:c1] = recent_delta_px
        projected_1y_hr[r0:r1, c0:c1] = projected_1y_px

        if int(row.get("abandonment_cluster", 0)) == 1:
            binary_hr[r0:r1, c0:c1] = 1.0

        onset_val = pd.to_numeric(pd.Series([row.get("onset_index")]), errors="coerce").iloc[0]
        if np.isfinite(onset_val):
            onset_hr[r0:r1, c0:c1] = float(onset_val)

    # Save high-res maps
    np.save(output_dir / "highres_binary_map.npy", binary_hr)
    np.save(output_dir / "highres_onset_map.npy", onset_hr)
    np.save(output_dir / "highres_full_slope_map.npy", full_slope_hr)
    np.save(output_dir / "highres_recent_slope_map.npy", recent_slope_hr)
    np.save(output_dir / "highres_recent_delta_map.npy", recent_delta_hr)
    np.save(output_dir / "highres_projected_1y_delta_map.npy", projected_1y_hr)

    # Plot high-res
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    axs2 = axes2.flatten()

    imh0 = axs2[0].imshow(binary_hr, cmap="YlOrRd", vmin=0, vmax=1)
    axs2[0].set_title("High-res binary abandonment map")
    fig2.colorbar(imh0, ax=axs2[0], fraction=0.046, pad=0.04)

    imh1 = axs2[1].imshow(np.ma.masked_where(~np.isfinite(onset_hr), onset_hr), cmap="viridis")
    axs2[1].set_title("High-res onset index map")
    fig2.colorbar(imh1, ax=axs2[1], fraction=0.046, pad=0.04)

    fs_lo, fs_hi = robust_limits(full_slope_hr, symmetric=True)
    imh2 = axs2[2].imshow(full_slope_hr, cmap="RdYlGn", vmin=fs_lo, vmax=fs_hi)
    axs2[2].set_title("High-res full-series slope")
    fig2.colorbar(imh2, ax=axs2[2], fraction=0.046, pad=0.04)

    rs_lo, rs_hi = robust_limits(recent_slope_hr, symmetric=True)
    imh3 = axs2[3].imshow(recent_slope_hr, cmap="RdYlGn", vmin=rs_lo, vmax=rs_hi)
    axs2[3].set_title("High-res recent slope")
    fig2.colorbar(imh3, ax=axs2[3], fraction=0.046, pad=0.04)

    rd_lo, rd_hi = robust_limits(recent_delta_hr, symmetric=True)
    imh4 = axs2[4].imshow(recent_delta_hr, cmap="RdYlGn", vmin=rd_lo, vmax=rd_hi)
    axs2[4].set_title("High-res recent-vs-early delta")
    fig2.colorbar(imh4, ax=axs2[4], fraction=0.046, pad=0.04)

    pr_lo, pr_hi = robust_limits(projected_1y_hr, symmetric=True)
    imh5 = axs2[5].imshow(projected_1y_hr, cmap="RdYlGn", vmin=pr_lo, vmax=pr_hi)
    axs2[5].set_title("High-res projected 1-year delta")
    fig2.colorbar(imh5, ax=axs2[5], fraction=0.046, pad=0.04)

    for ax in axs2:
        ax.set_xlabel("Mosaic x (pixel)")
        ax.set_ylabel("Mosaic y (pixel)")

    fig2.suptitle("High-Spatial Temporal-Change Heatmaps (Pixel-Level Mosaic)", fontsize=15)
    fig2.tight_layout(rect=[0, 0.02, 1, 0.96])

    fig2.savefig(output_dir / "temporal_change_heatmaps_highres.png", dpi=220)
    plt.close(fig2)

def run_all_visualizations(project_root: Path) -> None:
    """Run all consolidated visualization tasks based on available outputs."""
    import json
    
    print("Running consolidated visualization suite...")
    
    # 1. Model Comparisons from Reports
    # We check multiple possible report locations to create separated graphs
    report_types = {
        "improved": [
            project_root / "outputs" / "reports" / "run_report_improved.json",
            project_root / "outputs" / "run_report_improved.json"
        ],
        "standard": [
            project_root / "outputs" / "reports" / "run_report.json",
            project_root / "outputs" / "run_report.json"
        ],
        "regression": [
            project_root / "outputs" / "reports" / "run_report_regression.json",
            project_root / "outputs" / "run_report_regression.json"
        ]
    }

    models_plotted = False
    for pipeline_name, paths in report_types.items():
        report_json = next((p for p in paths if p.exists()), None)
        
        if report_json:
            print(f"\nFound {report_json.name} ({pipeline_name} pipeline)! Generating model evaluation plots ...")
            out_dir = project_root / "outputs" / "figures" / pipeline_name
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                with open(report_json, "r", encoding="utf-8") as rf:
                    data = json.load(rf)
                models = data.get("all_models", {})
                if models:
                    plot_model_comparison(models, out_dir / "model_comparison.png")
                    plot_confusion_matrices(models, out_dir / "model_confusion_matrices.png")
                    print(f" -> Created model evaluation plots in `outputs/figures/{pipeline_name}`.")
                    models_plotted = True
            except Exception as e:
                print(f"Error reading JSON or plotting models for {pipeline_name}: {e}")

    if not models_plotted:
         print("\nNo run_report*.json found in outputs. Skipping model evaluation plots.")
        
    # 2. Temporal Change Heatmaps
    summary_csv_main = project_root / "outputs" / "tables" / "tile_abandonment_summary.csv"
    if not summary_csv_main.exists():
        summary_csv_main = project_root / "outputs" / "tile_abandonment_summary.csv"
        
    if summary_csv_main.exists():
        print(f"Found {summary_csv_main.name}! Generating temporal heatmaps ...")
        try:
            summary = pd.read_csv(summary_csv_main)
            time_series_dir = project_root / "time_series"
            heatmaps_out = project_root / "outputs" / "maps"
            heatmaps_out.mkdir(parents=True, exist_ok=True)
            
            plot_temporal_change_heatmaps(
                summary=summary,
                time_series_dir=time_series_dir,
                smoothed_dir=time_series_dir / "smoothed",
                raw_dir=time_series_dir / "raw",
                output_dir=heatmaps_out
            )
            print(" -> Created temporal change heatmaps.")
        except Exception as e:
            print(f"Error creating temporal change heatmaps: {e}")
    else:
        print("No tile_abandonment_summary.csv found. Temporal heatmaps skipped.")

    # 3. High Resolution Maps & Prediction vs Actual
    maps_dir = project_root / "outputs" / "maps"
    pred_path = maps_dir / "highres_projected_1y_delta_map.npy"
    act_path = maps_dir / "highres_recent_delta_map.npy"
    binary_path = maps_dir / "highres_binary_map.npy"

    if pred_path.exists() and act_path.exists() and binary_path.exists():
        print("Found high-res heatmaps! Generating Prediction vs Actual Accuracy map ...")
        try:
            pred_change = np.load(pred_path).astype(np.float32)
            actual_change = np.load(act_path).astype(np.float32)
            pred_binary = np.load(binary_path).astype(np.float32)

            metrics = compute_metrics(pred_change, actual_change, pred_binary)
            
            # Save metrics table
            summary_df = pd.DataFrame([metrics])
            summary_csv = maps_dir / "highres_prediction_vs_actual_metrics.csv"
            summary_df.to_csv(summary_csv, index=False)
            
            out_png = maps_dir / "highres_prediction_vs_actual.png"
            plot_prediction_vs_actual_highres(pred_change, actual_change, pred_binary, metrics, out_png)
            
            print(f" -> Created high-resolution accuracy maps: F1={metrics.get('f1_decline', 0):.3f}")
        except Exception as e:
            print(f"Error creating prediction vs actual map: {e}")
            
    # 4. Validation Overlays (Raw vs Smoothed Mean Curves for Sampled tiles)
    if summary_csv_main.exists() and (project_root / "time_series").exists():
        print("Generating validation overlays (NDVI curve phase tracking) ...")
        try:
            summary = pd.read_csv(summary_csv_main)
            time_series_dir = project_root / "time_series"
            smoothed_dir = time_series_dir / "smoothed"
            
            base_figures_dir = project_root / "outputs" / "figures"
            base_figures_dir.mkdir(parents=True, exist_ok=True)
            out_png_overlays = base_figures_dir / "validation_overlays.png"
            plot_validation_overlays(summary, time_series_dir, smoothed_dir, out_png_overlays)
            print(" -> Created validation overlays.")
        except Exception as e:
            print(f"Error creating validation overlays: {e}")

    print("\nVisualizations successfully generated! Check `outputs/figures` and `outputs/maps`.")