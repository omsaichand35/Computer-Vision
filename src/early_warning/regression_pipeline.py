from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

try:
    from scipy.stats import pearsonr
except Exception:
    pearsonr = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception:
    torch = None
    nn = None
    DataLoader = None
    Dataset = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


@dataclass
class RegressionConfig:
    project_root: Path
    use_smoothed: bool = True
    tile_pixel_size: int = 512
    window_length: int = 64
    future_window: int = 32
    stride: int = 4
    batch_size: int = 64
    epochs: int = 20
    lstm_hidden: int = 128
    dropout: float = 0.3
    lr: float = 1e-3
    use_gpu: bool = True
    random_state: int = 42


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, noise_std: float = 0.01, mask_prob: float = 0.05):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.noise_std = float(noise_std)
        self.mask_prob = float(mask_prob)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        seq = self.x[idx].clone()
        target = self.y[idx]

        if self.noise_std > 0:
            seq = seq + torch.randn_like(seq) * self.noise_std
        if self.mask_prob > 0:
            keep = (torch.rand_like(seq) > self.mask_prob).float()
            seq = seq * keep

        return seq[:, None], target


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size: int = 128, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(1)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _zscore_window(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    std = float(np.std(x))
    if std < 1e-6:
        std = 1.0
    return ((x - mu) / std).astype(np.float32)


def _window_features(x: np.ndarray) -> np.ndarray:
    t = np.arange(x.size, dtype=np.float32)
    slope = np.polyfit(t, x, 1)[0]
    amp = float(np.max(x) - np.min(x))
    mean = float(np.mean(x))
    var = float(np.var(x))
    trend_strength = float(abs(slope) * x.size)
    return np.array([mean, var, slope, amp, trend_strength], dtype=np.float32)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    if pearsonr is not None and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        corr = float(pearsonr(y_true, y_pred)[0])
    else:
        corr = 0.0

    true_decline = (y_true < 0).astype(np.uint8)
    pred_decline = (y_pred < 0).astype(np.uint8)
    tp = int(np.sum((true_decline == 1) & (pred_decline == 1)))
    fp = int(np.sum((true_decline == 0) & (pred_decline == 1)))
    fn = int(np.sum((true_decline == 1) & (pred_decline == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_corr": corr,
        "f1_decline": f1,
        "iou_decline": iou,
        "precision_decline": precision,
        "recall_decline": recall,
    }


def _load_tile_curves(cfg: RegressionConfig) -> Dict[str, np.ndarray]:
    remote = cfg.project_root / "Remote Sensing"
    input_dir = remote / "features" / "smoothed" if cfg.use_smoothed else remote / "time_series"

    curves: Dict[str, np.ndarray] = {}
    for p in sorted(input_dir.glob("tile_*.npy")):
        tile_id = p.stem.replace("tile_", "")
        arr = np.load(p)
        if arr.ndim == 3:
            curve = np.nanmean(arr.astype(np.float32) / 10000.0, axis=(1, 2))
        elif arr.ndim == 1:
            curve = (arr.astype(np.float32) / 10000.0)
        else:
            continue

        if np.any(~np.isfinite(curve)):
            idx = np.arange(curve.size)
            valid = np.isfinite(curve)
            if valid.any():
                curve[~valid] = np.interp(idx[~valid], idx[valid], curve[valid])
            else:
                curve[:] = 0.0
        curves[tile_id] = curve.astype(np.float32)

    if not curves:
        raise RuntimeError(f"No tile curves found in {input_dir}")
    return curves


def _build_windows(curves: Dict[str, np.ndarray], cfg: RegressionConfig):
    seqs = []
    feats = []
    y = []
    tile_ids = []

    win = cfg.window_length
    fut = cfg.future_window
    stride = cfg.stride

    for tile_id, curve in curves.items():
        max_start = curve.size - (win + fut)
        if max_start < 0:
            continue

        for s in range(0, max_start + 1, stride):
            x = curve[s : s + win]
            fut_slice = curve[s + win : s + win + fut]
            target_delta = float(np.mean(fut_slice) - np.mean(x))

            x_norm = _zscore_window(x)
            seqs.append(x_norm)
            feats.append(_window_features(x_norm))
            y.append(target_delta)
            tile_ids.append(tile_id)

    if not seqs:
        raise RuntimeError("No windows created. Check window_length/future_window against curve length.")

    return (
        np.vstack(seqs).astype(np.float32),
        np.vstack(feats).astype(np.float32),
        np.array(y, dtype=np.float32),
        np.array(tile_ids),
    )


def _split_by_tile(tile_ids: np.ndarray, y: np.ndarray, seed: int):
    unique_tiles, first_idx = np.unique(tile_ids, return_index=True)
    tile_targets = y[first_idx]
    tile_bins = (tile_targets < 0).astype(np.int32)

    tile_train, tile_temp = train_test_split(
        unique_tiles,
        test_size=0.30,
        random_state=seed,
        stratify=tile_bins if np.unique(tile_bins).size > 1 else None,
    )

    temp_bins = (tile_targets[np.isin(unique_tiles, tile_temp)] < 0).astype(np.int32)
    tile_val, tile_test = train_test_split(
        tile_temp,
        test_size=0.50,
        random_state=seed,
        stratify=temp_bins if np.unique(temp_bins).size > 1 else None,
    )

    tr = np.isin(tile_ids, tile_train)
    va = np.isin(tile_ids, tile_val)
    te = np.isin(tile_ids, tile_test)
    return tr, va, te


def _train_rf_regressor(X_train, y_train, X_val, y_val, X_test, y_test):
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    pred_val = rf.predict(X_val)
    pred_test = rf.predict(X_test)

    return {
        "val_metrics": _compute_metrics(y_val, pred_val),
        "test_metrics": _compute_metrics(y_test, pred_test),
        "pred_val": pred_val,
        "pred_test": pred_test,
    }


def _train_lstm_regressor(cfg: RegressionConfig, X_train, y_train, X_val, y_val, X_test, y_test):
    if torch is None or nn is None:
        return {
            "status": "torch_unavailable",
            "val_metrics": {},
            "test_metrics": {},
            "pred_val": np.zeros_like(y_val),
            "pred_test": np.zeros_like(y_test),
            "val_loss_curve": [],
            "device": "unavailable",
        }

    device = torch.device("cuda" if (cfg.use_gpu and torch.cuda.is_available()) else "cpu")
    model = LSTMRegressor(hidden_size=cfg.lstm_hidden, dropout=cfg.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_ds = WindowDataset(X_train, y_train, noise_std=0.01, mask_prob=0.05)
    val_x = torch.tensor(X_val[:, :, None], dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32)
    test_x = torch.tensor(X_test[:, :, None], dtype=torch.float32)
    test_y = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(val_x, val_y)), batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(list(zip(test_x, test_y)), batch_size=cfg.batch_size, shuffle=False)

    val_curve: List[float] = []

    epoch_iter = range(cfg.epochs)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="LSTM regression", unit="epoch")

    for ep in epoch_iter:
        model.train()
        tr_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item())
            n += 1

        model.eval()
        v_loss = 0.0
        vn = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                v_loss += float(loss.item())
                vn += 1

        tr_loss = tr_loss / max(n, 1)
        v_loss = v_loss / max(vn, 1)
        val_curve.append(v_loss)

        if tqdm is not None:
            epoch_iter.set_postfix(train_mse=f"{tr_loss:.4f}", val_mse=f"{v_loss:.4f}")
        else:
            print(f"Epoch {ep + 1}/{cfg.epochs} train_mse={tr_loss:.4f} val_mse={v_loss:.4f}")

    def infer(loader):
        out_p = []
        out_y = []
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                pred = model(xb).detach().cpu().numpy()
                out_p.append(pred)
                out_y.append(yb.numpy())
        return np.concatenate(out_p), np.concatenate(out_y)

    pred_val, yy_val = infer(val_loader)
    pred_test, yy_test = infer(test_loader)

    return {
        "status": "ok",
        "val_metrics": _compute_metrics(yy_val, pred_val),
        "test_metrics": _compute_metrics(yy_test, pred_test),
        "pred_val": pred_val,
        "pred_test": pred_test,
        "val_loss_curve": val_curve,
        "device": str(device),
    }


def _build_tile_level_maps(curves: Dict[str, np.ndarray], cfg: RegressionConfig, rf_model_pack, lstm_model_pack):
    tile_ids = sorted(curves.keys())
    n_row = max(int(t.split("_")[0]) for t in tile_ids)
    n_col = max(int(t.split("_")[1]) for t in tile_ids)

    rf_pred_map = np.full((n_row, n_col), np.nan, dtype=np.float32)
    lstm_pred_map = np.full((n_row, n_col), np.nan, dtype=np.float32)
    actual_map = np.full((n_row, n_col), np.nan, dtype=np.float32)

    # For spatial map, use latest available window per tile.
    for tile_id in tile_ids:
        curve = curves[tile_id]
        if curve.size < (cfg.window_length + cfg.future_window):
            continue

        x = curve[-(cfg.window_length + cfg.future_window) : -cfg.future_window]
        future = curve[-cfg.future_window:]
        target_actual = float(np.mean(future) - np.mean(x))
        x_norm = _zscore_window(x)
        feat = _window_features(x_norm)[None, :]

        rr = int(tile_id.split("_")[0]) - 1
        cc = int(tile_id.split("_")[1]) - 1

        actual_map[rr, cc] = target_actual

        rf = rf_model_pack.get("model")
        if rf is not None:
            rf_pred_map[rr, cc] = float(rf.predict(feat)[0])

        lstm = lstm_model_pack.get("model")
        device = lstm_model_pack.get("device")
        if lstm is not None and device is not None and torch is not None:
            with torch.no_grad():
                xb = torch.tensor(x_norm[None, :, None], dtype=torch.float32, device=device)
                val = float(lstm(xb).cpu().numpy().ravel()[0])
            lstm_pred_map[rr, cc] = val

    return rf_pred_map, lstm_pred_map, actual_map


def _plot_regression_maps(maps_dir: Path, actual_map: np.ndarray, rf_pred: np.ndarray, lstm_pred: np.ndarray) -> None:
    rf_res = rf_pred - actual_map
    lstm_res = lstm_pred - actual_map

    def lims(a):
        valid = a[np.isfinite(a)]
        if valid.size == 0:
            return -1.0, 1.0
        q = float(np.percentile(np.abs(valid), 98))
        q = max(q, 1e-6)
        return -q, q

    lo, hi = lims(actual_map)
    rlo, rhi = lims(rf_res)
    llo, lhi = lims(lstm_res)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axs = axes.flatten()

    im0 = axs[0].imshow(actual_map, cmap="RdYlGn", vmin=lo, vmax=hi)
    axs[0].set_title("Actual delta map")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(rf_pred, cmap="RdYlGn", vmin=lo, vmax=hi)
    axs[1].set_title("RF predicted delta map")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(lstm_pred, cmap="RdYlGn", vmin=lo, vmax=hi)
    axs[2].set_title("LSTM predicted delta map")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    im3 = axs[3].imshow(rf_res, cmap="bwr", vmin=rlo, vmax=rhi)
    axs[3].set_title("RF residual (pred-actual)")
    fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    im4 = axs[4].imshow(lstm_res, cmap="bwr", vmin=llo, vmax=lhi)
    axs[4].set_title("LSTM residual (pred-actual)")
    fig.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.04)

    mask = (actual_map < 0).astype(np.float32)
    im5 = axs[5].imshow(mask, cmap="YlGnBu", vmin=0, vmax=1)
    axs[5].set_title("Actual decline mask (delta < 0)")
    fig.colorbar(im5, ax=axs[5], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel("Tile column")
        ax.set_ylabel("Tile row")

    fig.suptitle("Regression Validation: Predicted vs Actual Temporal NDVI Change", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(maps_dir / "regression_prediction_vs_actual_maps.png", dpi=220)
    plt.close(fig)


def _robust_limits(arr: np.ndarray, symmetric: bool = False) -> Tuple[float, float]:
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


def _expand_tile_map(tile_map: np.ndarray, tile_pixel_size: int) -> np.ndarray:
    return np.kron(tile_map, np.ones((tile_pixel_size, tile_pixel_size), dtype=np.float32))


def _plot_single_highres_regression_comparison(
    maps_dir: Path,
    model_name: str,
    pred_change_hr: np.ndarray,
    actual_change_hr: np.ndarray,
) -> Dict[str, float]:
    valid = np.isfinite(pred_change_hr) & np.isfinite(actual_change_hr)
    if not np.any(valid):
        raise RuntimeError(f"No valid high-res pixels available for {model_name} comparison.")

    diff = pred_change_hr - actual_change_hr
    abs_diff = np.abs(diff)

    pred_decline = (pred_change_hr < 0).astype(np.uint8)
    actual_decline = (actual_change_hr < 0).astype(np.uint8)

    tp = int(np.sum((pred_decline == 1) & (actual_decline == 1) & valid))
    tn = int(np.sum((pred_decline == 0) & (actual_decline == 0) & valid))
    fp = int(np.sum((pred_decline == 1) & (actual_decline == 0) & valid))
    fn = int(np.sum((pred_decline == 0) & (actual_decline == 1) & valid))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0
    iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    rmse = float(np.sqrt(np.mean((diff[valid]) ** 2)))
    mae = float(np.mean(abs_diff[valid]))

    pred_flat = pred_change_hr[valid].ravel()
    act_flat = actual_change_hr[valid].ravel()
    if np.std(pred_flat) > 1e-8 and np.std(act_flat) > 1e-8:
        corr = float(np.corrcoef(pred_flat, act_flat)[0, 1])
    else:
        corr = 0.0

    metrics = {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision_decline": precision,
        "recall_decline": recall,
        "f1_decline": f1,
        "iou_decline": iou,
    }

    p_lo, p_hi = _robust_limits(pred_change_hr, symmetric=True)
    a_lo, a_hi = _robust_limits(actual_change_hr, symmetric=True)
    d_lo, d_hi = _robust_limits(diff, symmetric=True)
    e_lo, e_hi = _robust_limits(abs_diff, symmetric=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axs = axes.flatten()

    im0 = axs[0].imshow(pred_change_hr, cmap="RdYlGn", vmin=p_lo, vmax=p_hi)
    axs[0].set_title(f"{model_name} predicted change (1-year NDVI delta)")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(actual_change_hr, cmap="RdYlGn", vmin=a_lo, vmax=a_hi)
    axs[1].set_title("Actual observed change")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(diff, cmap="bwr", vmin=d_lo, vmax=d_hi)
    axs[2].set_title("Residual map (predicted - actual)")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    im3 = axs[3].imshow(abs_diff, cmap="magma", vmin=e_lo, vmax=e_hi)
    axs[3].set_title("Absolute error map")
    fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    im4 = axs[4].imshow(pred_decline.astype(np.float32), cmap="YlOrRd", vmin=0, vmax=1)
    axs[4].set_title(f"{model_name} predicted decline mask")
    fig.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.04)

    im5 = axs[5].imshow(actual_decline.astype(np.float32), cmap="YlGnBu", vmin=0, vmax=1)
    axs[5].set_title(
        "Actual decline mask (delta < 0)\n"
        f"F1={f1:.3f}, IoU={iou:.3f}, RMSE={rmse:.3f}, Corr={corr:.3f}"
    )
    fig.colorbar(im5, ax=axs[5], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel("Mosaic x (pixel)")
        ax.set_ylabel("Mosaic y (pixel)")

    fig.suptitle(f"High-Resolution Regression Prediction vs Actual: {model_name}", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(maps_dir / f"highres_regression_prediction_vs_actual_{model_name.lower()}.png", dpi=220)
    plt.close(fig)

    return metrics


def _plot_regression_maps_highres(
    maps_dir: Path,
    actual_map: np.ndarray,
    rf_pred_map: np.ndarray,
    lstm_pred_map: np.ndarray,
    tile_pixel_size: int,
) -> None:
    actual_hr = _expand_tile_map(actual_map, tile_pixel_size)
    rf_hr = _expand_tile_map(rf_pred_map, tile_pixel_size)
    lstm_hr = _expand_tile_map(lstm_pred_map, tile_pixel_size)

    rf_metrics = _plot_single_highres_regression_comparison(maps_dir, "RF", rf_hr, actual_hr)
    lstm_metrics = _plot_single_highres_regression_comparison(maps_dir, "LSTM", lstm_hr, actual_hr)

    pd.DataFrame([rf_metrics, lstm_metrics]).to_csv(
        maps_dir / "highres_regression_prediction_vs_actual_metrics.csv",
        index=False,
    )


def run_regression_pipeline(cfg: RegressionConfig) -> Dict[str, object]:
    _seed_everything(cfg.random_state)

    remote = cfg.project_root / "Remote Sensing"
    out_dir = remote / "outputs" / "regression_redesign"
    tables_dir = out_dir / "tables"
    maps_dir = out_dir / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    curves = _load_tile_curves(cfg)
    X_seq, X_feat, y, tile_ids = _build_windows(curves, cfg)

    tr, va, te = _split_by_tile(tile_ids, y, seed=cfg.random_state)

    X_seq_tr, X_seq_va, X_seq_te = X_seq[tr], X_seq[va], X_seq[te]
    X_feat_tr, X_feat_va, X_feat_te = X_feat[tr], X_feat[va], X_feat[te]
    y_tr, y_va, y_te = y[tr], y[va], y[te]

    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=18,
        min_samples_leaf=2,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    rf_model.fit(X_feat_tr, y_tr)

    rf_val_pred = rf_model.predict(X_feat_va)
    rf_test_pred = rf_model.predict(X_feat_te)
    rf_val_metrics = _compute_metrics(y_va, rf_val_pred)
    rf_test_metrics = _compute_metrics(y_te, rf_test_pred)

    lstm_pack = {"model": None, "device": None}
    lstm_results = _train_lstm_regressor(cfg, X_seq_tr, y_tr, X_seq_va, y_va, X_seq_te, y_te)

    # Train a final LSTM model for tile-level map inference.
    if torch is not None and nn is not None and lstm_results.get("status") == "ok":
        device = torch.device("cuda" if (cfg.use_gpu and torch.cuda.is_available()) else "cpu")
        model = LSTMRegressor(hidden_size=cfg.lstm_hidden, dropout=cfg.dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        X_all = np.concatenate([X_seq_tr, X_seq_va], axis=0)
        y_all = np.concatenate([y_tr, y_va], axis=0)
        all_loader = DataLoader(WindowDataset(X_all, y_all, noise_std=0.005, mask_prob=0.03), batch_size=cfg.batch_size, shuffle=True)

        for _ in range(max(8, cfg.epochs // 2)):
            model.train()
            for xb, yb in all_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

        lstm_pack = {"model": model, "device": device}

    rf_pred_map, lstm_pred_map, actual_map = _build_tile_level_maps(curves, cfg, {"model": rf_model}, lstm_pack)
    _plot_regression_maps(maps_dir, actual_map, rf_pred_map, lstm_pred_map)
    _plot_regression_maps_highres(
        maps_dir=maps_dir,
        actual_map=actual_map,
        rf_pred_map=rf_pred_map,
        lstm_pred_map=lstm_pred_map,
        tile_pixel_size=cfg.tile_pixel_size,
    )

    np.save(maps_dir / "rf_pred_delta_map.npy", rf_pred_map)
    np.save(maps_dir / "lstm_pred_delta_map.npy", lstm_pred_map)
    np.save(maps_dir / "actual_delta_map.npy", actual_map)

    report = {
        "config": {
            "window_length": cfg.window_length,
            "future_window": cfg.future_window,
            "stride": cfg.stride,
            "epochs": cfg.epochs,
            "ndvi_scaling": "ndvi/10000",
            "input_normalization": "per-window zscore",
            "tile_split": "train/val/test by tile groups",
        },
        "dataset": {
            "samples_total": int(y.size),
            "samples_train": int(y_tr.size),
            "samples_val": int(y_va.size),
            "samples_test": int(y_te.size),
            "unique_tiles_total": int(np.unique(tile_ids).size),
            "unique_tiles_train": int(np.unique(tile_ids[tr]).size),
            "unique_tiles_val": int(np.unique(tile_ids[va]).size),
            "unique_tiles_test": int(np.unique(tile_ids[te]).size),
        },
        "rf_regression": {
            "val": rf_val_metrics,
            "test": rf_test_metrics,
        },
        "lstm_regression": {
            "val": lstm_results.get("val_metrics", {}),
            "test": lstm_results.get("test_metrics", {}),
            "device": lstm_results.get("device", "unknown"),
            "val_loss_curve": lstm_results.get("val_loss_curve", []),
        },
    }

    report_path = out_dir / "regression_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    pd.DataFrame([
        {"split": "val", **rf_val_metrics},
        {"split": "test", **rf_test_metrics},
    ]).to_csv(tables_dir / "rf_regression_metrics.csv", index=False)

    pd.DataFrame([
        {"split": "val", **lstm_results.get("val_metrics", {})},
        {"split": "test", **lstm_results.get("test_metrics", {})},
    ]).to_csv(tables_dir / "lstm_regression_metrics.csv", index=False)

    return {
        "report_json": str(report_path),
        "maps_figure": str(maps_dir / "regression_prediction_vs_actual_maps.png"),
        "maps_figure_highres_rf": str(maps_dir / "highres_regression_prediction_vs_actual_rf.png"),
        "maps_figure_highres_lstm": str(maps_dir / "highres_regression_prediction_vs_actual_lstm.png"),
        "maps_dir": str(maps_dir),
        "tables_dir": str(tables_dir),
    }


def main() -> None:
    cfg = RegressionConfig(project_root=Path(__file__).resolve().parents[2])
    outputs = run_regression_pipeline(cfg)
    print("Regression redesign pipeline completed.")
    for k, v in outputs.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
