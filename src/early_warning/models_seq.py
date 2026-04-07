from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency guard
    tqdm = None

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency guard
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass
class SeqResult:
    model_name: str
    accuracy: float
    train_size: int
    test_size: int
    device: str
    train_report: Dict[str, Dict[str, float]]
    test_report: Dict[str, Dict[str, float]]
    train_confusion_matrix: List[List[int]]
    test_confusion_matrix: List[List[int]]
    validation_loss_curve: List[float]


if nn is not None:
    class AugmentedSequenceDataset(torch.utils.data.Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray, noise_std: float = 0.02, mask_prob: float = 0.08):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
            self.noise_std = float(noise_std)
            self.mask_prob = float(mask_prob)

        def __len__(self) -> int:
            return self.X.shape[0]

        def __getitem__(self, idx: int) -> tuple[Any, Any]:
            x = self.X[idx].clone()
            y = self.y[idx]

            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            if self.mask_prob > 0:
                keep = torch.rand_like(x) > self.mask_prob
                x = x * keep.float()

            return x[:, None], y


    class LSTMClassifier(nn.Module):
        def __init__(self, input_size: int = 1, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, x: Any) -> Any:
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            return self.head(last).squeeze(1)



def _train_epoch(model: Any, loader: Any, criterion: Any, optimizer: Any) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for xb, yb in loader:
        xb = xb.to(next(model.parameters()).device)
        yb = yb.to(next(model.parameters()).device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _eval_loss(model: Any, loader: Any, criterion: Any) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += float(loss.item())
            n_batches += 1
    return total_loss / max(n_batches, 1)



def _eval_accuracy(model: Any, loader: Any) -> float:
    model.eval()
    total = 0
    correct = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            total += yb.size(0)
            correct += int((preds == yb.long()).sum().item())
    return correct / max(total, 1)


def _predict_labels(model: Any, loader: Any) -> np.ndarray:
    model.eval()
    preds_all: List[np.ndarray] = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for xb, _yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            preds_all.append(preds)
    return np.concatenate(preds_all, axis=0) if preds_all else np.array([], dtype=np.int64)



def train_lstm_baseline(
    X_seq: np.ndarray,
    y: np.ndarray,
    random_state: int,
    test_size: float,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    use_gpu: bool,
    group_ids: np.ndarray | None = None,
) -> SeqResult:
    if torch is None or nn is None:
        return SeqResult(
            model_name="LSTM",
            accuracy=0.0,
            train_size=0,
            test_size=0,
            device="unavailable",
            train_report={},
            test_report={},
            train_confusion_matrix=[],
            test_confusion_matrix=[],
            validation_loss_curve=[],
        )

    if X_seq.shape[0] < 10 or len(np.unique(y)) < 2:
        return SeqResult(
            model_name="LSTM",
            accuracy=0.0,
            train_size=int(X_seq.shape[0]),
            test_size=0,
            device="cpu",
            train_report={},
            test_report={},
            train_confusion_matrix=[],
            test_confusion_matrix=[],
            validation_loss_curve=[],
        )

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if group_ids is not None and len(group_ids) == len(y):
        unique_groups, first_idx = np.unique(group_ids, return_index=True)
        group_labels = y[first_idx]
        group_counts = np.bincount(group_labels)
        can_group_stratify = group_counts.size > 1 and int(group_counts.min()) >= 2

        try:
            g_train, g_test = train_test_split(
                unique_groups,
                test_size=test_size,
                random_state=random_state,
                stratify=group_labels if can_group_stratify else None,
            )
        except ValueError:
            g_train, g_test = train_test_split(
                unique_groups,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )

        train_mask = np.isin(group_ids, g_train)
        test_mask = np.isin(group_ids, g_test)
        X_train, X_test = X_seq[train_mask], X_seq[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
    else:
        class_counts = np.bincount(y) if y.size > 0 else np.array([])
        can_stratify = len(np.unique(y)) > 1 and class_counts.size > 0 and int(class_counts.min()) >= 2
        stratify_y = y if can_stratify else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_y,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_seq,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )

    # Fit normalization on train only, then apply to both splits (no leakage).
    train_mean = np.mean(X_train, axis=0, keepdims=True)
    train_std = np.std(X_train, axis=0, keepdims=True)
    train_std = np.where(train_std < 1e-6, 1.0, train_std)
    X_train = ((X_train - train_mean) / train_std).astype(np.float32)
    X_test = ((X_test - train_mean) / train_std).astype(np.float32)

    X_train_t = torch.tensor(X_train[:, :, None], dtype=torch.float32)
    X_test_t = torch.tensor(X_test[:, :, None], dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = AugmentedSequenceDataset(X_train, y_train, noise_std=0.02, mask_prob=0.08)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=bool(use_gpu and torch.cuda.is_available()),
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=bool(use_gpu and torch.cuda.is_available()),
    )

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model = LSTMClassifier().to(device)
    class_counts = np.bincount(y_train.astype(np.int64))
    if class_counts.size > 1 and class_counts[1] > 0:
        pos_weight_val = float(class_counts[0] / max(class_counts[1], 1))
    else:
        pos_weight_val = 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_iter = range(epochs)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="LSTM epochs", unit="epoch")

    running_epoch_secs = []
    val_loss_curve: List[float] = []
    for i in epoch_iter:
        ep_start = time.perf_counter()
        mean_loss = _train_epoch(model, train_loader, criterion, optimizer)
        val_loss = _eval_loss(model, test_loader, criterion)
        val_loss_curve.append(float(val_loss))
        ep_sec = time.perf_counter() - ep_start
        running_epoch_secs.append(ep_sec)
        avg_ep = float(np.mean(running_epoch_secs))
        eta = avg_ep * (epochs - (i + 1))
        if tqdm is not None:
            epoch_iter.set_postfix(loss=f"{mean_loss:.4f}", val_loss=f"{val_loss:.4f}", epoch_s=f"{ep_sec:.2f}", eta_s=f"{eta:.1f}")
        else:
            print(
                f"Epoch {i + 1}/{epochs} | loss={mean_loss:.4f} | val_loss={val_loss:.4f} "
                f"| epoch_s={ep_sec:.2f} | eta_s={eta:.1f}"
            )

    acc = _eval_accuracy(model, test_loader)
    y_train_pred = _predict_labels(model, DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size))
    y_test_pred = _predict_labels(model, test_loader)

    train_report = classification_report(y_train.astype(np.int64), y_train_pred.astype(np.int64), zero_division=0, output_dict=True)
    test_report = classification_report(y_test.astype(np.int64), y_test_pred.astype(np.int64), zero_division=0, output_dict=True)
    train_cm = confusion_matrix(y_train.astype(np.int64), y_train_pred.astype(np.int64), labels=[0, 1]).tolist()
    test_cm = confusion_matrix(y_test.astype(np.int64), y_test_pred.astype(np.int64), labels=[0, 1]).tolist()

    return SeqResult(
        model_name="LSTM",
        accuracy=float(acc),
        train_size=len(X_train),
        test_size=len(X_test),
        device=str(device),
        train_report=train_report,
        test_report=test_report,
        train_confusion_matrix=train_cm,
        test_confusion_matrix=test_cm,
        validation_loss_curve=val_loss_curve,
    )
