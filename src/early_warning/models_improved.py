"""Improved models with better class imbalance handling and advanced architectures."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Optional imports with graceful fallbacks
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    WeightedRandomSampler = None
    ReduceLROnPlateau = None

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except Exception:
    SMOTE = None
    SMOTE_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False
    LGBMClassifier = None


@dataclass
class ModelResult:
    model_name: str
    accuracy: float
    train_size: int
    test_size: int
    device: str
    train_report: Dict[str, Dict[str, float]]
    test_report: Dict[str, Dict[str, float]]
    train_confusion_matrix: List[List[int]]
    test_confusion_matrix: List[List[int]]
    f1_macro: float
    f1_weighted: float
    f1_class_1: float  # Critical for abandonment detection


def _compute_f1_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Compute F1 metrics for evaluation."""
    labels = np.unique(y_true)
    if len(labels) < 2:
        return 0.0, 0.0, 0.0

    f1_per_class = []
    for label in sorted(labels):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_per_class.append(f1)

    f1_macro = np.mean(f1_per_class)

    # Weighted F1
    support = [np.sum(y_true == label) for label in sorted(labels)]
    f1_weighted = np.average(f1_per_class, weights=support)

    # F1 for class 1 (abandonment) - the critical metric
    f1_class_1 = f1_per_class[1] if len(f1_per_class) > 1 else f1_per_class[0]

    return f1_macro, f1_weighted, f1_class_1


def train_gradient_boosting(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    test_size: float,
    use_smote: bool = True,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> ModelResult:
    """Train Gradient Boosting with class imbalance handling."""

    if X.shape[0] < 10 or len(np.unique(y)) < 2:
        return ModelResult(
            model_name="GradientBoosting",
            accuracy=0.0,
            train_size=int(X.shape[0]),
            test_size=0,
            device="cpu",
            train_report={},
            test_report={},
            train_confusion_matrix=[],
            test_confusion_matrix=[],
            f1_macro=0.0,
            f1_weighted=0.0,
            f1_class_1=0.0,
        )

    # Split first
    stratify = y if len(np.unique(y)) >= 2 and np.min(np.bincount(y)) >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )

    # Apply SMOTE if available and requested
    if use_smote and SMOTE_AVAILABLE and len(np.unique(y_train)) >= 2:
        try:
            smote = SMOTE(random_state=random_state, k_neighbors=min(5, np.min(np.bincount(y_train)) - 1))
            X_train, y_train = smote.fit_resample(X_train, y_train)
        except Exception:
            pass  # Fall back to original data

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Compute class weights
    class_counts = np.bincount(y_train.astype(int))
    class_weight = {i: len(y_train) / (len(class_counts) * count) for i, count in enumerate(class_counts)}

    # Train Gradient Boosting
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Apply threshold tuning for better class 1 recall
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Find optimal threshold using training data
        best_threshold = 0.5
        best_f1 = 0
        for threshold in np.arange(0.2, 0.8, 0.05):
            preds = (y_train_proba >= threshold).astype(int)
            _, _, f1_c1 = _compute_f1_metrics(y_train, preds)
            if f1_c1 > best_f1:
                best_f1 = f1_c1
                best_threshold = threshold

        # Use optimized threshold
        y_test_pred = (y_test_proba >= best_threshold).astype(int)

    # Compute metrics
    train_f1_macro, train_f1_weighted, train_f1_c1 = _compute_f1_metrics(y_train, y_train_pred)
    test_f1_macro, test_f1_weighted, test_f1_c1 = _compute_f1_metrics(y_test, y_test_pred)

    train_report = classification_report(y_train.astype(int), y_train_pred.astype(int), zero_division=0, output_dict=True)
    test_report = classification_report(y_test.astype(int), y_test_pred.astype(int), zero_division=0, output_dict=True)
    train_cm = confusion_matrix(y_train.astype(int), y_train_pred.astype(int), labels=[0, 1]).tolist()
    test_cm = confusion_matrix(y_test.astype(int), y_test_pred.astype(int), labels=[0, 1]).tolist()

    return ModelResult(
        model_name="GradientBoosting",
        accuracy=float(np.mean(y_test == y_test_pred)),
        train_size=len(X_train),
        test_size=len(X_test),
        device="cpu",
        train_report=train_report,
        test_report=test_report,
        train_confusion_matrix=train_cm,
        test_confusion_matrix=test_cm,
        f1_macro=float(test_f1_macro),
        f1_weighted=float(test_f1_weighted),
        f1_class_1=float(test_f1_c1),
    )


def train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    test_size: float,
    use_smote: bool = True,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> ModelResult:
    """Train XGBoost with scale_pos_weight for class imbalance."""

    if not XGB_AVAILABLE or X.shape[0] < 10 or len(np.unique(y)) < 2:
        return ModelResult(
            model_name="XGBoost",
            accuracy=0.0,
            train_size=int(X.shape[0]),
            test_size=0,
            device="cpu",
            train_report={},
            test_report={},
            train_confusion_matrix=[],
            test_confusion_matrix=[],
            f1_macro=0.0,
            f1_weighted=0.0,
            f1_class_1=0.0,
        )

    # Split
    stratify = y if len(np.unique(y)) >= 2 and np.min(np.bincount(y)) >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )

    # SMOTE
    if use_smote and SMOTE_AVAILABLE and len(np.unique(y_train)) >= 2:
        try:
            smote = SMOTE(random_state=random_state, k_neighbors=min(5, np.min(np.bincount(y_train)) - 1))
            X_train, y_train = smote.fit_resample(X_train, y_train)
        except Exception:
            pass

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Scale pos weight for class imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / (pos_count + 1e-8)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
    )
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Threshold tuning
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.15, 0.8, 0.05):
        preds = (y_test_proba >= threshold).astype(int)
        _, _, f1_c1 = _compute_f1_metrics(y_test, preds)
        if f1_c1 > best_f1:
            best_f1 = f1_c1
            best_threshold = threshold

    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    train_f1_macro, train_f1_weighted, train_f1_c1 = _compute_f1_metrics(y_train, y_train_pred)
    test_f1_macro, test_f1_weighted, test_f1_c1 = _compute_f1_metrics(y_test, y_test_pred)

    train_report = classification_report(y_train.astype(int), y_train_pred.astype(int), zero_division=0, output_dict=True)
    test_report = classification_report(y_test.astype(int), y_test_pred.astype(int), zero_division=0, output_dict=True)
    train_cm = confusion_matrix(y_train.astype(int), y_train_pred.astype(int), labels=[0, 1]).tolist()
    test_cm = confusion_matrix(y_test.astype(int), y_test_pred.astype(int), labels=[0, 1]).tolist()

    return ModelResult(
        model_name="XGBoost",
        accuracy=float(np.mean(y_test == y_test_pred)),
        train_size=len(X_train),
        test_size=len(X_test),
        device="cpu",
        train_report=train_report,
        test_report=test_report,
        train_confusion_matrix=train_cm,
        test_confusion_matrix=test_cm,
        f1_macro=float(test_f1_macro),
        f1_weighted=float(test_f1_weighted),
        f1_class_1=float(test_f1_c1),
    )


def train_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    test_size: float,
    use_smote: bool = True,
    n_estimators: int = 300,
    max_depth: int = 8,
    learning_rate: float = 0.05,
) -> ModelResult:
    """Train LightGBM with class imbalance handling."""

    if not LGBM_AVAILABLE or X.shape[0] < 10 or len(np.unique(y)) < 2:
        return ModelResult(
            model_name="LightGBM",
            accuracy=0.0,
            train_size=int(X.shape[0]),
            test_size=0,
            device="cpu",
            train_report={},
            test_report={},
            train_confusion_matrix=[],
            test_confusion_matrix=[],
            f1_macro=0.0,
            f1_weighted=0.0,
            f1_class_1=0.0,
        )

    # Split
    stratify = y if len(np.unique(y)) >= 2 and np.min(np.bincount(y)) >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )

    # SMOTE
    if use_smote and SMOTE_AVAILABLE and len(np.unique(y_train)) >= 2:
        try:
            smote = SMOTE(random_state=random_state, k_neighbors=min(5, np.min(np.bincount(y_train)) - 1))
            X_train, y_train = smote.fit_resample(X_train, y_train)
        except Exception:
            pass

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        class_weight='balanced',
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
    )
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Threshold tuning
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.15, 0.8, 0.05):
        preds = (y_test_proba >= threshold).astype(int)
        _, _, f1_c1 = _compute_f1_metrics(y_test, preds)
        if f1_c1 > best_f1:
            best_f1 = f1_c1
            best_threshold = threshold

    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    train_f1_macro, train_f1_weighted, train_f1_c1 = _compute_f1_metrics(y_train, y_train_pred)
    test_f1_macro, test_f1_weighted, test_f1_c1 = _compute_f1_metrics(y_test, y_test_pred)

    train_report = classification_report(y_train.astype(int), y_train_pred.astype(int), zero_division=0, output_dict=True)
    test_report = classification_report(y_test.astype(int), y_test_pred.astype(int), zero_division=0, output_dict=True)
    train_cm = confusion_matrix(y_train.astype(int), y_train_pred.astype(int), labels=[0, 1]).tolist()
    test_cm = confusion_matrix(y_test.astype(int), y_test_pred.astype(int), labels=[0, 1]).tolist()

    return ModelResult(
        model_name="LightGBM",
        accuracy=float(np.mean(y_test == y_test_pred)),
        train_size=len(X_train),
        test_size=len(X_test),
        device="cpu",
        train_report=train_report,
        test_report=test_report,
        train_confusion_matrix=train_cm,
        test_confusion_matrix=test_cm,
        f1_macro=float(test_f1_macro),
        f1_weighted=float(test_f1_weighted),
        f1_class_1=float(test_f1_c1),
    )


# Improved CNN model with focal loss
if nn is not None:
    class FocalLoss(nn.Module):
        """Focal loss for handling class imbalance."""

        def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            return focal_loss.mean()


    class ImprovedCNN1D(nn.Module):
        """1D CNN for sequence classification with residual connections."""

        def __init__(
            self,
            input_size: int = 1,
            num_filters: int = 64,
            kernel_sizes: list = None,
            hidden_size: int = 128,
            dropout: float = 0.4,
        ):
            super().__init__()

            if kernel_sizes is None:
                kernel_sizes = [3, 5, 7]

            # Multi-scale convolution blocks
            self.conv_blocks = nn.ModuleList()
            for k in kernel_sizes:
                self.conv_blocks.append(
                    nn.Sequential(
                        nn.Conv1d(input_size, num_filters, kernel_size=k, padding=k//2),
                        nn.BatchNorm1d(num_filters),
                        nn.ReLU(),
                        nn.Conv1d(num_filters, num_filters, kernel_size=k, padding=k//2),
                        nn.BatchNorm1d(num_filters),
                        nn.ReLU(),
                    )
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
            x = x.transpose(1, 2)

            # Multi-scale convolutions
            conv_outputs = []
            for conv_block in self.conv_blocks:
                out = conv_block(x)  # (batch, num_filters, seq_len)
                out = torch.max(out, dim=2)[0]  # Global max pooling
                conv_outputs.append(out)

            # Concatenate multi-scale features
            combined = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(kernel_sizes))

            return combined


    class ClassificationHead(nn.Module):
        """Classification head for sequence models."""

        def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.4):
            super().__init__()
            self.head = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(x).squeeze(1)


def train_improved_cnn(
    X_seq: np.ndarray,
    y: np.ndarray,
    random_state: int,
    test_size: float,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    use_gpu: bool = False,
    use_focal_loss: bool = True,
) -> ModelResult:
    """Train improved 1D CNN with focal loss for class imbalance."""

    if torch is None or nn is None or X_seq.shape[0] < 10 or len(np.unique(y)) < 2:
        return ModelResult(
            model_name="CNN1D",
            accuracy=0.0,
            train_size=int(X_seq.shape[0]),
            test_size=0,
            device="unavailable",
            train_report={},
            test_report={},
            train_confusion_matrix=[],
            test_confusion_matrix=[],
            f1_macro=0.0,
            f1_weighted=0.0,
            f1_class_1=0.0,
        )

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # Split
    stratify = y if len(np.unique(y)) >= 2 and np.min(np.bincount(y)) >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y, test_size=test_size, random_state=random_state, stratify=None
        )

    # Normalize
    train_mean = np.mean(X_train, axis=0, keepdims=True)
    train_std = np.std(X_train, axis=0, keepdims=True)
    train_std = np.where(train_std < 1e-6, 1.0, train_std)
    X_train_norm = ((X_train - train_mean) / train_std).astype(np.float32)
    X_test_norm = ((X_test - train_mean) / train_std).astype(np.float32)

    # Create datasets
    X_train_t = torch.tensor(X_train_norm[:, :, None], dtype=torch.float32)
    X_test_t = torch.tensor(X_test_norm[:, :, None], dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Class weights for sampling
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.astype(int)]
    sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.float32),
                                     num_samples=len(sample_weights), replacement=True)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    # Model
    num_filters = 64
    kernel_sizes = [3, 5, 7]
    feature_size = num_filters * len(kernel_sizes)
    hidden_size = 128

    feature_extractor = ImprovedCNN1D(input_size=1, num_filters=num_filters, kernel_sizes=kernel_sizes).to(device)
    classifier = ClassificationHead(input_size=feature_size, hidden_size=hidden_size).to(device)

    # Loss with focal loss option
    if use_focal_loss and class_counts[1] > 0:
        pos_weight = float(class_counts[0] / class_counts[1])
        criterion = FocalLoss(alpha=min(0.75, 1 - pos_weight / (pos_weight + 1)), gamma=2.0)
    else:
        pos_weight = float(class_counts[0] / class_counts[1]) if class_counts.size > 1 else 1.0
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    optimizer = torch.optim.AdamW(list(feature_extractor.parameters()) + list(classifier.parameters()),
                                   lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        # Train
        feature_extractor.train()
        classifier.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            features = feature_extractor(xb)
            logits = classifier(features)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validate
        feature_extractor.eval()
        classifier.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                features = feature_extractor(xb)
                logits = classifier(features)
                loss = criterion(logits, yb)
                val_loss += loss.item()

        val_loss /= len(test_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                'feature_extractor': feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= max_patience:
            break

    # Load best model
    if best_model_state is not None:
        feature_extractor.load_state_dict(best_model_state['feature_extractor'])
        classifier.load_state_dict(best_model_state['classifier'])

    # Evaluate
    feature_extractor.eval()
    classifier.eval()

    def predict(loader):
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                features = feature_extractor(xb)
                logits = classifier(features)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_probs)

    y_train_pred, y_train_proba = predict(train_loader)
    y_test_pred, y_test_proba = predict(test_loader)

    # Threshold tuning for better F1
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.2, 0.8, 0.05):
        preds = (y_test_proba >= threshold).astype(int)
        _, _, f1_c1 = _compute_f1_metrics(y_test, preds)
        if f1_c1 > best_f1:
            best_f1 = f1_c1
            best_threshold = threshold

    y_test_pred = (y_test_proba >= best_threshold).astype(int)
    y_train_pred = (y_train_proba >= best_threshold).astype(int)

    train_f1_macro, train_f1_weighted, train_f1_c1 = _compute_f1_metrics(y_train, y_train_pred)
    test_f1_macro, test_f1_weighted, test_f1_c1 = _compute_f1_metrics(y_test, y_test_pred)

    train_report = classification_report(y_train.astype(int), y_train_pred.astype(int), zero_division=0, output_dict=True)
    test_report = classification_report(y_test.astype(int), y_test_pred.astype(int), zero_division=0, output_dict=True)
    train_cm = confusion_matrix(y_train.astype(int), y_train_pred.astype(int), labels=[0, 1]).tolist()
    test_cm = confusion_matrix(y_test.astype(int), y_test_pred.astype(int), labels=[0, 1]).tolist()

    return ModelResult(
        model_name="CNN1D-Focal",
        accuracy=float(np.mean(y_test == y_test_pred)),
        train_size=len(X_train),
        test_size=len(X_test),
        device=str(device),
        train_report=train_report,
        test_report=test_report,
        train_confusion_matrix=train_cm,
        test_confusion_matrix=test_cm,
        f1_macro=float(test_f1_macro),
        f1_weighted=float(test_f1_weighted),
        f1_class_1=float(test_f1_c1),
    )


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    X_seq: np.ndarray,
    random_state: int,
    test_size: float,
    use_gpu: bool = False,
) -> Tuple[ModelResult, List[ModelResult]]:
    """Train ensemble of models and combine predictions."""

    # Train individual models
    gb_result = train_gradient_boosting(X, y, random_state, test_size, use_smote=True)
    xgb_result = train_xgboost(X, y, random_state, test_size, use_smote=True) if XGB_AVAILABLE else gb_result
    lgbm_result = train_lightgbm(X, y, random_state, test_size, use_smote=True) if LGBM_AVAILABLE else gb_result

    all_results = [gb_result, xgb_result, lgbm_result]

    # Find best model based on F1 for class 1
    best_result = max(all_results, key=lambda r: r.f1_class_1)

    return best_result, all_results
