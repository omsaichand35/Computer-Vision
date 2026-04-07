from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


@dataclass
class RFResult:
    report: str
    train_size: int
    test_size: int
    train_report: Dict[str, Dict[str, float]]
    test_report: Dict[str, Dict[str, float]]
    train_confusion_matrix: List[List[int]]
    test_confusion_matrix: List[List[int]]



def train_rf_baseline(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    test_size: float,
    group_ids: np.ndarray | None = None,
) -> RFResult:
    if X.shape[0] < 8 or len(np.unique(y)) < 2:
        return RFResult(
            report="Skipped RF training: insufficient samples or single class in weak labels.",
            train_size=int(X.shape[0]),
            test_size=0,
            train_report={},
            test_report={},
            train_confusion_matrix=[],
            test_confusion_matrix=[],
        )

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
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
    else:
        class_counts = np.bincount(y) if y.size > 0 else np.array([])
        can_stratify = len(np.unique(y)) > 1 and class_counts.size > 0 and int(class_counts.min()) >= 2
        stratify_y = y if can_stratify else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_y,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=None,
            )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0)
    train_report = classification_report(y_train, y_train_pred, zero_division=0, output_dict=True)
    test_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    train_cm = confusion_matrix(y_train, y_train_pred, labels=[0, 1]).tolist()
    test_cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    return RFResult(
        report=report,
        train_size=len(X_train),
        test_size=len(X_test),
        train_report=train_report,
        test_report=test_report,
        train_confusion_matrix=train_cm,
        test_confusion_matrix=test_cm,
    )
