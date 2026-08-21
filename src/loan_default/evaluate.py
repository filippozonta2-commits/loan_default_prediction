"""Classification metrics for imbalanced credit-risk models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_probabilities(
    y_true: Any, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    """Evaluate probabilities at a configurable decision threshold."""
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }
