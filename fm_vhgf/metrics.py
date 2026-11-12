"""Patient-level binary metrics from continuous positive-class probabilities."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def binary_metrics(labels, probabilities, threshold=0.5):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("Expected equally sized one-dimensional labels and probabilities.")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("AUC requires both classes in the evaluation set.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Expected finite positive-class probabilities in [0,1].")
    predicted = probabilities >= threshold
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "acc": float(accuracy_score(labels, predicted)),
        "sen": float(recall_score(labels, predicted, zero_division=0)),
        "pre": float(precision_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }
