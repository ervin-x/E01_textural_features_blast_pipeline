from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class BinaryMetrics:
    pr_auc: float
    roc_auc: float
    balanced_accuracy: float
    macro_f1: float
    recall_blast: float
    specificity: float
    matthews_corrcoef: float
    brier_score: float
    expected_calibration_error: float


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        if upper == 1.0:
            mask = (y_score >= lower) & (y_score <= upper)
        else:
            mask = (y_score >= lower) & (y_score < upper)
        if not np.any(mask):
            continue
        accuracy = float(np.mean(y_true[mask]))
        confidence = float(np.mean(y_score[mask]))
        weight = float(np.mean(mask))
        ece += abs(accuracy - confidence) * weight
    return float(ece)


def optimize_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    candidate_thresholds = np.unique(np.clip(y_score, 0.0, 1.0))
    candidate_thresholds = np.concatenate(([0.0], candidate_thresholds, [1.0]))

    best_threshold = 0.5
    best_value = -np.inf
    for threshold in candidate_thresholds:
        y_pred = (y_score >= threshold).astype(int)
        value = matthews_corrcoef(y_true, y_pred)
        if np.isnan(value):
            value = -1.0
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
    return best_threshold


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> BinaryMetrics:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = float(average_precision_score(y_true, y_score))
    except ValueError:
        pr_auc = float("nan")

    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    matthews = float(matthews_corrcoef(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_score))
    ece = expected_calibration_error(y_true, y_score)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall_blast = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return BinaryMetrics(
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        balanced_accuracy=balanced_accuracy,
        macro_f1=macro_f1,
        recall_blast=recall_blast,
        specificity=specificity,
        matthews_corrcoef=matthews,
        brier_score=brier,
        expected_calibration_error=ece,
    )


def bootstrap_metric_interval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    n_bootstrap: int = 100,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(random_state)
    metric_samples: dict[str, list[float]] = {
        "pr_auc": [],
        "roc_auc": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "recall_blast": [],
        "specificity": [],
        "matthews_corrcoef": [],
        "brier_score": [],
        "expected_calibration_error": [],
    }

    if len(y_true) == 0:
        return {name: (float("nan"), float("nan")) for name in metric_samples}

    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, len(y_true), size=len(y_true))
        sample_true = y_true[sample_idx]
        sample_score = y_score[sample_idx]
        if len(np.unique(sample_true)) < 2:
            continue
        metrics = compute_binary_metrics(sample_true, sample_score, threshold)
        for name in metric_samples:
            metric_samples[name].append(float(getattr(metrics, name)))

    result: dict[str, tuple[float, float]] = {}
    for name, values in metric_samples.items():
        if not values:
            result[name] = (float("nan"), float("nan"))
            continue
        result[name] = (
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
    return result


def recall_at_fixed_fpr(y_true: np.ndarray, y_score: np.ndarray, max_fpr: float = 0.10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    mask = fpr <= max_fpr
    if not np.any(mask):
        return 0.0
    return float(np.max(tpr[mask]))


def ranking_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    positive_total = max(int(np.sum(y_true_sorted)), 1)
    cumulative_recall = np.cumsum(y_true_sorted) / positive_total
    screened_fraction = (np.arange(len(y_true_sorted)) + 1) / max(len(y_true_sorted), 1)
    return screened_fraction.astype(float), cumulative_recall.astype(float)
