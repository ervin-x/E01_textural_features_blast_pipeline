from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.metrics import compute_binary_metrics, ranking_recall_curve, recall_at_fixed_fpr


def _soft_noisy_or(scores: Iterable[float]) -> float:
    values = np.clip(np.asarray(list(scores), dtype=float), 0.0, 1.0)
    if len(values) == 0:
        return 0.0
    return float(1.0 - np.prod(1.0 - values))


def aggregate_object_scores(
    prediction_df: pd.DataFrame,
    threshold: float,
    group_columns: list[str],
    target_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in prediction_df.groupby(group_columns, sort=False):
        scores = group["y_score"].astype(float).to_numpy()
        scores_sorted = np.sort(scores)[::-1]
        top3 = scores_sorted[: min(3, len(scores_sorted))]
        group_key = keys if isinstance(keys, tuple) else (keys,)
        key_payload = dict(zip(group_columns, group_key, strict=True))
        image_contains_blast = int(group[target_column].max())
        rows.extend(
            [
                {
                    **key_payload,
                    "aggregation_method": "max_object_score",
                    "score": float(scores.max()),
                    target_column: image_contains_blast,
                    "n_objects": int(len(scores)),
                },
                {
                    **key_payload,
                    "aggregation_method": "mean_top3_scores",
                    "score": float(np.mean(top3)),
                    target_column: image_contains_blast,
                    "n_objects": int(len(scores)),
                },
                {
                    **key_payload,
                    "aggregation_method": "count_objects_above_threshold",
                    "score": float(np.sum(scores >= threshold)),
                    target_column: image_contains_blast,
                    "n_objects": int(len(scores)),
                },
                {
                    **key_payload,
                    "aggregation_method": "soft_noisy_or",
                    "score": _soft_noisy_or(scores),
                    target_column: image_contains_blast,
                    "n_objects": int(len(scores)),
                },
            ]
        )
    return pd.DataFrame(rows)


def compute_aggregate_metrics(score_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in score_df.groupby("aggregation_method", sort=False):
        y_true = group[target_column].astype(int).to_numpy()
        y_score = group["score"].astype(float).to_numpy()
        if len(y_score) == 0:
            continue
        if float(np.nanmax(y_score)) > 1.0 or float(np.nanmin(y_score)) < 0.0:
            score_min = float(np.nanmin(y_score))
            score_max = float(np.nanmax(y_score))
            if score_max > score_min:
                y_score_prob = (y_score - score_min) / (score_max - score_min)
            else:
                y_score_prob = np.zeros_like(y_score, dtype=float)
        else:
            y_score_prob = y_score
        threshold = float(np.median(y_score_prob))
        metrics = compute_binary_metrics(y_true, y_score_prob, threshold)
        screened_fraction, cumulative_recall = ranking_recall_curve(y_true, y_score)
        rows.append(
            {
                "aggregation_method": method,
                "n_groups": int(len(group)),
                "positive_groups": int(np.sum(y_true)),
                "pr_auc": metrics.pr_auc,
                "roc_auc": metrics.roc_auc,
                "balanced_accuracy": metrics.balanced_accuracy,
                "recall_blast": metrics.recall_blast,
                "specificity": metrics.specificity,
                "matthews_corrcoef": metrics.matthews_corrcoef,
                "brier_score": metrics.brier_score,
                "expected_calibration_error": metrics.expected_calibration_error,
                "recall_at_fpr_10pct": recall_at_fixed_fpr(y_true, y_score_prob, max_fpr=0.10),
                "top_10pct_recall": float(cumulative_recall[np.searchsorted(screened_fraction, 0.10, side="left")])
                if len(cumulative_recall)
                else float("nan"),
                "top_20pct_recall": float(cumulative_recall[np.searchsorted(screened_fraction, 0.20, side="left")])
                if len(cumulative_recall)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)
