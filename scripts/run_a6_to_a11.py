from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
VENDOR_ROOT = PROJECT_ROOT / "vendor"
VENV_SITE_PACKAGES = sorted((PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages"))

for candidate in [*VENV_SITE_PACKAGES, SRC_ROOT, VENDOR_ROOT]:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.append(candidate_str)

from evaluation.aggregate_scores import aggregate_object_scores, compute_aggregate_metrics
from evaluation.metrics import bootstrap_metric_interval, compute_binary_metrics, optimize_threshold
from evaluation.visualization import (
    save_bar_plot,
    save_confusion_matrix,
    save_example_rois,
    save_pipeline_overview,
    save_pr_roc_summary,
    save_ranking_curve,
)
from models.baseline_pipeline import (
    FEATURE_GROUPS,
    KEY_COLUMNS,
    build_model,
    feature_columns_by_group,
    fit_predict_probability,
    merge_task_with_features,
)
from utils.project import CHECKPOINTS_ROOT, CROPS_ROOT, FEATURES_ROOT, FIGURES_ROOT, PREDICTIONS_ROOT, REPORTS_ROOT, TABLES_ROOT, ensure_output_layout, write_markdown


TASK_FILES = {
    "a41_clean_binary": PROJECT_ROOT / "outputs" / "dataset_index" / "task_a41_clean_binary.parquet",
    "a42_strict_binary": PROJECT_ROOT / "outputs" / "dataset_index" / "task_a42_strict_binary.parquet",
    "a43_full_realistic_binary": PROJECT_ROOT / "outputs" / "dataset_index" / "task_a43_full_realistic_binary.parquet",
}
MODEL_NAMES = [
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "xgboost",
]


def _pick_feature_path(base_name: str) -> Path:
    candidates = [
        FEATURES_ROOT / f"{base_name}.parquet",
        FEATURES_ROOT / f"{base_name}_fast.parquet",
        FEATURES_ROOT / f"{base_name}_lite.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of the candidate feature files exist for {base_name}: {candidates}")


def _load_required_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    features_combined = pd.read_parquet(_pick_feature_path("features_combined"))
    features_bbox = pd.read_parquet(_pick_feature_path("features_bbox"))
    features_mask = pd.read_parquet(_pick_feature_path("features_mask"))
    tasks = {task_name: pd.read_parquet(path) for task_name, path in TASK_FILES.items()}
    return features_combined, features_bbox, features_mask, tasks


def _split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df.loc[df["split_name"] == "train"].copy()
    val_df = df.loc[df["split_name"] == "val"].copy()
    test_df = df.loc[df["split_name"] == "test"].copy()
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("One of the splits is empty.")
    return train_df, val_df, test_df


def _cap_frame(frame: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    sampled_groups: list[pd.DataFrame] = []
    remaining = max_rows
    groups = list(frame.groupby("target_binary", sort=True))
    for index, (_, group) in enumerate(groups):
        if index == len(groups) - 1:
            n_take = min(len(group), remaining)
        else:
            n_take = max(1, int(round(max_rows * len(group) / len(frame))))
            n_take = min(len(group), n_take)
            remaining -= n_take
        sampled_groups.append(group.sample(n=n_take, random_state=random_state))
    return pd.concat(sampled_groups, ignore_index=True)


def _select_best_row(leaderboard_df: pd.DataFrame, task_name: str | None = None) -> pd.Series:
    frame = leaderboard_df.copy()
    if task_name is not None:
        frame = frame.loc[frame["task_name"] == task_name].copy()
    if len(frame) == 0:
        raise ValueError("Leaderboard is empty for selection.")
    return frame.sort_values(
        ["validation_pr_auc", "test_pr_auc", "test_recall_blast"],
        ascending=[False, False, False],
    ).iloc[0]


def run_a6() -> dict[str, str]:
    ensure_output_layout()
    features_combined, features_bbox, features_mask, tasks = _load_required_artifacts()
    feature_groups = feature_columns_by_group(features_combined)

    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []

    for task_name, task_df in tasks.items():
        merged_df = merge_task_with_features(task_df, features_combined)
        train_df, val_df, test_df = _split_frame(merged_df)
        train_df = _cap_frame(train_df, max_rows=20000, random_state=42)
        val_df = _cap_frame(val_df, max_rows=5000, random_state=43)
        scale_pos_weight = float((train_df["target_binary"].eq(0).sum()) / max(train_df["target_binary"].eq(1).sum(), 1))

        for model_name in MODEL_NAMES:
            for feature_group_name, feature_columns in feature_groups.items():
                X_train = train_df[feature_columns]
                y_train = train_df["target_binary"].astype(int)
                X_val = val_df[feature_columns]
                y_val = val_df["target_binary"].astype(int)
                X_test = test_df[feature_columns]
                y_test = test_df["target_binary"].astype(int)

                pipeline, spec = build_model(model_name, scale_pos_weight=scale_pos_weight, use_class_weight=True)
                val_score = fit_predict_probability(pipeline, X_train, y_train, X_val)
                threshold = optimize_threshold(y_val.to_numpy(), val_score)
                test_score = np.asarray(pipeline.predict_proba(X_test)[:, 1], dtype=float) if hasattr(pipeline, "predict_proba") else fit_predict_probability(pipeline, X_train, y_train, X_test)

                val_metrics = compute_binary_metrics(y_val.to_numpy(), val_score, threshold)
                test_metrics = compute_binary_metrics(y_test.to_numpy(), test_score, threshold)
                ci = bootstrap_metric_interval(y_test.to_numpy(), test_score, threshold, n_bootstrap=60, random_state=42)

                run_id = f"{task_name}__{model_name}__{feature_group_name}"
                checkpoint_path = CHECKPOINTS_ROOT / f"{run_id}.joblib"
                joblib.dump(
                    {
                        "pipeline": pipeline,
                        "feature_columns": feature_columns,
                        "threshold": threshold,
                        "task_name": task_name,
                        "model_name": model_name,
                        "feature_group": feature_group_name,
                    },
                    checkpoint_path,
                )
                checkpoint_rows.append({"run_id": run_id, "checkpoint_path": str(checkpoint_path)})

                y_pred = (test_score >= threshold).astype(int)
                split_rows = test_df[KEY_COLUMNS + ["split_name", "target_binary"]].copy()
                split_rows["run_id"] = run_id
                split_rows["task_name"] = task_name
                split_rows["model_name"] = model_name
                split_rows["implementation"] = spec.implementation
                split_rows["feature_group"] = feature_group_name
                split_rows["threshold"] = threshold
                split_rows["split_name"] = "test"
                split_rows["y_true"] = y_test.to_numpy()
                split_rows["y_score"] = test_score
                split_rows["y_pred"] = y_pred
                prediction_rows.extend(split_rows.to_dict("records"))

                metrics_rows.append(
                    {
                        "run_id": run_id,
                        "task_name": task_name,
                        "model_name": model_name,
                        "implementation": spec.implementation,
                        "feature_group": feature_group_name,
                        "n_features": int(len(feature_columns)),
                        "threshold": threshold,
                        "validation_pr_auc": val_metrics.pr_auc,
                        "validation_roc_auc": val_metrics.roc_auc,
                        "validation_recall_blast": val_metrics.recall_blast,
                        "test_pr_auc": test_metrics.pr_auc,
                        "test_pr_auc_ci_low": ci["pr_auc"][0],
                        "test_pr_auc_ci_high": ci["pr_auc"][1],
                        "test_roc_auc": test_metrics.roc_auc,
                        "test_roc_auc_ci_low": ci["roc_auc"][0],
                        "test_roc_auc_ci_high": ci["roc_auc"][1],
                        "test_balanced_accuracy": test_metrics.balanced_accuracy,
                        "test_balanced_accuracy_ci_low": ci["balanced_accuracy"][0],
                        "test_balanced_accuracy_ci_high": ci["balanced_accuracy"][1],
                        "test_macro_f1": test_metrics.macro_f1,
                        "test_macro_f1_ci_low": ci["macro_f1"][0],
                        "test_macro_f1_ci_high": ci["macro_f1"][1],
                        "test_recall_blast": test_metrics.recall_blast,
                        "test_recall_blast_ci_low": ci["recall_blast"][0],
                        "test_recall_blast_ci_high": ci["recall_blast"][1],
                        "test_specificity": test_metrics.specificity,
                        "test_specificity_ci_low": ci["specificity"][0],
                        "test_specificity_ci_high": ci["specificity"][1],
                        "test_matthews_corrcoef": test_metrics.matthews_corrcoef,
                        "test_matthews_corrcoef_ci_low": ci["matthews_corrcoef"][0],
                        "test_matthews_corrcoef_ci_high": ci["matthews_corrcoef"][1],
                        "test_brier_score": test_metrics.brier_score,
                        "test_brier_score_ci_low": ci["brier_score"][0],
                        "test_brier_score_ci_high": ci["brier_score"][1],
                        "test_expected_calibration_error": test_metrics.expected_calibration_error,
                        "test_expected_calibration_error_ci_low": ci["expected_calibration_error"][0],
                        "test_expected_calibration_error_ci_high": ci["expected_calibration_error"][1],
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["task_name", "test_pr_auc", "test_recall_blast"],
        ascending=[True, False, False],
    )
    predictions_df = pd.DataFrame(prediction_rows)

    if len(metrics_df) != len(TASK_FILES) * len(MODEL_NAMES) * len(FEATURE_GROUPS):
        raise ValueError("A6 did not produce the expected number of model runs.")
    if predictions_df.empty:
        raise ValueError("A6 predictions are empty.")

    object_metrics_path = TABLES_ROOT / "object_level_metrics_baselines.csv"
    leaderboard_path = TABLES_ROOT / "baseline_model_leaderboard.csv"
    feature_ablation_path = TABLES_ROOT / "feature_group_ablation.csv"
    predictions_path = PREDICTIONS_ROOT / "object_level_baseline_predictions.parquet"
    checkpoint_manifest_path = TABLES_ROOT / "checkpoint_manifest.csv"

    metrics_df.to_csv(object_metrics_path, index=False)
    metrics_df.to_csv(leaderboard_path, index=False)
    metrics_df.to_csv(feature_ablation_path, index=False)
    predictions_df.to_parquet(predictions_path, index=False)
    pd.DataFrame(checkpoint_rows).to_csv(checkpoint_manifest_path, index=False)

    save_pr_roc_summary(predictions_df, FIGURES_ROOT / "pr_curves_baselines.png", FIGURES_ROOT / "roc_curves_baselines.png")
    best_full_row = _select_best_row(metrics_df, task_name="a43_full_realistic_binary")
    best_test_predictions = predictions_df.loc[
        (predictions_df["run_id"] == best_full_row["run_id"]) & (predictions_df["split_name"] == "test")
    ].copy()
    save_confusion_matrix(
        best_test_predictions["y_true"].to_numpy(),
        best_test_predictions["y_pred"].to_numpy(),
        FIGURES_ROOT / "confusion_matrix_best_baseline.png",
        title=f"Best baseline: {best_full_row['model_name']} | {best_full_row['feature_group']}",
    )

    report_lines = [
        "# A6 Baseline QC",
        "",
        f"- Total baseline runs: `{len(metrics_df)}`",
        f"- Prediction rows saved: `{len(predictions_df)}`",
        f"- Leaderboard path: `{leaderboard_path}`",
        f"- Best full-realistic run: `{best_full_row['run_id']}`",
    ]
    write_markdown(REPORTS_ROOT / "a6_baseline_qc.md", report_lines)

    return {
        "object_level_metrics_csv": str(object_metrics_path),
        "feature_group_ablation_csv": str(feature_ablation_path),
        "object_level_predictions_parquet": str(predictions_path),
        "baseline_model_leaderboard_csv": str(leaderboard_path),
        "pr_curves_png": str(FIGURES_ROOT / "pr_curves_baselines.png"),
        "roc_curves_png": str(FIGURES_ROOT / "roc_curves_baselines.png"),
        "confusion_matrix_png": str(FIGURES_ROOT / "confusion_matrix_best_baseline.png"),
        "a6_qc_report_md": str(REPORTS_ROOT / "a6_baseline_qc.md"),
    }


def run_a7(leaderboard_df: pd.DataFrame, full_task_df: pd.DataFrame, features_combined: pd.DataFrame) -> None:
    best_row = _select_best_row(leaderboard_df, task_name="a43_full_realistic_binary")
    feature_group = best_row["feature_group"]
    feature_columns = feature_columns_by_group(features_combined)[feature_group]

    merged_df = merge_task_with_features(full_task_df, features_combined)
    train_df, val_df, test_df = _split_frame(merged_df)
    train_df = _cap_frame(train_df, max_rows=20000, random_state=42)
    val_df = _cap_frame(val_df, max_rows=5000, random_state=43)
    if len(train_df) > 10000:
        train_df = _cap_frame(train_df, max_rows=10000, random_state=44)
    scale_pos_weight = float((train_df["target_binary"].eq(0).sum()) / max(train_df["target_binary"].eq(1).sum(), 1))
    pipeline, spec = build_model(str(best_row["model_name"]), scale_pos_weight=scale_pos_weight, use_class_weight=True)
    X_train = train_df[feature_columns]
    y_train = train_df["target_binary"].astype(int)
    if len(test_df) > 2000:
        test_df = test_df.sample(n=2000, random_state=42)
    X_test = test_df[feature_columns]
    y_test = test_df["target_binary"].astype(int)
    pipeline.fit(X_train, y_train)

    importance = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=3,
        random_state=42,
        n_jobs=1,
        scoring="average_precision",
    )
    stability_rows: list[dict[str, object]] = []
    for repeat_idx in range(3):
        sample_df = train_df.sample(frac=0.6, replace=True, random_state=42 + repeat_idx)
        sample_pipeline, _ = build_model(str(best_row["model_name"]), scale_pos_weight=scale_pos_weight, use_class_weight=True)
        sample_pipeline.fit(sample_df[feature_columns], sample_df["target_binary"].astype(int))
        sample_importance = permutation_importance(
            sample_pipeline,
            X_test,
            y_test,
            n_repeats=2,
            random_state=100 + repeat_idx,
            n_jobs=1,
            scoring="average_precision",
        )
        ranking = np.argsort(-sample_importance.importances_mean)[:20]
        for rank_position, index in enumerate(ranking, start=1):
            stability_rows.append(
                {
                    "repeat_id": repeat_idx,
                    "feature_name": feature_columns[index],
                    "importance_mean": float(sample_importance.importances_mean[index]),
                    "importance_std": float(sample_importance.importances_std[index]),
                    "rank_position": rank_position,
                }
            )

    stability_df = pd.DataFrame(stability_rows)
    top_features_df = (
        stability_df.groupby("feature_name")
        .agg(
            top20_frequency=("feature_name", "size"),
            mean_importance=("importance_mean", "mean"),
            best_rank=("rank_position", "min"),
        )
        .reset_index()
        .sort_values(["top20_frequency", "mean_importance"], ascending=[False, False])
    )
    top_features_path = TABLES_ROOT / "top_features_stability.csv"
    top_features_df.to_csv(top_features_path, index=False)

    import matplotlib.pyplot as plt

    top_plot_df = top_features_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_plot_df["feature_name"], top_plot_df["top20_frequency"], color="#4c72b0")
    ax.set_title("Feature stability across repeated importance runs")
    ax.set_xlabel("Top-20 frequency")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "feature_importance_stability.png", dpi=180)
    plt.close(fig)

    shap_like_df = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    top_shap_df = shap_like_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_shap_df["feature_name"], top_shap_df["importance_mean"], xerr=top_shap_df["importance_std"], color="#55a868")
    ax.set_title("Top permutation-importance features for the best model")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FIGURES_ROOT / "shap_summary_top_model.png", dpi=180)
    plt.close(fig)

    notes_lines = [
        "# Feature Interpretation Notes",
        "",
        f"- Selected model: `{best_row['run_id']}`",
        f"- Implementation: `{spec.implementation}`",
        "",
        "## Stable top features",
        "",
    ]
    for _, row in top_features_df.head(15).iterrows():
        notes_lines.append(
            f"- `{row['feature_name']}`: frequency `{int(row['top20_frequency'])}`, mean importance `{row['mean_importance']:.6f}`."
        )
    write_markdown(REPORTS_ROOT / "feature_interpretation_notes.md", notes_lines)


def _evaluate_single_configuration(
    task_name: str,
    task_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    use_class_weight: bool,
    label: str,
) -> dict[str, object]:
    merged_df = merge_task_with_features(task_df, feature_df)
    train_df, val_df, test_df = _split_frame(merged_df)
    train_df = _cap_frame(train_df, max_rows=20000, random_state=42)
    val_df = _cap_frame(val_df, max_rows=5000, random_state=43)
    scale_pos_weight = float((train_df["target_binary"].eq(0).sum()) / max(train_df["target_binary"].eq(1).sum(), 1))
    pipeline, spec = build_model(model_name, scale_pos_weight=scale_pos_weight, use_class_weight=use_class_weight)
    val_score = fit_predict_probability(pipeline, train_df[feature_columns], train_df["target_binary"], val_df[feature_columns])
    threshold = optimize_threshold(val_df["target_binary"].to_numpy(), val_score)
    test_score = np.asarray(pipeline.predict_proba(test_df[feature_columns])[:, 1], dtype=float) if hasattr(pipeline, "predict_proba") else fit_predict_probability(pipeline, train_df[feature_columns], train_df["target_binary"], test_df[feature_columns])
    metrics = compute_binary_metrics(test_df["target_binary"].to_numpy(), test_score, threshold)
    return {
        "task_name": task_name,
        "label": label,
        "model_name": model_name,
        "implementation": spec.implementation,
        "use_class_weight": use_class_weight,
        "n_rows": int(len(test_df)),
        "n_features": int(len(feature_columns)),
        "pr_auc": metrics.pr_auc,
        "roc_auc": metrics.roc_auc,
        "balanced_accuracy": metrics.balanced_accuracy,
        "macro_f1": metrics.macro_f1,
        "recall_blast": metrics.recall_blast,
        "specificity": metrics.specificity,
        "matthews_corrcoef": metrics.matthews_corrcoef,
        "brier_score": metrics.brier_score,
        "expected_calibration_error": metrics.expected_calibration_error,
    }


def run_a8_mask_usage(
    leaderboard_df: pd.DataFrame,
    tasks: dict[str, pd.DataFrame],
    features_combined: pd.DataFrame,
    features_bbox: pd.DataFrame,
    features_mask: pd.DataFrame,
) -> pd.DataFrame:
    best_full_row = _select_best_row(leaderboard_df, task_name="a43_full_realistic_binary")
    best_model_name = str(best_full_row["model_name"])
    combined_full_columns = feature_columns_by_group(features_combined)["full"]
    bbox_full_columns = [column for column in features_bbox.columns if column not in KEY_COLUMNS]
    mask_full_columns = [column for column in features_mask.columns if column not in KEY_COLUMNS]

    mask_ablation_rows: list[dict[str, object]] = []
    for task_name, task_df in tasks.items():
        mask_keys = set(
            tuple(row)
            for row in features_mask[KEY_COLUMNS[:3]].drop_duplicates().itertuples(index=False, name=None)
        )
        task_mask_ready = task_df.loc[
            task_df[KEY_COLUMNS[:3]].apply(tuple, axis=1).isin(mask_keys)
        ].copy()
        for label, feature_df, feature_columns in [
            ("bbox_only", features_bbox, bbox_full_columns),
            ("mask_only", features_mask, mask_full_columns),
            ("bbox_plus_mask", features_combined, combined_full_columns),
        ]:
            mask_ablation_rows.append(
                _evaluate_single_configuration(
                    task_name=task_name,
                    task_df=task_mask_ready,
                    feature_df=feature_df,
                    feature_columns=feature_columns,
                    model_name=best_model_name,
                    use_class_weight=True,
                    label=label,
                )
            )
    mask_ablation_df = pd.DataFrame(mask_ablation_rows)
    mask_ablation_df.to_csv(TABLES_ROOT / "ablation_mask_usage.csv", index=False)
    return mask_ablation_df


def run_a8_dataset_regimes(
    leaderboard_df: pd.DataFrame,
    tasks: dict[str, pd.DataFrame],
    features_combined: pd.DataFrame,
) -> pd.DataFrame:
    best_full_row = _select_best_row(leaderboard_df, task_name="a43_full_realistic_binary")
    best_model_name = str(best_full_row["model_name"])
    combined_full_columns = feature_columns_by_group(features_combined)["full"]

    dataset_ablation_rows: list[dict[str, object]] = []
    for task_name, task_df in tasks.items():
        dataset_ablation_rows.append(
            _evaluate_single_configuration(
                task_name=task_name,
                task_df=task_df,
                feature_df=features_combined,
                feature_columns=combined_full_columns,
                model_name=best_model_name,
                use_class_weight=True,
                label="combined_with_class_weight",
            )
        )
    dataset_ablation_rows.append(
        _evaluate_single_configuration(
            task_name="a43_full_realistic_binary",
            task_df=tasks["a43_full_realistic_binary"],
            feature_df=features_combined,
            feature_columns=combined_full_columns,
            model_name=best_model_name,
            use_class_weight=False,
            label="combined_without_class_weight",
        )
    )
    top_features_df = pd.read_csv(TABLES_ROOT / "top_features_stability.csv").head(30)
    selected_columns = [column for column in top_features_df["feature_name"].tolist() if column in combined_full_columns]
    if not selected_columns:
        selected_columns = combined_full_columns[: min(30, len(combined_full_columns))]
    dataset_ablation_rows.append(
        _evaluate_single_configuration(
            task_name="a43_full_realistic_binary",
            task_df=tasks["a43_full_realistic_binary"],
            feature_df=features_combined,
            feature_columns=selected_columns,
            model_name=best_model_name,
            use_class_weight=True,
            label="combined_top30_features",
        )
    )
    dataset_ablation_df = pd.DataFrame(dataset_ablation_rows)
    dataset_ablation_df.to_csv(TABLES_ROOT / "ablation_dataset_regimes.csv", index=False)
    return dataset_ablation_df


def save_a8_summary_plot(mask_ablation_df: pd.DataFrame, dataset_ablation_df: pd.DataFrame) -> str:
    ablation_plot_df = pd.concat(
        [
            mask_ablation_df.assign(ablation_family="mask_usage"),
            dataset_ablation_df.assign(ablation_family="dataset_regime"),
        ],
        ignore_index=True,
    )
    save_bar_plot(
        ablation_plot_df[["label", "pr_auc"]].drop_duplicates(),
        x="label",
        y="pr_auc",
        hue=None,
        path=FIGURES_ROOT / "ablation_summary.png",
        title="A8 ablation summary (PR-AUC)",
        rotation=35,
    )
    return str(FIGURES_ROOT / "ablation_summary.png")


def run_a8(
    leaderboard_df: pd.DataFrame,
    tasks: dict[str, pd.DataFrame],
    features_combined: pd.DataFrame,
    features_bbox: pd.DataFrame,
    features_mask: pd.DataFrame,
) -> dict[str, str]:
    mask_ablation_df = run_a8_mask_usage(
        leaderboard_df=leaderboard_df,
        tasks=tasks,
        features_combined=features_combined,
        features_bbox=features_bbox,
        features_mask=features_mask,
    )
    dataset_ablation_df = run_a8_dataset_regimes(
        leaderboard_df=leaderboard_df,
        tasks=tasks,
        features_combined=features_combined,
    )
    plot_path = save_a8_summary_plot(mask_ablation_df, dataset_ablation_df)

    return {
        "ablation_mask_usage_csv": str(TABLES_ROOT / "ablation_mask_usage.csv"),
        "ablation_dataset_regimes_csv": str(TABLES_ROOT / "ablation_dataset_regimes.csv"),
        "ablation_summary_png": plot_path,
    }


def run_a6_a8() -> dict[str, dict[str, str]]:
    result_a6 = run_a6()
    features_combined, features_bbox, features_mask, tasks = _load_required_artifacts()
    leaderboard_df = pd.read_csv(TABLES_ROOT / "baseline_model_leaderboard.csv")
    run_a7(leaderboard_df, tasks["a43_full_realistic_binary"], features_combined)
    result_a8 = run_a8(leaderboard_df, tasks, features_combined, features_bbox, features_mask)
    return {"a6": result_a6, "a8": result_a8}


def run_a10_a11() -> dict[str, str]:
    leaderboard_df = pd.read_csv(TABLES_ROOT / "baseline_model_leaderboard.csv")
    predictions_df = pd.read_parquet(PREDICTIONS_ROOT / "object_level_baseline_predictions.parquet")
    best_full_row = _select_best_row(leaderboard_df, task_name="a43_full_realistic_binary")
    best_predictions = predictions_df.loc[
        (predictions_df["run_id"] == best_full_row["run_id"]) & (predictions_df["split_name"] == "test")
    ].copy()
    if best_predictions.empty:
        raise ValueError("No object-level predictions found for A10.")

    image_score_df = aggregate_object_scores(
        best_predictions.assign(image_contains_blast=best_predictions["y_true"]),
        threshold=float(best_full_row["threshold"]),
        group_columns=["patient_id", "image_id"],
        target_column="image_contains_blast",
    )
    patient_score_df = aggregate_object_scores(
        best_predictions.assign(patient_contains_blast=best_predictions["y_true"]),
        threshold=float(best_full_row["threshold"]),
        group_columns=["patient_id"],
        target_column="patient_contains_blast",
    )
    image_metrics_df = compute_aggregate_metrics(image_score_df, target_column="image_contains_blast")
    patient_metrics_df = compute_aggregate_metrics(patient_score_df, target_column="patient_contains_blast")

    image_scores_path = PREDICTIONS_ROOT / "image_level_scores.parquet"
    patient_scores_path = PREDICTIONS_ROOT / "patient_level_scores.parquet"
    image_metrics_path = TABLES_ROOT / "image_level_metrics.csv"
    patient_metrics_path = TABLES_ROOT / "patient_level_ranking_metrics.csv"
    image_score_df.to_parquet(image_scores_path, index=False)
    patient_score_df.to_parquet(patient_scores_path, index=False)
    image_metrics_df.to_csv(image_metrics_path, index=False)
    patient_metrics_df.to_csv(patient_metrics_path, index=False)

    save_ranking_curve(
        image_score_df,
        target_column="image_contains_blast",
        path=FIGURES_ROOT / "image_ranking_recall_curve.png",
        title="Image-level ranking recall",
    )
    save_ranking_curve(
        patient_score_df,
        target_column="patient_contains_blast",
        path=FIGURES_ROOT / "patient_ranking_curve.png",
        title="Patient-level ranking recall",
    )

    subset_summary_df = pd.read_csv(TABLES_ROOT / "subset_summary.csv").assign(section="subset_summary")
    split_summary_df = pd.read_csv(TABLES_ROOT / "split_summary.csv").assign(section="split_summary")
    task_prevalence_df = pd.read_csv(TABLES_ROOT / "task_prevalence_summary.csv").assign(section="task_prevalence")
    dataset_summary_df = pd.concat(
        [
            subset_summary_df.rename(columns={"subset_name": "name"}),
            split_summary_df.rename(columns={"split_name": "name"}),
            task_prevalence_df.rename(columns={"task_name": "name"}),
        ],
        ignore_index=True,
        sort=False,
    )
    dataset_summary_df.to_csv(TABLES_ROOT / "dataset_summary.csv", index=False)

    save_pipeline_overview(FIGURES_ROOT / "pipeline_overview.png")
    roi_manifest = pd.read_parquet(CROPS_ROOT / "roi_manifest.parquet")
    example_rows = roi_manifest.loc[roi_manifest["mask_crop_path"].notna()].head(4)
    save_example_rois(
        [Path(path) for path in example_rows["bbox_crop_path"].tolist()],
        [Path(path) for path in example_rows["mask_crop_path"].tolist()],
        FIGURES_ROOT / "example_rois.png",
    )

    mask_plot_df = subset_summary_df[["subset_name", "mask_ready_ratio"]].copy()
    save_bar_plot(
        mask_plot_df,
        x="subset_name",
        y="mask_ready_ratio",
        hue=None,
        path=FIGURES_ROOT / "mask_coverage.png",
        title="Mask coverage by subset",
    )

    figure_manifest = pd.DataFrame(
        [
            {"figure_name": "pipeline_overview", "path": str(FIGURES_ROOT / "pipeline_overview.png"), "stage": "A11"},
            {"figure_name": "example_rois", "path": str(FIGURES_ROOT / "example_rois.png"), "stage": "A11"},
            {"figure_name": "mask_coverage", "path": str(FIGURES_ROOT / "mask_coverage.png"), "stage": "A11"},
            {"figure_name": "pr_curves_baselines", "path": str(FIGURES_ROOT / "pr_curves_baselines.png"), "stage": "A6"},
            {"figure_name": "roc_curves_baselines", "path": str(FIGURES_ROOT / "roc_curves_baselines.png"), "stage": "A6"},
            {"figure_name": "feature_importance_stability", "path": str(FIGURES_ROOT / "feature_importance_stability.png"), "stage": "A7"},
            {"figure_name": "ablation_summary", "path": str(FIGURES_ROOT / "ablation_summary.png"), "stage": "A8"},
            {"figure_name": "image_ranking_recall_curve", "path": str(FIGURES_ROOT / "image_ranking_recall_curve.png"), "stage": "A10"},
            {"figure_name": "patient_ranking_curve", "path": str(FIGURES_ROOT / "patient_ranking_curve.png"), "stage": "A10"},
        ]
    )
    figure_manifest.to_csv(TABLES_ROOT / "figure_manifest.csv", index=False)

    table_manifest = pd.DataFrame(
        [
            {"table_name": path.stem, "path": str(path)}
            for path in sorted(TABLES_ROOT.glob("*.csv"))
        ]
    )
    table_manifest.to_csv(TABLES_ROOT / "table_manifest.csv", index=False)

    report_lines = [
        "# A10-A11 QC",
        "",
        f"- Best full-realistic baseline: `{best_full_row['run_id']}`",
        f"- Image-level rows: `{len(image_score_df)}`",
        f"- Patient-level rows: `{len(patient_score_df)}`",
        f"- Figure manifest rows: `{len(figure_manifest)}`",
        f"- Table manifest rows: `{len(table_manifest)}`",
    ]
    write_markdown(REPORTS_ROOT / "a10_a11_qc.md", report_lines)

    return {
        "image_level_metrics_csv": str(image_metrics_path),
        "patient_level_ranking_metrics_csv": str(patient_metrics_path),
        "image_level_scores_parquet": str(image_scores_path),
        "patient_level_scores_parquet": str(patient_scores_path),
        "figure_manifest_csv": str(TABLES_ROOT / "figure_manifest.csv"),
        "table_manifest_csv": str(TABLES_ROOT / "table_manifest.csv"),
        "a10_a11_qc_md": str(REPORTS_ROOT / "a10_a11_qc.md"),
    }


def main() -> None:
    result_a6_a8 = run_a6_a8()
    result_a10_a11 = run_a10_a11()
    print(json.dumps({"a6_a8": result_a6_a8, "a10_a11": result_a10_a11}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
