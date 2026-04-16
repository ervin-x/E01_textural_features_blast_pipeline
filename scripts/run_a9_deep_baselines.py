from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MPLCONFIGDIR = Path(__file__).resolve().parents[1] / "outputs" / "logs" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
VENDOR_ROOT = PROJECT_ROOT / "vendor"
VENV_SITE_PACKAGES = sorted((PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages"))

for candidate in [*VENV_SITE_PACKAGES, SRC_ROOT, VENDOR_ROOT]:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.append(candidate_str)

from models.baseline_pipeline import KEY_COLUMNS
from models.deep_baselines import save_training_history, train_deep_baseline
from utils.project import CHECKPOINTS_ROOT, FEATURES_ROOT, FIGURES_ROOT, PREDICTIONS_ROOT, REPORTS_ROOT, TABLES_ROOT, ensure_output_layout, write_markdown


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


def _split_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        df.loc[df["split_name"] == "train"].copy(),
        df.loc[df["split_name"] == "val"].copy(),
        df.loc[df["split_name"] == "test"].copy(),
    )


def _prepare_tabular_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not feature_columns:
        return train_df, val_df, test_df

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    for frame in [train_df, val_df, test_df]:
        for column in feature_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    medians = train_df[feature_columns].replace([float("inf"), float("-inf")], pd.NA).median()
    train_df[feature_columns] = train_df[feature_columns].replace([float("inf"), float("-inf")], pd.NA).fillna(medians)
    val_df[feature_columns] = val_df[feature_columns].replace([float("inf"), float("-inf")], pd.NA).fillna(medians)
    test_df[feature_columns] = test_df[feature_columns].replace([float("inf"), float("-inf")], pd.NA).fillna(medians)

    means = train_df[feature_columns].mean()
    stds = train_df[feature_columns].std().replace(0.0, 1.0).fillna(1.0)
    train_df[feature_columns] = (train_df[feature_columns] - means) / stds
    val_df[feature_columns] = (val_df[feature_columns] - means) / stds
    test_df[feature_columns] = (test_df[feature_columns] - means) / stds

    return train_df, val_df, test_df


def _comparison_row_from_handcrafted(best_handcrafted: pd.Series) -> dict[str, object]:
    return {
        "model_name": "handcrafted_best_full_realistic",
        "source_type": "handcrafted",
        "reference_run_id": str(best_handcrafted["run_id"]),
        "reference_model_name": str(best_handcrafted["model_name"]),
        "implementation": str(best_handcrafted["implementation"]),
        "feature_group": str(best_handcrafted["feature_group"]),
        "validation_pr_auc": float(best_handcrafted["validation_pr_auc"]),
        "test_pr_auc": float(best_handcrafted["test_pr_auc"]),
        "test_roc_auc": float(best_handcrafted["test_roc_auc"]),
        "test_balanced_accuracy": float(best_handcrafted["test_balanced_accuracy"]),
        "test_macro_f1": float(best_handcrafted["test_macro_f1"]),
        "test_recall_blast": float(best_handcrafted["test_recall_blast"]),
        "test_specificity": float(best_handcrafted["test_specificity"]),
        "test_matthews_corrcoef": float(best_handcrafted["test_matthews_corrcoef"]),
        "test_brier_score": float(best_handcrafted["test_brier_score"]),
        "test_expected_calibration_error": float(best_handcrafted["test_expected_calibration_error"]),
    }


def _comparison_row_from_deep(model_name: str, result) -> dict[str, object]:
    return {
        "model_name": model_name,
        "source_type": "deep",
        "reference_run_id": None,
        "reference_model_name": model_name,
        "implementation": "small_cnn" if model_name != "cnn_late_fusion" else "small_cnn_late_fusion",
        "feature_group": "image_only" if model_name != "cnn_late_fusion" else "image_plus_tabular",
        "validation_pr_auc": max((float(epoch["val_pr_auc"]) for epoch in result.history), default=float("nan")),
        "test_pr_auc": float(result.test_metrics["pr_auc"]),
        "test_roc_auc": float(result.test_metrics["roc_auc"]),
        "test_balanced_accuracy": float(result.test_metrics["balanced_accuracy"]),
        "test_macro_f1": float(result.test_metrics["macro_f1"]),
        "test_recall_blast": float(result.test_metrics["recall_blast"]),
        "test_specificity": float(result.test_metrics["specificity"]),
        "test_matthews_corrcoef": float(result.test_metrics["matthews_corrcoef"]),
        "test_brier_score": float(result.test_metrics["brier_score"]),
        "test_expected_calibration_error": float(result.test_metrics["expected_calibration_error"]),
    }


def run_a9_deep_baselines() -> dict[str, str]:
    ensure_output_layout()
    task_path = PROJECT_ROOT / "outputs" / "dataset_index" / "task_a43_full_realistic_modeling_workset.parquet"
    if not task_path.exists():
        task_path = PROJECT_ROOT / "outputs" / "dataset_index" / "task_a43_full_realistic_binary.parquet"
    roi_path = PROJECT_ROOT / "outputs" / "crops" / "roi_manifest_modeling_workset.parquet"
    if not roi_path.exists():
        roi_path = PROJECT_ROOT / "outputs" / "crops" / "roi_manifest.parquet"
    task_df = pd.read_parquet(task_path)
    roi_df = pd.read_parquet(roi_path)
    feature_path = FEATURES_ROOT / "features_combined.parquet"
    if not feature_path.exists():
        feature_path = FEATURES_ROOT / "features_combined_fast.parquet"
    if not feature_path.exists():
        feature_path = FEATURES_ROOT / "features_combined_lite.parquet"
    feature_df = pd.read_parquet(feature_path)
    handcrafted_df = pd.read_csv(TABLES_ROOT / "baseline_model_leaderboard.csv")

    merged_roi = task_df.merge(
        roi_df[KEY_COLUMNS + ["bbox_crop_path", "mask_crop_path"]].copy(),
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    mask_ready_df = merged_roi.loc[merged_roi["mask_crop_path"].notna()].copy()
    fusion_df = mask_ready_df.merge(feature_df, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    mask_ready_df = mask_ready_df.loc[mask_ready_df["target_binary"].notna()].copy()
    fusion_df = fusion_df.loc[fusion_df["target_binary"].notna()].copy()
    tabular_columns = [
        column
        for column in feature_df.columns
        if column not in KEY_COLUMNS and pd.api.types.is_numeric_dtype(feature_df[column])
    ]

    bbox_train, bbox_val, bbox_test = _split_frame(mask_ready_df)
    fusion_train, fusion_val, fusion_test = _split_frame(fusion_df)
    bbox_train = _cap_frame(bbox_train, max_rows=5000, random_state=42)
    bbox_val = _cap_frame(bbox_val, max_rows=1000, random_state=43)
    fusion_train = _cap_frame(fusion_train, max_rows=5000, random_state=42)
    fusion_val = _cap_frame(fusion_val, max_rows=1000, random_state=43)
    fusion_train, fusion_val, fusion_test = _prepare_tabular_splits(
        fusion_train,
        fusion_val,
        fusion_test,
        tabular_columns,
    )

    bbox_result = train_deep_baseline(
        bbox_train,
        bbox_val,
        bbox_test,
        image_column="bbox_crop_path",
        feature_columns=None,
        checkpoint_path=CHECKPOINTS_ROOT / "cnn_bbox_best.pt",
        max_epochs=6,
        patience=2,
    )
    mask_result = train_deep_baseline(
        bbox_train,
        bbox_val,
        bbox_test,
        image_column="mask_crop_path",
        feature_columns=None,
        checkpoint_path=CHECKPOINTS_ROOT / "cnn_mask_best.pt",
        max_epochs=6,
        patience=2,
    )
    fusion_result = train_deep_baseline(
        fusion_train,
        fusion_val,
        fusion_test,
        image_column="bbox_crop_path",
        feature_columns=tabular_columns,
        checkpoint_path=CHECKPOINTS_ROOT / "cnn_late_fusion_best.pt",
        max_epochs=6,
        patience=2,
    )

    best_handcrafted = handcrafted_df.loc[
        handcrafted_df["task_name"] == "a43_full_realistic_binary"
    ].sort_values(["validation_pr_auc", "test_pr_auc"], ascending=[False, False]).iloc[0]

    comparison_rows = [
        _comparison_row_from_handcrafted(best_handcrafted),
        _comparison_row_from_deep("cnn_bbox", bbox_result),
        _comparison_row_from_deep("cnn_mask", mask_result),
        _comparison_row_from_deep("cnn_late_fusion", fusion_result),
    ]
    deep_vs_path = TABLES_ROOT / "deep_vs_handcrafted.csv"
    pd.DataFrame(comparison_rows).to_csv(deep_vs_path, index=False)

    prediction_rows = []
    for model_name, result in [
        ("cnn_bbox", bbox_result),
        ("cnn_mask", mask_result),
        ("cnn_late_fusion", fusion_result),
    ]:
        frame = result.test_predictions[KEY_COLUMNS + ["split_name", "target_binary", "y_true", "y_score", "y_pred"]].copy()
        frame["model_name"] = model_name
        prediction_rows.append(frame)
    deep_predictions_path = PREDICTIONS_ROOT / "object_level_deep_predictions.parquet"
    pd.concat(prediction_rows, ignore_index=True).to_parquet(deep_predictions_path, index=False)

    history_payload = {
        "cnn_bbox": bbox_result.history,
        "cnn_mask": mask_result.history,
        "cnn_late_fusion": fusion_result.history,
    }
    history_path = PROJECT_ROOT / "outputs" / "logs" / "cnn_training_history.json"
    save_training_history(history_payload, history_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name, result in [
        ("cnn_bbox", bbox_result),
        ("cnn_mask", mask_result),
        ("cnn_late_fusion", fusion_result),
    ]:
        history_df = pd.DataFrame(result.history)
        axes[0].plot(history_df["epoch"], history_df["train_loss"], label=f"{model_name} train")
        axes[0].plot(history_df["epoch"], history_df["val_loss"], linestyle="--", label=f"{model_name} val")
        axes[1].plot(history_df["epoch"], history_df["val_pr_auc"], label=model_name)
    axes[0].set_title("Deep training loss")
    axes[1].set_title("Validation PR-AUC")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend(fontsize=8)
    fig.tight_layout()
    training_curves_path = FIGURES_ROOT / "deep_training_curves.png"
    fig.savefig(training_curves_path, dpi=180)
    plt.close(fig)

    report_lines = [
        "# A9 Deep Baseline QC",
        "",
        f"- Task source: `{task_path}`",
        f"- ROI source: `{roi_path}`",
        f"- Device: `CPU fallback`",
        f"- Mask-ready full-realistic rows: `{len(mask_ready_df)}`",
        f"- Fusion rows: `{len(fusion_df)}`",
        f"- BBox checkpoint: `{bbox_result.model_path}`",
        f"- Mask checkpoint: `{mask_result.model_path}`",
        f"- Fusion checkpoint: `{fusion_result.model_path}`",
        f"- Deep comparison table: `{deep_vs_path}`",
        f"- Best handcrafted reference run: `{best_handcrafted['run_id']}`",
    ]
    write_markdown(REPORTS_ROOT / "a9_deep_qc.md", report_lines)

    return {
        "deep_vs_handcrafted_csv": str(deep_vs_path),
        "deep_predictions_parquet": str(deep_predictions_path),
        "bbox_checkpoint": bbox_result.model_path,
        "mask_checkpoint": mask_result.model_path,
        "fusion_checkpoint": fusion_result.model_path,
        "history_json": str(history_path),
        "training_curves_png": str(training_curves_path),
        "a9_qc_md": str(REPORTS_ROOT / "a9_deep_qc.md"),
    }


def main() -> None:
    print(json.dumps(run_a9_deep_baselines(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
