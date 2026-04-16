from __future__ import annotations

import os
from pathlib import Path

MPLCONFIGDIR = Path(__file__).resolve().parents[2] / "outputs" / "logs" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

from evaluation.metrics import ranking_recall_curve


def save_pr_roc_summary(prediction_df: pd.DataFrame, pr_path: Path, roc_path: Path) -> None:
    top_runs = (
        prediction_df.groupby(["task_name", "model_name", "feature_group"], as_index=False)["y_score"]
        .mean()
        .sort_values(["task_name", "y_score"], ascending=[True, False])
        .groupby("task_name", group_keys=False)
        .head(2)
    )

    fig_pr, ax_pr = plt.subplots(figsize=(10, 8))
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    for _, run in top_runs.iterrows():
        run_df = prediction_df.loc[
            (prediction_df["task_name"] == run["task_name"])
            & (prediction_df["model_name"] == run["model_name"])
            & (prediction_df["feature_group"] == run["feature_group"])
        ]
        label = f"{run['task_name']} | {run['model_name']} | {run['feature_group']}"
        PrecisionRecallDisplay.from_predictions(
            run_df["y_true"],
            run_df["y_score"],
            name=label,
            ax=ax_pr,
        )
        RocCurveDisplay.from_predictions(
            run_df["y_true"],
            run_df["y_score"],
            name=label,
            ax=ax_roc,
        )
    fig_pr.tight_layout()
    fig_pr.savefig(pr_path, dpi=180)
    plt.close(fig_pr)

    fig_roc.tight_layout()
    fig_roc.savefig(roc_path, dpi=180)
    plt.close(fig_roc)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=[0, 1]), display_labels=["non-blast", "blast"]).plot(
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_bar_plot(data: pd.DataFrame, x: str, y: str, hue: str | None, path: Path, title: str, rotation: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if hue is None:
        ax.bar(data[x].astype(str), data[y].astype(float))
    else:
        pivot = data.pivot(index=x, columns=hue, values=y).fillna(0.0)
        pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=rotation)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_ranking_curve(score_df: pd.DataFrame, target_column: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for method, group in score_df.groupby("aggregation_method", sort=False):
        screened_fraction, cumulative_recall = ranking_recall_curve(
            group[target_column].astype(int).to_numpy(),
            group["score"].astype(float).to_numpy(),
        )
        ax.plot(screened_fraction, cumulative_recall, label=method)
    ax.set_xlabel("Screened fraction")
    ax.set_ylabel("Recall of positive groups")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_pipeline_overview(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    boxes = [
        (0.05, 0.5, "Data\n(images, labels, masks)"),
        (0.23, 0.5, "A0-A2\nindexing and split"),
        (0.41, 0.5, "A3-A5\nROI and features"),
        (0.59, 0.5, "A6-A9\nobject-level models"),
        (0.77, 0.5, "A10\nimage/patient aggregation"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f2f2f2", "edgecolor": "#333333"},
            transform=ax.transAxes,
        )
    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate(
            "",
            xy=(end[0] - 0.06, end[1]),
            xytext=(start[0] + 0.08, start[1]),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "linewidth": 2.0},
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_example_rois(bbox_paths: list[Path], mask_paths: list[Path], path: Path) -> None:
    pairs = list(zip(bbox_paths[:4], mask_paths[:4], strict=False))
    if not pairs:
        raise ValueError("No ROI pairs available for visualization.")

    fig, axes = plt.subplots(len(pairs), 2, figsize=(8, 2.5 * len(pairs)))
    if len(pairs) == 1:
        axes = np.asarray([axes])
    for row_idx, (bbox_path, mask_path) in enumerate(pairs):
        bbox_img = np.asarray(Image.open(bbox_path).convert("RGB"))
        mask_img = np.asarray(Image.open(mask_path).convert("RGB"))
        axes[row_idx, 0].imshow(bbox_img)
        axes[row_idx, 0].set_title("bbox crop")
        axes[row_idx, 1].imshow(mask_img)
        axes[row_idx, 1].set_title("mask crop")
        axes[row_idx, 0].axis("off")
        axes[row_idx, 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
