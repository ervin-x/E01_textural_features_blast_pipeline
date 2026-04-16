from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
VENDOR_ROOT = PROJECT_ROOT / "vendor"
VENV_SITE_PACKAGES = sorted((PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages"))

for candidate in [*VENV_SITE_PACKAGES, SRC_ROOT, SCRIPTS_ROOT, VENDOR_ROOT]:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.append(candidate_str)

from build_modeling_workset import build_modeling_workset
from dataset.build_index import MAIN_PROTOCOL_EXCLUDES, build_index_artifacts
from dataset.build_patient_summary import build_patient_summary
from dataset.build_subsets import build_subsets
from dataset.build_tasks import build_tasks
from dataset.make_group_splits import make_group_splits
from features.feature_registry import extract_features
from preprocessing.extract_bbox_crops import extract_bbox_crops
from preprocessing.extract_mask_crops import extract_mask_crops
from run_a6_to_a11 import (
    _load_required_artifacts,
    run_a10_a11,
    run_a6,
    run_a7,
    run_a8_dataset_regimes,
    run_a8_mask_usage,
    save_a8_summary_plot,
)
from run_a9_deep_baselines import run_a9_deep_baselines
from utils.dataset_paths import resolve_dataset_roots
from utils.project import (
    CROPS_ROOT,
    DATASET_INDEX_ROOT,
    FEATURES_ROOT,
    FIGURES_ROOT,
    PREDICTIONS_ROOT,
    PROJECT_ROOT as PROJECT_ROOT_FROM_UTILS,
    REPORTS_ROOT,
    TABLES_ROOT,
    ensure_dir,
    ensure_output_layout,
    utc_now_iso,
    write_json,
    write_markdown,
)


assert PROJECT_ROOT == PROJECT_ROOT_FROM_UTILS

HYPOTHESIS_REPORTS_ROOT = REPORTS_ROOT / "hypotheses"

FOUNDATION_REQUIRED = [
    DATASET_INDEX_ROOT / "patient_inventory.csv",
    DATASET_INDEX_ROOT / "image_index.parquet",
    DATASET_INDEX_ROOT / "object_index.parquet",
    DATASET_INDEX_ROOT / "subset_full_realistic.parquet",
    DATASET_INDEX_ROOT / "subset_clean_cell.parquet",
    DATASET_INDEX_ROOT / "subset_strict_morphology.parquet",
    DATASET_INDEX_ROOT / "subset_mask_ready.parquet",
    PROJECT_ROOT / "outputs" / "splits" / "split_v1.json",
    TABLES_ROOT / "subset_summary.csv",
    TABLES_ROOT / "split_summary.csv",
    TABLES_ROOT / "task_prevalence_summary.csv",
    DATASET_INDEX_ROOT / "task_a41_clean_binary.parquet",
    DATASET_INDEX_ROOT / "task_a42_strict_binary.parquet",
    DATASET_INDEX_ROOT / "task_a43_full_realistic_binary.parquet",
    CROPS_ROOT / "bbox_roi_manifest.parquet",
    CROPS_ROOT / "roi_manifest.parquet",
    FEATURES_ROOT / "features_bbox.parquet",
    FEATURES_ROOT / "features_mask.parquet",
    FEATURES_ROOT / "features_combined.parquet",
    FEATURES_ROOT / "feature_dictionary.md",
    TABLES_ROOT / "feature_missingness.csv",
]

H1_REQUIRED = [
    TABLES_ROOT / "baseline_model_leaderboard.csv",
    TABLES_ROOT / "object_level_metrics_baselines.csv",
    PREDICTIONS_ROOT / "object_level_baseline_predictions.parquet",
    TABLES_ROOT / "top_features_stability.csv",
    REPORTS_ROOT / "feature_interpretation_notes.md",
]

H2_REQUIRED = [TABLES_ROOT / "ablation_mask_usage.csv"]
H3_REQUIRED = [TABLES_ROOT / "ablation_dataset_regimes.csv"]
H4_REQUIRED = [
    TABLES_ROOT / "image_level_metrics.csv",
    TABLES_ROOT / "patient_level_ranking_metrics.csv",
    TABLES_ROOT / "figure_manifest.csv",
    TABLES_ROOT / "table_manifest.csv",
]
H5_REQUIRED = [
    DATASET_INDEX_ROOT / "task_a43_full_realistic_modeling_workset.parquet",
    CROPS_ROOT / "roi_manifest_modeling_workset.parquet",
    TABLES_ROOT / "deep_vs_handcrafted.csv",
    PREDICTIONS_ROOT / "object_level_deep_predictions.parquet",
    FIGURES_ROOT / "deep_training_curves.png",
]


def _ensure_reports_root() -> None:
    ensure_output_layout()
    ensure_dir(HYPOTHESIS_REPORTS_ROOT)


def _write_summary(slug: str, title: str, lines: list[str], payload: dict[str, object]) -> dict[str, str]:
    _ensure_reports_root()
    md_path = HYPOTHESIS_REPORTS_ROOT / f"{slug}_summary.md"
    json_path = HYPOTHESIS_REPORTS_ROOT / f"{slug}_summary.json"
    write_markdown(md_path, [f"# {title}", "", f"Generated at: `{utc_now_iso()}`", "", *lines])
    write_json(json_path, payload)
    return {"summary_md": str(md_path), "summary_json": str(json_path)}


def _require(paths: list[Path], message: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        missing_block = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"{message}\nMissing artifacts:\n{missing_block}")


def _all_exist(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _best_rows_by_task(leaderboard_df: pd.DataFrame) -> pd.DataFrame:
    ranked = leaderboard_df.sort_values(
        ["task_name", "validation_pr_auc", "test_pr_auc", "test_recall_blast"],
        ascending=[True, False, False, False],
    )
    return ranked.groupby("task_name", sort=False).head(1).reset_index(drop=True)


def _task_display_name(task_name: str) -> str:
    mapping = {
        "a41_clean_binary": "A4.1 clean_cell",
        "a42_strict_binary": "A4.2 strict_morphology",
        "a43_full_realistic_binary": "A4.3 full_realistic",
    }
    return mapping.get(task_name, task_name)


def run_foundation(
    data_root: str | Path | None = None,
    masks_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    _ensure_reports_root()
    resolved_data_root, resolved_masks_root, dataset_path_source = resolve_dataset_roots(
        data_root=data_root,
        masks_root=masks_root,
        config_path=config_path,
    )

    steps_run: list[str] = []

    if not _all_exist(
        [
            DATASET_INDEX_ROOT / "patient_inventory.csv",
            DATASET_INDEX_ROOT / "image_index.parquet",
            DATASET_INDEX_ROOT / "object_index.parquet",
        ]
    ):
        build_index_artifacts(
            data_root=resolved_data_root,
            masks_root=resolved_masks_root,
            output_root=PROJECT_ROOT / "outputs",
            main_protocol_excludes=MAIN_PROTOCOL_EXCLUDES,
        )
        steps_run.append("A0")

    if not _all_exist(
        [
            DATASET_INDEX_ROOT / "subset_full_realistic.parquet",
            DATASET_INDEX_ROOT / "subset_clean_cell.parquet",
            DATASET_INDEX_ROOT / "subset_strict_morphology.parquet",
            DATASET_INDEX_ROOT / "subset_mask_ready.parquet",
            TABLES_ROOT / "subset_summary.csv",
        ]
    ):
        build_subsets(
            object_index_path=DATASET_INDEX_ROOT / "object_index.parquet",
            image_index_path=DATASET_INDEX_ROOT / "image_index.parquet",
        )
        steps_run.append("A1")

    patient_summary_path = build_patient_summary(
        patient_inventory_path=DATASET_INDEX_ROOT / "patient_inventory.csv",
        image_index_path=DATASET_INDEX_ROOT / "image_index.parquet",
        object_index_path=DATASET_INDEX_ROOT / "object_index.parquet",
    )

    if not _all_exist(
        [
            PROJECT_ROOT / "outputs" / "splits" / "split_v1.json",
            TABLES_ROOT / "split_summary.csv",
        ]
    ):
        make_group_splits(
            patient_summary_path=patient_summary_path,
            object_index_path=DATASET_INDEX_ROOT / "object_index.parquet",
            random_state=42,
        )
        steps_run.append("A2")

    if not _all_exist(
        [
            DATASET_INDEX_ROOT / "task_a41_clean_binary.parquet",
            DATASET_INDEX_ROOT / "task_a42_strict_binary.parquet",
            DATASET_INDEX_ROOT / "task_a43_full_realistic_binary.parquet",
            TABLES_ROOT / "task_prevalence_summary.csv",
        ]
    ):
        build_tasks(PROJECT_ROOT / "outputs" / "splits" / "split_v1.json")
        steps_run.append("A4")

    if not (CROPS_ROOT / "bbox_roi_manifest.parquet").exists():
        extract_bbox_crops(DATASET_INDEX_ROOT / "subset_full_realistic.parquet")
        steps_run.append("A3_bbox")

    if not (CROPS_ROOT / "roi_manifest.parquet").exists():
        extract_mask_crops(CROPS_ROOT / "bbox_roi_manifest.parquet")
        steps_run.append("A3_mask")

    if not _all_exist(
        [
            FEATURES_ROOT / "features_bbox.parquet",
            FEATURES_ROOT / "features_mask.parquet",
            FEATURES_ROOT / "features_combined.parquet",
            FEATURES_ROOT / "feature_dictionary.md",
            TABLES_ROOT / "feature_missingness.csv",
        ]
    ):
        extract_features(CROPS_ROOT / "roi_manifest.parquet")
        steps_run.append("A5")

    _require(FOUNDATION_REQUIRED, "Foundation validation failed after stepwise preparation.")

    patient_inventory_df = pd.read_csv(DATASET_INDEX_ROOT / "patient_inventory.csv")
    image_index_df = pd.read_parquet(DATASET_INDEX_ROOT / "image_index.parquet")
    object_index_df = pd.read_parquet(DATASET_INDEX_ROOT / "object_index.parquet")
    subset_summary_df = pd.read_csv(TABLES_ROOT / "subset_summary.csv")
    split_summary_df = pd.read_csv(TABLES_ROOT / "split_summary.csv")
    feature_missingness_df = pd.read_csv(TABLES_ROOT / "feature_missingness.csv")
    combined_missingness_df = feature_missingness_df.loc[feature_missingness_df["matrix"].eq("combined")].copy()
    bbox_manifest_df = pd.read_parquet(CROPS_ROOT / "bbox_roi_manifest.parquet")
    roi_manifest_df = pd.read_parquet(CROPS_ROOT / "roi_manifest.parquet")
    features_combined_df = pd.read_parquet(FEATURES_ROOT / "features_combined.parquet")

    main_patient_df = patient_inventory_df.loc[patient_inventory_df["include_in_main_protocol"].eq(True)].copy()
    main_image_df = image_index_df.loc[
        image_index_df["include_in_main_protocol"].eq(True) & image_index_df["has_image"].eq(True)
    ].copy()
    main_object_df = object_index_df.loc[object_index_df["include_in_main_protocol"].eq(True)].copy()

    lines = [
        "## Что запускает этот блок",
        "",
        "- `A0`: аудит данных и построение индексов.",
        "- `A1`: формирование подмножеств `full_realistic`, `clean_cell`, `strict_morphology`, `mask_ready`.",
        "- `A2`: patient-level split `train / val / test`.",
        "- `A4`: подготовка трех бинарных задач.",
        "- `A3`: построение ROI по bbox и маскам.",
        "- `A5`: извлечение признаков `bbox`, `mask`, `combined`.",
        "",
        "## Ключевые результаты",
        "",
        f"- Dataset path source: `{dataset_path_source}`.",
        f"- Original dataset root: `{resolved_data_root}`.",
        f"- Masks dataset root: `{resolved_masks_root}`.",
        f"- Recomputed steps in this run: `{', '.join(steps_run) if steps_run else 'none; reused existing validated artifacts'}`.",
        f"- Main-protocol objects: `{len(main_object_df)}`.",
        f"- Main-protocol images: `{len(main_image_df)}`.",
        f"- Main-protocol patients: `{len(main_patient_df)}`.",
        f"- Full-realistic subset rows: `{int(subset_summary_df.loc[subset_summary_df['subset_name'].eq('full_realistic'), 'object_count'].iloc[0])}`.",
        f"- ROI rows in bbox manifest: `{len(bbox_manifest_df)}`.",
        f"- ROI rows in mask manifest: `{len(roi_manifest_df)}`.",
        f"- Combined feature rows: `{len(features_combined_df)}`.",
        f"- Combined numeric features: `{len(combined_missingness_df)}`.",
        f"- Combined features with missing values: `{int((combined_missingness_df['missing_fraction'] > 0).sum())}`.",
        "",
        "## Split summary",
        "",
        "| Split | Patients | Active patients | Images | Objects | Blast objects |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in split_summary_df.to_dict("records"):
        lines.append(
            f"| {row['split_name']} | {int(row['patient_count'])} | {int(row['active_patient_count'])} | "
            f"{int(row['image_count'])} | {int(row['object_count'])} | {int(row['blast_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Основные артефакты",
            "",
            f"- Dataset index: `{DATASET_INDEX_ROOT / 'object_index.parquet'}`.",
            f"- Split definition: `{PROJECT_ROOT / 'outputs' / 'splits' / 'split_v1.json'}`.",
            f"- ROI manifest: `{CROPS_ROOT / 'roi_manifest.parquet'}`.",
            f"- Combined features: `{FEATURES_ROOT / 'features_combined.parquet'}`.",
            f"- Feature dictionary: `{FEATURES_ROOT / 'feature_dictionary.md'}`.",
        ]
    )

    payload = {
        "block": "foundation",
        "dataset_roots": {
            "data_root": str(resolved_data_root),
            "masks_root": str(resolved_masks_root),
            "source": dataset_path_source,
        },
        "recomputed_steps": steps_run,
        "patient_summary_csv": str(patient_summary_path),
        "main_protocol_summary": {
            "patient_count": int(len(main_patient_df)),
            "image_count": int(len(main_image_df)),
            "object_count": int(len(main_object_df)),
        },
        "a3": {
            "bbox_manifest_rows": len(bbox_manifest_df),
            "roi_manifest_rows": len(roi_manifest_df),
            "bbox_roi_manifest_parquet": str(CROPS_ROOT / "bbox_roi_manifest.parquet"),
            "roi_manifest_parquet": str(CROPS_ROOT / "roi_manifest.parquet"),
        },
        "a5": {
            "features_bbox": str(FEATURES_ROOT / "features_bbox.parquet"),
            "features_mask": str(FEATURES_ROOT / "features_mask.parquet"),
            "features_combined": str(FEATURES_ROOT / "features_combined.parquet"),
            "feature_dictionary_md": str(FEATURES_ROOT / "feature_dictionary.md"),
            "feature_missingness_csv": str(TABLES_ROOT / "feature_missingness.csv"),
        },
    }
    payload.update(_write_summary("foundation", "Foundation Reproducibility Summary", lines, payload))
    return payload


def run_h1_interpretable_features() -> dict[str, object]:
    _require(
        [
            FEATURES_ROOT / "features_combined.parquet",
            DATASET_INDEX_ROOT / "task_a41_clean_binary.parquet",
            DATASET_INDEX_ROOT / "task_a42_strict_binary.parquet",
            DATASET_INDEX_ROOT / "task_a43_full_realistic_binary.parquet",
        ],
        "H1 requires foundation outputs. Run `scripts/run_foundation.py` first.",
    )

    h1_recomputed: list[str] = []
    if not _all_exist(
        [
            TABLES_ROOT / "baseline_model_leaderboard.csv",
            TABLES_ROOT / "object_level_metrics_baselines.csv",
            PREDICTIONS_ROOT / "object_level_baseline_predictions.parquet",
        ]
    ):
        a6_result = run_a6()
        h1_recomputed.append("A6")
    else:
        a6_result = {
            "baseline_model_leaderboard_csv": str(TABLES_ROOT / "baseline_model_leaderboard.csv"),
            "object_level_metrics_csv": str(TABLES_ROOT / "object_level_metrics_baselines.csv"),
            "object_level_predictions_parquet": str(PREDICTIONS_ROOT / "object_level_baseline_predictions.parquet"),
        }

    features_combined, _, _, tasks = _load_required_artifacts()
    leaderboard_df = pd.read_csv(TABLES_ROOT / "baseline_model_leaderboard.csv")
    if not _all_exist(
        [
            TABLES_ROOT / "top_features_stability.csv",
            REPORTS_ROOT / "feature_interpretation_notes.md",
        ]
    ):
        run_a7(leaderboard_df, tasks["a43_full_realistic_binary"], features_combined)
        h1_recomputed.append("A7")

    _require(H1_REQUIRED, "H1 validation failed after preparation.")
    best_rows = _best_rows_by_task(leaderboard_df)
    top_features_df = pd.read_csv(TABLES_ROOT / "top_features_stability.csv").head(10)

    lines = [
        "## Что проверяет блок H1",
        "",
        "- Полезны ли интерпретируемые признаки формы, цвета и текстуры для различения `blast / non-blast`.",
        "- Какие baseline-модели и группы признаков дают лучший результат на patient-level split.",
        f"- Recomputed steps in this run: `{', '.join(h1_recomputed) if h1_recomputed else 'none; reused existing validated artifacts'}`.",
        "",
        "## Лучшие конфигурации по задачам",
        "",
        "| Task | Best model | Feature group | Validation PR-AUC | Test PR-AUC | Recall blast | Specificity |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in best_rows.to_dict("records"):
        lines.append(
            f"| {_task_display_name(str(row['task_name']))} | {row['model_name']} | {row['feature_group']} | "
            f"{float(row['validation_pr_auc']):.4f} | {float(row['test_pr_auc']):.4f} | "
            f"{float(row['test_recall_blast']):.4f} | {float(row['test_specificity']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Наиболее стабильные признаки для лучшей full_realistic-модели",
            "",
            "| Feature | Top-20 frequency | Mean importance | Best rank |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in top_features_df.to_dict("records"):
        lines.append(
            f"| {row['feature_name']} | {int(row['top20_frequency'])} | "
            f"{float(row['mean_importance']):.6f} | {int(row['best_rank'])} |"
        )
    lines.extend(
        [
            "",
            "## Основные артефакты",
            "",
            f"- Baseline leaderboard: `{TABLES_ROOT / 'baseline_model_leaderboard.csv'}`.",
            f"- Object-level metrics: `{TABLES_ROOT / 'object_level_metrics_baselines.csv'}`.",
            f"- Predictions: `{PREDICTIONS_ROOT / 'object_level_baseline_predictions.parquet'}`.",
            f"- Stable features: `{TABLES_ROOT / 'top_features_stability.csv'}`.",
            f"- Interpretation notes: `{REPORTS_ROOT / 'feature_interpretation_notes.md'}`.",
        ]
    )

    payload = {
        "block": "h1",
        "hypothesis": "interpretable_features_help",
        "recomputed_steps": h1_recomputed,
        "a6": a6_result,
        "a7_report_md": str(REPORTS_ROOT / "feature_interpretation_notes.md"),
        "best_rows": best_rows.to_dict("records"),
        "top_features_preview": top_features_df.to_dict("records"),
    }
    payload.update(_write_summary("h1_interpretable_features", "H1 Interpretable Features Summary", lines, payload))
    return payload


def run_h2_mask_effect() -> dict[str, object]:
    _require(
        [
            TABLES_ROOT / "baseline_model_leaderboard.csv",
            FEATURES_ROOT / "features_bbox.parquet",
            FEATURES_ROOT / "features_mask.parquet",
            FEATURES_ROOT / "features_combined.parquet",
        ],
        "H2 requires foundation outputs and H1 baseline results. Run `scripts/run_foundation.py` and `scripts/run_h1_interpretable_features.py` first.",
    )

    leaderboard_df = pd.read_csv(TABLES_ROOT / "baseline_model_leaderboard.csv")
    features_combined, features_bbox, features_mask, tasks = _load_required_artifacts()

    if not _all_exist(H2_REQUIRED):
        mask_ablation_df = run_a8_mask_usage(
            leaderboard_df=leaderboard_df,
            tasks=tasks,
            features_combined=features_combined,
            features_bbox=features_bbox,
            features_mask=features_mask,
        )
        h2_recomputed = True
    else:
        mask_ablation_df = pd.read_csv(TABLES_ROOT / "ablation_mask_usage.csv")
        h2_recomputed = False

    pivot_df = (
        mask_ablation_df.pivot(index="task_name", columns="label", values="pr_auc")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pivot_df["delta_mask_vs_bbox"] = pivot_df["mask_only"] - pivot_df["bbox_only"]
    pivot_df["delta_bbox_plus_mask_vs_bbox"] = pivot_df["bbox_plus_mask"] - pivot_df["bbox_only"]

    lines = [
        "## Что проверяет блок H2",
        "",
        "- Улучшает ли использование маски клетки качество признаков и итоговой классификации.",
        f"- Recomputed in this run: `{'A8 mask usage' if h2_recomputed else 'none; reused existing validated artifact'}`.",
        "",
        "## Сравнение по PR-AUC",
        "",
        "| Task | bbox_only | mask_only | bbox_plus_mask | mask_only - bbox_only | bbox_plus_mask - bbox_only |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pivot_df.to_dict("records"):
        lines.append(
            f"| {_task_display_name(str(row['task_name']))} | {float(row['bbox_only']):.4f} | "
            f"{float(row['mask_only']):.4f} | {float(row['bbox_plus_mask']):.4f} | "
            f"{float(row['delta_mask_vs_bbox']):.4f} | {float(row['delta_bbox_plus_mask_vs_bbox']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Основные артефакты",
            "",
            f"- Mask ablation table: `{TABLES_ROOT / 'ablation_mask_usage.csv'}`.",
            f"- Combined ablation figure (если уже собрана совместно с H3): `{FIGURES_ROOT / 'ablation_summary.png'}`.",
        ]
    )

    payload = {
        "block": "h2",
        "hypothesis": "mask_improves_features",
        "recomputed_steps": ["A8_mask_usage"] if h2_recomputed else [],
        "ablation_mask_usage_csv": str(TABLES_ROOT / "ablation_mask_usage.csv"),
        "pr_auc_comparison": pivot_df.to_dict("records"),
    }
    payload.update(_write_summary("h2_mask_effect", "H2 Mask Effect Summary", lines, payload))
    return payload


def run_h3_noise_regimes() -> dict[str, object]:
    _require(
        [
            TABLES_ROOT / "baseline_model_leaderboard.csv",
            TABLES_ROOT / "top_features_stability.csv",
            FEATURES_ROOT / "features_combined.parquet",
        ],
        "H3 requires foundation outputs and H1 interpretation artifacts. Run `scripts/run_foundation.py` and `scripts/run_h1_interpretable_features.py` first.",
    )

    leaderboard_df = pd.read_csv(TABLES_ROOT / "baseline_model_leaderboard.csv")
    features_combined, _, _, tasks = _load_required_artifacts()

    if not _all_exist(H3_REQUIRED):
        dataset_ablation_df = run_a8_dataset_regimes(
            leaderboard_df=leaderboard_df,
            tasks=tasks,
            features_combined=features_combined,
        )
        h3_recomputed = True
    else:
        dataset_ablation_df = pd.read_csv(TABLES_ROOT / "ablation_dataset_regimes.csv")
        h3_recomputed = False

    primary_rows = dataset_ablation_df.loc[
        dataset_ablation_df["label"].eq("combined_with_class_weight")
    ].copy()
    extras_rows = dataset_ablation_df.loc[
        dataset_ablation_df["task_name"].eq("a43_full_realistic_binary")
        & dataset_ablation_df["label"].ne("combined_with_class_weight")
    ].copy()

    lines = [
        "## Что проверяет блок H3",
        "",
        "- Насколько шумовые классы и более реалистичный режим корпуса усложняют задачу.",
        f"- Recomputed in this run: `{'A8 dataset regimes' if h3_recomputed else 'none; reused existing validated artifact'}`.",
        "",
        "## Основные режимы корпуса",
        "",
        "| Task | PR-AUC | ROC-AUC | Recall blast | Specificity |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in primary_rows.to_dict("records"):
        lines.append(
            f"| {_task_display_name(str(row['task_name']))} | {float(row['pr_auc']):.4f} | "
            f"{float(row['roc_auc']):.4f} | {float(row['recall_blast']):.4f} | {float(row['specificity']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Дополнительные ablation-проверки для full_realistic",
            "",
            "| Label | PR-AUC | Recall blast | MCC |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in extras_rows.to_dict("records"):
        lines.append(
            f"| {row['label']} | {float(row['pr_auc']):.4f} | {float(row['recall_blast']):.4f} | "
            f"{float(row['matthews_corrcoef']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Основные артефакты",
            "",
            f"- Dataset regime ablation: `{TABLES_ROOT / 'ablation_dataset_regimes.csv'}`.",
            f"- Combined ablation figure (появляется после совместного запуска H2 и H3 или `all`): `{FIGURES_ROOT / 'ablation_summary.png'}`.",
        ]
    )

    payload = {
        "block": "h3",
        "hypothesis": "noisy_regimes_make_task_harder",
        "recomputed_steps": ["A8_dataset_regimes"] if h3_recomputed else [],
        "ablation_dataset_regimes_csv": str(TABLES_ROOT / "ablation_dataset_regimes.csv"),
        "primary_regime_rows": primary_rows.to_dict("records"),
        "full_realistic_extra_rows": extras_rows.to_dict("records"),
    }
    payload.update(_write_summary("h3_noise_regimes", "H3 Noise Regimes Summary", lines, payload))
    return payload


def run_h4_aggregation() -> dict[str, object]:
    _require(
        [
            TABLES_ROOT / "baseline_model_leaderboard.csv",
            PREDICTIONS_ROOT / "object_level_baseline_predictions.parquet",
        ],
        "H4 requires H1 outputs. Run `scripts/run_h1_interpretable_features.py` first.",
    )

    if not _all_exist(H4_REQUIRED):
        a10_a11_result = run_a10_a11()
        h4_recomputed = True
    else:
        a10_a11_result = {
            "image_level_metrics_csv": str(TABLES_ROOT / "image_level_metrics.csv"),
            "patient_level_ranking_metrics_csv": str(TABLES_ROOT / "patient_level_ranking_metrics.csv"),
            "figure_manifest_csv": str(TABLES_ROOT / "figure_manifest.csv"),
            "table_manifest_csv": str(TABLES_ROOT / "table_manifest.csv"),
        }
        h4_recomputed = False

    image_metrics_df = pd.read_csv(TABLES_ROOT / "image_level_metrics.csv")
    patient_metrics_df = pd.read_csv(TABLES_ROOT / "patient_level_ranking_metrics.csv")
    best_image = image_metrics_df.sort_values(["pr_auc", "roc_auc"], ascending=[False, False]).iloc[0]
    best_patient = patient_metrics_df.sort_values(["pr_auc", "roc_auc"], ascending=[False, False]).iloc[0]

    lines = [
        "## Что проверяет блок H4",
        "",
        "- Можно ли превратить object-level score в полезную оценку изображения и пациента.",
        f"- Recomputed in this run: `{'A10-A11' if h4_recomputed else 'none; reused existing validated artifacts'}`.",
        "",
        "## Лучший image-level метод",
        "",
        f"- Method: `{best_image['aggregation_method']}`.",
        f"- PR-AUC: `{float(best_image['pr_auc']):.4f}`.",
        f"- ROC-AUC: `{float(best_image['roc_auc']):.4f}`.",
        f"- Recall at FPR 10%: `{float(best_image['recall_at_fpr_10pct']):.4f}`.",
        "",
        "## Лучший patient-level метод",
        "",
        f"- Method: `{best_patient['aggregation_method']}`.",
        f"- PR-AUC: `{float(best_patient['pr_auc']):.4f}`.",
        f"- ROC-AUC: `{float(best_patient['roc_auc']):.4f}`.",
        f"- Recall at FPR 10%: `{float(best_patient['recall_at_fpr_10pct']):.4f}`.",
        "",
        "## Основные артефакты",
        "",
        f"- Image-level metrics: `{TABLES_ROOT / 'image_level_metrics.csv'}`.",
        f"- Patient-level metrics: `{TABLES_ROOT / 'patient_level_ranking_metrics.csv'}`.",
        f"- Image ranking figure: `{FIGURES_ROOT / 'image_ranking_recall_curve.png'}`.",
        f"- Patient ranking figure: `{FIGURES_ROOT / 'patient_ranking_curve.png'}`.",
    ]

    payload = {
        "block": "h4",
        "hypothesis": "object_scores_are_useful_after_aggregation",
        "recomputed_steps": ["A10_A11"] if h4_recomputed else [],
        "a10_a11": a10_a11_result,
        "best_image_method": best_image.to_dict(),
        "best_patient_method": best_patient.to_dict(),
    }
    payload.update(_write_summary("h4_aggregation", "H4 Aggregation Summary", lines, payload))
    return payload


def run_h5_deep_vs_handcrafted() -> dict[str, object]:
    _require(
        [
            TABLES_ROOT / "baseline_model_leaderboard.csv",
            FEATURES_ROOT / "features_combined.parquet",
            CROPS_ROOT / "roi_manifest.parquet",
            DATASET_INDEX_ROOT / "task_a43_full_realistic_binary.parquet",
        ],
        "H5 requires foundation outputs and H1 baseline leaderboard. Run `scripts/run_foundation.py` and `scripts/run_h1_interpretable_features.py` first.",
    )

    h5_recomputed: list[str] = []
    if not _all_exist(
        [
            DATASET_INDEX_ROOT / "task_a43_full_realistic_modeling_workset.parquet",
            CROPS_ROOT / "roi_manifest_modeling_workset.parquet",
        ]
    ):
        workset_result = build_modeling_workset()
        h5_recomputed.append("modeling_workset")
    else:
        workset_result = {
            "task_workset_parquet": str(DATASET_INDEX_ROOT / "task_a43_full_realistic_modeling_workset.parquet"),
            "roi_workset_parquet": str(CROPS_ROOT / "roi_manifest_modeling_workset.parquet"),
            "qc_report_md": str(REPORTS_ROOT / "modeling_workset_qc.md"),
        }

    if not _all_exist(H5_REQUIRED):
        a9_result = run_a9_deep_baselines()
        h5_recomputed.append("A9")
    else:
        a9_result = {
            "deep_vs_handcrafted_csv": str(TABLES_ROOT / "deep_vs_handcrafted.csv"),
            "deep_predictions_parquet": str(PREDICTIONS_ROOT / "object_level_deep_predictions.parquet"),
            "training_curves_png": str(FIGURES_ROOT / "deep_training_curves.png"),
            "a9_qc_md": str(REPORTS_ROOT / "a9_deep_qc.md"),
        }

    _require(H5_REQUIRED, "H5 validation failed after preparation.")
    comparison_df = pd.read_csv(TABLES_ROOT / "deep_vs_handcrafted.csv")
    handcrafted_row = comparison_df.loc[comparison_df["source_type"].eq("handcrafted")].iloc[0]
    best_deep_row = (
        comparison_df.loc[comparison_df["source_type"].eq("deep")]
        .sort_values(["test_pr_auc", "test_roc_auc"], ascending=[False, False])
        .iloc[0]
    )
    delta_pr_auc = float(best_deep_row["test_pr_auc"]) - float(handcrafted_row["test_pr_auc"])

    lines = [
        "## Что проверяет блок H5",
        "",
        "- Как lightweight deep-baseline соотносится с лучшим handcrafted baseline на тех же split и данных.",
        f"- Recomputed in this run: `{', '.join(h5_recomputed) if h5_recomputed else 'none; reused existing validated artifacts'}`.",
        "",
        "## Сравнение лучшего handcrafted и лучшего deep baseline",
        "",
        f"- Best handcrafted: `{handcrafted_row['model_name']}` with test PR-AUC `{float(handcrafted_row['test_pr_auc']):.4f}`.",
        f"- Best deep: `{best_deep_row['model_name']}` with test PR-AUC `{float(best_deep_row['test_pr_auc']):.4f}`.",
        f"- PR-AUC delta (deep - handcrafted): `{delta_pr_auc:.4f}`.",
        "",
        "## Полная таблица сравнения",
        "",
        "| Model | Source | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Recall blast | MCC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_df.to_dict("records"):
        lines.append(
            f"| {row['model_name']} | {row['source_type']} | {float(row['validation_pr_auc']):.4f} | "
            f"{float(row['test_pr_auc']):.4f} | {float(row['test_roc_auc']):.4f} | "
            f"{float(row['test_recall_blast']):.4f} | {float(row['test_matthews_corrcoef']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Основные артефакты",
            "",
            f"- Modeling workset QC: `{REPORTS_ROOT / 'modeling_workset_qc.md'}`.",
            f"- Deep comparison table: `{TABLES_ROOT / 'deep_vs_handcrafted.csv'}`.",
            f"- Deep predictions: `{PREDICTIONS_ROOT / 'object_level_deep_predictions.parquet'}`.",
            f"- Training curves: `{FIGURES_ROOT / 'deep_training_curves.png'}`.",
        ]
    )

    payload = {
        "block": "h5",
        "hypothesis": "deep_vs_handcrafted",
        "recomputed_steps": h5_recomputed,
        "workset": workset_result,
        "a9": a9_result,
        "handcrafted_row": handcrafted_row.to_dict(),
        "best_deep_row": best_deep_row.to_dict(),
        "delta_pr_auc": delta_pr_auc,
    }
    payload.update(_write_summary("h5_deep_vs_handcrafted", "H5 Deep vs Handcrafted Summary", lines, payload))
    return payload


def run_all_hypotheses(
    data_root: str | Path | None = None,
    masks_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    foundation = run_foundation(data_root=data_root, masks_root=masks_root, config_path=config_path)
    h1 = run_h1_interpretable_features()
    h2 = run_h2_mask_effect()
    h3 = run_h3_noise_regimes()
    if (TABLES_ROOT / "ablation_mask_usage.csv").exists() and (TABLES_ROOT / "ablation_dataset_regimes.csv").exists():
        save_a8_summary_plot(
            pd.read_csv(TABLES_ROOT / "ablation_mask_usage.csv"),
            pd.read_csv(TABLES_ROOT / "ablation_dataset_regimes.csv"),
        )
    h4 = run_h4_aggregation()
    h5 = run_h5_deep_vs_handcrafted()

    lines = [
        "## Выполненные блоки",
        "",
        f"- Foundation: `{foundation['summary_md']}`.",
        f"- H1: `{h1['summary_md']}`.",
        f"- H2: `{h2['summary_md']}`.",
        f"- H3: `{h3['summary_md']}`.",
        f"- H4: `{h4['summary_md']}`.",
        f"- H5: `{h5['summary_md']}`.",
        "",
        "## Сводные артефакты полного прогона",
        "",
        f"- Combined ablation figure: `{FIGURES_ROOT / 'ablation_summary.png'}`.",
        f"- Figure manifest: `{TABLES_ROOT / 'figure_manifest.csv'}`.",
        f"- Table manifest: `{TABLES_ROOT / 'table_manifest.csv'}`.",
        f"- Full detailed report: `{PROJECT_ROOT / 'reports' / 'E01_detailed_report.md'}`.",
    ]

    payload = {
        "block": "all",
        "foundation": foundation,
        "h1": h1,
        "h2": h2,
        "h3": h3,
        "h4": h4,
        "h5": h5,
    }
    payload.update(_write_summary("run_all_hypotheses", "Run-All Hypotheses Summary", lines, payload))
    return payload


def print_payload(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
