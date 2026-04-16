from __future__ import annotations

import argparse
from pathlib import Path

from utils.project import DATASET_INDEX_ROOT, REPORTS_ROOT, TABLES_ROOT, ensure_output_layout, write_markdown

import numpy as np
import pandas as pd


OBJECT_INDEX_PATH = DATASET_INDEX_ROOT / "object_index.parquet"
IMAGE_INDEX_PATH = DATASET_INDEX_ROOT / "image_index.parquet"

SUBSET_RULES = {
    "full_realistic": lambda df: df["include_in_main_protocol"].eq(True) & df["geometry_valid"].eq(True),
    "clean_cell": lambda df: df["include_in_main_protocol"].eq(True)
    & df["geometry_valid"].eq(True)
    & ~df["class_id"].isin([12, 14]),
    "strict_morphology": lambda df: df["include_in_main_protocol"].eq(True)
    & df["geometry_valid"].eq(True)
    & ~df["class_id"].isin([12, 14, 15]),
    "mask_ready": lambda df: df["include_in_main_protocol"].eq(True)
    & df["geometry_valid"].eq(True)
    & df["has_mask"].eq(True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build A1 subsets for experiment E01.")
    parser.add_argument("--object-index", type=Path, default=OBJECT_INDEX_PATH)
    parser.add_argument("--image-index", type=Path, default=IMAGE_INDEX_PATH)
    return parser.parse_args()


def _geometry_issue_columns(object_df: pd.DataFrame) -> pd.DataFrame:
    issue_df = pd.DataFrame(index=object_df.index)
    issue_df["finite"] = np.isfinite(object_df["x_center_norm"]) & np.isfinite(object_df["y_center_norm"])
    issue_df["finite"] &= np.isfinite(object_df["width_norm"]) & np.isfinite(object_df["height_norm"])
    issue_df["x_in_range"] = object_df["x_center_norm"].between(0, 1, inclusive="both")
    issue_df["y_in_range"] = object_df["y_center_norm"].between(0, 1, inclusive="both")
    issue_df["width_in_range"] = object_df["width_norm"].between(0, 1, inclusive="right")
    issue_df["height_in_range"] = object_df["height_norm"].between(0, 1, inclusive="right")
    return issue_df


def _geometry_issue_label(row: pd.Series) -> str:
    issues: list[str] = []
    if not bool(row["finite"]):
        issues.append("non_finite")
    if not bool(row["x_in_range"]):
        issues.append("x_out_of_range")
    if not bool(row["y_in_range"]):
        issues.append("y_out_of_range")
    if not bool(row["width_in_range"]):
        issues.append("width_out_of_range")
    if not bool(row["height_in_range"]):
        issues.append("height_out_of_range")
    return ",".join(issues) if issues else "ok"


def _unique_image_count(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame[["patient_id", "image_id"]].drop_duplicates().shape[0])


def build_subsets(object_index_path: Path, image_index_path: Path) -> dict[str, object]:
    ensure_output_layout()
    object_df = pd.read_parquet(object_index_path)
    image_df = pd.read_parquet(image_index_path)
    geometry_checks = _geometry_issue_columns(object_df)
    object_df = object_df.copy()
    object_df["geometry_valid"] = geometry_checks.all(axis=1)
    object_df["geometry_issue"] = geometry_checks.apply(_geometry_issue_label, axis=1)

    subset_paths: dict[str, str] = {}
    summary_rows: list[dict[str, object]] = []
    zero_object_patients = sorted(
        set(object_df.loc[object_df["include_in_main_protocol"].eq(True), "patient_id"].unique().tolist())
    )
    invalid_geometry_df = object_df.loc[
        object_df["include_in_main_protocol"].eq(True) & object_df["geometry_valid"].ne(True)
    ].copy()
    invalid_geometry_path = DATASET_INDEX_ROOT / "invalid_geometry_objects.csv"
    invalid_geometry_df.to_csv(invalid_geometry_path, index=False)

    for subset_name, predicate in SUBSET_RULES.items():
        subset_df = object_df.loc[predicate(object_df)].copy()
        subset_path = DATASET_INDEX_ROOT / f"subset_{subset_name}.parquet"
        subset_df.to_parquet(subset_path, index=False)
        subset_paths[subset_name] = str(subset_path)

        patient_count = subset_df["patient_id"].nunique()
        image_count = _unique_image_count(subset_df)
        class_counts = subset_df.groupby(["class_id", "class_name"]).size().reset_index(name="count")
        top_classes = (
            class_counts.sort_values(["count", "class_id"], ascending=[False, True])
            .head(10)
            .to_dict("records")
        )
        summary_rows.append(
            {
                "subset_name": subset_name,
                "object_count": int(len(subset_df)),
                "patient_count": int(patient_count),
                "image_count": int(image_count),
                "has_bad_cells": bool(subset_df["class_id"].eq(12).any()),
                "has_artifacts": bool(subset_df["class_id"].eq(14).any()),
                "has_gumpricht_shadows": bool(subset_df["class_id"].eq(15).any()),
                "mask_ready_ratio": float(subset_df["has_mask"].mean()) if len(subset_df) else 0.0,
                "top_classes": top_classes,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_ROOT / "subset_summary.csv", index=False)

    class_distribution_rows: list[dict[str, object]] = []
    for subset_name, subset_path in subset_paths.items():
        subset_df = pd.read_parquet(subset_path)
        grouped = subset_df.groupby(["class_id", "class_name"]).size().reset_index(name="count")
        grouped["subset_name"] = subset_name
        class_distribution_rows.extend(grouped.to_dict("records"))

    pd.DataFrame(class_distribution_rows).to_csv(TABLES_ROOT / "subset_class_distribution.csv", index=False)

    excluded_images = _unique_image_count(image_df.loc[image_df["include_in_main_protocol"].eq(False)].copy())
    patient_inventory_df = pd.read_csv(DATASET_INDEX_ROOT / "patient_inventory.csv")
    main_protocol_patients = patient_inventory_df.loc[
        patient_inventory_df["include_in_main_protocol"].eq(True), "patient_id"
    ].tolist()
    zero_object_patient_dirs = sorted(set(main_protocol_patients) - set(zero_object_patients))
    report_lines = [
        "# A1 Subset QC Report",
        "",
        f"Generated at: `{pd.Timestamp.utcnow().isoformat()}`",
        "",
        "## Summary",
        "",
        f"- Source object index: `{object_index_path}`",
        f"- Source image index: `{image_index_path}`",
        f"- Images excluded from main protocol: `{excluded_images}`",
        f"- Main protocol patient directories: `{len(main_protocol_patients)}`",
        f"- Patient directories with zero labeled objects: `{len(zero_object_patient_dirs)}`",
        f"- Zero-object patient directories: `{', '.join(zero_object_patient_dirs) if zero_object_patient_dirs else 'none'}`",
        f"- Main-protocol objects with invalid geometry: `{len(invalid_geometry_df)}`",
        f"- Invalid-geometry artifact: `{invalid_geometry_path}`",
        "",
        "## Subset Checks",
        "",
        "| Subset | Objects | Patients | Images | Has bad cells | Has artifacts | Has gumpricht shadows | Mask-ready ratio |",
        "|---|---:|---:|---:|---|---|---|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['subset_name']} | {row['object_count']} | {row['patient_count']} | {row['image_count']} | "
            f"{row['has_bad_cells']} | {row['has_artifacts']} | {row['has_gumpricht_shadows']} | {row['mask_ready_ratio']:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Geometry validation",
            "",
            "| Issue | Count |",
            "|---|---:|",
        ]
    )
    issue_counts = (
        invalid_geometry_df["geometry_issue"].value_counts().sort_values(ascending=False).to_dict()
        if len(invalid_geometry_df)
        else {}
    )
    for issue, count in issue_counts.items():
        report_lines.append(f"| {issue} | {int(count)} |")

    report_lines.extend(
        [
            "",
            "Note:",
            "Object-level subsets can contain fewer patient ids than the main protocol if a patient directory exists but contains zero labeled objects. "
            "This is currently expected for the zero-object directories listed above.",
            "All A1 subsets additionally exclude objects with non-finite or out-of-range bounding-box geometry; these rows are preserved in invalid_geometry_objects.csv for auditability.",
        ]
    )

    write_markdown(REPORTS_ROOT / "a1_subset_qc.md", report_lines)

    return {
        "subset_paths": subset_paths,
        "invalid_geometry_path": str(invalid_geometry_path),
        "summary_path": str(TABLES_ROOT / "subset_summary.csv"),
        "class_distribution_path": str(TABLES_ROOT / "subset_class_distribution.csv"),
        "qc_report_path": str(REPORTS_ROOT / "a1_subset_qc.md"),
    }


def main() -> None:
    args = parse_args()
    result = build_subsets(args.object_index, args.image_index)
    print(result)


if __name__ == "__main__":
    main()
