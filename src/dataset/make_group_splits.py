from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from utils.project import REPORTS_ROOT, SPLITS_ROOT, TABLES_ROOT, ensure_output_layout, write_json, write_markdown

import pandas as pd


PATIENT_SUMMARY_PATH = SPLITS_ROOT / "patient_summary.csv"
OBJECT_INDEX_PATH = Path(__file__).resolve().parents[2] / "outputs" / "dataset_index" / "object_index.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create patient-level split_v1 for A2.")
    parser.add_argument("--patient-summary", type=Path, default=PATIENT_SUMMARY_PATH)
    parser.add_argument("--object-index", type=Path, default=OBJECT_INDEX_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _stratified_patient_split(patient_summary_df: pd.DataFrame, random_state: int) -> tuple[list[str], list[str], list[str]]:
    patient_ids = patient_summary_df["patient_id"].to_numpy()
    strata = patient_summary_df["patient_has_blast"].astype(int).to_numpy()

    train_ids, temp_ids, train_y, temp_y = train_test_split(
        patient_ids,
        strata,
        test_size=0.30,
        random_state=random_state,
        stratify=strata,
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_y,
    )
    return sorted(train_ids.tolist()), sorted(val_ids.tolist()), sorted(test_ids.tolist())


def _split_image_counts(object_df: pd.DataFrame) -> pd.Series:
    image_counts = (
        object_df[["split_name", "patient_id", "image_id"]]
        .drop_duplicates()
        .groupby("split_name")
        .size()
    )
    return image_counts.astype(int)


def make_group_splits(patient_summary_path: Path, object_index_path: Path, random_state: int = 42) -> dict[str, str]:
    ensure_output_layout()
    patient_summary_df = pd.read_csv(patient_summary_path)
    object_df = pd.read_parquet(object_index_path)
    object_df = object_df.loc[object_df["include_in_main_protocol"].eq(True)].copy()

    train_ids, val_ids, test_ids = _stratified_patient_split(patient_summary_df, random_state=random_state)

    split_payload = {
        "version": "split_v1",
        "random_state": random_state,
        "train_patients": train_ids,
        "validation_patients": val_ids,
        "test_patients": test_ids,
    }

    split_json_path = SPLITS_ROOT / "split_v1.json"
    write_json(split_json_path, split_payload)
    for name, values in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        path = SPLITS_ROOT / f"split_v1_{name}_patients.txt"
        path.write_text("\n".join(values) + "\n", encoding="utf-8")

    patient_to_split = {patient_id: "train" for patient_id in train_ids}
    patient_to_split.update({patient_id: "val" for patient_id in val_ids})
    patient_to_split.update({patient_id: "test" for patient_id in test_ids})
    object_df["split_name"] = object_df["patient_id"].map(patient_to_split)

    class_distribution = (
        object_df.groupby(["split_name", "class_id", "class_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["split_name", "count", "class_id"], ascending=[True, False, True])
    )
    class_distribution.to_csv(TABLES_ROOT / "class_distribution_by_split.csv", index=False)

    split_summary = (
        object_df.groupby("split_name")
        .agg(
            object_count=("object_id_within_image", "size"),
            blast_count=("is_blast", "sum"),
            active_patient_count=("patient_id", "nunique"),
        )
        .reset_index()
    )
    split_summary["image_count"] = split_summary["split_name"].map(_split_image_counts(object_df))
    split_summary["blast_fraction"] = split_summary["blast_count"] / split_summary["object_count"]
    split_summary["patient_count"] = split_summary["split_name"].map(
        {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}
    )
    split_summary["zero_object_patient_count"] = split_summary["patient_count"] - split_summary["active_patient_count"]
    split_summary.to_csv(TABLES_ROOT / "split_summary.csv", index=False)

    report_lines = [
        "# A2 Split QC Report",
        "",
        f"Generated at: `{pd.Timestamp.utcnow().isoformat()}`",
        "",
        f"- Random state: `{random_state}`",
        f"- Train patients: `{len(train_ids)}`",
        f"- Validation patients: `{len(val_ids)}`",
        f"- Test patients: `{len(test_ids)}`",
        "",
        "## Split-level summary",
        "",
        "| Split | Patients in split list | Patients with objects | Zero-object patients | Images | Objects | Blast objects | Blast fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_summary.to_dict("records"):
        report_lines.append(
            f"| {row['split_name']} | {int(row['patient_count'])} | {int(row['active_patient_count'])} | {int(row['zero_object_patient_count'])} | "
            f"{int(row['image_count'])} | {int(row['object_count'])} | {int(row['blast_count'])} | {float(row['blast_fraction']):.6f} |"
        )

    intersections = {
        "train_val": sorted(set(train_ids) & set(val_ids)),
        "train_test": sorted(set(train_ids) & set(test_ids)),
        "val_test": sorted(set(val_ids) & set(test_ids)),
    }
    report_lines.extend(
        [
            "",
            "## Intersection checks",
            "",
            f"- train ∩ val: `{len(intersections['train_val'])}`",
            f"- train ∩ test: `{len(intersections['train_test'])}`",
            f"- val ∩ test: `{len(intersections['val_test'])}`",
        ]
    )
    write_markdown(REPORTS_ROOT / "a2_split_qc.md", report_lines)

    return {
        "split_json": str(split_json_path),
        "split_summary_csv": str(TABLES_ROOT / "split_summary.csv"),
        "class_distribution_csv": str(TABLES_ROOT / "class_distribution_by_split.csv"),
        "qc_report_md": str(REPORTS_ROOT / "a2_split_qc.md"),
    }


def main() -> None:
    args = parse_args()
    result = make_group_splits(args.patient_summary, args.object_index, args.random_state)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
