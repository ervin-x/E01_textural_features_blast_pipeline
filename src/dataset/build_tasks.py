from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.project import DATASET_INDEX_ROOT, REPORTS_ROOT, SPLITS_ROOT, TABLES_ROOT, ensure_output_layout, write_markdown

import pandas as pd


SPLIT_JSON_PATH = SPLITS_ROOT / "split_v1.json"


TASK_SPECS = {
    "a41_clean_binary": "subset_clean_cell.parquet",
    "a42_strict_binary": "subset_strict_morphology.parquet",
    "a43_full_realistic_binary": "subset_full_realistic.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build binary task datasets for A4.")
    parser.add_argument("--split-json", type=Path, default=SPLIT_JSON_PATH)
    return parser.parse_args()


def build_tasks(split_json_path: Path) -> dict[str, str]:
    ensure_output_layout()
    split_payload = json.loads(split_json_path.read_text(encoding="utf-8"))
    patient_to_split = {patient_id: "train" for patient_id in split_payload["train_patients"]}
    patient_to_split.update({patient_id: "val" for patient_id in split_payload["validation_patients"]})
    patient_to_split.update({patient_id: "test" for patient_id in split_payload["test_patients"]})

    summary_rows: list[dict[str, object]] = []
    output_paths: dict[str, str] = {}

    for task_name, subset_file in TASK_SPECS.items():
        subset_path = DATASET_INDEX_ROOT / subset_file
        task_df = pd.read_parquet(subset_path).copy()
        task_df["target_binary"] = task_df["class_id"].eq(7).astype(int)
        task_df["split_name"] = task_df["patient_id"].map(patient_to_split)
        task_output_path = DATASET_INDEX_ROOT / f"task_{task_name}.parquet"
        task_df.to_parquet(task_output_path, index=False)
        output_paths[task_name] = str(task_output_path)

        class_grouped = (
            task_df.groupby(["split_name", "target_binary"])
            .size()
            .reset_index(name="count")
            .sort_values(["split_name", "target_binary"])
        )
        split_counts = {
            row["split_name"]: {
                "positive": int(class_grouped.loc[(class_grouped["split_name"] == row["split_name"]) & (class_grouped["target_binary"] == 1), "count"].sum()),
                "negative": int(class_grouped.loc[(class_grouped["split_name"] == row["split_name"]) & (class_grouped["target_binary"] == 0), "count"].sum()),
            }
            for row in class_grouped.to_dict("records")
        }

        hard_negative_rows = (
            task_df.loc[task_df["target_binary"].eq(0)]
            .groupby(["class_id", "class_name"])
            .size()
            .reset_index(name="count")
            .sort_values(["count", "class_id"], ascending=[False, True])
            .head(10)
        )

        summary_rows.append(
            {
                "task_name": task_name,
                "task_path": str(task_output_path),
                "object_count": int(len(task_df)),
                "positive_count": int(task_df["target_binary"].sum()),
                "negative_count": int((1 - task_df["target_binary"]).sum()),
                "positive_fraction": float(task_df["target_binary"].mean()),
                "hard_negative_top10": json.dumps(hard_negative_rows.to_dict("records"), ensure_ascii=False),
                "train_positive": split_counts.get("train", {}).get("positive", 0),
                "train_negative": split_counts.get("train", {}).get("negative", 0),
                "val_positive": split_counts.get("val", {}).get("positive", 0),
                "val_negative": split_counts.get("val", {}).get("negative", 0),
                "test_positive": split_counts.get("test", {}).get("positive", 0),
                "test_negative": split_counts.get("test", {}).get("negative", 0),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_ROOT / "task_prevalence_summary.csv", index=False)

    report_lines = [
        "# A4 Task Definition Report",
        "",
        f"Generated at: `{pd.Timestamp.utcnow().isoformat()}`",
        "",
        "## Task summary",
        "",
        "| Task | Objects | Positive | Negative | Positive fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['task_name']} | {row['object_count']} | {row['positive_count']} | {row['negative_count']} | {row['positive_fraction']:.6f} |"
        )
    write_markdown(REPORTS_ROOT / "task_definition_report.md", report_lines)

    output_paths["task_prevalence_summary_csv"] = str(TABLES_ROOT / "task_prevalence_summary.csv")
    output_paths["task_definition_report_md"] = str(REPORTS_ROOT / "task_definition_report.md")
    return output_paths


def main() -> None:
    args = parse_args()
    result = build_tasks(args.split_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
