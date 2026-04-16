from __future__ import annotations

import argparse
from pathlib import Path

from utils.project import DATASET_INDEX_ROOT, SPLITS_ROOT, REPORTS_ROOT, ensure_output_layout, write_markdown

import pandas as pd


PATIENT_INVENTORY_PATH = DATASET_INDEX_ROOT / "patient_inventory.csv"
IMAGE_INDEX_PATH = DATASET_INDEX_ROOT / "image_index.parquet"
OBJECT_INDEX_PATH = DATASET_INDEX_ROOT / "object_index.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build patient summary for A2.")
    parser.add_argument("--patient-inventory", type=Path, default=PATIENT_INVENTORY_PATH)
    parser.add_argument("--image-index", type=Path, default=IMAGE_INDEX_PATH)
    parser.add_argument("--object-index", type=Path, default=OBJECT_INDEX_PATH)
    return parser.parse_args()


def build_patient_summary(patient_inventory_path: Path, image_index_path: Path, object_index_path: Path) -> Path:
    ensure_output_layout()
    inventory_df = pd.read_csv(patient_inventory_path)
    image_df = pd.read_parquet(image_index_path)
    object_df = pd.read_parquet(object_index_path)

    inventory_df = inventory_df.loc[inventory_df["include_in_main_protocol"].eq(True)].copy()
    image_df = image_df.loc[image_df["include_in_main_protocol"].eq(True)].copy()
    object_df = object_df.loc[object_df["include_in_main_protocol"].eq(True)].copy()

    object_grouped = (
        object_df.groupby("patient_id")
        .agg(
            objects_count=("object_id_within_image", "size"),
            blast_objects_count=("is_blast", "sum"),
            mask_missing_objects_count=("has_mask", lambda s: int((~s).sum())),
            quality_or_artifact_objects_count=("is_quality_or_artifact", "sum"),
        )
        .reset_index()
    )
    object_grouped["blast_fraction"] = object_grouped["blast_objects_count"] / object_grouped["objects_count"]
    object_grouped["mask_missing_fraction"] = (
        object_grouped["mask_missing_objects_count"] / object_grouped["objects_count"]
    )

    image_grouped = (
        image_df.loc[image_df["has_image"].eq(True)]
        .groupby("patient_id")
        .agg(
            images_count=("image_id", "nunique"),
            images_with_blast_count=("contains_blast", "sum"),
            images_without_masks_count=("has_any_mask", lambda s: int((~s).sum())),
        )
        .reset_index()
    )
    image_grouped["images_with_blast_fraction"] = (
        image_grouped["images_with_blast_count"] / image_grouped["images_count"]
    )
    image_grouped["images_without_masks_fraction"] = (
        image_grouped["images_without_masks_count"] / image_grouped["images_count"]
    )

    patient_summary_df = (
        inventory_df[["patient_id", "class_mapping_variant_id"]]
        .merge(object_grouped, on="patient_id", how="left")
        .merge(image_grouped, on="patient_id", how="left")
        .fillna(0)
    )
    patient_summary_df["patient_has_blast"] = patient_summary_df["blast_objects_count"] > 0
    patient_summary_df["blast_stratum"] = patient_summary_df["patient_has_blast"].map({True: "blast", False: "non_blast"})
    patient_summary_df = patient_summary_df.sort_values(["patient_has_blast", "blast_fraction", "patient_id"], ascending=[False, False, True])

    output_path = SPLITS_ROOT / "patient_summary.csv"
    patient_summary_df.to_csv(output_path, index=False)

    report_lines = [
        "# A2 Patient Summary QC Report",
        "",
        f"Generated at: `{pd.Timestamp.utcnow().isoformat()}`",
        "",
        f"- Patients in main protocol: `{patient_summary_df['patient_id'].nunique()}`",
        f"- Patients with blast: `{int(patient_summary_df['patient_has_blast'].sum())}`",
        f"- Patients without blast: `{int((~patient_summary_df['patient_has_blast']).sum())}`",
        f"- Mean blast fraction: `{patient_summary_df['blast_fraction'].mean():.6f}`",
        f"- Mean mask missing fraction: `{patient_summary_df['mask_missing_fraction'].mean():.6f}`",
    ]
    write_markdown(REPORTS_ROOT / "a2_patient_summary_qc.md", report_lines)

    return output_path


def main() -> None:
    args = parse_args()
    output_path = build_patient_summary(args.patient_inventory, args.image_index, args.object_index)
    print(output_path)


if __name__ == "__main__":
    main()
