from __future__ import annotations

import argparse
from pathlib import Path

from utils.project import CROPS_ROOT, DATASET_INDEX_ROOT, REPORTS_ROOT, ensure_output_layout, write_markdown
from preprocessing.image_io import crop_array, load_rgb_image, save_rgb_image, yolo_to_bbox_pixels

import pandas as pd


OBJECT_INDEX_PATH = DATASET_INDEX_ROOT / "object_index.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract bbox crops for A3.")
    parser.add_argument("--object-index", type=Path, default=OBJECT_INDEX_PATH)
    return parser.parse_args()


def extract_bbox_crops(object_index_path: Path) -> pd.DataFrame:
    ensure_output_layout()
    object_df = pd.read_parquet(object_index_path)
    object_df = object_df.loc[object_df["include_in_main_protocol"].eq(True)].copy()
    geometry_valid = (
        pd.Series(True, index=object_df.index)
        & object_df["x_center_norm"].map(lambda value: pd.notna(value) and float("-inf") < float(value) < float("inf"))
        & object_df["y_center_norm"].map(lambda value: pd.notna(value) and float("-inf") < float(value) < float("inf"))
        & object_df["width_norm"].map(lambda value: pd.notna(value) and float("-inf") < float(value) < float("inf"))
        & object_df["height_norm"].map(lambda value: pd.notna(value) and float("-inf") < float(value) < float("inf"))
        & object_df["x_center_norm"].between(0, 1, inclusive="both")
        & object_df["y_center_norm"].between(0, 1, inclusive="both")
        & object_df["width_norm"].between(0, 1, inclusive="right")
        & object_df["height_norm"].between(0, 1, inclusive="right")
    )
    invalid_geometry_df = object_df.loc[~geometry_valid].copy()
    invalid_geometry_path = CROPS_ROOT / "invalid_bbox_objects.csv"
    invalid_geometry_df.to_csv(invalid_geometry_path, index=False)
    object_df = object_df.loc[geometry_valid].copy()
    object_df = object_df.sort_values(["image_path", "object_id_within_image"])

    rows: list[dict[str, object]] = []
    current_image_path = None
    current_image = None

    bbox_root = CROPS_ROOT / "bbox" / "main_protocol"
    for row in object_df.to_dict("records"):
        image_path = row["image_path"]
        if image_path != current_image_path:
            current_image = load_rgb_image(image_path)
            current_image_path = image_path

        assert current_image is not None
        image_height, image_width = current_image.shape[:2]
        bbox = yolo_to_bbox_pixels(
            image_width=image_width,
            image_height=image_height,
            x_center_norm=float(row["x_center_norm"]),
            y_center_norm=float(row["y_center_norm"]),
            width_norm=float(row["width_norm"]),
            height_norm=float(row["height_norm"]),
        )
        crop = crop_array(current_image, bbox)
        bbox_path = (
            bbox_root
            / str(row["patient_id"])
            / f"{row['image_id']}__obj_{int(row['object_id_within_image']):04d}.png"
        )
        save_rgb_image(crop, bbox_path)

        rows.append(
            {
                "patient_id": row["patient_id"],
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "object_id_within_image": int(row["object_id_within_image"]),
                "class_id": int(row["class_id"]),
                "class_name": row["class_name"],
                "bbox_x1": bbox[0],
                "bbox_y1": bbox[1],
                "bbox_x2": bbox[2],
                "bbox_y2": bbox[3],
                "bbox_width_px": bbox[2] - bbox[0],
                "bbox_height_px": bbox[3] - bbox[1],
                "bbox_crop_path": str(bbox_path),
                "has_mask": bool(row["has_mask"]),
                "mask_path": row["mask_path"],
                "image_width_px": image_width,
                "image_height_px": image_height,
            }
        )

    bbox_manifest_df = pd.DataFrame(rows)
    bbox_manifest_df.to_parquet(CROPS_ROOT / "bbox_roi_manifest.parquet", index=False)
    unique_image_count = int(bbox_manifest_df[["patient_id", "image_id"]].drop_duplicates().shape[0]) if len(bbox_manifest_df) else 0

    report_lines = [
        "# A3 BBox Crop QC",
        "",
        f"Generated at: `{pd.Timestamp.utcnow().isoformat()}`",
        "",
        f"- Objects processed: `{len(bbox_manifest_df)}`",
        f"- Objects skipped due to invalid geometry: `{len(invalid_geometry_df)}`",
        f"- Unique images processed: `{unique_image_count}`",
        f"- BBox crop directory: `{bbox_root}`",
        f"- Invalid-geometry artifact: `{invalid_geometry_path}`",
    ]
    write_markdown(REPORTS_ROOT / "a3_bbox_qc.md", report_lines)
    return bbox_manifest_df


def main() -> None:
    args = parse_args()
    manifest_df = extract_bbox_crops(args.object_index)
    print({"bbox_manifest_rows": len(manifest_df)})


if __name__ == "__main__":
    main()
