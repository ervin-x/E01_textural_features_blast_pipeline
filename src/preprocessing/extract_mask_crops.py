from __future__ import annotations

import argparse
from pathlib import Path

from utils.project import CROPS_ROOT, REPORTS_ROOT, ensure_output_layout, write_markdown
from preprocessing.image_io import crop_array, load_binary_mask, load_rgb_image, save_rgb_image

import numpy as np
import pandas as pd


BBOX_MANIFEST_PATH = CROPS_ROOT / "bbox_roi_manifest.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract mask-aware crops for A3.")
    parser.add_argument("--bbox-manifest", type=Path, default=BBOX_MANIFEST_PATH)
    return parser.parse_args()


def _mask_tight_bbox(binary_crop: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(binary_crop)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def extract_mask_crops(bbox_manifest_path: Path) -> pd.DataFrame:
    ensure_output_layout()
    bbox_manifest_df = pd.read_parquet(bbox_manifest_path)

    mask_root = CROPS_ROOT / "mask" / "main_protocol"
    mask_tight_root = CROPS_ROOT / "mask_tight" / "main_protocol"

    bbox_manifest_df["mask_crop_path"] = None
    bbox_manifest_df["mask_tight_crop_path"] = None
    bbox_manifest_df["mask_tight_width_px"] = None
    bbox_manifest_df["mask_tight_height_px"] = None
    bbox_manifest_df["mask_area_px"] = None
    bbox_manifest_df["mask_bbox_area_px"] = bbox_manifest_df["bbox_width_px"] * bbox_manifest_df["bbox_height_px"]
    bbox_manifest_df["mask_status"] = np.where(bbox_manifest_df["has_mask"], "pending", "missing")

    current_full_image_path = None
    current_full_image = None
    for index, row in bbox_manifest_df.loc[bbox_manifest_df["has_mask"].eq(True)].iterrows():
        full_image_path = row.get("image_path")
        if pd.isna(full_image_path):
            full_image_path = None

        if full_image_path is None:
            continue

        if full_image_path != current_full_image_path:
            current_full_image = load_rgb_image(full_image_path)
            current_full_image_path = full_image_path

        assert current_full_image is not None
        full_mask = load_binary_mask(row["mask_path"])
        bbox = (int(row["bbox_x1"]), int(row["bbox_y1"]), int(row["bbox_x2"]), int(row["bbox_y2"]))
        image_crop = crop_array(current_full_image, bbox)
        mask_crop = crop_array(full_mask.astype(np.uint8), bbox).astype(bool)
        masked_rgb = image_crop.copy()
        masked_rgb[~mask_crop] = 0

        mask_crop_path = (
            mask_root
            / str(row["patient_id"])
            / f"{row['image_id']}__obj_{int(row['object_id_within_image']):04d}.png"
        )
        save_rgb_image(masked_rgb, mask_crop_path)

        tight_bbox = _mask_tight_bbox(mask_crop)
        mask_tight_crop_path = None
        mask_tight_width = None
        mask_tight_height = None
        if tight_bbox is not None:
            tx1, ty1, tx2, ty2 = tight_bbox
            tight_rgb = masked_rgb[ty1:ty2, tx1:tx2]
            mask_tight_crop_path = (
                mask_tight_root
                / str(row["patient_id"])
                / f"{row['image_id']}__obj_{int(row['object_id_within_image']):04d}.png"
            )
            save_rgb_image(tight_rgb, mask_tight_crop_path)
            mask_tight_width = tx2 - tx1
            mask_tight_height = ty2 - ty1

        bbox_manifest_df.at[index, "mask_crop_path"] = str(mask_crop_path)
        bbox_manifest_df.at[index, "mask_tight_crop_path"] = str(mask_tight_crop_path) if mask_tight_crop_path else None
        bbox_manifest_df.at[index, "mask_tight_width_px"] = mask_tight_width
        bbox_manifest_df.at[index, "mask_tight_height_px"] = mask_tight_height
        bbox_manifest_df.at[index, "mask_area_px"] = int(mask_crop.sum())
        bbox_manifest_df.at[index, "mask_status"] = "matched"

    output_manifest_path = CROPS_ROOT / "roi_manifest.parquet"
    bbox_manifest_df.to_parquet(output_manifest_path, index=False)

    report_lines = [
        "# A3 Mask Crop QC",
        "",
        f"Generated at: `{pd.Timestamp.utcnow().isoformat()}`",
        "",
        f"- ROI manifest rows: `{len(bbox_manifest_df)}`",
        f"- Rows with matched masks: `{int((bbox_manifest_df['mask_status'] == 'matched').sum())}`",
        f"- Rows without masks: `{int((bbox_manifest_df['mask_status'] == 'missing').sum())}`",
        f"- Mask crop directory: `{mask_root}`",
        f"- Mask tight crop directory: `{mask_tight_root}`",
    ]
    write_markdown(REPORTS_ROOT / "a3_mask_qc.md", report_lines)
    return bbox_manifest_df


def main() -> None:
    args = parse_args()
    manifest_df = extract_mask_crops(args.bbox_manifest)
    print({"roi_manifest_rows": len(manifest_df)})


if __name__ == "__main__":
    main()
