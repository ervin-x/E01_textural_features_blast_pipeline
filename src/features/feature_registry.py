from __future__ import annotations

import argparse
import os
from pathlib import Path

from utils.project import CROPS_ROOT, FEATURES_ROOT, REPORTS_ROOT, TABLES_ROOT, ensure_output_layout, write_markdown
from features.extract_color import color_features
from features.extract_morphology import morphology_from_bbox, morphology_from_mask
from features.extract_texture_glcm import texture_glcm_features
from features.extract_texture_lbp import texture_lbp_features
from features.extract_texture_wavelet import texture_wavelet_features
from preprocessing.image_io import load_binary_mask, load_rgb_image

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


ROI_MANIFEST_PATH = CROPS_ROOT / "roi_manifest.parquet"
KEY_COLUMNS = {"patient_id", "image_id", "object_id_within_image", "class_id", "class_name", "has_mask"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract A5 feature matrices.")
    parser.add_argument("--roi-manifest", type=Path, default=ROI_MANIFEST_PATH)
    parser.add_argument("--output-suffix", type=str, default="")
    parser.add_argument("--texture-profile", choices=["full", "lite"], default="full")
    return parser.parse_args()


def _gray(image_rgb: np.ndarray) -> np.ndarray:
    return np.dot(image_rgb[..., :3].astype(np.float32), np.array([0.299, 0.587, 0.114], dtype=np.float32))


def _record_key(row: dict[str, object]) -> dict[str, object]:
    return {
        "patient_id": row["patient_id"],
        "image_id": row["image_id"],
        "object_id_within_image": int(row["object_id_within_image"]),
        "class_id": int(row["class_id"]),
        "class_name": row["class_name"],
        "has_mask": bool(row["has_mask"]),
    }


def _texture_bundle(
    gray_image: np.ndarray,
    binary_mask: np.ndarray | None,
    texture_profile: str,
) -> dict[str, float]:
    features: dict[str, float] = {}
    if texture_profile == "full":
        features.update(texture_glcm_features(gray_image, binary_mask=binary_mask))
        features.update(texture_lbp_features(gray_image, binary_mask=binary_mask))
        features.update(texture_wavelet_features(gray_image, binary_mask=binary_mask))
        return features

    features.update(texture_lbp_features(gray_image, binary_mask=binary_mask))
    return features


def _feature_family(feature_name: str) -> str:
    if feature_name == "mask_missing":
        return "quality_flag"
    normalized = feature_name
    if normalized.startswith("bbox_"):
        normalized = normalized[len("bbox_") :]
    elif normalized.startswith("mask_"):
        normalized = normalized[len("mask_") :]

    if normalized.startswith("morph_"):
        return "morphology"
    if normalized.startswith(("color_", "hsv_", "lab_")):
        return "color"
    if normalized.startswith(("texture_", "wavelet_")):
        return "texture"
    return "other"


def _feature_region(feature_name: str, matrix_name: str) -> str:
    if feature_name.startswith("bbox_"):
        return "bbox_crop"
    if feature_name.startswith("mask_"):
        return "mask_crop"
    if matrix_name == "bbox":
        return "bbox_crop"
    if matrix_name == "mask":
        return "mask_crop"
    return "combined"


def _feature_description(feature_name: str) -> str:
    normalized = feature_name
    if normalized.startswith("bbox_"):
        normalized = normalized[len("bbox_") :]
    elif normalized.startswith("mask_"):
        normalized = normalized[len("mask_") :]

    if normalized == "mask_missing":
        return "Indicator that mask-derived features are unavailable for this object."
    if normalized.startswith("morph_bbox_area_px"):
        return "Bounding-box area in pixels."
    if normalized.startswith("morph_bbox_width_px"):
        return "Bounding-box width in pixels."
    if normalized.startswith("morph_bbox_height_px"):
        return "Bounding-box height in pixels."
    if normalized.startswith("morph_bbox_perimeter_px"):
        return "Bounding-box perimeter in pixels."
    if normalized.startswith("morph_bbox_aspect_ratio"):
        return "Bounding-box aspect ratio."
    if normalized.startswith("morph_bbox_diagonal_px"):
        return "Bounding-box diagonal length in pixels."
    if normalized.startswith("morph_area_px"):
        return "Mask area in pixels."
    if normalized.startswith("morph_perimeter_px"):
        return "Mask perimeter in pixels."
    if normalized.startswith("morph_extent"):
        return "Mask extent relative to its enclosing box."
    if normalized.startswith("morph_eccentricity"):
        return "Mask eccentricity from regionprops."
    if normalized.startswith("morph_solidity"):
        return "Mask solidity from regionprops."
    if normalized.startswith("morph_roundness"):
        return "Roundness computed as 4*pi*area/perimeter^2."
    if normalized.startswith("morph_bbox_fill_ratio"):
        return "Ratio between mask area and enclosing bounding-box area."
    if normalized.startswith("color_"):
        return "RGB channel summary statistic computed on selected pixels."
    if normalized.startswith("hsv_"):
        return "HSV channel summary statistic computed on selected pixels."
    if normalized.startswith("lab_"):
        return "Lab channel summary statistic computed on selected pixels."
    if normalized.startswith("texture_glcm_"):
        return "GLCM/Haralick-style texture statistic."
    if normalized.startswith("texture_lbp_"):
        return "Local Binary Pattern histogram bin."
    if normalized.startswith("texture_sobel_"):
        return "Sobel-gradient summary statistic."
    if normalized.startswith("texture_laplace_"):
        return "Laplacian summary statistic."
    if normalized.startswith("wavelet_"):
        return "Wavelet coefficient summary statistic."
    return "Derived feature from the configured extraction pipeline."


def _build_feature_dictionary(matrix_name: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        if column in KEY_COLUMNS:
            continue
        rows.append(
            {
                "matrix": matrix_name,
                "feature_name": column,
                "feature_family": _feature_family(column),
                "source_region": _feature_region(column, matrix_name),
                "description": _feature_description(column),
            }
        )
    return rows


def _build_missingness_rows(matrix_name: str, frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        if column in KEY_COLUMNS:
            continue
        rows.append(
            {
                "matrix": matrix_name,
                "feature_name": column,
                "missing_fraction": float(frame[column].isna().mean()),
                "missing_count": int(frame[column].isna().sum()),
                "is_constant": bool(frame[column].nunique(dropna=False) <= 1),
            }
        )
    return rows


def _extract_single_row(
    row: dict[str, object],
    texture_profile: str,
) -> tuple[dict[str, object], dict[str, object] | None, dict[str, object]]:
    bbox_rgb = load_rgb_image(row["bbox_crop_path"])
    bbox_gray = _gray(bbox_rgb)

    bbox_features = {}
    bbox_features.update(morphology_from_bbox(int(row["bbox_width_px"]), int(row["bbox_height_px"])))
    bbox_features.update(color_features(bbox_rgb))
    bbox_features.update(_texture_bundle(bbox_gray, binary_mask=None, texture_profile=texture_profile))
    bbox_prefixed = {f"bbox_{key}": value for key, value in bbox_features.items()}

    bbox_row = {**_record_key(row), **bbox_prefixed}
    combined_row = {**_record_key(row), **bbox_prefixed}

    mask_row: dict[str, object] | None = None
    if row["has_mask"] and row["mask_crop_path"]:
        full_mask = load_binary_mask(row["mask_path"])
        bbox = (
            int(row["bbox_x1"]),
            int(row["bbox_y1"]),
            int(row["bbox_x2"]),
            int(row["bbox_y2"]),
        )
        mask_crop = full_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
        mask_rgb = load_rgb_image(row["mask_crop_path"])
        mask_gray = _gray(mask_rgb)

        mask_features = {}
        mask_features.update(morphology_from_mask(mask_crop))
        mask_features.update(color_features(mask_rgb, binary_mask=mask_crop))
        mask_features.update(_texture_bundle(mask_gray, binary_mask=mask_crop, texture_profile=texture_profile))
        mask_prefixed = {f"mask_{key}": value for key, value in mask_features.items()}
        mask_row = {**_record_key(row), **mask_prefixed}
        combined_row.update(mask_prefixed)
    else:
        combined_row["mask_missing"] = 1

    return bbox_row, mask_row, combined_row


def extract_features(
    roi_manifest_path: Path,
    output_suffix: str = "",
    texture_profile: str = "full",
) -> dict[str, str]:
    ensure_output_layout()
    roi_df = pd.read_parquet(roi_manifest_path)
    roi_records = roi_df.to_dict("records")
    n_jobs = min(8, max(1, os.cpu_count() or 1))
    extracted_rows = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=64)(
        delayed(_extract_single_row)(row, texture_profile) for row in roi_records
    )
    bbox_rows = [bbox_row for bbox_row, _, _ in extracted_rows]
    mask_rows = [mask_row for _, mask_row, _ in extracted_rows if mask_row is not None]
    combined_rows = [combined_row for _, _, combined_row in extracted_rows]

    features_bbox_df = pd.DataFrame(bbox_rows)
    features_mask_df = pd.DataFrame(mask_rows)
    features_combined_df = pd.DataFrame(combined_rows)

    suffix = output_suffix.strip()
    features_bbox_path = FEATURES_ROOT / f"features_bbox{suffix}.parquet"
    features_mask_path = FEATURES_ROOT / f"features_mask{suffix}.parquet"
    features_combined_path = FEATURES_ROOT / f"features_combined{suffix}.parquet"
    features_bbox_df.to_parquet(features_bbox_path, index=False)
    features_mask_df.to_parquet(features_mask_path, index=False)
    features_combined_df.to_parquet(features_combined_path, index=False)

    dictionary_rows = []
    for frame_name, frame in [("bbox", features_bbox_df), ("mask", features_mask_df), ("combined", features_combined_df)]:
        dictionary_rows.extend(_build_feature_dictionary(frame_name, frame))
    feature_dictionary_csv = FEATURES_ROOT / f"feature_dictionary{suffix}.csv"
    pd.DataFrame(dictionary_rows).to_csv(feature_dictionary_csv, index=False)
    dictionary_lines = [
        "# Feature Dictionary",
        "",
        "| Matrix | Feature name | Family | Source region | Description |",
        "|---|---|---|---|---|",
    ]
    for row in dictionary_rows:
        dictionary_lines.append(
            f"| {row['matrix']} | {row['feature_name']} | {row['feature_family']} | {row['source_region']} | {row['description']} |"
        )
    feature_dictionary_md = FEATURES_ROOT / f"feature_dictionary{suffix}.md"
    write_markdown(feature_dictionary_md, dictionary_lines)

    missingness_rows = []
    for frame_name, frame in [("bbox", features_bbox_df), ("mask", features_mask_df), ("combined", features_combined_df)]:
        missingness_rows.extend(_build_missingness_rows(frame_name, frame))
    missingness_df = pd.DataFrame(missingness_rows)
    missingness_path = TABLES_ROOT / f"feature_missingness{suffix}.csv"
    missingness_df.to_csv(missingness_path, index=False)
    combined_missingness_df = missingness_df.loc[missingness_df["matrix"].eq("combined")].copy()

    report_lines = [
        "# A5 Feature Extraction QC",
        "",
        f"Generated at: `{pd.Timestamp.utcnow().isoformat()}`",
        "",
        f"- BBox feature rows: `{len(features_bbox_df)}`",
        f"- Mask feature rows: `{len(features_mask_df)}`",
        f"- Combined feature rows: `{len(features_combined_df)}`",
        f"- Parallel workers used: `{n_jobs}`",
        f"- Texture profile: `{texture_profile}`",
        f"- Numeric feature count (combined): `{int(len(combined_missingness_df))}`",
        f"- Combined features with missing values: `{int((combined_missingness_df['missing_fraction'] > 0).sum())}`",
        f"- Combined constant features: `{int(combined_missingness_df['is_constant'].sum())}`",
        f"- Total dictionary rows: `{len(dictionary_rows)}`",
    ]
    qc_report_path = REPORTS_ROOT / f"feature_extraction_qc{suffix}.md"
    write_markdown(qc_report_path, report_lines)

    return {
        "features_bbox": str(features_bbox_path),
        "features_mask": str(features_mask_path),
        "features_combined": str(features_combined_path),
        "feature_dictionary_csv": str(feature_dictionary_csv),
        "feature_dictionary_md": str(feature_dictionary_md),
        "feature_missingness_csv": str(missingness_path),
        "feature_extraction_qc_md": str(qc_report_path),
    }


def main() -> None:
    args = parse_args()
    result = extract_features(
        args.roi_manifest,
        output_suffix=args.output_suffix,
        texture_profile=args.texture_profile,
    )
    print(result)


if __name__ == "__main__":
    main()
