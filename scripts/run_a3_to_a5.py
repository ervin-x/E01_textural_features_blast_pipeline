from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
VENDOR_ROOT = PROJECT_ROOT / "vendor"
VENV_SITE_PACKAGES = sorted((PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages"))

for candidate in [*VENV_SITE_PACKAGES, SRC_ROOT, VENDOR_ROOT]:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.append(candidate_str)

from features.feature_registry import extract_features
from preprocessing.extract_bbox_crops import extract_bbox_crops
from preprocessing.extract_mask_crops import extract_mask_crops


def main() -> None:
    bbox_manifest_df = extract_bbox_crops(PROJECT_ROOT / "outputs" / "dataset_index" / "subset_full_realistic.parquet")
    roi_manifest_df = extract_mask_crops(PROJECT_ROOT / "outputs" / "crops" / "bbox_roi_manifest.parquet")
    feature_result = extract_features(PROJECT_ROOT / "outputs" / "crops" / "roi_manifest.parquet")
    print(
        json.dumps(
            {
                "a3": {
                    "bbox_manifest_rows": len(bbox_manifest_df),
                    "roi_manifest_rows": len(roi_manifest_df),
                    "roi_manifest_path": str(PROJECT_ROOT / "outputs" / "crops" / "roi_manifest.parquet"),
                },
                "a5": feature_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
