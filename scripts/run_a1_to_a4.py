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

from dataset.build_patient_summary import build_patient_summary
from dataset.build_subsets import build_subsets
from dataset.build_tasks import build_tasks
from dataset.make_group_splits import make_group_splits


def main() -> None:
    subset_result = build_subsets(
        object_index_path=PROJECT_ROOT / "outputs" / "dataset_index" / "object_index.parquet",
        image_index_path=PROJECT_ROOT / "outputs" / "dataset_index" / "image_index.parquet",
    )
    patient_summary_path = build_patient_summary(
        patient_inventory_path=PROJECT_ROOT / "outputs" / "dataset_index" / "patient_inventory.csv",
        image_index_path=PROJECT_ROOT / "outputs" / "dataset_index" / "image_index.parquet",
        object_index_path=PROJECT_ROOT / "outputs" / "dataset_index" / "object_index.parquet",
    )
    split_result = make_group_splits(
        patient_summary_path=patient_summary_path,
        object_index_path=PROJECT_ROOT / "outputs" / "dataset_index" / "object_index.parquet",
        random_state=42,
    )
    task_result = build_tasks(PROJECT_ROOT / "outputs" / "splits" / "split_v1.json")

    print(
        json.dumps(
            {
                "a1": subset_result,
                "a2": {"patient_summary_csv": str(patient_summary_path), **split_result},
                "a4": task_result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
