from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
VENDOR_ROOT = PROJECT_ROOT / "vendor"
VENV_SITE_PACKAGES = sorted((PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages"))

for candidate in [*VENV_SITE_PACKAGES, SRC_ROOT, VENDOR_ROOT]:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.append(candidate_str)

from utils.project import CROPS_ROOT, DATASET_INDEX_ROOT, REPORTS_ROOT, ensure_output_layout, write_markdown


KEY_COLUMNS = [
    "patient_id",
    "image_id",
    "object_id_within_image",
    "class_id",
    "class_name",
    "has_mask",
]


def _cap_frame(frame: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    sampled_groups: list[pd.DataFrame] = []
    remaining = max_rows
    groups = list(frame.groupby("target_binary", sort=True))
    for index, (_, group) in enumerate(groups):
        if index == len(groups) - 1:
            n_take = min(len(group), remaining)
        else:
            n_take = max(1, int(round(max_rows * len(group) / len(frame))))
            n_take = min(len(group), n_take)
            remaining -= n_take
        sampled_groups.append(group.sample(n=n_take, random_state=random_state))
    return pd.concat(sampled_groups, ignore_index=True)


def build_modeling_workset() -> dict[str, str]:
    ensure_output_layout()
    full_task = pd.read_parquet(DATASET_INDEX_ROOT / "task_a43_full_realistic_binary.parquet")
    roi_manifest = pd.read_parquet(CROPS_ROOT / "roi_manifest.parquet")

    train_df = _cap_frame(full_task.loc[full_task["split_name"] == "train"].copy(), max_rows=20000, random_state=42)
    val_df = _cap_frame(full_task.loc[full_task["split_name"] == "val"].copy(), max_rows=5000, random_state=43)
    test_df = full_task.loc[full_task["split_name"] == "test"].copy()
    workset_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    workset_keys = workset_df[KEY_COLUMNS].drop_duplicates()
    workset_roi = roi_manifest.merge(workset_keys, on=KEY_COLUMNS, how="inner", validate="one_to_one")

    workset_task_path = DATASET_INDEX_ROOT / "task_a43_full_realistic_modeling_workset.parquet"
    workset_roi_path = CROPS_ROOT / "roi_manifest_modeling_workset.parquet"
    workset_df.to_parquet(workset_task_path, index=False)
    workset_roi.to_parquet(workset_roi_path, index=False)

    report_lines = [
        "# Modeling Workset QC",
        "",
        f"- Train rows: `{len(train_df)}`",
        f"- Validation rows: `{len(val_df)}`",
        f"- Test rows: `{len(test_df)}`",
        f"- Total workset rows: `{len(workset_df)}`",
        f"- ROI rows matched: `{len(workset_roi)}`",
        f"- Task artifact: `{workset_task_path}`",
        f"- ROI artifact: `{workset_roi_path}`",
    ]
    report_path = REPORTS_ROOT / "modeling_workset_qc.md"
    write_markdown(report_path, report_lines)

    return {
        "task_workset_parquet": str(workset_task_path),
        "roi_workset_parquet": str(workset_roi_path),
        "qc_report_md": str(report_path),
    }


def main() -> None:
    print(json.dumps(build_modeling_workset(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
