from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
VENDOR_ROOT = PROJECT_ROOT / "vendor"
VENV_SITE_PACKAGES = sorted((PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages"))

for candidate in [*VENV_SITE_PACKAGES, SRC_ROOT, VENDOR_ROOT]:
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.append(candidate_str)

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DATASET_INDEX_ROOT = OUTPUT_ROOT / "dataset_index"
SPLITS_ROOT = OUTPUT_ROOT / "splits"
TABLES_ROOT = OUTPUT_ROOT / "tables"
REPORTS_ROOT = OUTPUT_ROOT / "reports"
PREDICTIONS_ROOT = OUTPUT_ROOT / "predictions"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
FEATURES_ROOT = OUTPUT_ROOT / "features"
CROPS_ROOT = OUTPUT_ROOT / "crops"
LOGS_ROOT = OUTPUT_ROOT / "logs"
CHECKPOINTS_ROOT = OUTPUT_ROOT / "checkpoints"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_layout() -> None:
    for directory in [
        OUTPUT_ROOT,
        DATASET_INDEX_ROOT,
        SPLITS_ROOT,
        TABLES_ROOT,
        REPORTS_ROOT,
        PREDICTIONS_ROOT,
        FIGURES_ROOT,
        FEATURES_ROOT,
        CROPS_ROOT,
        LOGS_ROOT,
        CHECKPOINTS_ROOT,
    ]:
        ensure_dir(directory)


def write_markdown(path: Path, lines: list[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
