from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets" / "local_paths.json"
TEMPLATE_CONFIG_PATH = PROJECT_ROOT / "configs" / "datasets" / "local_paths.template.json"


def _normalize_path(path_value: str | Path, field_name: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configured path for `{field_name}` does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Configured path for `{field_name}` is not a directory: {path}")
    return path


def _load_config(config_path: Path) -> tuple[Path, Path, str]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if "data_root" not in payload or "masks_root" not in payload:
        raise KeyError(
            f"Dataset config must contain `data_root` and `masks_root`: {config_path}"
        )
    data_root = _normalize_path(payload["data_root"], "data_root")
    masks_root = _normalize_path(payload["masks_root"], "masks_root")
    return data_root, masks_root, f"config:{config_path}"


def resolve_dataset_roots(
    data_root: str | Path | None = None,
    masks_root: str | Path | None = None,
    config_path: str | Path | None = None,
) -> tuple[Path, Path, str]:
    if (data_root is None) ^ (masks_root is None):
        raise ValueError("Pass both `data_root` and `masks_root`, or neither of them.")

    if data_root is not None and masks_root is not None:
        return (
            _normalize_path(data_root, "data_root"),
            _normalize_path(masks_root, "masks_root"),
            "cli",
        )

    env_data_root = os.environ.get("E01_DATA_ROOT")
    env_masks_root = os.environ.get("E01_MASKS_ROOT")
    if env_data_root and env_masks_root:
        return (
            _normalize_path(env_data_root, "E01_DATA_ROOT"),
            _normalize_path(env_masks_root, "E01_MASKS_ROOT"),
            "env",
        )

    candidate_config = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    if candidate_config.exists():
        return _load_config(candidate_config)

    legacy_data_root = PROJECT_ROOT.parent / "Data" / "data"
    legacy_masks_root = PROJECT_ROOT.parent / "Data" / "data_masks"
    if legacy_data_root.exists() and legacy_masks_root.exists():
        return legacy_data_root.resolve(), legacy_masks_root.resolve(), "legacy_default"

    raise FileNotFoundError(
        "Dataset paths are not configured. "
        f"Create `{DEFAULT_CONFIG_PATH}` from `{TEMPLATE_CONFIG_PATH}`, "
        "or set `E01_DATA_ROOT` and `E01_MASKS_ROOT`, "
        "or pass `--data-root` and `--masks-root` to the entrypoint script."
    )
