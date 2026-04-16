from __future__ import annotations

import argparse
from pathlib import Path

from reproducibility_blocks import print_payload, run_foundation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E01 foundation block (A0-A5).")
    parser.add_argument("--data-root", type=Path, default=None, help="Absolute path to the original dataset `data`.")
    parser.add_argument("--masks-root", type=Path, default=None, help="Absolute path to the masks dataset `data_masks`.")
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=None,
        help="Optional JSON config with `data_root` and `masks_root`.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_payload(run_foundation(data_root=args.data_root, masks_root=args.masks_root, config_path=args.paths_config))
