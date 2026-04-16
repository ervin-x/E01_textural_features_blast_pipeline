from __future__ import annotations

import re
from pathlib import Path


MASK_FILENAME_RE = re.compile(
    r"^(?P<image_id>.+?)_class_(?P<class_id>[^_]+)_obj_(?P<object_id>\d+)\.png$"
)


def parse_mask_filename(mask_path: Path) -> dict[str, object] | None:
    """Parse mask filename into image id, class id and object index."""
    match = MASK_FILENAME_RE.match(mask_path.name)
    if not match:
        return None

    return {
        "image_id": match.group("image_id"),
        "class_id": int(float(match.group("class_id"))),
        "object_id_within_image": int(match.group("object_id")),
        "mask_path": str(mask_path.resolve()),
    }


def build_mask_index(mask_dir: Path) -> dict[str, list[dict[str, object]]]:
    """Build an image_id -> sorted mask entries index."""
    index: dict[str, list[dict[str, object]]] = {}
    if not mask_dir.exists():
        return index

    for mask_path in sorted(mask_dir.glob("*.png")):
        parsed = parse_mask_filename(mask_path)
        if parsed is None:
            continue

        image_id = str(parsed["image_id"])
        index.setdefault(image_id, []).append(parsed)

    for image_id, entries in index.items():
        entries.sort(key=lambda item: int(item["object_id_within_image"]))

    return index
