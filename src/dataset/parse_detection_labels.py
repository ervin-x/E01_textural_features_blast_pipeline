from __future__ import annotations

from pathlib import Path


def normalize_class_id(raw_class_id: str) -> tuple[int, bool]:
    """Normalize class id and report whether it was stored as a float-like token."""
    class_id = int(float(raw_class_id))
    was_float_like = raw_class_id.strip() != str(class_id)
    return class_id, was_float_like


def parse_label_file(label_path: Path) -> list[dict[str, object]]:
    """Parse YOLO-like detection labels into structured rows."""
    rows: list[dict[str, object]] = []

    for object_id, raw_line in enumerate(
        line for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()
    ):
        parts = raw_line.split()
        if len(parts) < 5:
            raise ValueError(f"Malformed detection line in {label_path}: {raw_line!r}")

        class_id, class_id_was_float = normalize_class_id(parts[0])
        rows.append(
            {
                "object_id_within_image": object_id,
                "raw_class_id": parts[0],
                "class_id": class_id,
                "class_id_was_float": class_id_was_float,
                "x_center_norm": float(parts[1]),
                "y_center_norm": float(parts[2]),
                "width_norm": float(parts[3]),
                "height_norm": float(parts[4]),
                "raw_line": raw_line,
            }
        )

    return rows
