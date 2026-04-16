from __future__ import annotations

import re
from pathlib import Path


CLASS_WITH_ID_RE = re.compile(r"^\s*(?P<class_id>\d+(?:\.0+)?)\s+(?P<class_name>.+?)\s*$")


def parse_predefined_classes(path: Path) -> dict[int, str]:
    """Parse predefined_classes.txt in either indexed or line-only format."""
    mapping: dict[int, str] = {}
    next_index = 0

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = CLASS_WITH_ID_RE.match(line)
        if match:
            class_id = int(float(match.group("class_id")))
            class_name = match.group("class_name").strip()
        else:
            class_id = next_index
            class_name = line

        mapping[class_id] = class_name
        next_index = max(next_index, class_id + 1)

    return mapping


def mapping_signature(mapping: dict[int, str]) -> tuple[tuple[int, str], ...]:
    """Return a hashable signature for a class mapping."""
    return tuple(sorted(mapping.items()))


def normalize_mapping_for_json(mapping: dict[int, str]) -> dict[str, str]:
    """Convert class mapping keys to strings for JSON serialization."""
    return {str(class_id): class_name for class_id, class_name in sorted(mapping.items())}
