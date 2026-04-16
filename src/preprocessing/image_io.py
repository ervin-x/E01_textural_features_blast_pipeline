from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_binary_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    return mask > 0


def save_rgb_image(array: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def yolo_to_bbox_pixels(
    image_width: int,
    image_height: int,
    x_center_norm: float,
    y_center_norm: float,
    width_norm: float,
    height_norm: float,
) -> tuple[int, int, int, int]:
    bbox_width = max(1, int(round(width_norm * image_width)))
    bbox_height = max(1, int(round(height_norm * image_height)))
    center_x = x_center_norm * image_width
    center_y = y_center_norm * image_height

    x1 = int(round(center_x - bbox_width / 2))
    y1 = int(round(center_y - bbox_height / 2))
    x2 = x1 + bbox_width
    y2 = y1 + bbox_height

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))
    return x1, y1, x2, y2


def crop_array(array: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return array[y1:y2, x1:x2]
