from __future__ import annotations

import numpy as np
from skimage.color import rgb2hsv, rgb2lab


def _stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q50": float(np.quantile(values, 0.50)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
    }


def color_features(image_rgb: np.ndarray, binary_mask: np.ndarray | None = None) -> dict[str, float]:
    rgb = image_rgb.astype(np.float32) / 255.0
    if binary_mask is None:
        pixels = rgb.reshape(-1, 3)
    else:
        pixels = rgb[binary_mask]
        if pixels.size == 0:
            pixels = rgb.reshape(-1, 3)

    hsv_pixels = rgb2hsv(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    lab_pixels = rgb2lab(pixels.reshape(-1, 1, 3)).reshape(-1, 3)

    features: dict[str, float] = {}
    for index, channel in enumerate(["r", "g", "b"]):
        features.update(_stats(f"color_{channel}", pixels[:, index]))
    for index, channel in enumerate(["h", "s", "v"]):
        features.update(_stats(f"hsv_{channel}", hsv_pixels[:, index]))
    for index, channel in enumerate(["l", "a", "b"]):
        features.update(_stats(f"lab_{channel}", lab_pixels[:, index]))
    return features
