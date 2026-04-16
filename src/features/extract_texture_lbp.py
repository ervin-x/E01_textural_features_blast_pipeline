from __future__ import annotations

import numpy as np
from skimage.feature import local_binary_pattern


def _downsample(gray_image: np.ndarray, binary_mask: np.ndarray | None = None, max_side: int = 96) -> tuple[np.ndarray, np.ndarray | None]:
    working = gray_image
    mask = binary_mask
    max_dim = max(working.shape[:2])
    if max_dim <= max_side:
        return working, mask
    step = int(np.ceil(max_dim / max_side))
    working = working[::step, ::step]
    if mask is not None:
        mask = mask[::step, ::step]
    return working, mask


def texture_lbp_features(gray_image: np.ndarray, binary_mask: np.ndarray | None = None) -> dict[str, float]:
    gray_image, binary_mask = _downsample(gray_image, binary_mask)
    working = np.clip(gray_image, 0, 255).astype(np.uint8, copy=False)
    lbp_r1 = local_binary_pattern(working, P=8, R=1, method="uniform")
    lbp_r2 = local_binary_pattern(working, P=16, R=2, method="uniform")

    if binary_mask is None:
        values_r1 = lbp_r1.ravel()
        values_r2 = lbp_r2.ravel()
    else:
        values_r1 = lbp_r1[binary_mask]
        values_r2 = lbp_r2[binary_mask]
        if values_r1.size == 0:
            values_r1 = lbp_r1.ravel()
            values_r2 = lbp_r2.ravel()

    hist_r1, _ = np.histogram(values_r1, bins=np.arange(0, 8 + 3), density=True)
    hist_r2, _ = np.histogram(values_r2, bins=np.arange(0, 16 + 3), density=True)

    features: dict[str, float] = {}
    for idx, value in enumerate(hist_r1):
        features[f"texture_lbp_r1_bin_{idx:02d}"] = float(value)
    for idx, value in enumerate(hist_r2):
        features[f"texture_lbp_r2_bin_{idx:02d}"] = float(value)
    return features
