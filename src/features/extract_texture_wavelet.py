from __future__ import annotations

import numpy as np
import pywt


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


def texture_wavelet_features(gray_image: np.ndarray, binary_mask: np.ndarray | None = None) -> dict[str, float]:
    gray_image, binary_mask = _downsample(gray_image, binary_mask)
    working = gray_image.astype(np.float32)
    if binary_mask is not None:
        working = working.copy()
        working[~binary_mask] = 0.0

    coeffs2 = pywt.dwt2(working, "haar")
    cA, (cH, cV, cD) = coeffs2
    features: dict[str, float] = {}
    for name, coeff in [("ca", cA), ("ch", cH), ("cv", cV), ("cd", cD)]:
        features[f"wavelet_{name}_mean"] = float(np.mean(coeff))
        features[f"wavelet_{name}_std"] = float(np.std(coeff))
        features[f"wavelet_{name}_energy"] = float(np.mean(np.square(coeff)))
    return features
