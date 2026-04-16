from __future__ import annotations

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import laplace, sobel


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


def texture_glcm_features(gray_image: np.ndarray, binary_mask: np.ndarray | None = None) -> dict[str, float]:
    gray_image, binary_mask = _downsample(gray_image, binary_mask)
    if binary_mask is not None:
        working = gray_image.copy()
        working[~binary_mask] = 0
    else:
        working = gray_image

    quantized = np.clip((working / 8).astype(np.uint8), 0, 31)
    glcm = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )
    features: dict[str, float] = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
        values = graycoprops(glcm, prop)
        features[f"texture_glcm_{prop.lower()}_mean"] = float(np.mean(values))
        features[f"texture_glcm_{prop.lower()}_std"] = float(np.std(values))

    sobel_img = sobel(working.astype(np.float32) / 255.0)
    laplace_img = laplace(working.astype(np.float32) / 255.0)
    features["texture_sobel_mean"] = float(np.mean(sobel_img))
    features["texture_sobel_std"] = float(np.std(sobel_img))
    features["texture_laplace_mean"] = float(np.mean(laplace_img))
    features["texture_laplace_std"] = float(np.std(laplace_img))
    return features
