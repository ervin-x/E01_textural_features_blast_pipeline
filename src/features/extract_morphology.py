from __future__ import annotations

import numpy as np
from skimage.measure import perimeter, regionprops


def morphology_from_mask(binary_mask: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    area = float(binary_mask.sum())
    features["morph_area_px"] = area
    if area <= 0:
        features.update(
            {
                "morph_perimeter_px": 0.0,
                "morph_extent": 0.0,
                "morph_eccentricity": 0.0,
                "morph_solidity": 0.0,
                "morph_roundness": 0.0,
                "morph_bbox_fill_ratio": 0.0,
            }
        )
        return features

    props = regionprops(binary_mask.astype(np.uint8))[0]
    perim = float(perimeter(binary_mask, neighborhood=8))
    bbox_area = float((props.bbox[2] - props.bbox[0]) * (props.bbox[3] - props.bbox[1]))
    features["morph_perimeter_px"] = perim
    features["morph_extent"] = float(props.extent)
    features["morph_eccentricity"] = float(props.eccentricity)
    features["morph_solidity"] = float(props.solidity)
    features["morph_roundness"] = float((4.0 * np.pi * area / (perim ** 2)) if perim > 0 else 0.0)
    features["morph_bbox_fill_ratio"] = float(area / bbox_area) if bbox_area > 0 else 0.0
    return features


def morphology_from_bbox(width_px: int, height_px: int) -> dict[str, float]:
    area = float(width_px * height_px)
    perimeter_px = float(2 * (width_px + height_px))
    aspect_ratio = float(width_px / height_px) if height_px > 0 else 0.0
    return {
        "morph_bbox_area_px": area,
        "morph_bbox_width_px": float(width_px),
        "morph_bbox_height_px": float(height_px),
        "morph_bbox_perimeter_px": perimeter_px,
        "morph_bbox_aspect_ratio": aspect_ratio,
        "morph_bbox_diagonal_px": float(np.hypot(width_px, height_px)),
    }
