from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Detection:
    """Dataclass that represents a detection of a single vhicle."""

    anchor: Tuple[int, int]
    mask: np.array
    box: Tuple[float, float, float, float]
    label: int  # class
    score: float


def merge_masks(detections: List[Detection], dimensions: Tuple[int, int]) -> np.ndarray:
    """Merge bool masks of all detections into a single mask with specified dimensions."""

    mask = np.zeros(dimensions, dtype=bool)

    for detection in detections:
        h, w = detection.mask.shape
        x, y = detection.anchor

        mask[y : y + h, x : x + w] |= detection.mask

    return mask
