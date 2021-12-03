import os
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import torch
from dotenv import load_dotenv


@dataclass
class Detection:
    """Dataclass that represents a detection of a single vhicle."""

    anchor: Tuple[int, int]
    mask: np.array
    box: Tuple[float, float, float, float]
    label: int  # class
    score: float


load_dotenv()


_detections_path = os.getenv(
    'DETECTIONS_PATH',
    './data/detections',
)
if not os.path.exists(_detections_path):
    raise Exception('Detections path is invalid')


detection_list_paths = [
    os.path.join(_detections_path, name)
    for name in sorted(os.listdir(_detections_path))
    if name.endswith('.detection')
]


def load_detection_list(index: int) -> List[Detection]:
    """Load single detection list."""

    detections = torch.load(detection_list_paths[index])

    return detections


def load_detection_lists(indices: List[int] = None) -> Iterator[List[Detection]]:
    """Lazily load all detection lists."""

    if indices is None:
        indices = range(len(detection_list_paths))

    for i in indices:
        yield load_detection_list(i)


def merge_masks(detections: List[Detection], dimensions: Tuple[int, int]) -> np.ndarray:
    """Merge bool masks of all detections into a single mask with specified dimensions."""

    mask = np.zeros(dimensions, dtype=bool)

    for detection in detections:
        h, w = detection.mask.shape
        x, y = detection.anchor

        mask[y : y + h, x : x + w] |= detection.mask

    return mask