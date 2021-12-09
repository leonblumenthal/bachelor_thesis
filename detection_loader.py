import os
from typing import Iterator, List

import torch
from dotenv import load_dotenv

from data_models import Detection


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