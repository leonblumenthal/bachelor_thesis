import os
from typing import Iterator, List

import torch

from data_models import Detection


class DetectionLoader:
    """Helper class to load detection lists from a directory by index."""

    def __init__(self, dir_path: str) -> None:
        self.paths = [
            os.path.join(dir_path, name)
            for name in sorted(os.listdir(dir_path))
            if name.endswith('.detection')
        ]

    def load_detection_list(self, index: int) -> List[Detection]:
        """Load single detection list."""

        detections = torch.load(self.paths[index])

        return detections

    def load_detection_lists(
        self, indices: List[int] = None
    ) -> Iterator[List[Detection]]:
        """Lazily load all detection lists."""

        if indices is None:
            indices = range(len(self.paths))

        for i in indices:
            yield self.load_detection_list(i)
