import os
from typing import Iterator, List

import cv2
import numpy as np


class FrameLoader:
    """Helper class to load frames from a directory by index."""

    def __init__(self, dir_path: str, ending: str = '.jpg') -> None:
        self.paths = [
            os.path.join(dir_path, name)
            for name in sorted(os.listdir(dir_path))
            if name.endswith(ending)
        ]

    def load_frame(self, index: int, rgb: bool = True) -> np.ndarray:
        """Load single frame by index and convert to RGB if necessary."""

        image = cv2.imread(self.paths[index])

        # Convert from BGR to RGB.
        if rgb:
            image = image[:, :, ::-1]

        return image

    def load_frames(
        self, indices: List[int] = None, rgb: bool = True
    ) -> Iterator[np.ndarray]:
        """Lazily load all frames and convert them to RGB if necessary."""

        if indices is None:
            indices = range(len(self.paths))

        for i in indices:
            yield self.load_frame(i, rgb)
