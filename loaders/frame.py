import cv2
import numpy as np

from .loader import Loader


class FrameLoader(Loader):
    """Helper class to load frames from a directory by index."""

    def __init__(self, dir_path: str, ending: str = '.jpg'):
        super().__init__(dir_path, ending)

    @classmethod
    def _load_item(cls, path: str, **kwargs) -> np.ndarray:
        image = cv2.imread(path)

        # Convert from BGR to RGB.
        if kwargs.get('rgb', True):
            image = image[:, :, ::-1]

        return image
