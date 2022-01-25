import torch

from ..models import Detection
from .loader import Loader


class DetectionsLoader(Loader):
    """Helper class to load detection lists from a directory by index"""

    def __init__(self, dir_path: str):
        super().__init__(dir_path, '.detection')

    @classmethod
    def _load_item(cls, path: str, **kwargs) -> Detection:
        detections = torch.load(path)

        return detections
