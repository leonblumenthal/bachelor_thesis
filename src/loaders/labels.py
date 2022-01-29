import json
from typing import Dict, List, Tuple

import numpy as np

from ..models import Vehicle
from .loader import Loader


class LabelsLoader(Loader):
    """Helper class to load label lists and meta data from a directory by index"""

    def __init__(self, dir_path: str):
        super().__init__(dir_path, '.json')

    def _load_item(self, path: str, **kwargs) -> Tuple[List[Vehicle], Dict]:
        with open(path) as f:
            data = json.load(f)

        labels_data = data.pop('labels')
        labels = [self._create_vehicle(label_data) for label_data in labels_data]

        return labels, data

    @staticmethod
    def _create_vehicle(label_data: Dict) -> Vehicle:
        """Create vehicle from single label data."""

        location = np.array(
            [label_data['box3d']['location'][c] for c in 'xyz']
        ).reshape(3, 1)
        dimensions = tuple(
            label_data['box3d']['dimension'][key]
            for key in ('length', 'width', 'height')
        )
        yaw = label_data['box3d']['orientation']['rotationYaw']

        label = Vehicle(location, dimensions, yaw, label_data['category'])

        return label
