import json
from typing import Dict, List, Tuple

from .loader import Loader
from ..models import Box3D, Dimension, Label, Location, ProjectedBox


class LabelsLoader(Loader):
    """Helper class to load label lists and meta data from a directory by index"""

    def __init__(self, dir_path: str):
        super().__init__(dir_path, '.json')

    def _load_item(self, path: str, **kwargs) -> Tuple[List[Label], Dict]:
        with open(path) as f:
            data = json.load(f)

        labels_data = data.pop('labels')
        labels = [self._create_label(label_data) for label_data in labels_data]

        return labels, data

    @staticmethod
    def _create_label(
        label_data: Dict, projected_scale: Tuple[float, float] = (2 * 1920, 2 * 1200)
    ) -> Label:
        """Create label from single label data."""
        px, py = projected_scale
        projected_box = ProjectedBox(
            **{
                k: (x * px, y * py)
                for k, (x, y) in label_data['box3d_projected'].items()
            }
        )

        location = Location(**label_data['box3d']['location'])
        dimension = Dimension(**label_data['box3d']['dimension'])
        box3d = Box3D(
            dimension, location, yaw=label_data['box3d']['orientation']['rotationYaw']
        )

        label = Label(label_data['id'], label_data['category'], box3d, projected_box)

        return label
