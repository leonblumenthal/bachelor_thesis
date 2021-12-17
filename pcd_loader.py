import json
import os
from typing import Dict, Iterator, List, Tuple

from dotenv import load_dotenv

from data_models import Box3D, Dimension, Label, Location, ProjectedBox
from frame_loader import FrameLoader


load_dotenv()


dataset_path = os.getenv(
    'PROVIDENTIA_CAMERA_DATASET_PATH',
    './data/providentia_camera_dataset',
)
if not os.path.exists(dataset_path):
    raise Exception('Providentia camera dataset path is invalid')


_image_loader = FrameLoader(os.path.join(dataset_path, 'images'))
load_image = _image_loader.load_frame
load_images = _image_loader.load_frames


_labels_dir = os.path.join(dataset_path, 'labels')
label_paths = [
    os.path.join(_labels_dir, name)
    for name in sorted(os.listdir(_labels_dir))
    if name.endswith('.json')
]


def _create_label(
    label_data: Dict, projected_scale: Tuple[float, float] = (2 * 1920, 2 * 1200)
) -> Label:
    """Create label from single label data."""
    px, py = projected_scale
    projected_box = ProjectedBox(
        **{k: (x * px, y * py) for k, (x, y) in label_data['box3d_projected'].items()}
    )

    location = Location(**label_data['box3d']['location'])
    dimension = Dimension(**label_data['box3d']['dimension'])
    box3d = Box3D(
        dimension, location, yaw=label_data['box3d']['orientation']['rotationYaw']
    )

    label = Label(label_data['id'], label_data['category'], box3d, projected_box)

    return label


def load_label_list(index: int) -> Tuple[List[Label], Dict]:
    """Load labels list and info data."""

    with open(label_paths[index]) as f:
        data = json.load(f)

    labels_data = data.pop('labels')
    labels = [_create_label(label_data) for label_data in labels_data]

    return labels, data


def load_label_lists(indices: List[int] = None) -> Iterator[Tuple[List[Label], Dict]]:
    """Lazily load all labels."""

    if indices is None:
        indices = range(len(label_paths))

    for i in indices:
        yield load_label_list(i)
