import json
import os
from typing import Dict, Iterator, List

import cv2
import numpy as np
from dotenv import load_dotenv


load_dotenv()


dataset_path = os.getenv(
    'PROVIDENTIA_CAMERA_DATASET_PATH',
    './data/providentia_camera_dataset',
)
if not os.path.exists(dataset_path):
    raise Exception('Providentia camera dataset path is invalid')


_images_dir = os.path.join(dataset_path, 'images')
image_paths = [
    os.path.join(_images_dir, name)
    for name in sorted(os.listdir(_images_dir))
    if name.endswith('.jpg')
]

_labels_dir = os.path.join(dataset_path, 'labels')
label_paths = [
    os.path.join(_labels_dir, name)
    for name in sorted(os.listdir(_labels_dir))
    if name.endswith('.json')
]


def load_image(index: int, rgb: bool = True) -> np.ndarray:
    """Load single image by index and convert to RGB if necessary."""

    image = cv2.imread(image_paths[index])

    # Convert from BGR to RGB.
    if rgb:
        image = image[:, :, ::-1]

    return image


def load_images(indices: List[int] = None, rgb: bool = True) -> Iterator[np.ndarray]:
    """Lazily load all images and convert them to RGB if necessary."""

    if indices is None:
        indices = range(len(image_paths))

    for i in indices:
        yield load_image(i, rgb)


def load_label(index: int) -> Dict:
    """Load single label json."""

    with open(label_paths[index]) as f:
        label = json.load(f)

    return label


def load_labels(indices: List[int] = None) -> Iterator[Dict]:
    """Lazily load all labels."""

    if indices is None:
        indices = range(len(label_paths))

    for i in indices:
        yield load_label(i)
