import json
import os
from typing import Dict

import numpy as np

from .perspective import Perspective


def parse_perspective(path: str) -> Perspective:
    """Parse single perspective from JSON file."""

    with open(path) as f:
        data = json.load(f)

    keys = {'rotation_matrix', 'translation', 'intrinsic_matrix'}
    if not keys.issubset(data):
        return None

    rotation_matrix = np.array(data['rotation_matrix']).T
    translation = np.array(data['extrinsic_matrix'])[:, -1:]
    translation = -rotation_matrix.T @ translation
    intrinsic_matrix = np.array(data['intrinsic_matrix'])
    image_shape = data['image_height'], data['image_width']

    perspective = Perspective(
        rotation_matrix, translation, intrinsic_matrix, image_shape
    )

    return perspective


def parse_perspectives(dir_path: str) -> Dict[str, Perspective]:
    """Parse perspectives from JSON files in specified directory."""
    perspectives = {}

    for filename in os.listdir(dir_path):
        if not filename.endswith('.json'):
            continue

        path = os.path.join(dir_path, filename)

        perspective = parse_perspective(path)
        if perspective is None:
            continue

        name = filename.split('.')[0]
        perspectives[name] = perspective

    return perspectives


def match_perspective(
    perspectives: Dict[str, Perspective], labels_data: Dict
) -> Perspective:
    """Get perspective from dict by matching name to labels data filename."""

    for name, perspective in perspectives.items():
        if name in labels_data['image_file_name']:
            return perspective

    return None
