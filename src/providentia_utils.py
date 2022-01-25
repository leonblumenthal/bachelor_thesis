import json
import os
from typing import Dict

import numpy as np

from .perspective import Perspective


def parse_perspectives(dir_path: str) -> Dict[str, Perspective]:
    """Parse perspectives from JSON files in specified directory."""
    perspectives = {}

    for filename in os.listdir(dir_path):
        if not filename.endswith('.json'):
            continue

        with open(os.path.join(dir_path, filename)) as f:
            data = json.load(f)

        keys = {'rotation_matrix', 'translation', 'intrinsic_matrix'}
        if not keys.issubset(data):
            continue

        name = filename.split('.')[0]

        rotation_matrix = np.array(data['rotation_matrix']).T
        translation = np.array(data['extrinsic_matrix'])[:, -1:]
        translation = -rotation_matrix.T @ translation
        intrinsic_matrix = np.array(data['intrinsic_matrix'])

        perspectives[name] = Perspective(rotation_matrix, translation, intrinsic_matrix)

    return perspectives


def match_perspective(
    perspectives: Dict[str, Perspective], labels_data: Dict
) -> Perspective:
    """Get perspective from dict by matching name to labels data filename."""

    for name, perspective in perspectives.items():
        if name in labels_data['image_file_name']:
            return perspective

    return None
