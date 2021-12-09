from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Dimension:
    length: float
    width: float
    height: float


@dataclass
class Location:
    x: float
    y: float
    z: float


@dataclass
class Box3D:
    dimension: Dimension
    location: Location
    yaw: float


@dataclass
class ProjectedBox:
    bottom_left_front: Tuple[float, float]
    bottom_left_back: Tuple[float, float]
    bottom_right_back: Tuple[float, float]
    bottom_right_front: Tuple[float, float]
    top_left_front: Tuple[float, float]
    top_left_back: Tuple[float, float]
    top_right_back: Tuple[float, float]
    top_right_front: Tuple[float, float]


@dataclass
class Label:
    """Single Label of providentia camera dataset"""

    id: int
    category: str
    box3d: Box3D
    projected_box: ProjectedBox


@dataclass
class Detection:
    """Detection of a single vehicle"""

    anchor: Tuple[int, int]
    mask: np.array
    box: Tuple[float, float, float, float]
    label: int  # class
    score: float
