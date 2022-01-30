from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class Vehicle:
    """Representation of a single vehicle"""

    location: np.ndarray  # x, y, z
    dimensions: Tuple[float, float, float]  # length, width, height
    yaw: float
    category: str


@dataclass
class Detection:
    """Detection of a single vehicle"""

    anchor: Tuple[int, int]
    mask: np.array
    box: Tuple[float, float, float, float]
    label: int  # class
    score: float
    category: str = None


@dataclass
class DirectionLine:
    """
    Line on ground plane specified by unit normal vector and bias.
    Direction of vehicles: (down |---normal---> up)
    """

    normal_vector: np.ndarray
    bias: float
    direction_vector: np.ndarray = field(init=False)

    def __post_init__(self):
        self.direction_vector = self.normal_vector[[1, 0]].copy()
        self.direction_vector[0] *= -1
