from dataclasses import dataclass
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
