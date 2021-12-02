from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Detection:
    """Dataclass that represents a detection of a single vhicle."""

    anchor: Tuple[int, int]
    mask: np.array
    box: Tuple[float, float, float, float]
    label: int  # class
    score: float
