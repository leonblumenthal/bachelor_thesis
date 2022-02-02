from typing import List, Tuple

import numpy as np
import torch

from .models import Detection


def _get_mask_box(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """Get bounding box from bool mask as (x min, y min, x max, y max)."""

    x_min, x_max = torch.where(mask.any(dim=0))[0][[0, -1]].tolist()
    y_min, y_max = torch.where(mask.any(dim=1))[0][[0, -1]].tolist()

    return x_min, y_min, x_max, y_max


def _trim_mask(
    bounding_box: Tuple[int, int, int, int], mask: torch.Tensor
) -> np.ndarray:
    """Trim mask based on bounding boxand and convert to numpy."""

    x_min, y_min, x_max, y_max = bounding_box

    return mask[y_min : y_max + 1, x_min : x_max + 1].numpy()


def create_detections(
    masks: torch.Tensor,
    boxes: torch.Tensor,
    classes: torch.Tensor,
    scores: torch.Tensor,
) -> List[Detection]:
    """Create detections from masks, boxes, classes, and scores."""

    detections = []

    for mask, box, label, score in zip(
        masks.to('cpu'),
        boxes.to('cpu'),
        classes.to('cpu'),
        scores.to('cpu'),
    ):
        # Some masks are empty.
        if not mask.any():
            continue

        mask_box = _get_mask_box(mask)

        detection = Detection(
            anchor=mask_box[:2],
            mask=_trim_mask(mask_box, mask).astype(bool),
            box=tuple(box.tolist()),
            label=label.item(),
            score=score.item(),
        )

        detections.append(detection)

    return detections
