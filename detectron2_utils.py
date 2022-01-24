from typing import List, Tuple

import numpy as np
import torch
from detectron2.structures import Instances
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from src.models import Detection


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


def create_detections(instances: Instances) -> List[Detection]:
    """Create detections from detectron2 instances."""

    detections = []

    for mask, box, label, score in zip(
        instances.pred_masks,
        instances.pred_boxes,
        instances.pred_classes,
        instances.scores,
    ):  
        # Some masks are empty.
        if not mask.any():
            continue

        mask_box = _get_mask_box(mask)

        detection = Detection(
            anchor=mask_box[:2],
            mask=_trim_mask(mask_box, mask),
            box=tuple(box.tolist()),
            label=label.item(),
            score=score.item(),
        )

        detections.append(detection)

    return detections


def load_default_predictor(
    model_name: str = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    input_format: str = 'RGB',
) -> DefaultPredictor:
    """Load default predictor with model from model zoo."""

    cfg = get_cfg()
    cfg.INPUT.FORMAT = input_format
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)

    return DefaultPredictor(cfg)
