from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

DEFAULT_CONFIG = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'


def load_default_predictor(
    model_name: str,
    input_format: str = 'RGB',
) -> DefaultPredictor:
    """Load default predictor with model from model zoo."""

    cfg = get_cfg()
    cfg.INPUT.FORMAT = input_format
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)

    return DefaultPredictor(cfg)
