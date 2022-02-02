import os

import gdown

from .yolact_edge_predictor import YolactEdgePredictor

# name: (model config, weights google drive download id)
CONFIGS = {
    'coco_resnet50': ('yolact_resnet50_config', '15TRS8MNNe3pmjilonRy9OSdJdCPl5DhN'),
    'coco_resnet50_edge': (
        'yolact_edge_resnet50_config',
        '15TRS8MNNe3pmjilonRy9OSdJdCPl5DhN',
    ),
    'coco_resnet101': ('yolact_base_config', '1EAzO-vRDZ2hupUJ4JFSUi40lAZ5Jo-Bp'),
    'coco_resnet101_edge': ('yolact_edge_config', '1EAzO-vRDZ2hupUJ4JFSUi40lAZ5Jo-Bp'),
}


def load_predictor(config: str, weights_dir: str) -> YolactEdgePredictor:
    """Download weights and load Yolact Edge predictor based on config."""
    if config not in CONFIGS:
        raise ValueError(f'{config} config is unknown')

    model_config, gdrive_id = CONFIGS[config]

    url = f'https://drive.google.com/u/0/uc?id={gdrive_id}'
    # Weights for edge and non-edge variants are equal.
    weights_path = os.path.join(weights_dir, '_'.join(config.split('_')[:2]) + '.pth')

    gdown.cached_download(url, weights_path)

    predictor = YolactEdgePredictor(model_config, weights_path)

    return predictor
