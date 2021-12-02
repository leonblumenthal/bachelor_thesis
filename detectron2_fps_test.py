from argparse import ArgumentParser
import time

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

assert torch.__version__.startswith("1.10") and torch.cuda.is_available()

import providentia_camera_dataset as pcd


def test_fps(device: str):
    torch.cuda.set_device(device)

    print('Loading images...')
    images = list(pcd.load_images())

    print('Loading model...')
    model_name = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    cfg = get_cfg()
    cfg.INPUT.FORMAT = 'RGB'
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    predictor = DefaultPredictor(cfg)

    print('Runnning inference...')
    start_time = time.perf_counter()
    for image in images:
        predictor(image)
    elapsed_time = time.perf_counter() - start_time

    print(f'Elapsed time: {elapsed_time:.0f} s')
    print(f'Number of images: {len(images)}')
    print(f'Dimensions of images: {images[0].shape}')
    print(
        f'Performance: {len(images) / elapsed_time:.2f} fps, {elapsed_time / len(images):.2f} s/img'
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('cuda_device', type=int)
    args = parser.parse_args()

    test_fps(device=f'cuda:{args.cuda_device}')
