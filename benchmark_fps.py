"""Measure FPS using all frames in a directory."""

import time
import warnings
from argparse import ArgumentParser

import torch

from src import detectron2_utils, yolact_utils
from src.loaders import FrameLoader

# Ignore Detectron2 warning.
warnings.simplefilter('ignore', UserWarning)


def test_fps(
    src_dir: str, model: str, config: str, weights_dir: str, num_warmup_frames: int = 10
):

    frame_loader = FrameLoader(src_dir)
    num_frames = len(frame_loader.paths)
    print(f'Found {num_frames} frames')

    print(f'Loading {model} model {config}...')
    if model == 'detectron2':
        if config == 'default':
            config = detectron2_utils.DEFAULT_CONFIG
        predictor = detectron2_utils.load_default_predictor(config, input_format='BGR')
    elif model == 'yolact':
        predictor = yolact_utils.load_predictor(config, weights_dir)

    print('Running inference')
    total_time = 0
    for i, frame in enumerate(frame_loader.load_items(rgb=False)):
        start_time = time.perf_counter()
        predictor(frame)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        if i >= num_warmup_frames:
            total_time += end_time - start_time

    fps = (num_frames - num_warmup_frames) / total_time
    print('Finished')
    print(f'Total seconds: {total_time:.02f}')
    print(f'FPS: {fps:.02f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str)
    parser.add_argument('model', choices=['detectron2', 'yolact'])
    parser.add_argument(
        'config', type=str, help='flexible for detectron2, limited choices for yolact'
    )
    parser.add_argument('--weights_dir', default='weights')
    args = parser.parse_args()

    test_fps(args.src_dir, args.model, args.config, args.weights_dir)
