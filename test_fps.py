"""
Load every frame in a specified directory into cpu memeory 
and measure inference fps.
"""

import time
import warnings
from argparse import ArgumentParser

from src import detectron2_utils, yolact_utils
from src.loaders import FrameLoader

# Ignore Detectron2 warning.
warnings.simplefilter('ignore', UserWarning)


def test_fps(src_dir: str, model: str, config: str, weights_dir: str):

    frame_loader = FrameLoader(src_dir)
    num_frames = len(frame_loader.paths)

    print(f'Loading {num_frames} frames...')

    if num_frames == 0:
        return

    frames = list(frame_loader.load_items(rgb=False))

    print(f'Loading {model} model {config}...')

    if model == 'detectron2':
        if config == 'default':
            config = detectron2_utils.DEFAULT_CONFIG
        predictor = detectron2_utils.load_default_predictor(config, input_format='BGR')
    elif model == 'yolact':
        predictor = yolact_utils.load_predictor(config, weights_dir)

    print('Runnning inference...')

    start_time = time.perf_counter()
    for frame in frames:
        predictor(frame)
    elapsed_time = time.perf_counter() - start_time

    print(f'Elapsed time: {elapsed_time:.0f} s')
    print(f'Number of frames: {num_frames}')
    print(f'Dimensions of frames: {frames[0].shape}')
    print(
        f'Performance: {len(frames) / elapsed_time:.2f} fps, {elapsed_time / num_frames:.2f} seconds/frame'
    )


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
