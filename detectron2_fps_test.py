"""Load every image in a specified directory into memeory and measure inference fps."""

import time
import warnings
from argparse import ArgumentParser

from detectron2_utils import DEFAULT_MODEL, load_default_predictor
from src.loaders import FrameLoader

# Ignore Detectron2 warning.
warnings.simplefilter('ignore', UserWarning)


def test_fps(src_dir: str, model_name: str):
    loader = FrameLoader(src_dir)

    print(f'Loading {len(loader.paths)} frames...')
    images = list(loader.load_items())

    print(f'Loading model "{model_name}"...')
    predictor = load_default_predictor(model_name)

    print('Runnning inference...')
    start_time = time.perf_counter()
    for image in images:
        predictor(image)
    elapsed_time = time.perf_counter() - start_time

    print(f'Elapsed time: {elapsed_time:.0f} s')
    print(f'Number of frames: {len(images)}')
    print(f'Dimensions of frames: {images[0].shape}')
    print(
        f'Performance: {len(images) / elapsed_time:.2f} fps, {elapsed_time / len(images):.2f} seconds/frame'
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str)
    parser.add_argument('-m', '--model', type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    test_fps(args.src_dir, args.model)
