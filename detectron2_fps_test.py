import time
import warnings

import pcd_loader as pcd
from detectron2_utils import load_default_predictor


# Ignore Detectron2 warning.
warnings.simplefilter('ignore', UserWarning)


def test_fps():
    print('Loading images...')
    images = list(pcd.load_images())

    print('Loading model...')
    predictor = load_default_predictor()

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
    test_fps()
