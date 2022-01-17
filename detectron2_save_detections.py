"""Save detections for every image in a specified directory."""

import os
import warnings
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from detectron2_utils import create_detections, load_default_predictor
from frame_loader import FrameLoader

# Ignore Detectron2 warning.
warnings.simplefilter('ignore', UserWarning)


def save_detections(src_dir: str, dst_dir: str):
    loader = FrameLoader(src_dir)
    print(f'Found {len(loader.paths)} frames')

    os.makedirs(dst_dir, exist_ok=True)

    predictor = load_default_predictor()

    # loader.load_frames() is better if paths are not needed.
    for i, frame_path in enumerate(tqdm(loader.paths)):
        out = predictor(loader.load_frame(i))
        instances = out['instances'].to('cpu')

        detections = create_detections(instances)

        name = os.path.basename(frame_path).split('.')[0]
        path = os.path.join(dst_dir, f'{name}.detection')

        # TODO: Save without torch.
        torch.save(detections, path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str)
    parser.add_argument('dst_dir', type=str)
    args = parser.parse_args()

    save_detections(args.src_dir, args.dst_dir)
