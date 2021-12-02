"""Save detections for every image of the providentia camera dataset."""

import os
import warnings
from argparse import ArgumentParser

import torch
from tqdm import tqdm

import providentia_camera_dataset as pcd
from detectron2_utils import create_detections, load_default_predictor


# Ignore Detectron2 warning.
warnings.simplefilter("ignore", UserWarning)


def save_detections(dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)

    predictor = load_default_predictor()

    for i, image in tqdm(enumerate(pcd.load_images()), total=len(pcd.image_paths)):
        out = predictor(image)
        instances = out['instances'].to('cpu')

        detections = create_detections(instances)

        path = os.path.join(dst_dir, f'{i}.detection')

        # TODO: Save without torch.
        torch.save(detections, path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dst_dir', type=str)
    args = parser.parse_args()

    save_detections(args.dst_dir)
