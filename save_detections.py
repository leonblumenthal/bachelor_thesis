"""
Create and save detections for every image in a specified directory 
with either a Detectron2 Mask-RCNN or Yolact Edge.
"""

import os
import pickle
import warnings
from argparse import ArgumentParser
from dataclasses import astuple

from tqdm import tqdm

from src import detectron2_utils, yolact_utils
from src.detection_utils import create_detections
from src.loaders import FrameLoader

# Ignore Detectron2 warning.
warnings.simplefilter('ignore', UserWarning)


def save_detections(
    src_dir: str, dst_dir: str, model: str, config: str, weights_dir: str
):

    frame_loader = FrameLoader(src_dir)
    num_frames = len(frame_loader.paths)

    print(f'Found {num_frames} frames')

    if num_frames == 0:
        return

    print(f'Loading {model} model {config}...')

    if model == 'detectron2':
        if config == 'default':
            config = detectron2_utils.DEFAULT_CONFIG
        predictor = detectron2_utils.load_default_predictor(config, input_format='BGR')
    elif model == 'yolact':
        predictor = yolact_utils.load_predictor(config, weights_dir)

    os.makedirs(dst_dir, exist_ok=True)

    print(f'Creating and saving detections...')

    # loader.load_frames() is better if paths are not needed.
    for i, frame_path in enumerate(tqdm(frame_loader.paths)):
        frame = frame_loader.load_item(i, rgb=False)

        output = predictor(frame)

        if model == 'detectron2':
            instances = output['instances'].to('cpu')
            masks = instances.pred_masks
            boxes = instances.pred_boxes
            classes = instances.pred_classes
            scores = instances.scores
        elif model == 'yolact':
            classes, scores, boxes, masks = output

        detections = create_detections(masks, boxes, classes, scores)

        name = os.path.basename(frame_path).split('.')[0]
        path = os.path.join(dst_dir, f'{name}.detection')

        # Convert detections to normal tuples.
        # This allows loading the detections without the specific code.
        detections = [astuple(detection) for detection in detections]
        pickle.dump(detections, open(path, 'wb'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('src_dir', type=str)
    parser.add_argument('dst_dir', type=str)
    parser.add_argument('model', choices=['detectron2', 'yolact'])
    parser.add_argument(
        'config', type=str, help='flexible for detectron2, limited choices for yolact'
    )
    parser.add_argument('--weights_dir', default='weights')
    args = parser.parse_args()

    save_detections(
        args.src_dir, args.dst_dir, args.model, args.config, args.weights_dir
    )
