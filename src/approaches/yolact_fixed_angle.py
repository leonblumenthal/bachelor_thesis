from typing import Dict, List, Tuple

import numpy as np
import torch
from ..models import DirectionLine, Vehicle
from ..perspective import Perspective
from ..yolact_edge_predictor import YolactEdgePredictor


def produce_ground_contours(
    valid_masks: torch.Tensor,
    perspective: Perspective,
    vertical_contour_shift: float,
) -> List[np.ndarray]:
    """
    Calculate bottom contours of detection masks on image and shift them.
    Return contours projected onto ground plane.
    """

    image_height = perspective.image_shape[0]

    # Use (arg)max on vertically flipped tensors to get y values of lowest
    # masked pixel for each x value and also if masked value exists.
    max_values, y_values = valid_masks.flip(1).max(1)
    max_values = max_values.bool()
    x_range = torch.arange(end=max_values.shape[1])

    # Transform y values and max values into real bottom contour
    # by only taking into account y values with max values == 1.
    image_contours = []
    for ms, ys in zip(max_values, y_values):
        y = image_height - ys[ms].cpu().numpy()
        y += vertical_contour_shift
        x = x_range[ms].cpu().numpy()
        image_contours.append(np.array([x, y]))

    # Project each contour from image to ground plane.
    ground_contours = [
        perspective.project_to_ground(points) for points in image_contours
    ]

    return ground_contours


def process_dimension(
    value: float, dimenion_values: Tuple[float, float, float]
) -> float:
    """
    Limit dimension based on min and max.
    Use mean if value is way smaller than min.
    """

    min_value, mean_value, max_value = dimenion_values

    if value < min_value / 2:
        return mean_value
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value

    return value


def estimate_height(
    mask_height: int,
    ground_contour: np.ndarray,
    perspective: Perspective,
    length_by_width: float,
) -> float:
    """Estimate height of vehicle based on distance to camera and focal length."""

    # Assume mid point of contour is roughly nearest point to camera.
    index = ground_contour.shape[1] // 2
    mid_point = ground_contour[:2, index]

    camera_point = perspective.translation[:2, 0]
    camera_height = perspective.translation[2, 0]

    camera_distance = np.linalg.norm(mid_point - camera_point)
    view_angle = np.tan(camera_height / camera_distance)
    estimated_height = (
        camera_distance
        * mask_height
        / (length_by_width * np.sin(view_angle) + np.cos(view_angle))
        / perspective.intrinsic_matrix[1, 1]
    )

    return estimated_height


def estimate_bounding_boxes(
    ground_contours: np.ndarray,
    categories: List[str],
    mask_heights: List[float],
    perspective: Perspective,
    direction_line: DirectionLine,
    dimension_values_mapping: Dict[str, Tuple],
) -> List[Vehicle]:
    """
    Estimate bounding boxes with fixed angle in direction line coordinate frame
    and dimension limits per category.
    """

    # Create rotation matrix to rotate into direction line coordiante system.
    rotation_matrix = np.hstack(
        (direction_line.direction_vector, direction_line.normal_vector)
    ).T

    # Rotate camera in direction line coordiante system.
    rotated_camera = rotation_matrix @ perspective.translation[:2]
    cam_x, cam_y = rotated_camera[:, 0]

    # Calculate yaw on one side of direction line.
    direction_yaw = np.arctan2(*direction_line.direction_vector[[1, 0], 0])

    predictions = []
    for points, category, mask_height in zip(ground_contours, categories, mask_heights):
        # Calculate naive bounding box in direciton line coordiate frame.
        rotated_points = rotation_matrix @ points[:2]
        min_x, min_y = rotated_points.min(1)
        max_x, max_y = rotated_points.max(1)

        # Estimate height and change category to CAR or VAN for shorter vehicles.
        length_by_width = (
            dimension_values_mapping['CAR'][0][1]
            / dimension_values_mapping['CAR'][1][1]
        )
        estimated_height = estimate_height(
            mask_height, points, perspective, length_by_width
        )
        if estimated_height <= dimension_values_mapping['CAR'][2][2]:
            category = 'CAR'
        elif estimated_height <= dimension_values_mapping['VAN'][2][2]:
            category = 'VAN'

        # Estimate dimensions with limits.
        length_values, width_values, height_values = dimension_values_mapping[category]
        length = process_dimension(max_x - min_x, length_values)
        width = process_dimension(max_y - min_y, width_values)
        height = process_dimension(estimated_height, height_values)

        # Resize naive bounding box based on camera position.
        if cam_x <= (min_x + max_x) / 2:
            max_x = min_x + length
        else:
            min_x = max_x - length
        if cam_y <= (min_y + max_y) / 2:
            max_y = min_y + width
        else:
            min_y = max_y - width

        # Calculate location in orginal coordinate frame.
        rotated_x = min_x + length / 2
        rotated_y = min_y + width / 2
        location = np.array([[rotated_x, rotated_y]]).T
        x, y = (rotation_matrix.T @ location)[:, 0]
        location = np.array([[x, y, 0]]).T

        # Flip direction based on direction line side.
        yaw = (
            direction_yaw
            if rotated_y + direction_line.bias > 0
            else (direction_yaw + np.pi) % (2 * np.pi)
        )

        prediction = Vehicle(location, (length, width, height), yaw, category)
        predictions.append(prediction)

    return predictions


def create_predictions(
    predictor: YolactEdgePredictor,
    frame: np.ndarray,
    perspective: Perspective,
    direction_line: DirectionLine,
    thresholds: Dict[str, int],
    vertical_contour_shift: float,
    label_mapping: Dict[int, str],
    dimension_values_mapping: Dict[str, Tuple],
) -> List[Vehicle]:
    """
    Use Yolact Edge predictor and create vehicle predictions from bottom ground contours
    of instance masks based on fixed direction line.
    """

    # Run Yolact Edge on image.
    classes, scores, boxes, masks = predictor(frame)

    # Calculate mask heights before boxes are altered for margin checks.
    mask_heights = boxes[:, 3] - boxes[:, 1]

    image_height, image_width = perspective.image_shape

    # Check margins to image edges.
    boxes[:, 2:] *= -1
    boxes[:, 2] += image_width
    boxes[:, 3] += image_height
    valid_margins = boxes.min(1).values >= thresholds['edge_margin']

    # Check if class labels have valid category mapping.
    valid_labels = torch.zeros_like(valid_margins)
    for label in label_mapping:
        valid_labels |= classes == label

    # Check estimated mask widths and scores
    valid_mask_widths = masks.sum((1, 2)) ** 0.5 >= thresholds['mask_width']
    valid_scores = scores >= thresholds['score']

    # Combine all checks above to single bool index mask.
    valid_detections = valid_scores & valid_labels & valid_mask_widths & valid_margins

    ground_contours = produce_ground_contours(
        masks[valid_detections], perspective, vertical_contour_shift
    )

    # Map class labels to catergories and calculate mask heights for valid detections.
    categories = [
        label_mapping[label.item()] for label in classes[valid_detections].cpu()
    ]
    mask_heights = mask_heights[valid_detections].cpu().tolist()

    predictions = estimate_bounding_boxes(
        ground_contours,
        categories,
        mask_heights,
        perspective,
        direction_line,
        dimension_values_mapping,
    )

    return predictions
