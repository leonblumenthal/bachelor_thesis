from typing import Dict, List, Tuple

import numpy as np

from .. import utils
from ..models import Detection, DirectionLine, Vehicle
from ..perspective import Perspective


def calculate_image_edge_margin(detection: Detection, perspective: Perspective) -> int:
    """Calculate minimum margin of detection mask to frame edge."""

    image_height, image_width = perspective.image_shape
    mask_height, mask_width = detection.mask.shape

    left_margin, top_margin = detection.anchor
    right_margin = image_width - mask_width - left_margin
    bottom_margin = image_height - mask_height - top_margin

    margin = min(left_margin, right_margin, top_margin, bottom_margin)

    return margin


def preprocess_detections(
    detections: List[Detection],
    perspective: Perspective,
    score_threshold: float,
    span_threshold: float,
    margin_threshold: int,
    label_mapping: Dict[int, str],
) -> List[Detection]:
    """
    Filter out detections without sufficient score, diagonal mask span, image edge margin, label.
    Return valid detections with mapped category.
    """

    valid_detections = []

    for detection in detections:
        width, height = detection.mask.shape
        span = (width ** 2 + height ** 2) ** 0.5
        margin = calculate_image_edge_margin(detection, perspective)

        if (
            detection.score >= score_threshold
            and span >= span_threshold
            and margin >= margin_threshold
            and detection.label in label_mapping
        ):
            detection.category = label_mapping[detection.label]
            valid_detections.append(detection)

    return valid_detections


def produce_ground_contours(
    detections: List[Detection], perspective: Perspective
) -> List[np.ndarray]:
    """
    Calculate bottom contours of detection masks on image.
    Return contours projected onto ground plane.
    """

    image_contours = [
        utils.calculate_bottom_contour(detection, anchored=False)
        for detection in detections
    ]

    ground_contours = [
        perspective.project_to_ground(image_contour) for image_contour in image_contours
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
    detection: Detection,
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
        * detection.mask.shape[0]
        / (length_by_width * np.sin(view_angle) + np.cos(view_angle))
        / perspective.intrinsic_matrix[1, 1]
    )

    return estimated_height


def estimate_bounding_boxes(
    ground_contours: np.ndarray,
    detections: List[Detection],
    perspective: Perspective,
    direction_line: DirectionLine,
    length_by_width: float,
    car_height_threshold: float,
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
    for points, detection in zip(ground_contours, detections):
        # Calculate naive bounding box in direciton line coordiate frame.
        rotated_points = rotation_matrix @ points[:2]
        min_x, min_y = rotated_points.min(1)
        max_x, max_y = rotated_points.max(1)

        # Estimate height and change category to CAR for short vehicles.
        estimated_height = estimate_height(
            detection, points, perspective, length_by_width
        )
        if estimated_height < car_height_threshold and detection.category != 'CAR':
            detection.category = 'CAR'

        # Estimate dimensions with limits.
        length_values, width_values, height_values = dimension_values_mapping[
            detection.category
        ]
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

        prediction = Vehicle(location, (length, width, height), yaw, detection.category)
        predictions.append(prediction)

    return predictions


def create_predictions(
    detections: List[Detection],
    perspective: Perspective,
    direction_line: DirectionLine,
    threshold_kwargs: Dict[str, float],
    label_mappings: Dict[int, str],
    length_by_width: float,
    car_height_threshold: float,
    dimension_values_mapping: Dict[str, Tuple],
) -> List[Vehicle]:
    """
    Create vehicle predictions from bottom ground contours
    of detections based on fixed direction line.
    """

    valid_detections = preprocess_detections(
        detections, perspective, **threshold_kwargs, label_mapping=label_mappings
    )

    ground_contours = produce_ground_contours(valid_detections, perspective)

    predictions = estimate_bounding_boxes(
        ground_contours,
        valid_detections,
        perspective,
        direction_line,
        length_by_width,
        car_height_threshold,
        dimension_values_mapping,
    )

    return predictions
