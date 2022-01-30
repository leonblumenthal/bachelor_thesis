from typing import List, Tuple

import numpy as np
from scipy import interpolate

from .models import Detection, Vehicle
from .perspective import Perspective


def merge_masks(detections: List[Detection], dimensions: Tuple[int, int]) -> np.ndarray:
    """Merge bool masks of all detections into a single mask with specified dimensions."""

    mask = np.zeros(dimensions, dtype=bool)

    for det in detections:
        h, w = det.mask.shape
        x, y = det.anchor

        mask[y : y + h, x : x + w] |= det.mask

    return mask


def calculate_mask_centers(
    detections: List[Detection], anchored: bool = False
) -> np.ndarray:
    """Calculate global mask centers for every detection (like center of mass)."""

    centers = []

    for det in detections:
        mask = det.mask.astype(float)
        mask /= mask.sum()
        h, w = mask.shape
        x = (mask.sum(0) * np.arange(w)).sum()
        y = (mask.sum(1) * np.arange(h)).sum()

        if not anchored:
            x += det.anchor[0]
            y += det.anchor[1]

        centers.append([x, y])

    return np.array(centers).T


def match_detections_and_labels(
    detections: List[Detection],
    labels: List[Vehicle],
    perspective: Perspective,
) -> Tuple[List[Detection], List[Vehicle]]:
    """
    Match detections and labels together based on
    pixel distance between mask- and 3D box centers.
    """

    if not detections or not labels:
        return [], []

    # Calculate centers for masks and labels.
    mask_centers = calculate_mask_centers(detections)
    label_centers = np.hstack(
        [
            label.location + np.array([[0, 0, label.dimensions[2] / 2]]).T
            for label in labels
        ]
    )
    label_centers = perspective.project_to_image(label_centers)

    # Get nearest mask for each label.
    distances = np.linalg.norm(
        mask_centers[..., None] - label_centers[:, None, :], axis=0
    )
    det_indices = distances.argmin(axis=0)

    # Save duplicate mask indices.
    values, counts = np.unique(det_indices, return_counts=True)
    duplicates = values[counts > 1]

    # Create matching pairs without duplicate masks.
    matched_detections = []
    matched_labels = []
    for det_index, label in zip(det_indices, labels):
        if det_index in duplicates:
            continue
        matched_detections.append(detections[det_index])
        matched_labels.append(label)

    return matched_detections, matched_labels


def calculate_bottom_contour(detection: Detection, anchored: bool = True) -> np.ndarray:
    """Calculate (anchored) bottom contour (2 x w) for detection mask (h x w)."""

    # This is way faster than using an edge detection kernel.
    mask = detection.mask
    x = np.arange(mask.shape[1])
    y = mask.shape[0] - mask[::-1].argmax(0)

    contour = np.array([x, y])

    if not anchored:
        contour += np.array([detection.anchor]).T

    return contour


def interpolate_path(points: np.ndarray, n: int, smoothing: int = 0) -> np.ndarray:
    """Interpolate path of points (2 x m) with n points using splines."""

    tck, _ = interpolate.splprep(points, s=smoothing)
    unew = np.linspace(0, 1, n)
    interpolated_points = np.array(interpolate.splev(unew, tck))

    return interpolated_points
