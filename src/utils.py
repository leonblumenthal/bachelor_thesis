from dataclasses import astuple
from typing import List, Tuple

import numpy as np
from scipy import interpolate

from .models import Detection, Label


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

    return np.array(centers)


def match_detections_and_labels(
    detections: List[Detection],
    labels: List[Label],
    min_span: float = 0,
    bounds: Tuple[int, int] = None,
) -> List[Tuple[Detection, Label]]:
    """
    Match detections and labels together based on
    pixel distance between mask- and projection centers.
    """

    if not detections or not labels:
        return []

    # Convert projected keypoints to numpy and compute min/max corners.
    label_keypoints = np.array([astuple(l.projected_box) for l in labels])
    label_maxs = label_keypoints.max(1)
    label_mins = label_keypoints.min(1)

    # Check if projection span is below threshold for every label.
    spans = np.linalg.norm((label_maxs - label_mins), axis=-1)
    valid_indices = spans >= min_span

    # Check out of bounds boxes for every label.
    if bounds is not None:
        above_zero = np.all(label_mins > 0, axis=-1)
        below_max = np.all(label_maxs < bounds, axis=-1)
        valid_indices &= above_zero & below_max

    # Ignore labels without sufficient span or out bounds.
    labels = [label for label, valid in zip(labels, valid_indices) if valid]
    label_keypoints = label_keypoints[valid_indices]

    # Calculate centers for masks and labels.
    mask_centers = calculate_mask_centers(detections)
    label_centers = label_keypoints.mean(1)

    # Get nearest mask for each label.
    distances = np.linalg.norm(
        mask_centers[None, ...] - label_centers[:, None, :], axis=-1
    )
    det_indices = distances.argmin(axis=1)

    # Save duplicate mask indices.
    values, counts = np.unique(det_indices, return_counts=True)
    duplicates = values[counts > 1]

    # Create matching pairs without duplicate masks.
    pairs = [
        (detections[det_index], label)
        for det_index, label in zip(det_indices, labels)
        if det_index not in duplicates
    ]

    return pairs


def calculate_bottom_contour(detection: Detection, anchored: bool = True) -> np.ndarray:
    """Calculate (anchored) bottom contour (2 x w) for detection mask (h x w)."""

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
