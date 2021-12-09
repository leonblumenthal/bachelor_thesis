from dataclasses import astuple
from typing import List, Tuple

import numpy as np

from data_models import Detection, Label


def merge_masks(detections: List[Detection], dimensions: Tuple[int, int]) -> np.ndarray:
    """Merge bool masks of all detections into a single mask with specified dimensions."""

    mask = np.zeros(dimensions, dtype=bool)

    for det in detections:
        h, w = det.mask.shape
        x, y = det.anchor

        mask[y : y + h, x : x + w] |= det.mask

    return mask


def calculate_mask_centers(detections: List[Detection]) -> np.ndarray:
    """Calculate global mask centers for every detection (like center of mass)."""

    centers = []

    for det in detections:
        mask = det.mask.astype(float)
        mask /= mask.sum()
        h, w = mask.shape
        x = (mask.sum(0) * np.arange(w)).sum() + det.anchor[0]
        y = (mask.sum(1) * np.arange(h)).sum() + det.anchor[1]

        centers.append([x, y])

    return np.array(centers)


def match_detections_and_labels(
    detections: List[Detection],
    labels: List[Label],
    min_span: float = 100,
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
