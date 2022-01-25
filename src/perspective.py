import numpy as np


class Perspective:
    """
    This class represents a single camera perspective including
    utility functions for translation between camera- and ground frame.

    This class relies on:
    1. rotation matrix, from ground- to camera frame (3x3)
    2. translation (i.e. camera position) in ground frame (3x1)
    3. intrinsic matrix of the camera (3x3)
    """

    def __init__(
        self,
        rotation_matrix: np.ndarray,
        translation: np.ndarray,
        intrinsic_matrix: np.ndarray,
    ):

        assert rotation_matrix.shape == (3, 3)
        assert translation.shape == (3, 1)
        assert intrinsic_matrix.shape == (3, 3)

        self.rotation_matrix = rotation_matrix
        self.translation = translation
        self.intrinsic_matrix = intrinsic_matrix

        self.extrinsic_matrix = np.hstack(
            (self.rotation_matrix, -self.rotation_matrix @ self.translation)
        )
        self.projection_matrix = self.intrinsic_matrix @ self.extrinsic_matrix

    # TODO: Add functionality to project batches of points.
    def project_to_ground(self, image_points: np.ndarray) -> np.ndarray:
        """Project image points (2xn) into ground frame (3xn) i.e. z=0."""

        assert image_points.shape[0] == 2

        # "Transform" points into ground frame.
        # The real ground point is somewhere on the line going through the camera position and the respective point.
        augmented_points = np.vstack((image_points, np.ones(image_points.shape[1])))
        ground_points = (
            self.rotation_matrix.T
            @ np.linalg.inv(self.intrinsic_matrix)
            @ augmented_points
        )
        # Find intersection of line with ground plane i.e. z=0.
        ground_points *= -self.translation[2] / ground_points[2]
        ground_points += self.translation

        return ground_points

    # TODO: Add functionality to project batches of points.
    def project_to_image(self, ground_points: np.ndarray):
        """Project ground points (3xn) into image frame (2xn)."""

        # Transform points using the projection matrix.
        augmented_points = np.vstack((ground_points, np.ones(ground_points.shape[1])))
        transformed_points = self.projection_matrix @ augmented_points
        # Divide x and y values by z (camera pinhole model).
        image_points = transformed_points[:2] / transformed_points[2]

        return image_points
