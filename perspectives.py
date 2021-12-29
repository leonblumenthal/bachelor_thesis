import numpy as np


class Perspective:
    """
    This class represents a single camera perspective including
    utility functions for translation between camera- and ground frame.

    This class relies on:
    1. rotation matrix, from ground- to camera frame (3x3)
    2. camera position in ground frame (3x1)
    3. intrinsic matrix of the camera (3x3)
    """

    def __init__(
        self,
        rotation_matrix: np.ndarray,
        camera_position: np.ndarray,
        intrinsic_matrix: np.ndarray,
    ):

        assert rotation_matrix.shape == (3, 3)
        assert camera_position.shape == (3, 1)
        assert intrinsic_matrix.shape == (3, 3)

        self.rotation_matrix = rotation_matrix
        self.camera_position = camera_position
        self.intrinsic_matrix = intrinsic_matrix

        self.extrinsic_matrix = np.hstack(
            (self.rotation_matrix, -self.rotation_matrix @ self.camera_position)
        )
        self.projection_matrix = self.intrinsic_matrix @ self.extrinsic_matrix

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
        ground_points *= -self.camera_position[2] / ground_points[2]
        ground_points += self.camera_position

        return ground_points

    def project_to_image(self, ground_points: np.ndarray):
        """Project ground points (3xn) into image frame (2xn)."""

        # Transform points using the projection matrix.
        augmented_points = np.vstack((ground_points, np.ones(ground_points.shape[1])))
        transformed_points = self.projection_matrix @ augmented_points
        # Divide x and y values by z (camera pinhole model).
        image_points = transformed_points[:2] / transformed_points[2]

        return image_points


s40_near = Perspective(
    rotation_matrix=np.array(
        [
            [0.16281282, 0.98651019, -0.0170187],
            [0.21592079, -0.05245554, -0.97500083],
            [-0.96274098, 0.15506795, -0.2215485],
        ]
    ),
    camera_position=np.array([[455.90735878, -14.77386387, 8.434]]).T,
    intrinsic_matrix=np.array(
        [
            [2.78886072e03, 0.00000000e00, 9.07839058e02],
            [0.00000000e00, 2.78331261e03, 5.89071478e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
)


s40_far = Perspective(
    rotation_matrix=np.array(
        [
            [7.54789063e-02, 9.97141308e-01, 3.48514044e-03],
            [5.62727716e-02, -7.69991457e-04, -9.98415135e-01],
            [-9.95558291e-01, 7.55554009e-02, -5.61700232e-02],
        ]
    ),
    camera_position=np.array([[465.72356842, -14.60582418, 8.25]]).T,
    intrinsic_matrix=np.array(
        [
            [9.02348282e03, 0.00000000e00, 1.22231430e03],
            [0.00000000e00, 9.01450436e03, 5.57541182e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
)

s50_near = Perspective(
    rotation_matrix=np.array(
        [
            [-6.67942916e-04, -9.99986586e-01, -5.13624571e-03],
            [-1.94233505e-01, 5.16816386e-03, -9.80941709e-01],
            [9.80955096e-01, 3.42417940e-04, -1.94234351e-01],
        ]
    ),
    camera_position=np.array([[0.82720991e00, -5.59124063e00, 8.4336e00]]).T,
    intrinsic_matrix=np.array(
        [
            [2.82046004e03, 0.00000000e00, 9.60667182e02],
            [0.00000000e00, 2.81622151e03, 5.25863289e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
)

s50_far = Perspective(
    rotation_matrix=np.array(
        [
            [0.07011563, -0.99750083, -0.00871162],
            [-0.05890341, 0.0045778, -0.99825319],
            [0.99579827, 0.0705063, -0.05843522],
        ]
    ),
    camera_position=np.array([[0.80730991, -5.20124063, 8.2375]]).T,
    intrinsic_matrix=np.array(
        [
            [8.87857970e03, 0.00000000e00, 5.84217565e02],
            [0.00000000e00, 8.81172402e03, 4.65520403e02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
)
