import numpy as np


class Camera:
    """
    A Camera class to simulate a 3D camera, providing functionalities such as positioning, rotation, and projection matrix calculation.
    """

    def __init__(
        self,
        position: np.array,
        target: np.array,
        up_vector: np.array,
        fov_deg: float,
        aspect_ratio: float,
        near_plane: float,
        far_plane: float,
    ) -> None:
        """
        Initializes the Camera object.

        Parameters:
        - position (np.array): The camera's position in 3D space.
        - target (np.array): The point in 3D space the camera is looking at.
        - up_vector (np.array): The camera's up direction vector.
        - fov_deg (float): The field of view in degrees.
        - aspect_ratio (float): The aspect ratio (width/height).
        - near_plane (float): The distance to the near clipping plane.
        - far_plane (float): The distance to the far clipping plane.
        """
        self.position: np.array[np.float32] = np.array(position[:3], dtype=np.float32)
        self.target: np.array[np.float32] = np.array(target[:3], dtype=np.float32)
        self.up_vector: np.array[np.float32] = np.array(up_vector[:3], dtype=np.float32)
        self.direction: np.array[np.float32] = self.target - self.position
        self.direction = self.direction / np.linalg.norm(
            self.direction
        )  # Normalize direction
        self.fov_deg: float = fov_deg
        self.aspect_ratio: float = aspect_ratio
        self.near_plane: float = near_plane
        self.far_plane: float = far_plane
        self.fix_up_vector = True
        self.rot_speed = 0.05
        self.update()

    def get_view_matrix(self) -> np.matrix:
        """
        Calculates the view matrix using the camera's position, direction and up vector.

        Returns:
        - np.matrix: The view matrix used in transforming world coordinates to camera space.
        """
        zaxis = -self.direction / np.linalg.norm(self.direction)
        xaxis = np.cross(self.up_vector, zaxis) / np.linalg.norm(
            np.cross(self.up_vector, zaxis)
        )
        yaxis = np.cross(zaxis, xaxis)

        view_matrix = np.matrix(
            [
                [xaxis[0], xaxis[1], xaxis[2], -np.dot(xaxis, self.position)],
                [yaxis[0], yaxis[1], yaxis[2], -np.dot(yaxis, self.position)],
                [zaxis[0], zaxis[1], zaxis[2], -np.dot(zaxis, self.position)],
                [0, 0, 0, 1],
            ]
        )
        return view_matrix

    def get_projection_matrix(self) -> np.matrix:
        """
        Calculates the perspective projection matrix.

        Returns:
        - np.matrix: The projection matrix used to project 3D points to 2D.
        """
        f = 1.0 / np.tan(np.radians(self.fov_deg) / 2.0)
        proj_matrix = np.matrix(
            [
                [f / self.aspect_ratio, 0, 0, 0],
                [0, f, 0, 0],
                [
                    0,
                    0,
                    (self.far_plane + self.near_plane)
                    / (self.near_plane - self.far_plane),
                    (2 * self.far_plane * self.near_plane)
                    / (self.near_plane - self.far_plane),
                ],
                [0, 0, -1, 0],
            ]
        )
        return proj_matrix

    def update(self) -> None:
        """
        Updates the projection and view matrices, and calculates the normal view matrix.
        """
        self.proj = self.get_projection_matrix()
        self.view = self.get_view_matrix()
        self.view_normal = np.transpose(np.linalg.inv(self.view))

    def forward(self, by: float) -> None:
        """
        Moves the camera forward by a specified distance, along its current direction.

        Parameters:
        - by (float): The step size.
        """
        self.position += by * self.direction

    def backward(self, by: float) -> None:
        """
        Moves the camera backward by a specified distance, opposite to its current direction.

        Parameters:
        - by (float): The step size.
        """
        self.position -= by * self.direction

    def leftward(self, by: float) -> None:
        """
        Strafes the camera left by a specified amount.

        Parameters:
        - by (float): The step size.
        """
        left = np.cross(self.direction, self.up_vector)
        self.position -= by * left / np.linalg.norm(left)

    def rightward(self, by: float) -> None:
        """
        Strafes the camera right by a specified amount.

        Parameters:
        - by (float): The step size.
        """
        right = np.cross(self.direction, self.up_vector)
        self.position += by * right / np.linalg.norm(right)

    def upward(self, by: float) -> None:
        """
        Moves the camera upward by a specified amount.

        Parameters:
        - by (float): The step size.
        """
        upward = np.cross(np.cross(self.direction, self.up_vector), self.direction)
        self.position += by * upward / np.linalg.norm(upward)

    def downward(self, by: float) -> None:
        """
        Moves the camera downward by a specified amount.

        Parameters:
        - by (float): The step size.
        """
        downward = np.cross(np.cross(self.direction, self.up_vector), self.direction)
        self.position -= by * downward / np.linalg.norm(downward)

    def rotate(self, v: np.array, angle: float, axis: np.array):
        """
        Rotates a vector around an axis by a given angle using Rodrigues' rotation formula.

        Parameters:
        - v (np.array): Vector to rotate.
        - angle (float): Angle in degrees.
        - axis (np.array): Axis around which to rotate.

        Returns:
        - np.array: The rotated vector.
        """
        axis = axis / np.linalg.norm(axis)
        v_rot = (
            v * np.cos(np.radians(angle))
            + np.cross(axis, v) * np.sin(np.radians(angle))
            + axis * np.dot(axis, v) * (1 - np.cos(np.radians(angle)))
        )
        return v_rot

    def yaw(self, angle: float) -> None:
        """
        Rotates the camera around its up vector by a given yaw angle.

        Parameters:
        - angle (float): The yaw angle in degrees.
        """
        self.direction = self.rotate(self.direction, angle, self.up_vector)
        self.direction /= np.linalg.norm(self.direction)

    def pitch(self, angle: float) -> None:
        """
        Rotates the camera around its horizontal axis by a given pitch angle.

        Parameters:
        - angle (float): The pitch angle in degrees.
        """
        rot_axis = np.cross(self.direction, self.up_vector)
        self.direction = self.rotate(
            self.direction, angle, rot_axis / np.linalg.norm(rot_axis)
        )
        self.direction /= np.linalg.norm(self.direction)
        if not self.fix_up_vector:
            self.up_vector = np.cross(
                np.cross(self.direction, self.up_vector), self.direction
            )
            self.up_vector /= np.linalg.norm(self.up_vector)

    def roll(self, angle: float) -> None:
        """
        Rolls the camera around its direction by a given roll angle.

        Parameters:
        - angle (float): The roll angle in degrees.
        """
        self.up_vector = self.rotate(self.up_vector, angle, self.direction)
        self.up_vector /= np.linalg.norm(self.up_vector)
