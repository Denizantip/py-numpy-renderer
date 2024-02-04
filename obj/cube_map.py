import numpy as np
from PIL import Image


class CubeMap:
    def __init__(self, left, right, top, bottom, front, back):
        self.left = self.load_texture(left)
        self.right = self.load_texture(right)
        self.top = self.load_texture(top)
        self.bottom = self.load_texture(bottom)
        self.front = self.load_texture(front)
        self.back = self.load_texture(back)

    @staticmethod
    def load_texture(name):
        texture = Image.open(name)
        texture = np.asarray(texture)[..., :3].copy()
        texture.setflags(write=1)
        return texture

    def __getitem__(self, item):
        match item:
            case 0:
                return self.left
            case 1:
                return self.right
            case 2:
                return self.top
            case 3:
                return self.bottom
            case 4:
                return self.front
            case 5:
                return self.back

    def get_cube_map_face_and_uv(self, direction):
        abs_direction = np.abs(direction)
        max_axis = np.argmax(abs_direction)

        if max_axis == 0:  # x-axis
            face_index = 0 if direction[0] > 0 else 1
        elif max_axis == 1:  # y-axis
            face_index = 2 if direction[1] > 0 else 3
        else:  # z-axis
            face_index = 4 if direction[2] > 0 else 5

        s = direction[(max_axis + 1) % 3] if direction[max_axis] > 0 else -direction[(max_axis + 1) % 3]
        t = direction[(max_axis + 2) % 3] if direction[max_axis] > 0 else -direction[(max_axis + 2) % 3]

        u = (s + 1) / 2
        v = (t + 1) / 2

        return self[face_index][u, v]

# view = np.linalg.inv(camera.lookat)
# view[:3, 3] = 0
# MVP = np.linalg.inv(view @ camera.projection)
# verts = verts @ MVP
# normalize(verts / verts[W_COL])

import numpy as np


class CubeMap:
    def __init__(self, images):
        # Check if exactly 6 images are provided
        if len(images) != 6:
            raise ValueError("CubeMap requires exactly 6 images.")

        # Assuming each image is a numpy array representing a face of the cube
        self.faces = np.array(images)

    def __getitem__(self, key):
        # Implementing the support interface of get item
        return self.faces[key]


def generate_rays(camera_pos, width, height, fov, near_clip, far_clip):
    aspect_ratio = width / height
    fov_rad = np.radians(fov)

    # Calculate the projection matrix
    projection_matrix = np.array([
        [1 / np.tan(0.5 * fov_rad), 0, 0, 0],
        [0, aspect_ratio / np.tan(0.5 * fov_rad), 0, 0],
        [0, 0, -(far_clip + near_clip) / (far_clip - near_clip), -2 * far_clip * near_clip / (far_clip - near_clip)],
        [0, 0, -1, 0]
    ])

    # Assuming an identity view matrix for simplicity
    view_matrix = np.eye(4)

    # Iterate through each pixel of the quad
    for y in range(height):
        for x in range(width):
            # Convert pixel coordinates to NDC
            ndc_x = (2 * x / width) - 1
            ndc_y = 1 - (2 * y / height)

            # Calculate the ray direction in view space
            ray_direction_view = np.linalg.inv(projection_matrix) @ np.array([ndc_x, ndc_y, -1, 1])

            # Calculate the ray direction in world space
            ray_direction_world = np.linalg.inv(view_matrix) @ np.array(
                [ray_direction_view[0], ray_direction_view[1], -1, 0])

            # Normalize the direction vector
            ray_direction_world /= np.linalg.norm(ray_direction_world)

            # Map the normalized direction to CubeMap coordinates
            cube_map_coordinates = (ray_direction_world + 1) / 2  # Mapping to [0, 1] range

            # Access the CubeMap using the coordinates
            cube_map_value = cube_map[cube_map_coordinates]
