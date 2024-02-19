import numpy as np
from PIL import Image

from obj.constants import XY, W_COL, XYZ
from obj.transformation import barycentric


class CubeMap:

    """
              ┌───────┐
              │ +Z ↑  │
              │  up   │
      ┌───────┼───────┼───────┬───────┐
      │ -X ←  │ +Y ↑  │ +X →  │ -Y ↓  │
      │   up  │   up  │  up   │  up   │
      └───────┼───────┼───────┴───────┘
              │ -Z ↓  │
              │  up   │
              └───────┘
    """
    def __init__(self, left, right, top, bottom, front, back):
        self.textures = np.array([
            np.flip(self.load_texture(right), axis=[0, 1]),
            np.rot90(self.load_texture(left).transpose((1, 0, 2)), -1),

            self.load_texture(top).transpose((1, 0, 2)),
            np.rot90(self.load_texture(bottom)),

            np.rot90(self.load_texture(front), -1),
            self.load_texture(back).transpose((1, 0, 2)),

            ])

        self.faces = [
            np.array([
                [-1, 1, 1, 1],
                [1, 1, 1, 1],
                [-1, -1, 1, 1]
                ]
                ),
            np.array([
                [1, 1, 1, 1],
                [1, -1, 1, 1],
                [-1, -1, 1, 1]
            ]
            )]

    @staticmethod
    def load_texture(name):
        texture = Image.open(name)
        texture = np.asarray(texture)[..., :3].copy()
        texture.setflags(write=1)
        return texture

    def __getitem__(self, vectors):
        # Find the major axis
        shape_idx = np.arange(vectors.shape[0])
        major_axis_indices = np.abs(vectors).argmax(axis=1)
        major_axis_amplitudes = vectors[shape_idx, major_axis_indices, np.newaxis]

        # Determine sign of major axis amplitudes
        sign = np.sign(major_axis_amplitudes)

        # Extract UV coordinates without major axis values
        uv_coordinates = np.delete(vectors, major_axis_indices + shape_idx * vectors.shape[1]).reshape(
            vectors.shape[0], -1)

        # Normalize UV coordinates
        normalized_uv = (uv_coordinates / major_axis_amplitudes + 1) / 2
        # Determine sides based on sign and major axis
        sides = (major_axis_amplitudes < 0).ravel() + (major_axis_indices * 2)
        # Create indices for texture retrieval
        texture_indices = np.row_stack([sides.astype(int), (normalized_uv.T * self.textures.shape[1] - 1).astype(int)])
        # Return the textures based on calculated indices
        return self.textures[tuple(texture_indices)]


def fill_frame_from_skybox(frame, camera: 'Camera', cubemap: CubeMap):
    height, width, _ = frame.shape
    p = np.mgrid[0: width, 0: height].reshape(2, -1).T
    for face in cubemap.faces:

        test = face @ camera.viewport
        bar = barycentric(*test[XY].astype(int), p)

        Bi = (bar >= 0).all(axis=1)
        bar = bar[Bi]
        y, x = p[Bi].T

        camera_view = camera.lookat
        camera_view[3, :3] = 0
        rays = face @ np.linalg.inv(camera_view @ camera.projection)
        rays /= rays[W_COL]

        rays = bar @ rays[XYZ]
        frame[x, y] = cubemap[rays]
