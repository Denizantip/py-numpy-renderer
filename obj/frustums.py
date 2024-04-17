import numpy as np

from obj.constants import *
from obj.line import bresenham_line
from plane_intersection import line_plane_intersection, extract_frustum_planes, is_visible, clipping


class Frustum:

    r"""
    Cube in clip space
        A                 B
          o-------------o
         /|            /|
        / |           / |
    C  o-------------o D|
       |  |          |  |
       |G o----------|--o H
       | /           | /
       |/            |/
       o-------------o
    E                F
                            X,    Y,    Z,   W
"""
    vertices = np.array([[-1.0, -1.0,  1.0, 1.0],   # E   0
                         [ 1.0, -1.0,  1.0, 1.0],   # F   1
                         [-1.0,  1.0,  1.0, 1.0],   # C   2
                         [ 1.0,  1.0,  1.0, 1.0],   # D   3

                         [-1.0,  1.0, -1.0, 1.0],   # A   4
                         [ 1.0,  1.0, -1.0, 1.0],   # B   5
                         [-1.0, -1.0, -1.0, 1.0],   # G   6
                         [ 1.0, -1.0, -1.0, 1.0]])  # H   7

    # segments =     [(E, F), (F, D), (D, C), (C, E), (B, A), (H, B), (G, H), (A, G), (C, A), (D, B), (F, H), (E, G)]
    edges = np.array([(0, 1), (1, 3), (3, 2), (2, 0), (5, 4), (7, 5), (6, 7), (4, 6), (2, 4), (3, 5), (1, 7), (0, 6)])
    triangles = np.array([(4, 6, 7), (7, 5, 4), (0, 6, 4),
                          (4, 2, 0), (7, 1, 3), (3, 5, 7),
                          (0, 2, 3), (3, 1, 0), (4, 5, 3),
                          (3, 2, 4), (6, 0, 7), (7, 0, 1)])
    #                 (E, C, D, F), (A, B, H, G), (A, C, E, G)]
    faces = np.array([(0, 2, 3, 1), (4, 5, 7, 6), (4, 2, 0, 6),
    #                 (D, B, H, F), (C, D, F, E), (A, G, H, B)
                      (3, 5, 7, 1), (2, 3, 1, 0), (4, 6, 7, 5)
                      ])


def split_triangle_lines(tri):
    edges = len(tri)
    for i in range(edges):
        start = tri[i]
        end = tri[(i + 1) % edges]
        yield start, end


def draw_view_frustum(frame, camera, positioned_object, z_buffer, sign):
    view_frustum_world = Frustum.vertices @ np.linalg.inv(positioned_object.MVP)
    view_frustum_world /= view_frustum_world[W_COL]
    planes = camera.frustum_planes

    for face in view_frustum_world[Frustum.faces]:
        face = clipping(face, planes)
        if not face.size:
            continue
        face = face @ camera.MVP
        face /= face[W_COL]
        face = face @ camera.viewport
        l = len(face)
        for i in range(l):
            start = face[i]
            end = face[(i + 1) % l]
            pxls = bresenham_line(start, end)
            y, x, z, w = pxls.T
            x = x.astype(np.int32) - 1
            y = y.astype(np.int32) - 1
            idx = (z_buffer[x, y] >= z)
            x = x[idx]
            y = y[idx]
            z = z[idx]
            z_buffer[x, y] = z
            frame[x, y] = (1., 0., 0.)
            clip_x, clip_y = np.array(camera.scene.resolution) - 1
            #  Lightweight AntiAliased line algo
            for i in [-1, 1]:
                z_buffer[np.clip(x + i, a_min=0, a_max=clip_x), y] = z
                z_buffer[x, np.clip(y + i, a_min=0, a_max=clip_y)] = z
                frame[np.clip(x + i, a_min=0, a_max=clip_x), y] = (frame[np.clip(x + i, a_min=0, a_max=clip_x), y]
                                                                   * 0.5 + (.5, 0., 0.))
                frame[x, np.clip(y + i, a_min=0, a_max=clip_y)] = (frame[x, np.clip(y + i, a_min=0, a_max=clip_y)]
                                                                   * 0.5 + (.5, 0., 0.))
