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
    #                 (E, F, C, D), (A, B, G, H), (C, A, G, E)]
    faces = np.array([(0, 1, 2, 3), (4, 5, 6, 7), (2, 4, 6, 0),
    #                 (D, B, H, F), (C, D, F, E), (A, B, H, G)
                      (3, 5, 7, 1), (2, 3, 1, 0), (4, 5, 7, 6)
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


    view_frustum_clip = view_frustum_world @ camera.MVP
    view_frustum_ndc = view_frustum_clip / view_frustum_clip[W_COL]

    view_frustum_screen = view_frustum_ndc @ camera.viewport
    view_frustum_screen[W_COL] = view_frustum_clip[W_COL]

    inv_viewport = np.linalg.inv(camera.viewport)

    # C-like approach (3 FOR LOOPS. My python eyes)
    for start, end in view_frustum_screen[Frustum.edges]:
        # for yy, xx, zz, ww in bresenham_line(start, end):
        #     _yy, _xx, _zz, _ww = (np.array((yy, xx, zz, 1)) @ inv_viewport) * ww
        #     if -_ww < _xx < _ww and -_ww < _yy < _ww and -_ww < _zz < _ww:
        #         for i in [-1, 0, 1]:
        #             xx = max(0, min(frame.shape[0] - 3, int(xx)))
        #             yy = max(0, min(frame.shape[1] - 3, int(yy)))
        #
        #             if (z_buffer[xx + i, yy + i] - zz) * sign < 0:
        #                 frame[xx + i, yy + i] = (255, 0, 0)
        #                 z_buffer[xx + i, yy + i] = zz

        # NumPy version. Should be faster then 3 FOR LOOPS. (but it HUGE. How someone can explain WTF is going on here?!)
        pxls = bresenham_line(start, end)
        _pxls = np.copy(pxls)
        _pxls[W] = 1
        pxls_clip = (_pxls @ inv_viewport)
        pxls_clip = pxls_clip * pxls[W_COL]

        idx = (
                (-pxls_clip[W] < pxls_clip[X]) & (pxls_clip[X] < pxls_clip[W]) &
                (-pxls_clip[W] < pxls_clip[Y]) & (pxls_clip[Y] < pxls_clip[W]) &
                (-pxls_clip[W] < pxls_clip[Z]) & (pxls_clip[Z] < pxls_clip[W])
        )
        if not idx.any():
            continue
        y, x, z, w = pxls[idx].T
        x = x.astype(np.int32)
        y = y.astype(np.int32)
        idx = (z_buffer[x, y] > z)
        x = x[idx]
        y = y[idx]
        z = z[idx]
        z_buffer[x, y] = z
        frame[x, y] = (1., 0., 0.)
        clip_x, clip_y = camera.scene.resolution
        #  Lightweight AntiAliased line algo
        for i in [-1, 1]:
            z_buffer[np.clip(x + i, a_min=0, a_max=clip_x - 1), y] = z
            z_buffer[x, np.clip(y + i, a_min=0, a_max=clip_y - 1)] = z
            frame[np.clip(x + i, a_min=0, a_max=clip_x - 1), y] = frame[np.clip(x + i, a_min=0, a_max=clip_x - 1), y] * 0.5 + (.5, 0., 0.)
            frame[x, np.clip(y + i, a_min=0, a_max=clip_y - 1)] = frame[x, np.clip(y + i, a_min=0, a_max=clip_y - 1)] * 0.5 + (.5, 0., 0.)
