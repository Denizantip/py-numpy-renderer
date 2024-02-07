import numpy as np

from obj.constants import W_COL, XYZ, Z, W
from obj.triangular import bresenham_line


def get_view_frustum():
    r"""
    Cube in clip space
         A                 B
            o------------o
           /|           /|
         /  |         /  |
     C  o------------o D |
        |   |       |    |
        | G o-------|----o H
        |  /        |   /
        |/          | /
        o------------o
      E              F
    """
    verts = np.array([[-1.0, -1.0,  1.0, 1.0],
                      [ 1.0, -1.0,  1.0, 1.0],
                      [-1.0,  1.0,  1.0, 1.0],
                      [ 1.0,  1.0,  1.0, 1.0],

                      [-1.0,  1.0, -1.0, 1.0],
                      [ 1.0,  1.0, -1.0, 1.0],
                      [-1.0, -1.0, -1.0, 1.0],
                      [ 1.0, -1.0, -1.0, 1.0]])
    return verts


# segments = [(A, B), (B, D), (D, C), (C, A), (E, F), (G, E), (F, H), (H, G), (C, E), (D, F), (B, H), (A, G)]
edges = np.array([(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (2, 4), (3, 5), (1, 7), (0, 6)])
faces = np.array([(4, 6, 7), (7, 5, 4), (0, 6, 4),
                  (4, 2, 0), (7, 1, 3), (3, 5, 7),
                  (0, 2, 3), (3, 1, 0), (4, 5, 3),
                  (3, 2, 4), (6, 0, 7), (7, 0, 1)])


def split_triangle_lines(tri):
    edges = len(tri)
    for i in range(edges):
        start = tri[i]
        end = tri[(i + 1) % edges]
        yield start, end


def draw_view_frustum(frame, camera, debug_camera, z_buffer, sign):
    cube_frustum = get_view_frustum() @ np.linalg.inv(debug_camera.lookat @ debug_camera.projection) @ camera.lookat @ camera.projection
    cube_frustum /= cube_frustum[W_COL]
    cube_frustum = cube_frustum @ camera.viewport

    # for triangle in cube_frustum[faces]:
    for start, end in cube_frustum[edges]:
        # for start, end in split_triangle_lines(triangle):
        for yy, xx, zz in bresenham_line(start[XYZ], end[XYZ], camera.resolution):
            for i in range(3):
                xx = max(0, min(frame.shape[0] - 3, int(xx)))
                yy = max(0, min(frame.shape[1] - 3, int(yy)))

                # if z_buffer[xx + i, yy + i] > 1 / zz:
                if (z_buffer[xx + i, yy + i] - 1 / zz) * sign > 0:
                    frame[xx + i, yy + i] = (64, 64, 128)
                    z_buffer[xx + i, yy + i] = zz
