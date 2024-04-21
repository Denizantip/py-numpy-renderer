import numpy as np

from obj.constants import XY, W, W_COL, X, Y, Z


def bresenham_line(start_point, end_point):
    delta = end_point - start_point
    # if delta[X] > 0:
    #     delta = start_point - end_point
    steps = max(abs(delta[XY]))
    if steps == 0:
        return [start_point]
    step_size = delta / steps

    return start_point + np.arange(int(steps))[:, None] * step_size


def draw_line(start, end, camera, z_buffer, frame):
    inv_viewport = np.linalg.inv(camera.viewport)
    pxls = bresenham_line(start, end)
    _pxls = np.copy(pxls)
    _pxls[W] = 1
    pxls_ndc = _pxls @ inv_viewport
    pxls_clip = pxls_ndc / pxls[W_COL]
    idx = (
            (-pxls_clip[W] < pxls_clip[X]) & (pxls_clip[X] < pxls_clip[W]) &
            (-pxls_clip[W] < pxls_clip[Y]) & (pxls_clip[Y] < pxls_clip[W]) &
            (-pxls_clip[W] < pxls_clip[Z]) & (pxls_clip[Z] < pxls_clip[W])
    )
    if not idx.any():
        return
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
        frame[np.clip(x + i, a_min=0, a_max=clip_x - 1), y] = frame[np.clip(x + i, a_min=0,
                                                                            a_max=clip_x - 1), y] * 0.5 + (.5, 0., 0.)
        frame[x, np.clip(y + i, a_min=0, a_max=clip_y - 1)] = frame[x, np.clip(y + i, a_min=0,
                                                                               a_max=clip_y - 1)] * 0.5 + (.5, 0., 0.)


# def bresenham_line(start, end):
#     start_x, start_y, start_z, start_w = start
#     end_x, end_y, end_z, end_w = end
#
#     # Normalize homogeneous coordinates
#     start_x /= start_w
#     start_y /= start_w
#     start_z /= start_w
#
#     end_x /= end_w
#     end_y /= end_w
#     end_z /= end_w
#
#     dx = abs(end_x - start_x)
#     dy = abs(end_y - start_y)
#     dz = abs(end_z - start_z)
#
#     sx = 1 if start_x < end_x else -1
#     sy = 1 if start_y < end_y else -1
#     sz = 1 if start_z < end_z else -1
#
#     err1 = dx - dy
#     err2 = dx - dz
#
#     points = []
#
#     while True:
#         points.append((start_x, start_y, start_z))
#
#         if (start_x == end_x and start_y == end_y and start_z == end_z):
#             break
#
#         e2 = 2 * err1
#         e3 = 2 * err2
#
#         if e2 > -dy:
#             err1 -= dy
#             start_x += sx
#         if e3 > -dz:
#             err2 -= dz
#             start_x += sx
#
#         if e2 < dx:
#             err1 += dx
#             start_y += sy
#         if e3 < dx:
#             err2 += dx
#             start_z += sz
#
#     return np.array(points)