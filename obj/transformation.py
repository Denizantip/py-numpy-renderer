from dataclasses import dataclass
from typing import Annotated

import numpy as np
from numpy.typing import NDArray

U = X = (..., 0)  # pts[:, 0]
V = Y = (..., 1)  # pts[:, 1]
Z = (..., 2)  # pts[:, 2]
# W = (..., slice(3, 4))  # pts[:, 3]
W = (..., 3)  # pts[:, 3]
XY = (..., (0, 1))  # pts[:, :2]
XZ = (..., (0, 2))
YZ = (..., (1, 2))
XYZ = (..., slice(None, 3))  # pts[:, :3]
XYZW = None
mat3x3 = (slice(None, 3), slice(None, 3))  # ptx[:3, :3]
add_dim = (..., np.newaxis)


class PROJECTION:
    OPEN_GL_PERSPECTIVE = 1
    OPEN_GL_ORTHOGRAPHIC = 2


vec3 = Annotated[NDArray[np.float32 | np.int32], 3]
vec4 = Annotated[NDArray[np.float32 | np.int32], 4]


@dataclass
class ValueRange:
    lo: float
    hi: float


# def normalize(vec) -> NDArray:
#     norm = np.linalg.norm(vec)
#     if norm == 0:
#         vec = np.array((0, 0, 0))
#         return vec
#     return vec / norm
def barycentric(a, b, c, p):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1

    denom = (d00 * d11 - d01 * d01)
    if denom == 0:
        return
    invDenom = 1.0 / denom
    v = (d11 * d20 - d01 * d21) * invDenom
    w = (d00 * d21 - d01 * d20) * invDenom
    u = 1.0 - v - w
    return np.array([u, v, w]).T


def barycentric_weights(triangle, point):
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    area_abc = np.linalg.norm(normal)

    area_pbc = np.linalg.norm(np.cross(triangle[1] - point, triangle[2] - point))
    area_pca = np.linalg.norm(np.cross(triangle[2] - point, triangle[0] - point))
    area_pab = np.linalg.norm(np.cross(triangle[0] - point, triangle[1] - point))

    weight_a = area_pbc / area_abc
    weight_b = area_pca / area_abc
    weight_c = area_pab / area_abc

    return weight_a, weight_b, weight_c


def interpolate_texture_coordinates(texture_coords, weights):
    return sum(np.array(coord) * weight for coord, weight in zip(texture_coords, weights))


def bound_box(vert, width, height):
    out = np.zeros(4, dtype=np.int32)
    min_x = vert[X].min().max(initial=0)
    max_x = vert[X].max().min(initial=height)
    min_y = vert[Y].min().max(initial=0)
    max_y = vert[Y].max().min(initial=width)
    if min_x > max_x or min_y > max_y:
        return

    return np.array((min_x, max_x, min_y, max_y)).astype(int)
    # return map(int, (min_x, max_x, min_y, max_y))


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def gluPerspective(
        angleOfView, imageAspectRatio, near, far, core=PROJECTION.OPEN_GL_PERSPECTIVE
):
    """ bottom and Left should be positive top and Right are negative."""
    # left, right, top, bottom = get_borders(angleOfView, imageAspectRatio, near, far)
    scale = np.tan(np.deg2rad(angleOfView) * 0.5) * near
    right = imageAspectRatio * scale
    top = scale
    match core:
        case PROJECTION.OPEN_GL_PERSPECTIVE:
            perspective = gl_symmetric_perspective
            # perspective = gl_perspective
        case PROJECTION.OPEN_GL_ORTHOGRAPHIC:
            perspective = gl_symmetric_orthographic
            # perspective = gl_orthographic
    return perspective(top, right, near, far)


def gl_perspective(top, bottom, right, left, near, far):
    M = np.array(
        [
            [ 2 * near / (right - left), 0                         , (right + left) / (right - left), 0                             ],  # noqa
            [ 0                        , 2 * near / (top - bottom) , (top + bottom) / (top - bottom), 0                             ],  # noqa
            [ 0                        , 0                         , -(far + near) / (far - near)   , -2 * far * near / (far - near)],  # noqa
            [ 0                        , 0                         , -1                             , 0                             ],  # noqa
        ]
    )
    return M.T


def generate_perspective_projection_matrix_left_handed(fov_degrees, aspect_ratio, near, far):
    # Convert fov degrees to radians
    # https: // perry.cz / articles / ProjectionMatrix.xhtml
    fov_rad = np.radians(fov_degrees)

    # Calculate scale based on the field of view and aspect ratio
    scale = 1.0 / np.tan(fov_rad / 2.0)

    # Create the projection matrix
    projection_matrix = np.zeros((4, 4))
    projection_matrix[0, 0] = aspect_ratio * scale  # X scaling
    projection_matrix[1, 1] = scale  # Y scaling
    projection_matrix[2, 2] = -far / (far - near)  # Z scaling
    projection_matrix[3, 2] = -1.0  # Perspective projection
    projection_matrix[2, 3] = -near * far / (far - near)  # Translation

    return projection_matrix.T


def gl_symmetric_perspective(top, right, near, far):
    M = np.array(
        [
            [ -near / right, 0          , 0                           , 0                             ],  # noqa
            [ 0           , near / top , 0                           , 0                             ],  # noqa
            [ 0           , 0          , -(far + near) / (far - near), -2 * far * near / (far - near)],  # noqa
            [ 0           , 0          , -1                          , 0                             ],  # noqa
        ]
    )
    return M.T

def make_orthographic_matrix(left, right, bottom, top, near, far):
    # Compute width, height and depth of the volume
    w = right - left
    h = top - bottom
    d = far - near  # For left-handed system

    # Create the matrix
    ortho_matrix = np.array([
        [2/w,   0,    0,      -(right+left)/w],
        [0,     2/h,  0,      -(top+bottom)/h],
        [0,     0,    -2/d,   -(far+near)/d],
        [0,     0,    0,      1]
    ], dtype=np.float32)

    return ortho_matrix.T

def gl_symmetric_orthographic(top, right, near, far):
    M = np.array(
        [
            [1 / right, 0       , 0                , 0                           ],  # noqa
            [0        , 1 / top , 0                , 0                           ],  # noqa
            [0        , 0       , -2 / (far - near), -(far + near) / (far - near)],  # noqa
            [0        , 0       , 0                , 1                           ],  # noqa
        ],
        dtype=np.float32,
    )
    return M.T


def scale(factor):
    scaling_matrix = np.array(
        [
            [factor, 0     , 0     , 0],  # noqa
            [0     , factor, 0     , 0],  # noqa
            [0     , 0     , factor, 0],  # noqa
            [0     , 0     , 0     , 1],  # noqa
        ]
    )
    return scaling_matrix


def translation(vec):
    x, y, z = vec
    translation_matrix = np.array(
        [[1, 0, 0, x],
         [0, 1, 0, y],
         [0, 0, 1, z],
         [0, 0, 0, 1]]
    )
    return translation_matrix.T


def rotate(a):
    x, y, z = np.deg2rad(a)

    rotate_x = np.array(
        [
            [1, 0        , 0         , 0],  # noqa
            [0, np.cos(y), -np.sin(y), 0],  # noqa
            [0, np.sin(y), np.cos(y) , 0],  # noqa
            [0, 0        , 0         , 1],  # noqa
        ],
        dtype=np.float32,
    ).T

    rotate_y = np.array(
        [
            [np.cos(x) , 0, np.sin(x), 0],  # noqa
            [0         , 1, 0        , 0],  # noqa
            [-np.sin(x), 0, np.cos(x), 0],  # noqa
            [0         , 0, 0        , 1],  # noqa
        ],
        dtype=np.float32,
    ).T

    rotate_z = np.array(
        [
            [np.cos(z) , np.sin(z), 0, 0],  # noqa
            [-np.sin(z), np.cos(z), 0, 0],  # noqa
            [0         , 0        , 1, 0],  # noqa
            [0         , 0        , 0, 1],  # noqa
        ],
        dtype=np.float32,
    ).T

    return rotate_z @ rotate_y @ rotate_x


def embed(vertices, init=1):
    return np.hstack([vertices, np.full((vertices.shape[0], 1), init)])


def FPSViewRH(
        eye: vec3,
        pitch: Annotated[float, ValueRange(-90, 90)],
        yaw: Annotated[float, ValueRange(0, 360)],
):
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    xaxis: vec3 = np.array([cos_yaw, 0, -sin_yaw])
    yaxis: vec3 = np.array([sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch])
    zaxis: vec3 = np.array([sin_yaw * cos_pitch, -sin_pitch, cos_pitch * cos_yaw])

    viewMatrix = np.array(
        [
            [xaxis[0]      , yaxis[0]      , zaxis[0]      , 0],  # noqa
            [xaxis[1]      , yaxis[1]      , zaxis[1]      , 0],  # noqa
            [xaxis[2]      , yaxis[2]      , zaxis[2]      , 0],  # noqa
            [-(xaxis @ eye), -(yaxis @ eye), -(zaxis @ eye), 1],
        ]
    )  # noqa

    return viewMatrix

