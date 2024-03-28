from dataclasses import dataclass
from constants import *
import numpy as np


@dataclass
class ValueRange:
    lo: float
    hi: float


def barycentric(a, b, c, p):
    """
    https://ceng2.ktu.edu.tr/~cakir/files/grafikler/Texture_Mapping.pdf
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.float32(v0 @ v0)
    d01 = np.float32(v0 @ v1)
    d11 = np.float32(v1 @ v1)
    d20 = v2 @ v0
    d21 = v2 @ v1

    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return
    invDenom = 1.0 / denom
    v = (d11 * d20 - d01 * d21) * invDenom
    w = (d00 * d21 - d01 * d20) * invDenom
    u = 1.0 - v - w
    return np.array([u, v, w]).T


def bound_box(vert, height, width):
    min_x = vert[X].min().max(initial=0)
    max_x = vert[X].max().min(initial=width - 1)
    min_y = vert[Y].min().max(initial=0)
    max_y = vert[Y].max().min(initial=height - 1)
    if min_x > max_x or min_y > max_y:
        return

    return np.ceil((min_x, max_x, min_y, max_y), ).astype(np.int32)


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def lookAtLH(eye, center, up=np.array([0, 1, 0])):
    forward = normalize(center - eye).ravel()
    # forward = normalize(self.position - self.scene.center).squeeze()
    right = normalize(np.cross(up, forward)).ravel()
    # right = normalize(np.cross(forward, self.up)).squeeze()
    new_up = np.cross(forward, right)
    # new_up = np.cross(right, forward)

    # Create the view matrix
    view_matrix = np.eye(4)
    # rot = np.row_stack((right, new_up, -forward))
    rot = np.column_stack((right, new_up, -forward))
    view_matrix[mat3x3] = rot
    # view_matrix[:3, 3] = rot @ -eye
    view_matrix[3, :3] = -eye @ rot

    result = np.array([[right[0], right[1], right[2], -right @ eye],
                       [new_up[0], new_up[1], new_up[2], -new_up @ eye],
                       [-forward[0], -forward[1], -forward[2], forward @ eye],
                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32).T

    # return result
    return view_matrix


def looka_at_translate(eye):
    tr = np.eye(4)
    tr[3, :3] = -eye
    return tr


def look_at_rotate_lh(eye, center, up):
    forward = normalize(center - eye).ravel()
    right = normalize(np.cross(up, forward)).ravel()
    new_up = np.cross(forward, right)
    rot = np.eye(4)
    rot[mat3x3] = np.column_stack((right, new_up, -forward))
    return rot


def look_at_rotate_rh(eye, center, up):
    forward = normalize(center - eye).ravel()
    right = normalize(np.cross(up, forward)).ravel()
    new_up = np.cross(forward, right)
    rot = np.eye(4)
    rot[mat3x3] = np.column_stack((right, new_up, forward))
    return rot


def lookAtRH(eye, center, up=np.array([0, 1, 0])):
    # forward = normalize(eye - center).squeeze()
    forward = normalize(center - eye).squeeze()
    right = normalize(np.cross(up, forward)).squeeze()
    # right = normalize(np.cross(forward, self.up)).squeeze()
    new_up = np.cross(forward, right)
    # new_up = np.cross(right, forward)

    view_matrix = np.eye(4)
    rot = np.column_stack((right, new_up, forward))
    view_matrix[mat3x3] = rot
    # view_matrix[3, :3] = rot @ self.position
    view_matrix[3, :3] = eye @ rot

    result = np.array([[right[0], right[1], right[2], -right @ eye],
                       [new_up[0], new_up[1], new_up[2], -new_up @ eye],
                       [forward[0], forward[1], forward[2], -forward @ eye],
                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32).T

    return view_matrix


def ViewPort(resolution, far, near, x_offset=0, y_offset=0):
    # width, height = resolution
    height, width = resolution
    depth = far - near
    m = np.array(
        [
            [           width / 2,                     0,         0, 0],  # noqa
            [                   0,            height / 2,         0, 0],  # noqa
            [                   0,                     0, depth / 2, 0],  # noqa
            # [        0,           0,                          1,                          0],  # noqa
            [width / 2 + x_offset, height / 2 + y_offset, depth / 2, 1],  # noqa
        ]
    )
    return m


def opengl_orthographicLH(fov, aspect_ratio, z_near, z_far):
    half_fov_rad = np.radians(fov / 2.0)
    half_height = np.tan(half_fov_rad) * z_near
    half_width = half_height * aspect_ratio

    # Calculate orthographic matrix parameters
    right = half_width
    top = half_height

    ortho_matrix = np.array([
        [1 / right, 0      , 0                                  , 0],  # noqa
        [0        , 1 / top, 0                                  , 0],  # noqa
        [0        , 0      , -2 / (z_far - z_near)              , 0],  # noqa
        [0        , 0      , (z_far + z_near) / (z_far - z_near), 1]  # noqa
    ], dtype=np.float32)
    return ortho_matrix


def opengl_perspectiveLH(fovy, aspect, z_near, z_far):
    f = 1.0 / np.tan(np.radians(fovy) / 2.0)
    perspective_matrix = np.zeros((4, 4))
    perspective_matrix[0, 0] = f / aspect
    perspective_matrix[1, 1] = f
    perspective_matrix[2, 2] = -(z_far + z_near) / (z_far - z_near)  # sign
    perspective_matrix[3, 2] = 2.0 * z_far * z_near / (z_far - z_near)
    perspective_matrix[2, 3] = 1.0
    return perspective_matrix


def opengl_perspectiveRH(fovy, aspect, z_near, z_far):
    f = 1.0 / np.tan(np.radians(fovy) / 2.0)
    perspective_matrix = np.zeros((4, 4))
    perspective_matrix[0, 0] = f / aspect
    perspective_matrix[1, 1] = f
    perspective_matrix[2, 2] = -(z_far + z_near) / (z_far - z_near)
    # perspective_matrix[2, 2] = -1
    perspective_matrix[3, 2] = -2.0 * z_far * z_near / (z_far - z_near)
    # perspective_matrix[3, 2] = -2.0 * z_near
    perspective_matrix[2, 3] = -1.0
    return perspective_matrix


def directx_perspectiveRH(fovy, aspect, z_near, z_far):
    """
    https://learn.microsoft.com/ru-ru/windows/win32/direct3d9/d3dxmatrixperspectivefovrh
    """
    f = 1.0 / np.tan(np.radians(fovy) / 2.0)
    perspective_matrix = np.zeros((4, 4))
    perspective_matrix[0, 0] = f / aspect
    perspective_matrix[1, 1] = f
    perspective_matrix[2, 2] = z_far / (z_near - z_far)  # sign?
    perspective_matrix[3, 2] = z_near * z_far / (z_near - z_far)  # sign?
    perspective_matrix[2, 3] = -1.0
    return perspective_matrix


def directx_perspectiveLH(fovy, aspect, z_near, z_far):
    """
    https://learn.microsoft.com/ru-ru/windows/win32/direct3d9/d3dxmatrixperspectivefovlh
    """
    f = 1.0 / np.tan(np.radians(fovy) / 2.0)
    perspective_matrix = np.zeros((4, 4))
    perspective_matrix[0, 0] = f / aspect
    perspective_matrix[1, 1] = f
    perspective_matrix[2, 2] = -z_far / (z_far - z_near)  # sign
    perspective_matrix[3, 2] = z_near * z_far / (z_far - z_near)
    perspective_matrix[2, 3] = 1.0
    return perspective_matrix


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


def rotate_xyz(a):
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


def perspective_matrix_3point(d, aspect_ratio, fov_y, angles):
    f = 1.0 / np.tan(fov_y / 2.0)

    perspective_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (d[1] + d[0]) / (d[1] - d[0]), -2 * d[0] * d[1] / (d[1] - d[0])],
        [0, 0, 1, 0]
    ])

    rotation_matrix = np.array([
        [np.cos(angles[0]), -np.sin(angles[0]), 0, 0],
        [np.sin(angles[0]), np.cos(angles[0]), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return np.dot(np.dot(rotation_matrix, perspective_matrix), np.linalg.inv(rotation_matrix))


def perspective_matrix_2point(d, aspect_ratio, fov_y, eye_sep):
    f = 1.0 / np.tan(fov_y / 2.0)
    perspective_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (d[1] + d[0]) / (d[1] - d[0]), -2 * d[0] * d[1] / (d[1] - d[0])],
        [0, 0, 1, 0]
    ])

    translation_matrix = np.array([
        [1, 0, -eye_sep / 2, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    return np.dot(perspective_matrix, translation_matrix)


# Example usage:
depth_range = [1, 100]
aspect_ratio = 16 / 9
fov_y = np.radians(45)
eye_separation = 0.1

matrix_2point = perspective_matrix_2point(depth_range, aspect_ratio, fov_y, eye_separation)

# Example usage:
angles = [np.radians(30), np.radians(45)]
matrix_3point = perspective_matrix_3point(depth_range, aspect_ratio, fov_y, angles)

perspectives = {
    SUBSYSTEM.DIRECTX: {
        PROJECTION_TYPE.PERSPECTIVE: {
            SYSTEM.LH: directx_perspectiveLH,
            SYSTEM.RH: directx_perspectiveRH},
        PROJECTION_TYPE.ORTHOGRAPHIC: {}
    },   # Directx
    SUBSYSTEM.OPENGL: {
        PROJECTION_TYPE.PERSPECTIVE: {
            SYSTEM.LH: opengl_perspectiveLH,
            SYSTEM.RH: opengl_perspectiveRH},
        PROJECTION_TYPE.ORTHOGRAPHIC: {
            SYSTEM.LH: opengl_orthographicLH,
        }
    }
}   # OpenGL
