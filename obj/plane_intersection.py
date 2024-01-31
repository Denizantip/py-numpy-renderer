from enum import IntFlag, auto

import numpy as np


class Intersect(IntFlag):
    left = auto()
    right = auto()
    bottom = auto()
    top = auto()
    near = auto()
    far = auto()


def normalize_plane(plane):
    """
    If needed to get true distance from point to plane
    """
    return plane / np.linalg.norm(plane)


def homogeneous_intersect(v1, v2, plane):
    t = np.dot(plane, v1) / (np.dot(plane, v1) - np.dot(plane, v2))
    intersection_point = (1 - t) * v1 + t * v2
    return intersection_point


def line_plane_intersection(line_point1, line_point2, plane_coefficients):
    line_direction = line_point2 - line_point1

    denominator = plane_coefficients @ line_direction
    if abs(denominator) < 1e-10:
        return None, None  # No intersection (parallel)
    # weight needed for texture/normals interpolation
    weight = -(plane_coefficients @ line_point1) / denominator

    if 0 <= weight <= 1:
        intersection_point = line_point1 + weight * line_direction

        return intersection_point, weight

def is_visible(point, plane):
    return plane @ point >= 0


def extract_frustum_planes(matrix):
    """
    Returns a list of frustum:
    left, right, bottom, top, near, far
    """
    planes = np.zeros((6, 4))

    planes[0] = normalize_plane(matrix[:, 3] + matrix[:, 0])  # left
    planes[1] = normalize_plane(matrix[:, 3] - matrix[:, 0])  # right
    planes[2] = normalize_plane(matrix[:, 3] + matrix[:, 1])  # bottom
    planes[3] = normalize_plane(matrix[:, 3] - matrix[:, 1])  # top
    planes[4] = normalize_plane(matrix[:, 3] + matrix[:, 2])  # near
    planes[5] = normalize_plane(matrix[:, 3] - matrix[:, 2])  # far
    return planes


def clipping(polygon, planes):
    """
        w_coord = bar_screen @ self.vertices[W]
        perspective = bar_screen * self.vertices[W] / w_coord[add_dim]
    """
    result_polygon = polygon
    for plane in planes:
        new_polygon = []

        edges = len(result_polygon)
        for i in range(edges):
            current_point = result_polygon[i]
            next_point = result_polygon[(i + 1) % edges]

            current_point_visible = is_visible(current_point, plane)
            next_point_visible = is_visible(next_point, plane)

            if current_point_visible:
                new_polygon.append(current_point)

            #  income or outcome
            if current_point_visible ^ next_point_visible:
                intersection_point, weight = line_plane_intersection(next_point, current_point, plane)
                new_polygon.append(intersection_point)

        result_polygon = new_polygon
    return result_polygon


def calculate_weight(vec_a, vec_b, vec_p):
    return np.linalg.norm(vec_p - vec_a) / np.linalg.norm(vec_b - vec_a)


def interpolate_coordinates(coord_a, coord_b, weight):
    return coord_a * (1 - weight) + coord_b * weight


def get_parameterized(matrix):
    """
    Returns the parameterized plane of the given matrix.
    Can be used for debugging purposes in https://www.geogebra.org/3d
    """
    for plane in extract_frustum_planes(matrix):
        vars = 'xyz '
        print(' + '.join((''.join((f"{coef:.2f}", var)) for coef, var in zip(plane, vars))).replace("+ -", "- "), "= 0",
              sep='')
