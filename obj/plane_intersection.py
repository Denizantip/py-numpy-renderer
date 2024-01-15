from enum import IntFlag, auto

import numpy as np

from core import Face


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


def line_plane_intersection(line_point1, line_point2, plane_coefficients):
    # cartesian_point1 = line_point1[:3] / line_point1[3]
    # cartesian_point2 = line_point2[:3] / line_point2[3]
    # line_direction = cartesian_point2 - cartesian_point1
    line_direction = line_point2 - line_point1
    # line_direction[3] = 1  # Homogenous

    denominator = plane_coefficients @ np.append(line_direction, 1)
    if abs(denominator) < 1e-10:
        return None  # No intersection (parallel)
    # weight needed for texture coordinate interpolation
    weight = -(plane_coefficients @ np.append(line_point1, 1)) / denominator
    # print('T coeff', t)
    if 0 <= weight <= 1:
        intersection_point = line_point1 + weight * line_direction

        return intersection_point, weight

    return None  # No intersection between segment points


def classify_point(plane, point):
    distance = plane[0] * point[0] + plane[1] * point[1] + plane[2] * point[2] + plane[3]
    if distance < 0:
        return False
    else:
        return True


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


def sutherland_hodgman_3d(polygon: Face, frustum_planes):
    output_list = polygon
    # Iterate over all planes
    for plane_coefficients in frustum_planes:
        input_list = output_list
        output_list = []

        if not input_list:
            break

        s = input_list[-1]
        previous_point_location = classify_point(plane_coefficients, s)

        for point in input_list:
            current_point_location = classify_point(plane_coefficients, point)

            if current_point_location:
                if not previous_point_location:
                    intersect_point, weight = line_plane_intersection(s, point, plane_coefficients)
                    uv = interpolate_coordinates(s, point, weight)
                    # need to add uv into uv buffer (index)
                    # intersection point to vertex buffer
                    if intersect_point is not None:
                        output_list.append(intersect_point)
                    # print('Weight: ', calculate_weight(s, point, intersect_point))
                output_list.append(point)
            elif previous_point_location:
                intersect_point, weight = line_plane_intersection(s, point, plane_coefficients)
                uv = interpolate_coordinates(s, point, weight)

                if intersect_point is not None:
                    output_list.append(intersect_point)
                # print('Weight: ', calculate_weight(s, point, intersect_point))
            s = point
            previous_point_location = current_point_location

    return output_list


def calculate_weight(vec_a, vec_b, vec_p):
    return np.linalg.norm(vec_p - vec_a) / np.linalg.norm(vec_b - vec_a)


def interpolate_coordinates(coord_a, coord_b, weight):
    return coord_a * (1 - weight) + coord_b * weight


def get_parametrized(matrix):
    """
    Returns the parametrized plane of the given matrix.
    Can be used for debugging purposes in https://www.geogebra.org/3d
    """
    for plane in extract_frustum_planes(matrix):
        vars = 'xyz '
        print(' + '.join((''.join((f"{coef:.2f}", var)) for coef, var in zip(plane, vars))).replace("+ -", "- "), "= 0",
              sep='')


def classify_point_on_planes(points, matrix):
    for point in points:
        print(point)
        flags = Intersect(0)
        for plane, flag in zip(extract_frustum_planes(matrix), Intersect):
            flags |= flag if classify_point(plane, point) else Intersect(0)
        print(flags)

