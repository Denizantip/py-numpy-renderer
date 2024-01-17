from enum import IntFlag, auto

import numpy as np

# from core import Face


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
    line_direction = line_point2 - line_point1

    denominator = plane_coefficients @ line_direction
    if abs(denominator) < 1e-10:
        return None  # No intersection (parallel)
    # weight needed for texture coordinate interpolation
    weight = -(plane_coefficients @ line_point1) / denominator
    # print('T coeff', t)
    if 0 <= weight <= 1:
        intersection_point = line_point1 + weight * line_direction

        return intersection_point

    return None  # No intersection between segment points


def classify_point(point):
    # return (point[3] > point[:3]).all() and (-point[3] > point[:3]).all()
    return ((point[3] > point[:3]) & (-point[3] < point[:3])).all()


def is_visible(point, plane):
    distance = plane @ point
    return distance >= 0


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


def sutherland_hodgman_3d(face, frustum_planes):
    output_list = list(face)

    # Iterate over all planes
    for plane_coefficients in frustum_planes:
        input_list = output_list
        output_list = []

        if not input_list:
            break

        s = input_list[-1]
        previous_point_location = is_visible(s, plane_coefficients)

        for point in input_list:
            current_point_location = is_visible(point, plane_coefficients)

            if current_point_location:
                if not previous_point_location:
                    intersect_point = line_plane_intersection(s, point, plane_coefficients)
                    # intersection point to vertex buffer
                    if intersect_point is not None:
                        output_list.append(intersect_point)
                    # print('Weight: ', calculate_weight(s, point, intersect_point))
                output_list.append(point)
            elif previous_point_location:
                intersect_point = line_plane_intersection(s, point, plane_coefficients)
                if intersect_point is not None:
                    output_list.append(intersect_point)

            s = point
            previous_point_location = current_point_location

    return output_list


def sutherland_hodgman_clip(polygon, clipping_planes):
    # Initial clipped polygon
    clipped_polygon = list(polygon)

    for plane in clipping_planes:
        new_clipped_polygon = []
        edges = len(clipped_polygon)
        # Iterate over each edge of the clipped polygon
        for idx in range(edges):
            current_point = clipped_polygon[idx]
            next_point = clipped_polygon[(idx + 1) % edges]

            # Check if the current point is inside the clipping window
            if is_visible(-plane, current_point):
                new_clipped_polygon.append(current_point)

            # Check if the edge intersects with the plane
            intersection_point = line_plane_intersection(current_point, next_point, plane)
            if intersection_point is not None:
                new_clipped_polygon.append(intersection_point)

        clipped_polygon = new_clipped_polygon

    return clipped_polygon


def clip_polygon_homogeneous(polygon, clipping_planes):
    """
    Clip a polygon against a convex clipping window in homogeneous coordinates using Sutherland-Hodgman algorithm.

    Parameters:
    - polygon: List of vertices of the input polygon in homogeneous coordinates.
    - clipping_planes: List of clipping planes (normal vectors) in homogeneous coordinates.

    Returns:
    - List of vertices of the clipped polygon in homogeneous coordinates.
    """
    result_polygon = polygon

    for plane in clipping_planes:
        new_polygon = []
        shape = len(result_polygon)
        for i in range(shape):
            current_vertex = result_polygon[i]
            next_vertex = result_polygon[(i + 1) % shape]

            current_distance = plane @ current_vertex
            next_distance = plane @ next_vertex

            if current_distance >= 0:
                new_polygon.append(current_vertex)

            if current_distance * next_distance < 0:
                # Crossing the clipping plane
                intersection_point = (current_vertex * abs(next_distance) + next_vertex * abs(current_distance)) / (abs(current_distance) + abs(next_distance))
                new_polygon.append(intersection_point)

        result_polygon = new_polygon

    return result_polygon


def clipping(polygon, planes):
    prev_point = polygon[-1]
    new_polygon = []
    for current_point in polygon:
        current_point_visible = classify_point(current_point)
        if not current_point_visible:
            new_polygon.append(current_point)
        for plane in planes:
            # intersection_point = line_plane_intersection(prev_point, current_point, plane)
            intersection_point = line_plane_intersection(current_point, prev_point, plane)
            if intersection_point is not None:
                new_polygon.append(intersection_point)
        prev_point = current_point
    return new_polygon


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

