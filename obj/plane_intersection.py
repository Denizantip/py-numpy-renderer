import numpy as np


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
    # weight needed for texture/normals interpolation
    weight = -(plane_coefficients @ line_point1) / denominator

    if 0 <= weight <= 1:
        intersection_point = line_point1 + weight * line_direction

        return intersection_point


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
        Sutherland–Hodgman algorithm to clip a polygon with a list of planes.
        Forward solution. Iterates through all edges and all planes.
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
                intersection_point = line_plane_intersection(next_point, current_point, plane)
                if intersection_point is not None:
                    new_polygon.append(intersection_point)

        result_polygon = new_polygon
    return np.array(result_polygon)


def get_parameterized(matrix):
    """
    Returns the parameterized plane of the given matrix.
    Can be used for debugging purposes in https://www.geogebra.org/3d
    """
    for plane in extract_frustum_planes(matrix):
        vars = 'xyz '
        print(' + '.join((''.join((f"{coef:.2f}", var)) for coef, var in zip(plane, vars))).replace("+ -", "- "), "= 0",
              sep='')
