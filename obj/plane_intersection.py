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


def clipping(polygon_vertices, clipping_planes):
    """
        Sutherland–Hodgman algorithm to clip a polygon with a list of planes.
        Forward solution. Iterates through all edges and all planes.
    """
    result_polygon = polygon_vertices
    for plane in clipping_planes:
        clipped_polygon = []
        num_edges = len(result_polygon)

        for i in range(num_edges):
            current_vertex = result_polygon[i]
            next_vertex = result_polygon[(i + 1) % num_edges]

            current_vertex_visible = is_visible(current_vertex, plane)
            next_vertex_visible = is_visible(next_vertex, plane)

            if current_vertex_visible:
                clipped_polygon.append(current_vertex)

            #  income or outcome
            if current_vertex_visible ^ next_vertex_visible:
                intersection_point = line_plane_intersection(next_vertex, current_vertex, plane)
                if intersection_point is not None:
                    clipped_polygon.append(intersection_point)

        result_polygon = clipped_polygon
    return np.array(result_polygon)


def get_parameterized(planes):
    """
    Returns the parameterized plane of the given matrix.
    Can be used for debugging purposes in https://www.geogebra.org/3d
    """
    for plane in planes:
        vars = 'xyz '
        print(' + '.join((''.join((f"{coef:.2f}", var)) for coef, var in zip(plane, vars))).replace("+ -", "- "), "= 0",
              sep='')

#########################################################
# All this kind of code is made for Sutherland hodgman clipping
# Below is example of usage:
# mvp_planes = extract_frustum_planes(self.camera.MVP)
# for model in self.models:
#     for face in model.faces:
#         # Fil up model silhouette.
#         visible_all = (face.vertices @ mvp_planes.T >= 0).all()
#         if visible_all:  # all vertices are visible (Inside view frustum)
#             rasterize(face, frame, z_buffer, self.light, self.camera)
#         else:  # some vertices are visible
#             polygon_vertices = clipping(face.vertices, mvp_planes) if model.clip else face.vertices
#             if len(polygon_vertices):
#                 bar = barycentric(*face.vertices[XYZ], polygon_vertices[XYZ])
#                 uvs = face.uv
#                 if bar is None:
#                     continue
#                 if uvs is not None:
#                     uvs = bar @ uvs
#                 normals = face.get_normals(bar)
#
#                 for idx in tringulate_args(polygon_vertices.shape[0]):
#                     face.uv = uvs[idx] if uvs is not None else None
#                     face.normals = normals[idx]
#                     face.vertices = polygon_vertices[idx]
#                     rasterize(face, frame, z_buffer, self.light, self.camera)
