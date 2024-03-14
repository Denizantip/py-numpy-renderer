from obj.constants import *
from obj.lightning import Lightning
from transformation import bound_box, barycentric, normalize

import numpy as np


class Errors:
    BACK_FACE_CULLING = 1
    WRONG_MIN_MAX = 1 << 1
    EMPTY_B = 1 << 2
    EMPTY_Z = 1 << 3
    CLIPPED = 1 << 4


def bresenham_line(start_point, end_point, bound):

    delta = end_point - start_point
    steps = max(abs(delta[:2])) + 1
    # if steps == 1:
    #     yield start_point
    #     return
    step_size = delta / steps
    for i in range(int(steps)):
        interpolated_point = start_point + i * step_size
        if (interpolated_point[:2] > bound).any() or (interpolated_point[:2] < 0).any():
            continue
        if interpolated_point.shape[0] >= 3:
            interpolated_point[2] = 1 / interpolated_point[2]
        yield interpolated_point


def triangle(p):
    length = len(p)
    for i in range(length):
        yield p[i], p[i - length + 1]


def make_triangles(p):
    length = len(p)
    for i in range(length):
        yield p[i], p[i - length + 1]


def rasterize(face, frame, z_buffer, light, camera, stencil_buffer):
    """
    Previously this conversion could be done on ALL vertices at once.
    But the clipping by Sutherland Hodgeman became impossible. Because of absence of Fragment shader where clipping
    performed by inequality  -W < vertex[XYZ] < W. GPU stuff.
    For correct light computation we still need vertices in World space (or camera space. I choose world space.)
    """
    face.world_vertices = face.vertices.copy()
    face.vertices = face.vertices @ camera.MVP
    depth = 1 / face.vertices[W_COL]
    face.vertices *= depth  # perspective division
    face.vertices = face.vertices @ camera.viewport
    face.vertices[W_COL] = depth

    # if camera.backface_culling and face.test @ camera.position < 0:
    if camera.backface_culling and face.test[2] < 0:
        # BTW if the vertices was in Camera space second condition Could be face.normal.Z < 0
        return Errors.BACK_FACE_CULLING

    height, width, _ = frame.shape
    # ┌────────────────────────────────────────────────────────────────────┐
    # │ ┌─┐                            Scene                               │  ┌──────────────────┐
    # │ │X│  ┌───────────┐  ┌────────────┐                  ╭────────────╮ │  │     Renderer     │
    # │ │Y│  │ ModelView │  │ Projection │  ╒════════════╕  │ Projection │ │  │┌────────────────┐│
    # │ │Z│=>│   Matrix  │=>│    Space   │=>│  CLipping  │=>│  Division  │ │=>││     Viewport   ││
    # │ │W│  └───────────┘  └────────────┘  ╘════════════╛  │    (NDC)   │ │  ││ Transformation ││
    # │ └─┘      Camera        Clip                         ╰────────────╯ │  │└────────────────┘│
    # │ World                                                              │  └──────────────────┘
    # └────────────────────────────────────────────────────────────────────┘

    #     ( b0 ╱ gl_Position[0].w, b1 ╱ gl_Position[1].w, b2 ╱ gl_Position[2].w )
    # B = -------------------------------------------------------------------------
    #      b0 ╱ gl_Position[0].w + b1 ╱ gl_Position[1].w + b2 ╱ gl_Position[2].w

    #  the depth of the fragment is not linear in window coordinates, but the depth inverse (1╱gl_Position.w) is.
    #  Accordingly, the attributes and the clip-space barycentric coordinates, when weighted by the depth inverse,
    #  vary linearly in window coordinates.
    box = bound_box(face.vertices[XY], height, width)
    if box is None:
        return Errors.EMPTY_Z
    min_x, max_x, min_y, max_y = box
    p = np.mgrid[min_x: max_x, min_y: max_y].reshape(2, -1).T

    bar_screen = barycentric(*face.vertices[XY], p)
    if bar_screen is None:
        return Errors.EMPTY_B

    Bi = (bar_screen >= 0).all(axis=1)

    bar_screen = bar_screen[Bi]
    if not bar_screen.size:
        return Errors.EMPTY_B

    y, x = p[Bi].T
    z = bar_screen @ face.vertices[Z]

    if camera.scene.system == SYSTEM.RH:
        Zi = (z_buffer[x, y] >= z)
    else:
        Zi = (z_buffer[x, y] <= z)

    if not Zi.any():
        return Errors.EMPTY_Z
    bar_screen = bar_screen[Zi]
    x, y, z = x[Zi], y[Zi], z[Zi]

    if face.model.depth_test:
        z_buffer[x, y] = z

    # Points
    # points_only(face, camera, image)

    # Wireframe render
    # wireframe_shading(face, camera, frame, z_buffer)

    #  General shading
    general_shading(face, bar_screen, light, camera, frame, x, y)
    # pbr(face, light, camera, frame, bar_screen, x, y)
    # flat_shading(face, light, frame, x, y)
    # gouraud(face, light, frame, bar_screen, x, y)

    return 0


def general_shading(face, bar, light, camera, frame, x, y):
    blinn = False
    perspective_corrected_barycentric = face.screen_perspective(bar)
    shininess_factor = face.get_specular(perspective_corrected_barycentric)
    object_color = face.get_object_color(perspective_corrected_barycentric)
    fragment_normals = face.get_normals(perspective_corrected_barycentric)
    fragment_position = perspective_corrected_barycentric @ face.world_vertices[XYZ]

    if light.light_type == Lightning.DIRECTIONAL_LIGHTNING:
        surface_light_dir = light.direction[np.newaxis]
    else:
        surface_light_dir = normalize(light.position - fragment_position)

    surface_view_dir = normalize(camera.position - fragment_position)
    if light.light_type == Lightning.SPOT_LIGHTNING:
        inLight = light.smoothstep(np.cos(np.deg2rad(20)),
                                   np.cos(np.deg2rad(10)),
                                   (light.direction * surface_light_dir).sum(axis=1))
        object_color = object_color * inLight[add_dim]

    if blinn:
        halfway_dir = normalize(surface_light_dir + surface_view_dir)
        spec = (fragment_normals * halfway_dir).sum(axis=1).clip(0)[add_dim] ** shininess_factor

    else:
        reflect_dir = light.reflect(-surface_light_dir, fragment_normals)
        spec = (surface_view_dir * reflect_dir).sum(axis=1).clip(0)[add_dim] ** shininess_factor

    specular = light.color * spec * light.specular_strength
    intensity_light = (fragment_normals * surface_light_dir).sum(axis=1).clip(0)[add_dim]
    diffuse = intensity_light * light.color
    attenuation = light.attenuation(fragment_position)
    frame[x, y] = (attenuation * (light.ambient + diffuse + specular) * object_color).clip(0.05, 1) * 255
    # frame[x, y] = (light.ambient * object_color).clip(0.05, 1) * 255


def flat_shading(face, light, frame, x, y):
    """Tested"""
    intensity = face.unit_normal @ light.direction
    frame[x, y] = intensity.clip(0.3, 1.) * 255


def gouraud(face, light, frame, bar, x, y):
    intensity = ((bar @ face.normals) * light.direction).sum(axis=1).clip(0, 1)[add_dim]
    frame[x, y] = (intensity * [255, 255, 255])


def fresnelSchlick(cosTheta, F0):
    return F0 + (1.0 - F0) * cosTheta.clip(0, 1) ** 5


def DistributionGGX(N, H, roughness):
    a = roughness * roughness
    a2 = a * a
    NdotH = (N * H).sum(axis=1).clip(0)
    NdotH2 = NdotH * NdotH

    num = a2
    denom = (NdotH2 * (a2 - 1.0) + 1.0)
    denom = np.pi * denom * denom

    return num / denom


def GeometrySchlickGGX(NdotV, roughness):
    r = (roughness + 1.0)
    k = (r * r) / 8.0

    num = NdotV
    denom = NdotV * (1.0 - k) + k

    return num / denom


def GeometrySmith(N, V, L, roughness):
    NdotV = (N * V).sum(axis=1).clip(0)
    NdotL = (N * L).sum(axis=1).clip(0)
    ggx2 = GeometrySchlickGGX(NdotV, roughness)
    ggx1 = GeometrySchlickGGX(NdotL, roughness)

    return ggx1 * ggx2


def pbr(face, light, camera, frame, bar, x, y):
    albedo = face.get_object_color(bar)
    metallic = face.material.Pm
    roughness = face.material.Pr
    ao = np.array(face.material.Ka)

    PI = 3.14159265359

    N = normalize(bar @ face.normals)
    V = normalize(camera.position - (bar @ face.vertices[XYZ]))

    F0 = 0.04
    # F0 = np.arange(F0, albedo, metallic)

    # calculate per-light radiance
    L = normalize(light.position - (bar @ face.vertices[XYZ]))
    H = normalize(V + L)
    distance = np.linalg.norm(light.position - (bar @ face.vertices[XYZ]), axis=1)
    attenuation = 1.0 / (distance * distance)
    radiance = light.color * attenuation[add_dim]

    # cook-torrance brdf
    NDF = DistributionGGX(N, H, roughness)[add_dim]
    G = GeometrySmith(N, V, L, roughness)[add_dim]
    F = fresnelSchlick((H * V).sum(axis=1).clip(0), F0)[add_dim]

    kS = F
    kD = (1.0) - kS
    kD *= 1.0 - metallic

    numerator = NDF * G * F
    denominator = 4.0 * (N * V).sum(axis=1).clip(0) * (N * L).sum(axis=1).clip(0) + 0.0001
    specular = numerator / denominator[add_dim]

    # add to outgoing radiance Lo
    NdotL = (N * L).sum(axis=1).clip(0)
    Lo = (kD * albedo / PI + specular) * radiance * NdotL[add_dim]

    ambient = albedo * ao
    color = ambient + Lo

    # color = color / (color + (1.0))
    # color = pow(color, (1.0/2.2))

    frame[x, y] = (color * 255).astype(np.uint8)


def wireframe_shading(face, camera, frame, z_buffer):
    if face.unit_normal[2] < 0:
        return Errors.BACK_FACE_CULLING
    for p1, p2 in make_triangles(face.vertices.astype(np.int32)):
        for yy, xx, zz in bresenham_line(p1[XYZ], p2[XYZ], camera.resolution):

            xx = max(0, min(frame.shape[0] - 1, int(xx)))
            yy = max(0, min(frame.shape[1] - 1, int(yy)))
            if (z_buffer[xx, yy] - 1 / zz) > 0:
                frame[xx, yy] = (64, 64, 128)
                z_buffer[xx, yy] = zz


def points_only(face, camera, image):
    if face.unit_normal @ -normalize(camera.position)[0] <= 0:
        return Errors.BACK_FACE_CULLING
    for p1, p2 in make_triangles(face.vertices.astype(np.int32)):
        w = p1[2]
        image[p1[1], p1[0]] = (255, 0, 0)
        image[p2[1], p2[0]] = (0, 0, 255)

class Edge(tuple):
    def __eq__(self, other):
        return (other[0] == self[0] and other[1] == self[1]) or (other[0] == self[1] and other[1] == self[0])
    def __hash__(self):
        return hash(frozenset(self))

def shadow_volumes(face, light, container):
    if face.unit_normal @ light.position > 0:
        for i in range(3):
            curr_i, next_i = i, (i + 1) % 3
            edge = Edge((face._vi[curr_i], face._vi[next_i]))
            if edge in container:
                container.discard(edge)
            else:
                container.add(edge)

def shadow_volumes2(face, light, container):
    if face.unit_normal @ light.position > 0:
        for i in range(3):
            curr_i, next_i = i, (i + 1) % 3
            edge = (face._vi[curr_i], face._vi[next_i])
            rev_edge = (face._vi[next_i], face._vi[curr_i])
            if edge in container or rev_edge in container:
                container.discard(edge)
                container.discard(rev_edge)
            else:
                container.add(edge)


def barycentric_shadow(verts, p):
    """
    https://ceng2.ktu.edu.tr/~cakir/files/grafikler/Texture_Mapping.pdf
    """
    a, b, c = verts.round()
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


def quadrilateral_area(v0, v1, v2, v3):
    # Calculate side lengths from vertex coordinates
    side_a = np.linalg.norm(np.array(v1) - np.array(v0))
    side_b = np.linalg.norm(np.array(v2) - np.array(v1))
    side_c = np.linalg.norm(np.array(v3) - np.array(v2))
    side_d = np.linalg.norm(np.array(v0) - np.array(v3))

    s = (side_a + side_b + side_c + side_d) / 2.0
    area = np.sqrt((s - side_a) * (s - side_b) * (s - side_c) * (s - side_d))
    return area


def normal(triangles):
    # The cross product of two sides is a normal vector
    return np.cross(triangles[:, :, 1] - triangles[:, :, 0],
                    triangles[:, :, 2] - triangles[:, :, 0], axis=2)


def area(triangles):
    # The norm of the cross product of two sides is twice the area
    return np.linalg.norm(normal(triangles), axis=1) / 2

def triangle_area_vectorized(v0, v1, v2):
    cross_product = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(cross_product, axis=-1)
    return area

def bars_q(v0, v1, v2, v3, points):
    quad_area = quadrilateral_area(v0, v1, v2, v3)
    triangles = np.array([[[v0, v1, point], [v1, v2, point], [v2, v3, point], [v3, v0, point]] for point in points])
    areas = np.zeros((triangles.shape[0], 4))
    v0, v1, v2 = np.split(triangles, 3, axis=-2)
    areas = triangle_area_vectorized(v0, v1, v2)
    # areas = area(triangles)
    coeffs = areas / quad_area
    return areas

# triangles = np.array([[[v0, v1, point], [v1, v2, point], [v2, v3, point], [v3, v0, point]] for point in p])