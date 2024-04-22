from enum import Flag, auto
from matplotlib.path import Path as mplPath
from numba import jit
from PIL import Image

from obj.constants import *
from obj.lightning import Lightning
from obj.line import bresenham_line
from obj.plane_intersection import clipping
from transformation import bound_box, barycentric, normalize

import numpy as np


class Errors(Flag):
    BACK_FACE_CULLING = auto()
    WRONG_MIN_MAX = auto()
    EMPTY_B = auto()
    EMPTY_Z = auto()
    CLIPPED = auto()


def make_triangles(p):
    length = len(p)
    for i in range(length):
        yield p[i], p[(i + 1) % length]


def rasterize(face, frame, z_buffer, light, camera, stencil_buffer=None, debug_camera=None):
    """
    Previously this conversion could be done on ALL vertices at once.
    But the clipping by Sutherland Hodgeman became impossible. Because of absence of Fragment shader where clipping
    performed by inequality  -W < vertex[XYZ] < W. GPU stuff.
    For correct light computation we still need vertices in World space (or camera space. I choose world space.)
    """
    face.world_vertices = face.vertices.copy()
    face.vertices = face.vertices @ camera.MVP

    clipping_space_debug = face.world_vertices @ debug_camera.MVP
    clipping_space = face.world_vertices @ camera.MVP

    depth = 1 / face.vertices[W_COL]  #     ↖
    face.vertices *= depth  # perspective division
    face.vertices = face.vertices @ camera.viewport
    face.vertices[W_COL] = depth

    if camera.backface_culling and face.unit_normal_current_space[2] < 0:
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

    if face.model.clip:
        persp = face.screen_perspective(bar_screen)
        if persp is not None and persp.size:
            for clip_space in (clipping_space, clipping_space_debug):
                clip_idx = persp @ clip_space
                Bi &= ((-clip_idx[W] < clip_idx[X]) & (clip_idx[X] < clip_idx[W]) &
                      (-clip_idx[W] < clip_idx[Y]) & (clip_idx[Y] < clip_idx[W]) &
                      (-clip_idx[W] < clip_idx[Z]) & (clip_idx[Z] < clip_idx[W]))

    bar_screen = bar_screen[Bi]
    if not bar_screen.size:
        return Errors.CLIPPED

    y, x = p[Bi].T
    # y = np.floor(bar_screen @ face.vertices[X]).astype(int)
    # x = np.floor(bar_screen @ face.vertices[Y]).astype(int)
    face.vertices[Z] = face.linearize_z(face.vertices[Z], camera)
    z = bar_screen @ face.vertices[Z]

    if camera.scene.system == SYSTEM.RH:
        #  Due to two passes we got here z-fighting :(
        Zi = (z_buffer[x, y] >= z)
    else:
        Zi = (z_buffer[x, y] <= z)

    if not Zi.any():
        return Errors.EMPTY_Z

    if stencil_buffer is not None:
        Zi &= (stencil_buffer[x, y] == 0)

    if not Zi.any():
        return Errors.EMPTY_Z

    bar_screen = bar_screen[Zi]
    x, y = x[Zi], y[Zi]

    if face.model.depth_test and stencil_buffer is None:
        z_buffer[x, y] = z[Zi]

    # Points
    # points_only(face, camera, image)

    # Wireframe render
    # wireframe_shading(face, frame, z_buffer)

    #  General shading
    general_shading(face, bar_screen, light, camera, frame, x, y, first_pass=stencil_buffer is None)
    # pbr(face, light, camera, frame, bar_screen, x, y)
    # flat_shading(face, light, frame, x, y)
    # gouraud(face, light, frame, bar_screen, x, y)

    return 0


def general_shading(face, bar, light, camera, frame, x, y, first_pass):
    perspective_corrected_barycentric = face.screen_perspective(bar)
    if perspective_corrected_barycentric is None:
        return Errors.EMPTY_B
    perspective_corrected_barycentric = perspective_corrected_barycentric[(perspective_corrected_barycentric >= 0).all(axis=1)]
    if perspective_corrected_barycentric.size == 0:
        return Errors.EMPTY_B
    object_color = face.get_object_color(perspective_corrected_barycentric)
    fragment_position = perspective_corrected_barycentric @ face.world_vertices[XYZ]
    attenuation = light.attenuation(fragment_position)
    if first_pass:
        frame[x, y] = (attenuation * light.ambient * object_color).clip(0.05, 1)
        return

    fragment_normals = face.get_normals(perspective_corrected_barycentric)

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

    specular_light = face.get_specular(perspective_corrected_barycentric)

    halfway_dir = normalize(surface_light_dir + surface_view_dir)
    specular_reflection = (fragment_normals * halfway_dir).sum(axis=1).clip(0)[add_dim] ** face.material.Ns

    specular = light.color * specular_reflection * light.specular_strength * specular_light
    intensity_light = (fragment_normals * surface_light_dir).sum(axis=1)[add_dim]
    diffuse = intensity_light * light.color
    frame[x, y] = (attenuation * object_color * (light.ambient + diffuse + specular)).clip(0.05, 1)


def flat_shading(face, light, frame, x, y):
    """Tested"""
    intensity = face.unit_normal_world_space @ light.direction
    frame[x, y] = intensity.clip(0.3, 1.) * 255


def gouraud(face, light, frame, bar, x, y):
    intensity = ((bar @ face.normals) * light.direction).sum(axis=1).clip(0, 1)[add_dim]
    frame[x, y] = (intensity * [255, 255, 255])


def fresnelSchlick(cosTheta, F0):
    # return F0 + (1.0 - F0) * cosTheta.clip(0, 1) ** 5
    return F0 + (1.0 - F0) * ((1 - cosTheta[:, None]) ** 5)


def DistributionGGX(N, H, roughness):
    a = roughness * roughness
    a2 = a * a
    NdotH = (N * H).sum(axis=1).clip(0)
    NdotH2 = NdotH * NdotH

    denom = (NdotH2 * (a2 - 1.0) + 1.0)
    denom = np.pi * denom * denom

    return a2 / denom


def GeometrySchlickGGX(NdotV, roughness):
    r = (roughness + 1.0)
    k = (r * r) / 8.0

    denom = NdotV * (1.0 - k) + k

    return NdotV / denom


def GeometrySmith(N, V, L, roughness):
    NdotV = (N * V).sum(axis=1).clip(0)
    NdotL = (N * L).sum(axis=1).clip(0)
    ggx2 = GeometrySchlickGGX(NdotV, roughness)
    ggx1 = GeometrySchlickGGX(NdotL, roughness)

    return ggx1 * ggx2


def pbr(face, light, camera, frame, bar, x, y):
    albedo = 1
    # albedo = face.get_object_color(bar)
    metallic = face.material.Pm
    roughness = face.material.Pr
    ao = np.array(face.material.Ka)

    PI = np.pi

    N = normalize(bar @ face.normals)
    V = normalize(camera.position - (bar @ face.vertices[XYZ]))

    F0 = np.array([0.04, 0.04, 0.04])
    F0 = mix(F0, albedo, metallic)

    # calculate per-light radiance
    L = normalize(light.position - (bar @ face.vertices[XYZ]))
    H = normalize(V + L)
    distance = np.linalg.norm(light.position - (bar @ face.vertices[XYZ]), axis=1)
    attenuation = 1.0 / (distance * distance)
    radiance = light.color * attenuation[add_dim]

    # cook-torrance brdf
    NDF = DistributionGGX(N, H, roughness)[add_dim]
    G = GeometrySmith(N, V, L, roughness)[add_dim]
    F = fresnelSchlick((H * V).sum(axis=1).clip(0), F0)

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

    color = color / (color + (1.0))
    # color = pow(color, (1.0/2.2))
    color = color ** (1.0/2.2)

    frame[x, y] = color


def wireframe_shading(face, frame, z_buffer):
    for p1, p2 in make_triangles(face.vertices):
        for yy, xx, zz in bresenham_line(p1[XYZ], p2[XYZ]):
            if 0 < xx < frame.shape[1] - 1 and 0 < yy < frame.shape[0] - 1 and (z_buffer[int(xx), int(yy)] - zz) > 0:
                frame[int(xx), int(yy)] = (64, 64, 128)
                z_buffer[int(xx), int(yy)] = zz


def points_only(face, camera, image):
    if face.unit_normal_world_space @ -normalize(camera.position)[0] <= 0:
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
    if face.unit_normal_world_space @ light.position > 0:
        for i in range(3):  # triple edge: AB BC CA
            curr_i, next_i = i, (i + 1) % 3
            edge = Edge((face._vi[curr_i], face._vi[next_i]))
            if edge in container:
                container.discard(edge)
            else:
                container.add(edge)


def inside(p, p0, p1):
    return np.cross(p - p0, p1 - p0) > 0


def outside(p, p0, p1):
    return np.cross(p - p0, p1 - p0) < 0


def quad_test(points, polygon, callback):
    l = len(polygon)
    is_inside_face = [callback(points, polygon[i], polygon[(i + 1) % l]) for i in range(l)]
    return np.logical_and.reduce(is_inside_face)


def resterize_quadrangle(quad, z_buffer, stencil_buffer, frame, camera):
    quad = clipping(quad, camera.frustum_planes)

    if quad.shape[0] < 3:
        return

    quad_ndc = quad @ camera.MVP

    quad = (quad_ndc / quad_ndc[W_COL]) @ camera.viewport
    a, b, c, *_ = quad[XYZ]
    plane_normal = np.cross(a - b, a - c)
    is_front = plane_normal[2] < 0

    Ax, By, Cz = plane_normal
    D = -a @ plane_normal

    height, width = camera.scene.resolution
    box = bound_box(quad, height, width)
    if box is None:
        return

    min_x, max_x, min_y, max_y = box
    p = np.mgrid[min_x: max_x, min_y: max_y].reshape(2, -1).T

    # Bi = ray_tracing_numpy_numba(p, quad[XY])
    # Bi = is_inside_sm_parallel(p, quad[XY])
    # Bi = points_inside_polygon(p, quad[XY])
    # Bi = mplPath(quad[XY], closed=False).contains_points(p)
    Bi = quad_test(p, quad[XY], inside if is_front else outside)

    y, x = p[Bi].T

    z = -(Ax * y + By * x + D) / Cz
    z = (                 (2 * camera.near * camera.far) /  # noqa
        #         -------------------------------------------------------------------
                   (camera.far + camera.near - z * (camera.far - camera.near)))

    if camera.scene.system == SYSTEM.RH:
        Zi = (z_buffer[x, y] >= z)
    else:
        Zi = (z_buffer[x, y] <= z)

    if not Zi.any():
        return
    x, y = x[Zi], y[Zi]

    if is_front:
        stencil_buffer[x, y] += 1
    else:
        stencil_buffer[x, y] -= 1

    # frame[x, y] = frame[x, y] * 0.5 + np.array([0.25, 0.25, 0.25])


def points_inside_polygon(points, vs):
    x, y = points[X], points[Y]
    x_vs = vs[X]
    y_vs = vs[Y]

    x_shifted = np.roll(x_vs, 1)
    y_shifted = np.roll(y_vs, 1)

    denom = y_shifted - y_vs
    # denom[denom == 0] = np.inf

    intersect = ((y_vs > y[:, np.newaxis]) != (y_shifted > y[:, np.newaxis])) & (
                x[:, np.newaxis] < (x_shifted - x_vs) * (y[:, np.newaxis] - y_vs) / denom + x_vs)
    inside = np.sum(intersect, axis=1) % 2 == 1

    return inside


def mix(x, y, a):
    """
    Linearly interpolate between x and y based on a (alpha).
    """
    return x * (1 - a) + y * a
