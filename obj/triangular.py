from obj.constants import *
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


def bresenham_line_with_thickness(start_point, end_point, thickness=1):
    delta = end_point - start_point
    steps = max(abs(delta)) + 1

    if steps == 1:
        yield start_point.astype(int)
        return

    step_size = delta / steps

    for t in range(-thickness // 2, thickness // 2 + 1):
        normal = np.array([-step_size[1], step_size[0]]) if delta[0] == 0 else np.array([0, 0])
        offset = t * normal

        for i in range(steps):
            yield (start_point + i * step_size + offset).astype(int)


def wu_antialiased_line(start_point, end_point):
    def ipart(x):
        return int(x)

    def fpart(x):
        return x - ipart(x)

    start_point = np.array(start_point)
    end_point = np.array(end_point)

    delta = end_point - start_point
    steps = max(abs(delta)) + 1

    # if steps == 1:
    #     yield start_point.astype(int)
    #     return

    step_size = delta / steps

    for i in range(steps):
        point = start_point + i * step_size
        x, y, z = map(int, point)
        intensity = 1 - fpart(point[1])
        yield (x, y, z, intensity)

        x, y, z = map(int, point + step_size)
        intensity = fpart(point[1])
        yield (x, y, z, intensity)


def wu_antialiased_line_with_thickness(start_point, end_point, thickness=1):
    def ipart(x):
        return int(x)

    def fpart(x):
        return x - ipart(x)

    start_point = np.array(start_point)
    end_point = np.array(end_point)

    delta = end_point - start_point
    steps = max(abs(delta)) + 1

    if steps == 1:
        yield tuple(start_point.astype(int))
        return

    step_size = delta / steps
    normal = np.array([-step_size[1], step_size[0], step_size[2]])

    for i in range(steps):
        point = start_point + i * step_size

        for t in range(-thickness // 2, thickness // 2 + 1):
            offset = t * normal
            x, y, z = map(int, point + offset)
            intensity = 1 - fpart(point[0])
            yield (x, y, z, intensity)


def triangle(p):
    length = len(p)
    for i in range(length):
        yield p[i], p[i - length + 1]


def make_triangles(p):
    length = len(p)
    for i in range(length):
        yield p[i], p[i - length + 1]


def shadow_texture(face, shadow_z_buffer, light):
    if face.shadow_normal[2] > 0:
        return

    height, width = light.resolution
    box = bound_box(face.shadow_vertices[XY], width, height)
    if box is None:
        return
    min_x, max_x, min_y, max_y = box
    p = np.mgrid[min_x: max_x, min_y: max_y].reshape(2, -1).T

    bar_screen = barycentric(*face.shadow_vertices[XY].astype(int), p)
    if bar_screen is None:
        return

    Bi = (bar_screen >= 0).all(axis=1)

    bar_screen = bar_screen[Bi]
    if not bar_screen.size:
        return
    y, x = p[Bi].T
    z = bar_screen @ face.shadow_vertices[Z]

    Zi = (shadow_z_buffer[x, y] < z)
    if not Zi.size:
        return

    x, y, z = x[Zi], y[Zi], z[Zi]
    shadow_z_buffer[x, y] = z


def rasterize(face, frame, z_buffer, light, camera):
    face.view_vertices = face.vertices @ camera.lookat
    face.vertices = face.view_vertices @ camera.projection

    depth = 1 / face.vertices[W_COL]
    face.vertices *= depth  # perspective division
    face.vertices = face.vertices @ camera.viewport
    face.vertices[W_COL] = depth

    if camera.backface_culling and face.unit_normal[2] < 0:
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

    bar_screen = barycentric(*face.vertices[XY].astype(np.int32), p)
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
    #

    return 0

def smoothstep(edge0, edge1, x_array):
    '''
                  limits in
     degrees | radians | dot space
     --------+---------+----------
        0    |   0.0   |    1.0
        22   |    .38  |     .93
        45   |    .79  |     .71
        67   |   1.17  |     .39
        90   |   1.57  |    0.0
       180   |   3.14  |   -1.0
    '''
    # Scale, and clamp x_array to 0-1 range element-wise
    x_array = np.clip((x_array - edge0) / (edge1 - edge0), 0.0, 1.0)
    # Evaluate polynomial element-wise
    return x_array * x_array * (3 - 2 * x_array)

def general_shading(face, bar, light, camera, frame, x, y):
    blinn = True

    shininess_factor = face.get_specular(bar)

    object_color = face.get_object_color(bar)
    if object_color is None:
        return

    norm = face.get_normals(bar)

    fragment_position = bar @ face.view_vertices[XYZ]

    light_dir = normalize(light.position - fragment_position)

    view_dir = normalize(camera.position - fragment_position)

    # inLight = smoothstep(np.cos(5), np.cos(2), (light.position * -light_dir).sum(axis=1))
    # shininess_factor *= inLight

    if blinn:
        halfway_dir = normalize(light_dir + view_dir)
        spec = (norm * halfway_dir).sum(axis=1).clip(0)[add_dim] ** shininess_factor

    else:
        reflect_dir = light.reflect(-light_dir, norm)
        spec = (view_dir * reflect_dir).sum(axis=1).clip(0)[add_dim] ** shininess_factor

    specular = light.specular * spec * face.material.Ns
    diff = (norm * light_dir).sum(axis=1)[add_dim]
    diffuse = diff * light.color

    # uniform_Mshadow = np.linalg.inv(camera.lookat)
    # sb_p = face.view_vertices @ uniform_Mshadow
    # sb_p = sb_p @ light.lookat @ light.projection
    # sb_p /= sb_p[W]
    # sb_p = sb_p @ light.viewport

    const = 0.000005 if light.projection_type == PROJECTION_TYPE.PERSPECTIVE else 0.005
    bias = const * np.tan(np.arccos(diff))
    bias = np.clip(bias, 0, const).squeeze()

    distance = np.linalg.norm((light.position - fragment_position), axis=1)

    attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance))[add_dim]

    # if face.model.shadowing:
    #     persp = face.screen_perspective(bar)
    #     sb_p = persp @ face.shadow_vertices[XYZ]
    #
    #     sb_p = sb_p[(sb_p[X] < 1500) & (sb_p[Y] < 1500)]
    #     _x, _y = sb_p[X].astype(int), sb_p[Y].astype(int)
    #
    #     shadow = (shadow_z_buffer[_x, _y]) >= sb_p[Z]
    #     diffuse[shadow] *= 0.3
    #     specular[shadow] = 0

    frame[x, y] = ((light.ambient * attenuation + diffuse * attenuation + specular * attenuation) * object_color ** 2.2).clip(0.05, 1) * 255


def flat_shading(face, light, frame, x, y):
    """Tested"""
    intensity = face.view_normal @ -light.direction.squeeze()
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
