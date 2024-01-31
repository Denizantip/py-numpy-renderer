import os.path
from copy import deepcopy

from functools import cache
import random
from typing import Iterator, Optional, List
from os import PathLike
from PIL import Image, ImageFont, ImageDraw

from obj.frustums import indexes, get_view_frustum
from obj.materials import Material
from obj.plane_intersection import *
from triangular import *
from numpy.typing import NDArray

'''
                ┌───────────────────────┐
                │ Vertex in WORLD SPACE │
                │        (x, y, z)      │
                └───────────────────────┘
                            ⇓
             ┌────────────────────────────┐  ┌────────────────────────────┐
             │ world-to-camera 4x4 matrix │  │ world-to-camera 4x4 matrix │
             └────────────────────────────┘  └────────────────────────────┘
                            ⇓                               ║
                ┌────────────────────────┐                  ║
                │ Vertex in CAMERA SPACE │                  ║
                │         (x, y, z)      │                  ║
                └────────────────────────┘                  ║
                            ∥                               ║
┌───────────────────────────────────────────────────────────────┐
│   VERTEX SHADER           ⇓                               ║   │
│               ┌────────────────────────┐                  ║   │
│               │ Vertex in CAMERA SPACE │                  ║   │
│               │       homogeneous      │                  ║   │
│               │     (x, y, z, w=1)     │                  ║   │
│               └────────────────────────┘                  ║   │
│                           ⇓                               ║   │
│              ┌────────────────────────────┐               ║   │
│              │    model @ Projection      │ <==═══════════╝   │
│              └────────────────────────────┘                   │
│                           ⇓                                   │
│              ┌────────────────────────────┐                   │
│              │   HOMOGENEOUS CLIP SPACE   │                   │
│              └────────────────────────────┘                   │
│                           ∥                                   │
└───────────────────────────────────────────────────────────────┘
                            ⇓                                              
                ┌────────────────────────┐                              
                │        Clipping        │                              
                │      -w <= x <= w      │                              
                │      -w <= y <= w      │
                │      -w <= z <= w      │                              
                └────────────────────────┘                              
                            ⇓                                              
                ┌────────────────────────┐                              
                │  Perspective division  │                              
                │          x / w         │                              
                │          y / w         │                              
                │          z / w         │                              
                └────────────────────────┘                                    
                            ⇓                                               
                ┌────────────────────────┐                                  
                │     View transform     │                                       
                └────────────────────────┘ 
'''


def triangulate(poligon):
    for i in range(len(poligon) - 2):
        yield np.array([poligon[0], *poligon[1 + i: 3 + i]], dtype=np.int32)


def triangulate_float(poligon):
    for i in range(len(poligon) - 2):
        yield np.array([poligon[0], *poligon[1 + i: 3 + i]])


class TextureMaps:
    texture_map = {
        'diffuse': 'map_Kd',
        'ambient': 'map_Ka',
        'specular': 'map_Ks',
        'shininess': 'map_Ns',
        'transparency': 'map_d',
        'normals': 'map_bump'
        }

    def __init__(self, model):
        self.model = model

    def register(self, attr_name: str, path: PathLike | str, normalize=True):
        if attr_name not in self.texture_map:
            raise ValueError(f"{attr_name} not recognized.\nSupported: {self.texture_map.keys()}")
        texture = self.load_texture(path)

        if normalize:
            setattr(self.model.materials['default'], self.texture_map[attr_name], (texture / 255) * 2 - 1)
        else:
            setattr(self.model.materials['default'], self.texture_map[attr_name], ((texture / 255)))

    @staticmethod
    def load_texture(name):
        texture = Image.open(name)
        texture = np.asarray(texture)[..., :3].copy()
        texture.setflags(write=1)
        return texture


class Face:
    def __init__(
        self, instance, Vi: NDArray, Ti: Optional[NDArray]=None, Ni: Optional[NDArray]=None,
            material: Optional[NDArray]=None,
    ):  # noqa
        self._vi = Vi
        self._ti = Ti
        self._ni = Ni
        self._bar = None

        self.model = instance
        self.vertices = instance.vertices[Vi]

        self.view_vertices = instance.view_tri[Vi]
        self.shadow_vertices = instance.shadow_vertices[Vi]

        self.uv = instance.uv[Ti] if instance.uv is not None else None
        self.normals = instance.normals[Ni] if instance.normals is not None else None
        self.textures = instance.textures
        self.material = instance.materials.get(instance.material_group[material[0]], instance.materials['default'])

    @property
    def unit_normal(self) -> NDArray:
        a, b, c = self.vertices[XYZ]
        return normalize(np.cross(b - a, c - a)).squeeze()

    @property
    def shadow_normal(self) -> NDArray:
        a, b, c = self.shadow_vertices[XYZ]
        return normalize(np.cross(b - a, c - a)).squeeze()

    def get_UV(self, shape, bar):
        b_persp = self.screen_perspective(bar)
        if b_persp is None:
            return
        V = (b_persp @ self.uv[X] * shape[1] - 1).astype(int)
        U = ((1.0 - (b_persp @ self.uv[Y])) * shape[0] - 1).astype(int)
        return U, V

    def get_specular(self, bar):
        if hasattr(self.material, 'map_Ks'):
            *shape, _ = self.material.map_Ks.shape
            U, V = self.get_UV(shape, bar)
            # shininess_factor = face.textures.specular[U, V][:, 2, None] * 255
            shininess_factor = self.material.map_Ks[U, V, 0, None] * 255
        else:
            shininess_factor = 16  # bigger -> smaller radius
        return shininess_factor

    def screen_perspective(self, bar_screen):
        w_coord = bar_screen @ self.vertices[W]
        perspective = bar_screen * self.vertices[W] / w_coord[add_dim]
        perspective = perspective[(perspective >= 0).all(axis=1)]

        if perspective.size:
            return perspective

    def get_object_color(self, bar):
        if hasattr(self.material, 'map_Kd'):
            *shape, _ = self.material.map_Kd.shape
            UV = self.get_UV(shape, bar)
            if UV is None:
                return
            U, V = UV

            object_color = self.material.map_Kd[U, V] ** (1 / 2.2)
        else:
            object_color = self.material.Kd
        return object_color

    def get_normals(self, bar):
        if hasattr(self.material, 'world_normal_map'):
            *shape, _ = self.textures.world_normal_map.shape
            U, V = self.get_UV(shape, bar)
            norm = self.textures.world_normal_map[U, V]

        elif hasattr(self.material, 'map_bump'):
            *shape, _ = self.material.map_bump.shape
            U, V = self.get_UV(shape, bar)
            norm = self.material.map_bump[U, V]
            norm = (self.tangent_(bar) @ norm[add_dim]).squeeze()

        elif self.normals is not None:
            norm = bar @ self.normals

        else:
            norm = bar @ np.array([self.unit_normal] * 3)

        return normalize(norm).squeeze()

    def tangent_(self, bar):
        a, b, c = self.view_vertices[XYZ]
        n = normalize(bar @ self.normals)

        #  interpolated normals
        #                      ←  3  →
        #                   _________
        #               ↗  /\__\__\__\  ↖
        #        n pixels /\/\__\__\__\   3
        #            ╱   /\/\/\__\__\__\    ↘
        #          ↙    /\/\/\/__/__/__/
        #              /\/\/\/__/__/__/
        #              \/\/\/__/__/__/
        #               \/\/__/__/__/
        #                \/__/__/__/
        #
        #                    ||
        #                    \/

        A = np.zeros((*n.shape, 3))
        A[:, 0] = b - a
        A[:, 1] = c - a
        A[:, 2] = n
        AI = np.linalg.inv(A)

        a_uv, b_uv, c_uv = self.uv.T
        i = AI @ np.array([a_uv[1] - a_uv[0], a_uv[2] - a_uv[0], 0])
        j = AI @ np.array([b_uv[1] - b_uv[0], b_uv[2] - b_uv[0], 0])

        B = np.empty((*n.shape, 3))
        B[..., 0] = normalize(i)
        B[..., 1] = normalize(j)
        B[..., 2] = normalize(n)
        return B


class Model:
    def __init__(self,
                 vertices: NDArray, uv: NDArray | None, normals: NDArray, faces: NDArray,
                 shadowing: bool = False, materials: dict = None, material_group: list = None
    ):
        self.vertices = vertices
        self.view_tri = np.empty_like(vertices)
        self.shadow_vertices = np.empty_like(vertices)

        self.shadowing = shadowing
        self.uv = uv
        self.normals = normals
        self._faces = faces
        self.materials = materials
        self.material_group = material_group
        self.textures = TextureMaps(self)
        self.shape = None

    @property
    def faces(self) -> Iterator[Face]:
        return (Face(self, *face.T) for face in self._faces)

    @classmethod
    def load_model(cls, name, shadowing=True):
        with open(name) as file:
            vertices = []
            faces = []
            normals = []
            uv = []
            mtl = 'default'
            mtl_group = ['default']
            materials = {'default': Material()}
            for file_line in file:
                if file_line.startswith("mtllib "):
                    mtllib = file_line.split()[1]
                    dirname = os.path.dirname(name)
                    mtl_path = os.path.join(dirname, mtllib)
                    materials |= cls.parse_mtl(mtl_path) if os.path.exists(mtl_path) else {}
                    continue
                if file_line.startswith('usemtl '):
                    mtl = file_line.split()[1]
                    mtl_group.append(mtl) if mtl not in mtl_group else None
                    continue
                if file_line.startswith("v "):
                    v = file_line.split()[1:]
                    if len(v) == 3:
                        v.append(1)  # adding w. Anyway it's going to be used.
                    vertices.append(v)
                    continue
                if file_line.startswith("f "):
                    _faces = []
                    for face in file_line.split()[1:]:
                        temp = []
                        for idx in face.split("/"):
                            if idx == '':
                                temp.append(0)
                            else:
                                temp.append(idx)
                        temp.append(mtl_group.index(mtl) + 1)
                        _faces.append(temp)

                    # _faces = [[0 if idx == '' else idx for idx in face.split("/")] for face in file_line.split()[1:]]
                    faces.extend(triangulate(_faces))
                    continue
                if file_line.startswith("vn "):
                    normals.append(file_line.split()[1:])
                    continue
                if file_line.startswith("vt "):
                    _uv = file_line.split()[1:]
                    if len(_uv) == 2:
                        _uv.append(0)
                    uv.append(_uv)
                    continue

            vertices = np.array(vertices, dtype=np.float32)
            faces = np.array(faces)
            faces = np.where(faces > 0, faces - 1, faces)
            normals = np.array(normals, dtype=np.float32) if normals else None
            uv = np.array(uv, dtype=np.float32) if uv else None

        return Model(vertices, uv, normals, faces, shadowing,
                     materials=materials, material_group=mtl_group)

    @staticmethod
    def parse_mtl(mtllib) -> dict:
        mtl_lib = {}
        with open(mtllib) as lib:
            for file_line in lib:
                if file_line.startswith('#') or file_line == '\n':
                    continue
                if file_line.startswith('newmtl '):
                    mtl_name = file_line.split()[1]
                    mtl_lib[mtl_name] = Material()
                    continue
                else:
                    key, *val = file_line.split()
                    if key.startswith('map'):
                        dir_name = os.path.dirname(mtllib)
                        path = os.path.join(dir_name, val[0])
                        if os.path.exists(path):
                            setattr(mtl_lib[mtl_name], key, TextureMaps.load_texture(path)/255)
                        else:
                            print(f"{key} {path} is not found. Recommend manually assign texture by descriptor "
                                  f"Model.texture.register")
                    else:
                        setattr(mtl_lib[mtl_name], key, val)
        return mtl_lib

    def __matmul__(self, other):
        self.vertices = self.vertices @ other
        return self


class PositionedObject:
    def __init__(self, position):
        self.scene = None
        self.position = position

    def direction_to(self, other):
        other = other.position if isinstance(other, PositionedObject) else np.array(other)
        return other - self.position

    def set_position(self, new_position: np.ndarray):
        self.position = new_position
        return self

    @property
    def direction(self):
        return normalize(-self.position)


class ProjectionMixin:
    """Extension for Positioned objects"""
    def __init__(self,
                 x_offset = 0,
                 y_offset = 0,
                 projection_type: PROJECTION_TYPE = PROJECTION_TYPE.PERSPECTIVE,
                 up=np.array([0, 1, 0]),
                 near=0.001,
                 center=np.array([0, 0, 0]),
                 far=10,
                 fovy=90):
        self.up = up
        self.center = np.array(center)
        self.projection_type = projection_type
        # self.resolution = resolution
        self.near = np.linalg.norm(self.position) if self.projection_type == PROJECTION_TYPE.ORTHOGRAPHIC else near
        self.far = far
        self.fovy = fovy
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.scene: Scene = None

    @property
    def projection(self):
        # width, height = self.resolution
        width, height = self.scene.resolution
        aspect_ratio = width / height
        perspective_func = perspectives[self.scene.subsystem][self.projection_type][self.scene.system]
        return perspective_func(self.fovy, aspect_ratio, self.near, self.far)

    @property
    @cache
    def lookat(self):
        if self.scene.system == SYSTEM.LH:
            return lookAtLH(self.position, self.center, self.up)
        elif self.scene.system == SYSTEM.RH:
            return lookAtRH(self.position, self.center, self.up)

    @property
    @cache
    def viewport(self):
        # return ViewPort(self.resolution, self.far, self.near, x_offset=self.x_offset, y_offset=self.y_offset)
        return ViewPort(self.scene.resolution, self.far, self.near, x_offset=self.x_offset, y_offset=self.y_offset)

    def LinearizeDepth(self, depth):
        # z = depth * 2 - 1
        z = depth
        result = (             2.0 * self.near * self.far /  # noqa
                  # ------------------------------------------------------
                      (self.far + self.near - z * (self.far - self.near))  )  # noqa
        return result

    def linearize_depth(self, z):
        return -(2 * self.far * self.near) / (z * (self.far - self.near) - self.far - self.near)


class Camera(PositionedObject, ProjectionMixin):
    def __init__(self, position, show=False, **kwargs):
        super(Camera, self).__init__(np.array(position))
        super(PositionedObject, self).__init__(**kwargs)
        self.show = show


class Light(PositionedObject, ProjectionMixin):
    """
    Here Projection mixin needs for shadow mapping
    """
    def __init__(self, position,
                 color=np.array([1., 1., 1]),
                 ambient_strength=0,
                 diffuse=1,
                 specular=2,
                 show=False,
                 constant=1,
                 linear=0.14,
                 quadratic=0.07,
                 **kwargs
                 ):
        self.color = color if isinstance(color, np.ndarray) else np.array(color)
        super(Light, self).__init__(np.array(position))
        self.ambient = ambient_strength * self.color
        self.show = show
        self.diffuse = diffuse
        self.specular = specular

        """
        Distance  | Constant  |  Linear   | Quadratic
        ----------+-----------+-----------+-----------
            7     |    1.0    |    0.7    |   1.8
            13    |    1.0    |    0.35   |   0.44
            20    |    1.0    |    0.22   |   0.20 ✓
            32    |    1.0    |    0.14   |   0.07  
            50    |    1.0    |    0.09   |   0.032 
            65    |    1.0    |    0.07   |   0.017
            100   |    1.0    |    0.045  |   0.0075
            160   |    1.0    |    0.027  |   0.0028
            200   |    1.0    |    0.022  |   0.0019
            325   |    1.0    |    0.014  |   0.0007
            600   |    1.0    |    0.007  |   0.0002
            3250  |    1.0    |    0.0014 |   0.000007

        """

        self.constant = constant,
        self.linear = linear,
        self.quadratic = quadratic,

        # distance = length(self.position - FragPos)
        # attenuation = 1.0 / (self.constant + self.linear * distance +
        #                       self.quadratic * (distance ** 2))
        super(PositionedObject, self).__init__(**kwargs)


class Bound:
    def __set__(self, instance: 'Scene', value: List[Camera | Light]):
        self.obj = value
        for obj in value:
            obj.scene = instance

            if isinstance(obj, Light) and obj.show:
                cube = Model.load_model('obj_loader_test/cone.obj', shadowing=False)
                cube = cube @ scale(0.1)
                # cube = cube @ translation(obj.position)
                cube.normals = -cube.normals
                cube = cube @ np.linalg.inv(obj.lookat)
                instance.add_model(cube)

            if isinstance(obj, Camera) and obj.show:
                camera = Model.load_model('obj_loader_test/camera.obj', shadowing=False)
                camera = camera @ scale(0.05)
                camera = camera @ np.linalg.inv(obj.lookat)
                instance.add_model(camera)

    def __get__(self, instance: 'Scene', owner):
        return self.obj


class Scene:
    cameras = Bound()
    light = Bound()

    def __init__(
            self,
            cameras=(Camera(position=(0, 0, 1)),),
            light=(Light(position=(1, 1, 1)), ),
            resolution=(1500, 3000),
            system=SYSTEM.RH,
            subsystem=SUBSYSTEM.DIRECTX
    ):
        self.system: SYSTEM = system
        self.subsystem: SUBSYSTEM = subsystem
        self.models: List[Model] = list()
        self.cameras: Camera | List[Camera] = cameras
        self.light: Light = light
        self.resolution = resolution

    def add_model(self, model: Model):
        self.models.append(model)

    def render(self):
        frame = np.zeros((*self.resolution, 3), dtype=np.uint8)
        z_buffer = np.full(self.resolution, -np.inf if self.system == SYSTEM.RH else np.inf, dtype=np.float32)
        # height, width, _ = self.frame.shape
        camera1, camera2 = self.cameras

        # shadow_z_buffer = np.full((height, width), -np.inf)

        ModelView = camera1.lookat
        ModelView2 = camera2.lookat
        # ShadowModelView = self.light[0].lookat

        Projection = camera1.projection
        Projection2 = camera2.projection
        # ShadowProjection = self.light[0].projection

        Viewport = camera1.viewport
        Viewport2 = camera2.viewport

        MVP = ModelView @ Projection
        MVP2 = ModelView2 @ Projection2
        print(extract_frustum_planes(MVP2))
        print(get_parameterized(MVP2))

        self.light[0].set_position((np.append(self.light[0].position, 1) @ ModelView)[XYZ])

        # shadow pass
        # shadow_transform = ShadowModelView @ ShadowProjection

        for model in self.models:
            total_faces = model._faces.shape[0]

            # model.view_tri = model.vertices @ ModelView
            # model.vertices = model.view_tri @ Projection

            depth = 1
            # if camera1.projection_type == PROJECTION_TYPE.PERSPECTIVE:
                # save for perspective correct interpolation (with zero division eps)
                # depth = 1 / model.vertices[W_COL]
                # model.vertices *= depth  # perspective division

            # model.vertices = model.vertices @ Viewport
            # model.vertices[W_COL] = depth

            if model.normals is not None:
                normals = model.normals @ ModelView[mat3x3]
                model.normals = normalize(normals)
            if hasattr(model.textures, 'world_normal_map'):
                model.textures.world_normal_map = normalize(model.textures.world_normal_map @ ModelView[mat3x3])

            rendered_faces = 0
            errors = [0, 0, 0, 0]

            for face in model.faces:
                print(face.vertices)
                print()
                new_polygon = clipping(face.vertices, extract_frustum_planes(MVP2))
                print(face.vertices)
                print()
                # print(new_polygon)
                if new_polygon:

                    # for idx in range(len(new_polygon)):
                    #     point = new_polygon[idx]
                    #     new_polygon[idx] = ((point / point[3]) @ Viewport).astype(int)

                    edges = len(new_polygon)
                    color = [random.randint(0, 255) for _ in range(3)]
                    for idx in range(edges):
                        current = new_polygon[idx] @ MVP
                        prev = new_polygon[(idx + 1) % edges] @ MVP

                        current = ((current / current[3]) @ Viewport).astype(int)

                        prev = ((prev / prev[3]) @ Viewport).astype(int)

                        for yy, xx, zz, _ in bresenham_line(prev, current):

                            for i in range(3):
                                xx = max(0, min(frame.shape[0] - 3, xx))
                                yy = max(0, min(frame.shape[1] - 3, yy))
                                if z_buffer[xx + i, yy + i] >= zz:
                                    frame[xx + i, yy + i] = color

                                    z_buffer[xx + i, yy + i] = zz
                else:
                    continue
                face.view_tri = face.vertices @ ModelView
                face.vertices = face.view_tri @ Projection

                depth = 1 / face.vertices[W_COL]
                face.vertices *= depth  # perspective division
                face.vertices = face.vertices @ Viewport
                face.vertices[W_COL] = depth

                code = rasterize(face, frame, z_buffer, self.light[0], camera1)
                if code:
                    for i in range(0, 4):
                        if (code >> i) == 1:
                            errors[i] += 1
                            break
                else:
                    rendered_faces += 1
            print('Total faces', total_faces)
            print('Face rendered', rendered_faces)
            print('CLipped', errors)

            x_axis = np.array([[0, 0, 0, 1], [1, 0, 0, 1]])
            x_letter = np.array([1.05, 0, 0, 1])

            y_axis = np.array([[0, 0, 0, 1], [0, 1, 0, 1]])
            y_letter = np.array([0, 1.05, 0, 1])

            z_axis = np.array([[0, 0, 0, 1], [0, 0, 1, 1]])
            z_letter = np.array([-0.05, 0, 1.1, 1])

            def transformer(vert, MVP, Viewport):
                vert = vert @ MVP
                vert /= vert[W_COL]
                vert = vert @ Viewport
                vert = vert.astype(int)
                return vert

            x_axis = transformer(x_axis, MVP, Viewport)
            x_letter = transformer(x_letter, MVP, Viewport)

            y_axis = transformer(y_axis, MVP, Viewport)
            y_letter = transformer(y_letter, MVP, Viewport)

            z_axis = transformer(z_axis, MVP, Viewport)
            z_letter = transformer(z_letter, MVP, Viewport)

            frame = Image.fromarray(frame)
            font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 20)
            font = ImageFont.TransposedFont(font, Image.Transpose.FLIP_TOP_BOTTOM)

            draw = ImageDraw.Draw(frame)
            R = (255, 0, 0)
            G = (0, 255, 0)
            B = (0, 0, 255)

            draw.text((x_letter[0], x_letter[1]), "X", font=font, fill=R)
            draw.text((y_letter[0], y_letter[1]), "Y", font=font, fill=G)
            draw.text((z_letter[0], z_letter[1]), "Z", font=ImageFont.TransposedFont(font, orientation=Image.Transpose.FLIP_TOP_BOTTOM), fill=B)
            frame = np.array(frame)

            for (start, end), color in zip([x_axis, y_axis, z_axis], [R, G, B]):
                for yy, xx in bresenham_line(start[:2], end[:2]):
                    for i in range(3):
                        xx = max(0, min(frame.shape[0] - 4, xx))
                        yy = max(0, min(frame.shape[1] - 4, yy))
                        frame[xx + i, yy + i] = color

            cube_frustum = get_view_frustum(camera2) @ np.linalg.inv(ModelView2 @ Projection2) @ MVP
            cube_frustum[-2] = np.append(camera2.position, 1) @ MVP
            cube_frustum[-1] = np.append(camera2.center, 1) @ MVP
            cube_frustum /= cube_frustum[W_COL]

            cube_frustum = (cube_frustum @ Viewport).astype(int)
            for start, end in cube_frustum[indexes]:

                for yy, xx, zz, _ in bresenham_line(start, end):
                    for i in range(3):
                        xx = max(0, min(frame.shape[0] - 3, xx))
                        yy = max(0, min(frame.shape[1] - 3, yy))
                        if z_buffer[xx+i, yy+i] > zz:
                            frame[xx + i, yy + i] = (128, 128, 128)
                            z_buffer[xx + i, yy + i] = zz

        return frame[::-1, ::-1]
        # return frame.transpose((1, 0, 2))
