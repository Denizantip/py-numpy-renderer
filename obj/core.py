import os.path

from typing import Iterator, Optional, List, Iterable
from os import PathLike

from functools import cached_property

from obj.cube_map import CubeMap, fill_frame_from_skybox
from obj.axes import draw_axis
from obj.frustums import draw_view_frustum

from obj.materials import Material
from obj.plane_intersection import extract_frustum_planes
from obj.transformation import ViewPort, perspectives, scale, looka_at_translate, look_at_rotate_lh, look_at_rotate_rh
from triangular import *
from numpy.typing import NDArray

total_bars = 0
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


def triangulate_int(poligon):
    for i in range(len(poligon) - 2):
        yield np.array([poligon[0], *poligon[1 + i: 3 + i]], dtype=np.int32)


class TextureMaps:
    texture_map = {
        'diffuse': 'map_Kd',
        'ambient': 'map_Ka',
        'specular': 'map_Ks',
        'shininess': 'map_Ns',
        'transparency': 'map_d',
        'normals': 'norm'
        }

    def __init__(self, model):
        self.model = model

    def register(self, attr_name: str, path: PathLike | str, normalize=True, tangent=False):
        if attr_name not in self.texture_map:
            raise ValueError(f"{attr_name} not recognized.\nSupported: {self.texture_map.keys()}")
        texture = self.load_texture(path)
        dt = np.dtype(np.float32, metadata={'tangent': tangent})

        if normalize:
            texture = texture * 2 - 1
        setattr(self.model.materials['default'], self.texture_map[attr_name], np.array(texture, dtype=dt))

    @staticmethod
    def load_texture(name):
        texture = Image.open(name)
        texture = texture.convert('RGB')
        texture = np.asarray(texture) / 255
        return texture


class Face:
    def __init__(
        self, instance, Vi: NDArray, Ti: Optional[NDArray]=None, Ni: Optional[NDArray]=None,
            material: Optional[NDArray]=['default'],
    ):  # noqa
        self._vi = Vi
        self._ti = Ti
        self._ni = Ni
        self._bar = None

        self.model = instance
        self.vertices = instance.vertices[Vi]
        self.world_vertices = self.vertices.copy()

        self.uv = instance.uv[Ti] if instance.uv is not None else None
        self.normals = instance.normals[Ni] if instance.normals is not None else None
        self.textures = instance.textures
        self.material = instance.materials.get(instance.material_group[material[0]], instance.materials['default'])

    @property
    def unit_normal_world_space(self) -> NDArray:
        a, b, c = self.world_vertices[XYZ]
        return normalize(np.cross(b - a, c - a)).squeeze()

    @property
    def unit_normal_current_space(self):
        "test in screen space O_o"
        a, b, c = self.vertices[XYZ]
        return normalize(np.cross(b - a, c - a)).squeeze()

    def get_UV(self, shape, perspective_bar):
        # cause of rounding errors in the perspective division there still can be a case when there is no UV in
        # range of texture resolution
        v = (perspective_bar @ self.uv[U]).clip(max=1.) * (shape[1] - 1)
        u = (1.0 - (perspective_bar @ self.uv[V])).clip(max=1.) * (shape[0] - 1)
        return np.array((u, v)).astype(np.int32)

    def get_specular(self, bar):
        if hasattr(self.material, 'map_Ks'):
            *shape, _ = self.material.map_Ks.shape
            U, V = self.get_UV(shape, bar)
            shininess_factor = self.material.map_Ks[U, V, 0, np.newaxis] * 255
            # shininess_factor = self.material.map_Ks[U, V, 1] * 255
        else:
            shininess_factor = self.material.Ks * 255  # the bigger -> smaller radius
        return shininess_factor

    def screen_perspective(self, bar_screen):
        w_coord = bar_screen @ self.vertices[W_COL]
        perspective = bar_screen * self.vertices[W] / w_coord

        if perspective.size:
            return perspective

    def get_object_color(self, bar):
        if hasattr(self.material, 'map_Kd'):
            *shape, _ = self.material.map_Kd.shape
            UV = self.get_UV(shape, bar)
            if UV is None:
                return
            U, V = UV
            object_color = self.material.map_Kd[U, V]
        else:
            object_color = self.material.Kd

        return object_color

    def get_normals(self, bar):
        if hasattr(self.material, 'norm'):
            *shape, _ = self.material.norm.shape
            U, V = self.get_UV(shape, bar)
            norm = self.material.norm[U, V]
            if self.material.norm.dtype.metadata['tangent']:
                norm = (self.tangent_(bar) @ norm[add_dim]).squeeze()

        elif self.normals is not None:
            norm = bar @ self.normals

        else:
            norm = bar @ np.array([self.unit_normal_world_space] * 3)

        return normalize(norm).squeeze()

    def tangent_(self, bar):
        a, b, c = self.world_vertices[XYZ]
        n = normalize(bar @ self.normals)

        #  interpolated normals
        #                      ←  3  →
        #                   _________
        #               ↗  /\__\__\__\  ↖
        #        n pixels /\/\__\__\__\   334
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
        B[..., 2] = n
        return B

    @staticmethod
    def linearize_z(depth, camera):
        return (2 * camera.near * camera.far) / (camera.far + camera.near - depth * (camera.far - camera.near))


class Model:
    def __init__(self,
                 vertices: NDArray, uv: NDArray | None, normals: NDArray | None, faces: NDArray,
                 shadowing: bool = False, materials: dict = None, material_group: list = None, clip=True,
                 depth_test=True
    ):
        self.vertices = vertices
        self.view_tri = np.empty_like(vertices)
        self.shadow_vertices = np.empty_like(vertices)
        self.clip = clip
        self.depth_test = depth_test

        self.shadowing = shadowing
        self.uv = uv
        self.normals = normals
        self._faces = faces
        self.materials = materials or {'default': Material()}
        self.material_group = material_group or ['default']
        self.textures = TextureMaps(self)
        self.shape = None
        self.silhouette = set()

    @property
    def faces(self) -> Iterator[Face]:
        return (Face(self, *face.T) for face in self._faces)

    @classmethod
    def load_model(cls, name, shadowing=True):
        """
        https://paulbourke.net/dataformats/obj/
        """
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
                                temp.append(-1)
                            else:
                                temp.append(idx)
                        temp.append(mtl_group.index(mtl) + 1)
                        _faces.append(temp)

                    faces.extend(triangulate_int(_faces))
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
                    material = Material()
                    mtl_lib[mtl_name] = material
                    continue
                else:
                    key, *val = file_line.split()
                    if key.startswith('map') or key == 'disp':
                        dir_name = os.path.dirname(mtllib)
                        path = os.path.join(dir_name, val[0])
                        if os.path.exists(path):
                            dt = np.float32
                            if key == 'map_bump':
                                key = 'norm'
                                dt = np.dtype(np.float32, metadata={'tangent': True})
                            setattr(material, key, np.array(TextureMaps.load_texture(path), dtype=dt))
                        else:
                            print(f"{key} {path} is not found. Recommend manually assign texture by descriptor "
                                  f"Model.texture.register")
                    else:
                        setattr(material, key, val)
        return mtl_lib

    def __matmul__(self, other):
        self.vertices = self.vertices @ other
        return self


class PositionedObject:
    def __init__(self, position, center=np.array([0, 0, 0]),):
        self.scene = None
        self.position = position
        self.center = center

    def direction_to(self, other):
        return normalize(self.direction - other)

    @property
    def direction(self):
        return normalize(self.position - self.center).ravel()

    def set_position(self, new_position: np.ndarray):
        self.position = new_position
        return self


class TransformationMatrixMixin:
    """Extension for Positioned objects"""
    def __init__(self,
                 x_offset=0,
                 y_offset=0,
                 # resolution=(1024, 1024),
                 projection_type: PROJECTION_TYPE = PROJECTION_TYPE.PERSPECTIVE,
                 up=np.array([0, 1, 0]),
                 near=0.001,
                 far=6,
                 fovy=90):
        self.up = up
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
        # width, height = self.scene.resolution
        height, width = self.scene.resolution
        aspect_ratio = width / height
        perspective_func = perspectives[self.scene.subsystem][self.projection_type][self.scene.system]
        return perspective_func(self.fovy, aspect_ratio, self.near, self.far)

    @property
    def rotate(self):
        if self.scene.system == SYSTEM.LH:
            # return look_at_rotate_lh(self.position, self.center, self.up)
            return look_at_rotate_lh(self.center, self.position, self.up)
        elif self.scene.system == SYSTEM.RH:
            # return look_at_rotate_rh(self.position, self.center, self.up)
            return look_at_rotate_rh(self.center, self.position, self.up)

    @property
    def translate(self):
        return looka_at_translate(self.position)

    @cached_property
    def lookat(self):
        return self.translate @ self.rotate

    @cached_property
    def MVP(self):
        return self.lookat @ self.projection

    @property
    def frustum_planes(self):
        return extract_frustum_planes(self.MVP)

    @property
    def viewport(self):
        return ViewPort(self.scene.resolution, self.far, self.near, x_offset=self.x_offset, y_offset=self.y_offset)


class Camera(PositionedObject, TransformationMatrixMixin):
    def __init__(self, position,
                 center,
                 show=False,
                 backface_culling=True,
                 **kwargs):
        super(Camera, self).__init__(np.array(position), center)
        super(PositionedObject, self).__init__(**kwargs)
        self.show = show
        self.backface_culling = backface_culling


class Light(PositionedObject, TransformationMatrixMixin):
    """
    Here Projection mixin needs for shadow mapping
    """
    def __init__(self, position,
                 light_type=Lightning.POINT_LIGHTNING,
                 center=(0, 0, 0),
                 color=(1., 1., 1.),
                 ambient_strength=0,
                 diffuse=1,
                 specular_strength=0.5,
                 show=False,
                 constant=1,
                 linear=0.14,
                 quadratic=0.07,
                 **kwargs
                 ):
        self.color = np.array(color)
        self.light_type = light_type
        super(Light, self).__init__(np.array(position), np.array(center))
        self.ambient = ambient_strength * self.color
        self.show = show
        self.diffuse = diffuse
        self.specular_strength = specular_strength

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

        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic

        super(PositionedObject, self).__init__(**kwargs)

    @staticmethod
    def reflect(I, N):
        return normalize(I - 2. * (N * I).sum(axis=1)[add_dim] * N)

    @staticmethod
    def smoothstep(edge0, edge1, x_array):
        """
        Smooth Hermite interpolation between 0 and 1 when edge0 < x < edge1.
        Results are undefined if edge0 >= edge1.
                      limits in
         degrees | radians | dot space
         --------+---------+----------
            0    |   0.0   |    1.0
            22   |    .38  |     .93
            45   |    .79  |     .71
            67   |   1.17  |     .39
            90   |   1.57  |    0.0
           180   |   3.14  |   -1.0
        """
        # assert edge0 < edge1
        x_array = np.clip((x_array - edge0) / (edge1 - edge0), 0.0, 1.0)
        # Evaluate polynomial element-wise
        return x_array * x_array * (3 - 2 * x_array)

    def attenuation(self, fragment_position):
        distance = np.linalg.norm((self.position - fragment_position), axis=1)

        denom = self.constant + distance * (self.linear + self.quadratic * distance)

        # attenuation = 1.0 / (self.constant + self.linear * distance + self.quadratic * (distance * distance))[
        #     add_dim]
        return 1.0 / denom[add_dim]


class Bound:
    def __set__(self, instance: 'Scene', value: Camera | Light):
        self.obj = value
        self.obj.scene = instance

        if isinstance(value, Light) and value.show:
            sub_model = Model.load_model('obj_loader_test/sphere.obj', shadowing=False)
            sub_model.clip = False
            sub_model = sub_model @ scale(0.1)
            try:
                sub_model = sub_model @ np.linalg.inv(value.lookat)
            except np.linalg.LinAlgError:
                sub_model = sub_model @ np.linalg.pinv(value.lookat)
            try:
                sub_model.normals = -sub_model.normals @ np.linalg.inv(value.lookat[mat3x3])
            except np.linalg.LinAlgError:
                sub_model.normals = -sub_model.normals @ np.linalg.pinv(value.lookat[mat3x3])
            instance.add_model(sub_model)

        elif isinstance(value, Camera) and value.show:
            sub_model = Model.load_model('obj_loader_test/camera.obj', shadowing=False)
            sub_model.clip = False
            sub_model = sub_model @ scale(0.1)
            sub_model = sub_model @ np.linalg.inv(value.lookat)
            sub_model.normals = -sub_model.normals @ np.linalg.inv(value.lookat[mat3x3])
            instance.add_model(sub_model)

    def __get__(self, instance: 'Scene', owner):
        return self.obj


class Scene:
    camera = Bound()
    light = Bound()
    debug_camera = Bound()

    def __init__(
            self,
            camera=Camera(position=(0, 0, 1),
                          center=(0, 0, 0)),
            light=Light(position=(1, 1, 1)),
            shadows=False,
            debug_camera=None,
            resolution=(1500, 1500),
            system=SYSTEM.RH,
            subsystem=SUBSYSTEM.DIRECTX,
            skymap=None
    ):
        self.system: SYSTEM = system
        self.subsystem: SUBSYSTEM = subsystem
        self.models: List[Model] = list()
        self.camera: Camera = camera
        self.light: Light = light
        self.debug_camera: Camera | None = debug_camera
        self.resolution = resolution
        self.skybox: 'CubeMap' = skymap

    def add_model(self, model: Model):
        self.models.append(model)

    def render(self):
        frame = np.zeros((*self.resolution, 3), dtype=np.float32)
        # z_buffer = np.full(self.resolution, self.camera.far if self.system == SYSTEM.RH else self.camera.near, dtype=np.float64)
        z_buffer = np.full(self.resolution, np.inf * self.system, dtype=np.float64)
        stencil_buffer = np.zeros(self.resolution, dtype=np.int16)
        # mvp_planes = self.debug_camera.frustum_planes.copy()
        mvp_planes = self.camera.frustum_planes

        if isinstance(self.skybox, CubeMap):
            fill_frame_from_skybox(frame, self.camera, self.skybox)
        elif isinstance(self.skybox, Iterable):
            frame[:] = np.array(self.skybox)
        else:
            frame[:] = [64/255, 0.5, 198/255]


        for model in self.models:
            for face in model.faces:
                shadow_volumes(face, self.light, model.silhouette)
                rasterize(face, frame, z_buffer, self.light, self.camera, debug_camera=self.debug_camera)#, self.debug_camera)

        # second pass to build shadow volumes
        ## fill the stencil buffer
        for model in self.models:
            for edge in model.silhouette:
                A, B = edge
                A, B = model.vertices[A], model.vertices[B]
                if self.light.light_type == Lightning.POINT_LIGHTNING:
                    C, D = (A + 1000 * normalize(A - (*self.light.position, 1)).squeeze(),
                            B + 1000 * normalize(B - (*self.light.position, 1)).squeeze())
                else:
                    C, D = (A + (*self.light.direction * -1000, 1),
                            B + (*self.light.direction * -1000, 1))

                quad = np.array((A, B, D, C))
                resterize_quadrangle(quad, z_buffer, stencil_buffer, frame, self.camera)

        for model in self.models:
            total_faces = len(model._faces)
            errors_count = {err: 0 for err in Errors}
            rendered_faces = 0
            for face in model.faces:
                error = rasterize(face, frame, z_buffer, self.light, self.camera, stencil_buffer, debug_camera=self.debug_camera)  # , self.debug_camera)
                if error:
                    errors_count[error] += 1
                else:
                    rendered_faces += 1
            print('Total faces', total_faces)
            print('Face rendered', rendered_faces)
            print('Discarded', errors_count)
        # TODO need to attach this function to camera
        draw_view_frustum(frame, self.camera, self.debug_camera, z_buffer, self.system)
        # frame = draw_axis(frame, self.camera, z_buffer, self.system)
        return (frame[::-1] ** 0.8 * 255).astype(np.uint8)
