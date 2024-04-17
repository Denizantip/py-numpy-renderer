import time
from tkinter import Tk, Canvas, NW
import numpy as np
from PIL import ImageTk, Image
from matplotlib import pyplot as plt

from core import Camera, Light, Model, Scene
from obj.constants import PROJECTION_TYPE
from obj.cube_map import CubeMap
from obj.lightning import Lightning
from transformation import scale, SYSTEM, SUBSYSTEM, translation

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    # teapot = Model.load_model("teapot.obj")
    head = Model.load_model("african_head/african_head.obj")
    head.textures.register("diffuse", "african_head/african_head_diffuse.tga", normalize=False)
    # head.textures.register("normals", "african_head/african_head_nm.tga")
    head.textures.register("normals", "african_head/african_head_nm_tangent.tga", tangent=True)
    head.textures.register("specular", "african_head/african_head_spec.tga", normalize=False)

    head_eye_inner = Model.load_model("african_head/african_head_eye_inner.obj")
    head_eye_inner.textures.register("diffuse", "african_head/african_head_eye_inner_diffuse.tga", normalize=False)
    head_eye_inner.textures.register("normals", "african_head/african_head_eye_inner_nm_tangent.tga", tangent=True)

    # head_eye_outer = Model.load_model("african_head/african_head_eye_outer.obj")
    # head_eye_outer.textures.register("diffuse", "african_head/african_head_eye_outer_diffuse.tga", normalize=False)
    # head_eye_outer.textures.register("normals", "african_head/african_head_eye_outer_nm_tangent.tga", tangent=True)
    # stage = Model.load_model("stage.obj")
    # stage = stage @ scale(0.15)
    # stage = stage @ scale(0.5)
    # stage = stage @ translation([0, -1.2, 0])
    # katana = Model.load_model("katana.obj")
    sword = Model.load_model("pbr/sword.obj")
    sword = sword @ scale(0.2)
    # sword = sword @ translation((0, -100, 0))

    # katana.textures.register('diffuse', 'handgrip_color.jpg')
    # katana = katana @ scale(0.1)
    # minicooper = Model.load_model('minicooper.obj')
    cube = Model.load_model('obj_loader_test/cube.obj', shadowing=False)
    cube.normals = -cube.normals

    cube = cube @ scale(0.25)

    deer = Model.load_model("deer.obj")
    deer = deer @ scale(0.001) @ translation([0, 0, 1])
    # deer.vertices[:, 1] -= 1
    floor = Model.load_model("floor.obj")
    floor.vertices[..., [0, 2]] = floor.vertices[..., [0, 2]] * 4
    floor.textures.register('diffuse', 'floor_diffuse.tga', normalize=False)
    # floor.textures.register('diffuse', 'grid.tga', normalize=False)
    # suit = Model.load_model("suit.obj")

    diablo = Model.load_model("diablo3_pose/diablo3_pose.obj")
    # diablo.normals = None
    diablo.textures.register('normals', 'diablo3_pose/diablo3_pose_nm_tangent.tga', tangent=True)
    # diablo.textures.register('normals', 'diablo3_pose/diablo3_pose_nm.tga')
    diablo.textures.register("specular", 'diablo3_pose/diablo3_pose_spec.tga', normalize=False)
    diablo.textures.register("diffuse", 'diablo3_pose/diablo3_pose_diffuse.tga', normalize=False)
    # diablo.textures.register("glow", 'diablo3_pose/diablo3_pose_glow.tga', normalize=False)

    # floor.vertices = floor.vertices @ scale(2)
    light = Light((0, 4, 3),
                  light_type=Lightning.DIRECTIONAL_LIGHTNING,
                  show=False,
                  center=(0, 0, 0),
                  fovy=90,
                  # linear=0.000000001,
                  # quadratic=0.0000000001,
                  ambient_strength=0.5,
                  specular_strength=0.5
                  )

    camera = Camera((0, 0, 5), up=np.array((0, 1, 0)),
                    show=False,
                    fovy=90,
                    near=0.0001,
                    far=2000,
                    backface_culling=True,
                    projection_type=PROJECTION_TYPE.PERSPECTIVE,
                    center=(0, 0, 0)
                    )

    camera2 = Camera((1, 1, 0), up=np.array((0, 1, 0)),
                     show=False,
                     fovy=45,
                     near=0.2,
                     far=6,
                     backface_culling=True,
                     center=(0, 0, 0),
                     projection_type=PROJECTION_TYPE.PERSPECTIVE,
                     )
    cube_map = CubeMap(back="skybox/back.jpg",
                       bottom="skybox/bottom.jpg",
                       front="skybox/front.jpg",
                       left="skybox/left.jpg",
                       right="skybox/right.jpg",
                       top="skybox/top.jpg")

    skymap = CubeMap(back="cubemap/neg-z.jpg",
                     front="cubemap/pos-z.jpg",
                     top="cubemap/pos-y.jpg",
                     bottom="cubemap/neg-y.jpg",
                     left="cubemap/neg-x.jpg",
                     right="cubemap/pos-x.jpg")

    cube_map_debug = CubeMap(back="cubemap_debug/back.png",
                             front="cubemap_debug/front.png",
                             top="cubemap_debug/top.png",
                             bottom="cubemap_debug/bottom.png",
                             left="cubemap_debug/left.png",
                             right="cubemap_debug/right.png",
                             normalize_input=True)

    height, width = (1500, 2500)
    scene = Scene(camera,
                  light,
                  shadows=True,
                  debug_camera=camera2,
                  # skymap=[64, 127, 198],
                  # skymap=cube_map_debug,
                  skymap=cube_map,
                  # skymap=None,
                  resolution=(height, width),
                  system=SYSTEM.RH,
                  subsystem=SUBSYSTEM.OPENGL)
    # scene.add_model(diablo)
    scene.add_model(floor)
    # scene.add_model(deer)
    # scene.add_model(head)
    # scene.add_model(head_eye_inner)
    # scene.add_model(head_eye_outer)

    # scene.add_model(sword)
    # scene.add_model(stage)
    # scene.add_model(minicooper)
    # scene.add_model(katana)
    # scene.add_model(sword)
    # scene.add_model(teapot)
    # scene.add_model(cube)
    # scene.add_model(spheres)

    # picture = scene.render()

    win = Tk()
    win.geometry(f"{width}x{height}")
    canvas = Canvas(win, width=width, height=height)

    canvas.pack()

    start = time.time()
    picture = scene.render()

    print(f"render took {time.time() - start}")
    img = ImageTk.PhotoImage(image=Image.fromarray(picture))
    image_container = canvas.create_image(0, 0, anchor=NW, image=img)

    win.mainloop()
