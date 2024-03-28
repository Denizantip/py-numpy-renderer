import time
from tkinter import Tk, Canvas, NW
import numpy as np
from PIL import ImageTk, Image
from core import Camera, Light, Model, Scene
from obj.constants import PROJECTION_TYPE
from obj.cube_map import CubeMap
from obj.lightning import Lightning
from transformation import scale, SYSTEM, SUBSYSTEM, translation

if __name__ == "__main__":
    teapot = Model.load_model("teapot.obj")
    stage = Model.load_model("stage.obj")
    stage = stage @ scale(0.15)
    stage = stage @ translation([0, -1.2, 0])
    # katana = Model.load_model("katana.obj")
    # sword = Model.load_model("pbr/sword.obj")
    # sword = sword @ scale(0.2)
    # sword = sword @ translation((0, -200, 0))

    # katana.textures.register('diffuse', 'handgrip_color.jpg')
    # katana = katana @ scale(0.1)
    # minicooper = Model.load_model('minicooper.obj')
    cube = Model.load_model('obj_loader_test/cube.obj', shadowing=False)
    cube.normals = None

    cube = cube @ scale(0.25)

    deer = Model.load_model("deer.obj")
    deer = deer @ scale(0.001) @ translation([0, 0, 1])
    deer.vertices[:, 1] -= 1
    floor = Model.load_model("floor.obj")
    floor.textures.register('diffuse', 'floor_diffuse.tga', normalize=False)
    # floor.textures.register('diffuse', 'grid.tga', normalize=False)
    spheres = Model.load_model("obj_loader_test/sphere.obj")
    # suit = Model.load_model("suit.obj")

    diablo = Model.load_model("diablo3_pose/diablo3_pose.obj")
    # diablo.normals = None
    diablo.textures.register('normals', 'diablo3_pose/diablo3_pose_nm_tangent.tga', tangent=True)
    # diablo.textures.register('normals', 'diablo3_pose/diablo3_pose_nm.tga')
    # diablo.textures.register("specular", 'diablo3_pose/diablo3_pose_spec.tga')
    diablo.textures.register("diffuse", 'diablo3_pose/diablo3_pose_diffuse.tga', normalize=False)
    # diablo.textures.register("glow", 'diablo3_pose/diablo3_pose_glow.tga', normalize=False)

    # floor.vertices = floor.vertices @ scale(2)
    light = Light((0, 2, 0),
                  light_type=Lightning.DIRECTIONAL_LIGHTNING,
                  show=False,
                  center=(0, 0, 0),
                  fovy=90,
                  linear=0.01,
                  quadratic=0.01,
                  ambient_strength=0.5,
                  specular_strength=2
                  )

    camera = Camera((0, 1, 3), up=np.array((0, 1, 0)),
                    show=False,
                    fovy=90,
                    near=0.001,
                    far=20,
                    backface_culling=True,
                    resolution=(1500, 1500),
                    projection_type=PROJECTION_TYPE.PERSPECTIVE,
                    center=(0, 0, 0)
                    )

    camera2 = Camera((2, 0, -2), up=np.array((0, 1, 0)),
                     show=False,
                     fovy=45,
                     near=0.001,
                     far=3,
                     backface_culling=False,
                     resolution=(1500, 1500),
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

    height, width = (1500, 2000)
    scene = Scene(camera,
                  light,
                  debug_camera=camera2,
                  # skymap=[64, 127, 198],
                  skymap=cube_map_debug,
                  # skymap=cube_map,
                  # skymap=None,
                  resolution=(height, width),
                  system=SYSTEM.RH,
                  subsystem=SUBSYSTEM.OPENGL)
    scene.add_model(diablo)
    # scene.add_model(deer)
    scene.add_model(floor)

    scene.add_model(stage)
    # scene.add_model(minicooper)
    # scene.add_model(katana)
    # scene.add_model(sword)
    # scene.add_model(teapot)
    # scene.add_model(cube)
    # scene.add_model(spheres)

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
