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
    # katana = Model.load_model("katana.obj")
    # sword = Model.load_model("pbr/sword.obj")
    # sword = sword @ scale(0.2)
    # sword = sword @ translation((0, -200, 0))

    # katana.textures.register('diffuse', 'handgrip_color.jpg')
    # katana = katana @ scale(0.1)
    # minicooper = Model.load_model('minicooper.obj')
    cube = Model.load_model('obj_loader_test/cube.obj', shadowing=False)
    cube.normals = None
    diablo = Model.load_model("diablo3_pose/diablo3_pose.obj")
    diablo.normals = None
    # deer = Model.load_model("deer.obj")
    floor = Model.load_model("floor.obj")
    # floor = floor @ scale(10)
    # floor.vertices[:, 0] = floor.vertices[:, 0] * 5
    # floor.vertices[:, 2] = floor.vertices[:, 2] * 5
    # floor.vertices[:, 1] += 4
    spheres = Model.load_model("obj_loader_test/sphere.obj")
    # suit = Model.load_model("suit.obj")

    floor.textures.register('diffuse', 'floor_diffuse.tga', normalize=False)
    # floor.textures.register('diffuse', 'grid.tga', normalize=False)

    # diablo.textures.register('normals', 'diablo3_pose/diablo3_pose_nm_tangent.tga', tangent=True)
    # diablo.textures.register('normals', 'diablo3_pose/diablo3_pose_nm.tga')
    # diablo.textures.register("specular", 'diablo3_pose/diablo3_pose_spec.tga')
    # diablo.textures.register("diffuse", 'diablo3_pose/diablo3_pose_diffuse.tga', normalize=False)
    # diablo.textures.register("glow", 'diablo3_pose/diablo3_pose_glow.tga', normalize=False)

    # floor.vertices = floor.vertices @ scale(2)
    light = Light((2, 2, -2),
                  light_type=Lightning.POINT_LIGHTNING,
                  show=False,
                  center=(0, 0, 0),
                  fovy=90,
                  ambient_strength=0,
                  specular_strength=1
                  )

    camera = Camera((-1.5, 3, 3), up=np.array((0, 1, 0)),
                    show=False,
                    fovy=90,
                    near=0.1,
                    far=10,
                    backface_culling=False,
                    resolution=(1500, 1500),
                    projection_type=PROJECTION_TYPE.PERSPECTIVE,
                    center=(0, 0, 0)
                    )

    camera2 = Camera((2, 0, -2), up=np.array((0, 1, 0)),
                     show=False,
                     fovy=45,
                     near=2,
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

    height, width = (1500, 1500)
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
    # scene.add_model(floor)
    # scene.add_model(diablo)

    # scene.add_model(cube_map)
    # scene.add_model(minicooper)
    # scene.add_model(katana)
    # scene.add_model(sword)
    # scene.add_model(deer)
    scene.add_model(cube)
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
