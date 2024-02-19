import time
from tkinter import Tk, Canvas, NW
import numpy as np
from PIL import ImageTk, Image
from core import Camera, Light, Model, Scene
from obj.constants import PROJECTION_TYPE
from obj.cube_map import CubeMap
from transformation import scale, SYSTEM, SUBSYSTEM, translation

if __name__ == "__main__":
    katana = Model.load_model("katana.obj")
    # sword = Model.load_model("pbr/sword.obj")
    # sword = sword @ scale(0.2)
    # sword = sword @ translation((0, -200, 0))

    # katana.textures.register('diffuse', 'handgrip_color.jpg')
    katana = katana @ scale(0.1)
    # minicooper = Model.load_model('minicooper.obj')
    cube = Model.load_model('obj_loader_test/cube.obj', shadowing=False)
    diablo = Model.load_model("diablo3_pose/diablo3_pose.obj")
    # deer = Model.load_model("deer.obj")
    floor = Model.load_model("floor.obj")
    # floor = floor @ scale(10)
    # floor.vertices[:, 0] = floor.vertices[:, 0] * 5
    # floor.vertices[:, 2] = floor.vertices[:, 2] * 5
    # floor.vertices[:, 1] += 4
    # spheres = Model.load_model("spheres.obj")
    # suit = Model.load_model("suit.obj")

    floor.textures.register('diffuse', 'floor_diffuse.tga', normalize=False)
    # floor.textures.register('diffuse', 'grid.tga', normalize=False)

    # diablo.textures.register('normals', 'diablo3_pose/diablo3_pose_nm_tangent.tga', tangent=True)
    # diablo.textures.register('normals', 'diablo3_pose/diablo3_pose_nm.tga')
    # diablo.textures.register("specular", 'diablo3_pose/diablo3_pose_spec.tga', normalize=False)
    # diablo.textures.register("diffuse", 'diablo3_pose/diablo3_pose_diffuse.tga', normalize=False)
    # diablo.textures.register("diffuse", 'grid.tga', normalize=False)
    # diablo.textures.register("glow", 'diablo3_pose/diablo3_pose_glow.tga', normalize=False)

    # floor.vertices = floor.vertices @ scale(2)
    light = Light((0, 0, 2), color=(1, 1, 1),
                  show=True,
                  fovy=45,
                  ambient_strength=0
                  )

    # cube_map = Model.load_model('floor.obj', shadowing=False)
    camera = Camera((2, 0, 2), up=np.array((0, 1, 0)),
                    show=False,
                    fovy=90,
                    near=0.1,
                    far=10,
                    backface_culling=False,
                    resolution=(2000, 1500),
                    projection_type=PROJECTION_TYPE.PERSPECTIVE,
                    center=(0, 0, 0)
                    )

    camera2 = Camera((2, 3, -2), up=np.array((0, 1, 0)),
                     show=False,
                     fovy=120,
                     near=0.1,
                     far=2,
                     backface_culling=False,
                     x_offset=1500,
                     resolution=(1500, 1500),
                     center=(2, 0, 1),
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
                             right="cubemap_debug/right.png")

    height, width = (1500, 2000)
    scene = Scene(camera,
                  light,
                  debug_camera=camera2,
                  # skymap=cube_map_debug,
                  # skymap=cube_map,
                  skymap=None,
                  resolution=(height, width),
                  system=SYSTEM.RH,
                  subsystem=SUBSYSTEM.DIRECTX)
    scene.add_model(floor)
    scene.add_model(diablo)

    # scene.add_model(cube_map)
    # scene.add_model(minicooper)
    # scene.add_model(katana)
    # scene.add_model(sword)
    # scene.add_model(deer)
    # scene.add_model(cube)
    # scene.add(spheres)

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
