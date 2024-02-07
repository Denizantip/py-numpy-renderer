import time
from tkinter import Tk, Canvas, NW
import numpy as np
from PIL import ImageTk, Image
from core import Camera, Light, Scene, Model, triangulate_int
from obj.constants import PROJECTION_TYPE
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
    cube_map = Model.load_model('floor.obj', shadowing=False)

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

    diablo.textures.register('normals', 'diablo3_pose/diablo3_pose_nm_tangent.tga', tangent=True)
    # diablo.textures.register('normals', 'diablo3_pose/diablo3_pose_nm.tga')
    # diablo.textures.register("specular", 'diablo3_pose/diablo3_pose_spec.tga', normalize=False)
    diablo.textures.register("diffuse", 'diablo3_pose/diablo3_pose_diffuse.tga', normalize=False)
    # diablo.textures.register("diffuse", 'grid.tga', normalize=False)
    # diablo.textures.register("glow", 'diablo3_pose/diablo3_pose_glow.tga', normalize=False)

    # floor.vertices = floor.vertices @ scale(2)
    light = Light((1., 0., -1), color=(1, 1, 1),
                  show=True
                  )
    # cube = cube @ translation((-0.5, 1, 2.5))
    # diablo = diablo @ translation((0, -0.1, 0))
    # minicooper = minicooper @ rotate((0, -90, 0))
    # cube.normals *= -1

    camera = Camera((1, 2, 3), up=np.array((0, 1, 0)),
                    show=False,
                    fovy=60,
                    near=0.5,
                    far=5,
                    backface_culling=True,
                    resolution=(1500, 1500),
                    projection_type=PROJECTION_TYPE.PERSPECTIVE,
                    center=(0, 0, 0)
                    )

    camera2 = Camera((2, 0, -2), up=np.array((0, 1, 0)),
                     show=True,
                     fovy=60,
                     near=.5,
                     far=4,
                     backface_culling=True,
                     x_offset=1500,
                     resolution=(1500, 1500),
                     center=(0, 0, 0),
                     projection_type=PROJECTION_TYPE.PERSPECTIVE,
                     )

    height, width = (1500, 2000)
    scene = Scene(camera,
                  light,
                  debug_camera=camera2,
                  resolution=(height, width),
                  system=SYSTEM.LH,
                  subsystem=SUBSYSTEM.OPENGL)
    scene.add_model(floor)
    scene.add_model(diablo)

    # cube_map = cube_map @ translation((0, 1, 0))
    # cube_map = cube_map @ rotate((0, 90, 0))
    # view = camera2.lookat
    # # view[:3, 3]= 0
    # view[3, :3]= 0
    # MVP = np.linalg.inv(view @ camera2.projection)
    # cube_map = cube_map @ MVP
    # cube_map.vertices /= cube_map.vertices[:, [3]]

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
