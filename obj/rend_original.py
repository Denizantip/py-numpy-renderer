import time
from tkinter import Tk, Canvas, NW
import numpy as np
from PIL import ImageTk, Image
from core import Camera, Light, Scene, Model
from transformation import PROJECTION, translation, scale, rotate

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

    diablo.textures.register('normals', 'diablo3_pose/diablo3_pose_nm_tangent.tga')
    # diablo.textures.register('world_normal_map', 'diablo3_pose/diablo3_pose_nm.tga')
    # diablo.textures.register("specular", 'diablo3_pose/diablo3_pose_spec.tga', normalize=False)
    diablo.textures.register("diffuse", 'diablo3_pose/diablo3_pose_diffuse.tga', normalize=False)
    # diablo.textures.register("diffuse", 'grid.tga', normalize=False)
    # diablo.textures.register("glow", 'diablo3_pose/diablo3_pose_glow.tga', normalize=False)

    width = 1500
    height = 1500

    # floor.vertices = floor.vertices @ scale(2)
    light = Light((0, 0.9, 1.9), color=(1, 1, 1),
                  show_cube=False,
                  # constant=1,
                  # linear=0.2,
                  # quadratic=0.2,
                  # projection_type=PROJECTION.OPEN_GL_PERSPECTIVE,
                  projection_type=PROJECTION.OPEN_GL_ORTHOGRAPHIC,
                  )
    # cube = cube @ translation((-0.5, 1, 2.5))
    # diablo = diablo @ translation((0, -0.1, 0))
    # minicooper = minicooper @ rotate((0, -90, 0))
    # cube.normals *= -1

    camera = Camera((0, 1, 2), up=np.array((0, 1, 0)),
                    projection_type=PROJECTION.OPEN_GL_PERSPECTIVE,
                    # projection_type=PROJECTION.OPEN_GL_ORTHOGRAPHIC,
                    fovy=90,
                    near=2.5,
                    far=3.2,
                    resolution=(height, width),
                    y_offset=0
                    )
    scene = Scene(camera, light, center=(0, 0, 0))
    scene.add_model(floor)
    # scene.add_model(diablo)
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
