import numpy as np
from PIL import Image, ImageFont, ImageDraw

from obj.constants import W_COL
from obj.triangular import bresenham_line


def transformer(vert, MVP, Viewport):
    vert = vert @ MVP
    vert /= vert[W_COL]
    vert = vert @ Viewport
    vert = vert.astype(int)
    return vert


def draw_axis(frame, camera):
    MVP = camera.lookat @ camera.projection
    x_axis = np.array([[0, 0, 0, 1], [1, 0, 0, 1]])
    x_letter = np.array([1.05, 0, 0, 1])

    y_axis = np.array([[0, 0, 0, 1], [0, 1, 0, 1]])
    y_letter = np.array([0, 1.05, 0, 1])

    z_axis = np.array([[0, 0, 0, 1], [0, 0, 1, 1]])
    z_letter = np.array([-0.05, 0, 1.1, 1])

    x_axis = transformer(x_axis, MVP, camera.viewport)
    x_letter = transformer(x_letter, MVP, camera.viewport)

    y_axis = transformer(y_axis, MVP, camera.viewport)
    y_letter = transformer(y_letter, MVP, camera.viewport)

    z_axis = transformer(z_axis, MVP, camera.viewport)
    z_letter = transformer(z_letter, MVP, camera.viewport)

    image = Image.fromarray(frame)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 20)
    font = ImageFont.TransposedFont(font, Image.Transpose.FLIP_TOP_BOTTOM)

    draw = ImageDraw.Draw(image)
    R = (255, 0, 0)
    G = (0, 255, 0)
    B = (0, 0, 255)

    draw.text((x_letter[0], x_letter[1]), "X", font=font, fill=R)
    draw.text((y_letter[0], y_letter[1]), "Y", font=font, fill=G)
    # draw.text((z_letter[0], z_letter[1]), "Z", font=ImageFont.TransposedFont(font, orientation=Image.Transpose.FLIP_TOP_BOTTOM), fill=B)
    draw.text((z_letter[0], z_letter[1]), "Z", font=font, fill=B)
    frame = np.array(image)

    for (start, end), color in zip([x_axis, y_axis, z_axis], [R, G, B]):
        for yy, xx in bresenham_line(start[:2], end[:2], camera.resolution):
            for i in range(3):
                xx = max(0, min(frame.shape[0] - 4, int(xx)))
                yy = max(0, min(frame.shape[1] - 4, int(yy)))
                frame[xx + i, yy + i] = color
    return frame