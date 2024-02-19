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


def draw_axis(frame, camera, z_buffer, sign):
    MVP = camera.lookat @ camera.projection
    x_axis = np.array([[-1, 0, 0, 1], [1, 0, 0, 1]])
    x_letter_pos = np.array([1.05, 0, 0, 1])
    x_letter_neg = np.array([-1.2, 0, 0, 1])

    y_axis = np.array([[0, -1, 0, 1], [0, 1, 0, 1]])
    y_letter_pos = np.array([0, 1.05, 0, 1])
    y_letter_neg = np.array([0, -1.2, 0, 1])

    z_axis = np.array([[0, 0, -1, 1], [0, 0, 1, 1]])
    z_letter_pos = np.array([-0.05, 0, 1.05, 1])
    z_letter_neg = np.array([-0.05, 0, -1.2, 1])

    x_axis = transformer(x_axis, MVP, camera.viewport)
    x_letter_pos = transformer(x_letter_pos, MVP, camera.viewport)
    x_letter_neg = transformer(x_letter_neg, MVP, camera.viewport)

    y_axis = transformer(y_axis, MVP, camera.viewport)
    y_letter_pos = transformer(y_letter_pos, MVP, camera.viewport)
    y_letter_neg = transformer(y_letter_neg, MVP, camera.viewport)

    z_axis = transformer(z_axis, MVP, camera.viewport)
    z_letter_pos = transformer(z_letter_pos, MVP, camera.viewport)
    z_letter_neg = transformer(z_letter_neg, MVP, camera.viewport)

    image = Image.fromarray(frame)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 20)
    font = ImageFont.TransposedFont(font, Image.Transpose.FLIP_TOP_BOTTOM)

    draw = ImageDraw.Draw(image)
    R = (255, 0, 0)
    G = (0, 255, 0)
    B = (0, 0, 255)

    draw.text((x_letter_pos[0], x_letter_pos[1]), "+X", font=font, fill=R)
    draw.text((y_letter_pos[0], y_letter_pos[1]), "+Y", font=font, fill=G)
    draw.text((z_letter_pos[0], z_letter_pos[1]), "+Z", font=font, fill=B)

    draw.text((x_letter_neg[0], x_letter_neg[1]), "-X", font=font, fill=R)
    draw.text((y_letter_neg[0], y_letter_neg[1]), "-Y", font=font, fill=G)
    draw.text((z_letter_neg[0], z_letter_neg[1]), "-Z", font=font, fill=B)

    frame = np.array(image)

    for (start, end), color in zip([x_axis, y_axis, z_axis], [R, G, B]):
        for yy, xx, zz in bresenham_line(start[:3], end[:3], camera.resolution):
            for i in range(3):
                xx = max(0, min(frame.shape[0] - 4, int(xx)))
                yy = max(0, min(frame.shape[1] - 4, int(yy)))
                if (z_buffer[xx + i, yy + i] - 1 / zz) * sign < 0:
                    frame[xx + i, yy + i] = color
                    z_buffer[xx + i, yy + i] = zz
    return frame