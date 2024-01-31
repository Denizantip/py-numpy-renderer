import numpy as np

def draw_antialiased_line(img, start, end, color):
    start, end = start.astype(int), end.astype(int)
    dimensions = len(start)

    steep = abs(end[1] - start[1]) > abs(end[0] - start[0])
    if steep:
        start[0], start[1] = start[1], start[0]
        end[0], end[1] = end[1], end[0]

    if start[0] > end[0]:
        start[0], end[0] = end[0], start[0]
        start[1], end[1] = end[1], start[1]

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    gradient = dy / dx if dx != 0 else 1  # Prevent division by zero

    xend = round(start[0])
    yend = start[1] + gradient * (xend - start[0])
    xgap = 1 - (start[0] + 0.5) % 1
    xpxl1 = int(xend)  # Used in the main loop
    ypxl1 = int(yend)

    if dimensions == 2:
        if steep:
            img[ypxl1, xpxl1, :] = np.round(color * (1 - yend % 1) * xgap).astype(np.uint8)
            img[ypxl1 + 1, xpxl1, :] = np.round(color * (yend % 1) * xgap).astype(np.uint8)
        else:
            img[xpxl1, ypxl1, :] = np.round(color * (1 - yend % 1) * xgap).astype(np.uint8)
            img[xpxl1, ypxl1 + 1, :] = np.round(color * (yend % 1) * xgap).astype(np.uint8)
    elif dimensions == 3:
        if steep:
            img[ypxl1, xpxl1, :] = np.round(color * (1 - yend % 1) * xgap).astype(np.uint8)
            img[ypxl1 + 1, xpxl1, :] = np.round(color * (yend % 1) * xgap).astype(np.uint8)
        else:
            img[xpxl1, ypxl1, :] = np.round(color * (1 - yend % 1) * xgap).astype(np.uint8)
            img[xpxl1, ypxl1 + 1, :] = np.round(color * (yend % 1) * xgap).astype(np.uint8)

    intery = yend + gradient  # Initial y-intersection for the main loop

    xend = round(end[0])
    yend = end[1] + gradient * (xend - end[0])
    xgap = (end[0] + 0.5) % 1
    xpxl2 = int(xend)  # Used in the main loop
    ypxl2 = int(yend)

    if dimensions == 2:
        if steep:
            img[ypxl2, xpxl2, :] = np.round(color * (1 - yend % 1) * xgap).astype(np.uint8)
            img[ypxl2 + 1, xpxl2, :] = np.round(color * (yend % 1) * xgap).astype(np.uint8)
        else:
            img[xpxl2, ypxl2, :] = np.round(color * (1 - yend % 1) * xgap).astype(np.uint8)
            img[xpxl2, ypxl2 + 1, :] = np.round(color * (yend % 1) * xgap).astype(np.uint8)
    elif dimensions == 3:
        if steep:
            img[ypxl2, xpxl2, :] = np.round(color * (1 - yend % 1) * xgap).astype(np.uint8)
            img[ypxl2 + 1, xpxl2, :] = np.round(color * (yend % 1) * xgap).astype(np.uint8)
        else:
            img[xpxl2, ypxl2, :] = np.round(color * (1 - yend % 1) * xgap).astype(np.uint8)
            img[xpxl2, ypxl2 + 1, :] = np.round(color * (yend % 1) * xgap).astype(np.uint8)

    # Main loop
    if dimensions == 2:
        if steep:
            for x in range(xpxl1 + 1, xpxl2):
                img[int(intery), x, :] = np.round(color * (1 - intery % 1)).astype(np.uint8)
                img[int(intery) + 1, x, :] = np.round(color * (intery % 1)).astype(np.uint8)
                intery += gradient
        else:
            for x in range(xpxl1 + 1, xpxl2):
                img[x, int(intery), :] = np.round(color * (1 - intery % 1)).astype(np.uint8)
                img[x, int(intery) + 1, :] = np.round(color * (intery % 1)).astype(np.uint8)
                intery += gradient
    elif dimensions == 3:
        if steep:
            for x in range(xpxl1 + 1, xpxl2):
                img[int(intery), x, :] = np.round(color * (1 - intery % 1)).astype(np.uint8)
                img[int(intery) + 1, x, :] = np.round(color * (intery % 1)).astype(np.uint8)
                intery += gradient
        else:
            for x in range(xpxl1 + 1, xpxl2):
                img[x, int(intery), :] = np.round(color * (1 - intery % 1)).astype(np.uint8)
                img[x, int(intery) + 1, :] = np.round(color * (intery % 1)).astype(np.uint8)
                intery += gradient
