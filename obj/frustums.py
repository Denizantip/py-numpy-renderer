import numpy as np

def get_view_frustum(camera: 'Camera'):
    verts = np.array([[-1.0, -1.0, 1.0, 1.0],
                      [1.0, -1.0, 1.0, 1.0],
                      [-1.0, 1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 1.0],
                      [-1.0, 1.0, -1.0, 1.0],
                      [1.0, 1.0, -1.0, 1.0],
                      [-1.0, -1.0, -1.0, 1.0],
                      [1.0, -1.0, -1.0, 1.0],
                      [*camera.position, 1],
                      [*camera.center, 1]])
    return verts

# segments = [(A, B), (B, D), (D, C), (C, A), (E, F), (G, E), (F, H), (H, G), (C, E), (D, F), (B, H), (A, G)]
indexes = np.array([(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (2, 4), (3, 5), (1, 7), (0, 6), (8, 9)])
