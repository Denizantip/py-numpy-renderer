import numpy as np
from numpy.typing import NDArray
from typing import Annotated, Literal

U = X = (..., 0)  # pts[:, 0]
V = Y = (..., 1)  # pts[:, 1]
Z = (..., 2)  # pts[:, 2]
W = (..., 3)  # pts[:, 3]
W_COL = (..., [3])  # pts[:, [3]]
XY = (..., (0, 1))  # pts[:, :2]
XZ = (..., (0, 2))  # pts[:, [0,2]]
YZ = (..., (1, 2))   # pts[:, 1:3]
XYZ = (..., slice(None, 3))  # pts[:, :3]
XYZW = None
mat3x3 = (slice(None, 3), slice(None, 3))  # pts[:3, :3]
add_dim = (..., np.newaxis)


class PROJECTION_TYPE:
    PERSPECTIVE = 1
    ORTHOGRAPHIC = 2


class SUBSYSTEM:
    DIRECTX = 1
    OPENGL = 2


class SYSTEM:
    LH = 1
    RH = 2


class Projection:
    projection_type: PROJECTION_TYPE = PROJECTION_TYPE.PERSPECTIVE
    system: SYSTEM = SYSTEM.LH
    subsystem: SUBSYSTEM = SUBSYSTEM.OPENGL


vec3 = Annotated[NDArray[np.float32 | np.int32], 3]
vec4 = Annotated[NDArray[np.float32 | np.int32], 4]
