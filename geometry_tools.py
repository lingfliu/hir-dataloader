import math
import numpy as np


def rotate_matrix(raw, pitch, yaw):
    # raw (z)
    cosR = math.cos(raw)
    sinR = math.sin(raw)

    # pitch (x)
    cosP = math.cos(pitch)
    sinP = math.sin(pitch)

    # yaw (y)
    cosY = math.cos(yaw)
    sinY = math.sin(yaw)
    #
    # rmat = np.mat([[cosR*cosY-sinR*sinP*sinY, -sinR*cosP, cosR*sinY+sinR*sinP*cosY],
    #                [sinR*cosY+cosR*sinP*sinY, cosR*cosP, sinR*sinY-cosR*sinP*cosY],
    #                [-cosP*sinY, sinP, cosP*cosY]])

    rmat = np.mat([[cosY*cosR, -cosY*sinR, sinY],
                   [sinP*sinY*cosR+cosP*sinR, -sinP*sinY*sinR+cosP*cosR, -sinP*cosY],
                   [-cosP*sinY*cosR + sinP*sinR, cosP*sinY*sinR + sinP*cosR, cosP*cosY]])

    return rmat
