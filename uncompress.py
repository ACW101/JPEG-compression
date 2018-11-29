from PIL import Image
from subprocess import Popen, PIPE
import numpy as np
from math import cos, sqrt, pi, radians, ceil

LQM = np.asarray([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58,  60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17,  22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35,  55,  64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def fromZigZag(zig_zag_in, N):
    m = np.zeros((8, 8), dtype=np.float32)
    res = [0] * (N * N)
    index = -1
    for i in range(2*(N-1) + 1):
        bound = 0 if i < N else i - N + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                m[j, i-j] = zig_zag_in[index]
            else:
                m[i-j, j] = zig_zag_in[index]
    return m


def dequantize(m, N):
    dequantized = np.zeros((8, 8), dtype=np.int16)
    if N == 8:
        for i in range(N):
            for j in range(N):
                dequantized[i][j] = m[i][j] * LQM[i][j]
    if N == 16:
        for i in range(N):
            for j in range(N):
                dequantized[i][j] = m[i][j] * LQM[i/2][j/2]
    return dequantized


def iDCT(DCT, N):
    # pre-calculate C(i) * C(j)
    coefficients = np.ones((8, 8), dtype=np.float32)
    for i in range(8):
        coefficients[0][i] = 1 / sqrt(2)
        coefficients[i][0] = 1 / sqrt(2)
    coefficients[0][0] = 1/2
    # pre-calculate cosine terms
    cosines = np.zeros((8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            cosines[i][j] = cos((2 * i + 1) * j * pi / 16)

    iDCT = np.zeros((8, 8), dtype=np.float32)
    for x in range(N):
        for y in range(N):
            temp = 0.0
            for i in range(N):
                for j in range(N):
                    temp += cosines[x][i] * cosines[y][j] * \
                        DCT[i][j] * coefficients[i][j]
            iDCT[x][y] = np.rint(temp / sqrt(2 * N))
    return iDCT


def restore_jpg(data, N):
    m = fromZigZag(data, N)
    dequantized = dequantize(m, N)
    centered = iDCT(dequantized, N)
    uncentered = (centered + 128)
    uncentered[uncentered > 255] = 255
    uncentered[uncentered < 0] = 0
    return uncentered.astype(np.uint8)


def read_compressed(file):
    data = []
    with open(file, "r") as f:
        for line in f:
            data.append([int(n) for n in line.split(",")])
    return data


data = read_compressed("test_raw.txt")
data_iterator = iter(data)

w = 768
h = 512
N = 8

h_elems = []
for _ in range(h // 8):
    w_elems = []
    for _ in range(w // 8):
        w_elems.append(restore_jpg(next(data_iterator), N))
    h_elems.append(np.concatenate(w_elems, axis=1))
restored_matrix = np.concatenate(h_elems, axis=0)
img = Image.fromarray(restored_matrix, "L")
img.save("test_uncompressed.png")
