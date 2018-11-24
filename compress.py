from PIL import Image
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


def DCT(m, N):
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

    dct = np.zeros((8, 8), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            temp = 0.0
            for x in range(N):
                for y in range(N):
                    temp += cosines[x][i] * cosines[y][j] * m[x][y]
            temp *= 1 / sqrt(2 * N) * coefficients[i][j]
            dct[i][j] = np.rint(temp)
    return dct.astype(int)


def quantize(m, N):
    if N == 8:
        for i in range(N):
            for j in range(N):
                m[i][j] = np.rint(m[i][j] / LQM[i][j])
    if N == 16:
        for i in range(N):
            for j in range(N):
                m[i][j] = np.rint(m[i][j] / LQM[i/2][j/2])


def toZigZag(m, N):
    res = [0] * (N * N)
    index = -1
    for i in range(2*(N-1) + 1):
        bound = 0 if i < N else i - N + 1
        for j in range(bound, i - bound + 1):
            index += 1
            if i % 2 == 1:
                res[index] = m[j, i-j]
            else:
                res[index] = m[i-j, j]
    return res


def jpg(m, N):
    centered = m - 128
    dct = DCT(m, N)
    quantize(m, N)
    zigzag = toZigZag(dct, N)
    return ",".join(str(e) for e in zigzag)


def write_compressed(data):
    from subprocess import Popen, PIPE
    output_file_name = 'pipe_out_test.txt.gz'

    gzip_output_file = open(output_file_name, 'wb', 0)

    # If gzip is supported
    output_stream = Popen(["gzip"], stdin=PIPE, stdout=gzip_output_file)

    for line in data:
        output_stream.stdin.write(str.encode(line + '\n'))

    output_stream.stdin.close()
    output_stream.wait()

    gzip_output_file.close()


f = "Kodak08gray.bmp"
img = Image.open(f)
img = img.convert("L")
m = np.asarray(img, dtype=np.int16)
N = 8

(h, w) = N * ceil(m.shape[0] / N), N * ceil(m.shape[1] / N)

padding = np.zeros((h, w))
padding[:m.shape[0], :m.shape[1]] = m
m = padding

res = []
for i in range(0, h, N):
    for j in range(0, w, N):
        sub = m[i:i+N, j:j+N]
        res.append(jpg(sub, N))

write_compressed(res)
