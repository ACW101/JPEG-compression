from PIL import Image
import numpy as np
from math import cos, sqrt, pi, radians, ceil
from bitarray import bitarray
from struct import pack
import sys

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
            dct[i][j] = temp
    return dct


def DCT_Matrix(m, N):
   # pre-calculate cosine terms
    C = np.ones((8, 8), dtype=np.float32)
    for j in range(N):
        C[0][j] = 1 / sqrt(N)
    for i in range(1, N):
        for j in range(N):
            C[i][j] = sqrt(2 / N) * cos((2 * j + 1) * i * pi / (2 * N))
    DCT = np.matmul(np.matmul(C, m), np.transpose(C))
    return DCT


def quantize(m, N):
    quantized = np.zeros((8, 8), dtype=np.int16)
    if N == 8:
        for i in range(N):
            for j in range(N):
                quantized[i][j] = np.rint(m[i][j] / LQM[i][j])
    if N == 16:
        for i in range(N):
            for j in range(N):
                quantized[i][j] = np.rint(m[i][j] / LQM[i/2][j/2])
    return quantized


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


huffman_table = {0: "0000", 1: "0001", 2: "0010", 3: "0011", 4: "0100", 5: "0101", 6: "0110", 7: "0111",
                 8: "1000", 9: "1001", 10: "1010", 11: "1011", 12: "1100", 13: "1101", 14: "1110", 15: "1111"}
rev_huffman_table = {value: key for (key, value) in huffman_table.items()}


def get_bit_len(num):
    if num == 0:
        return 0
    powers_of_two = 2
    bit_count = 1
    while abs(num) >= powers_of_two:
        powers_of_two *= 2
        bit_count += 1
    return bit_count


def to_twos_complement(num, bit_len):
    if bit_len == 0:
        return ""
    bit_format = "0%ib" % bit_len
    adjusted = num - pow(2, bit_len - 1) if num > 0 else num + \
        pow(2, bit_len - 1) - 1
    return format(adjusted, bit_format) if adjusted >= 0 else format((1 << bit_len) + adjusted, bit_format)


def DC_to_binary(DC):
    dc_bits = bitarray(endian="little")
    for dc in DC:
        bit_len = get_bit_len(dc)
        len_binary = huffman_table[bit_len]
        amplitude_binary = to_twos_complement(dc, bit_len)
        dc_bits += bitarray(len_binary + amplitude_binary, endian="little")
    dc_bits += bitarray("0000", endian="little")
    return dc_bits


def AC_to_binary(AC):
    ac_bits = bitarray(endian="little")
    for ac in AC:
        run_len = to_run_len(ac)
        for rl in run_len:
            zeros_binary = huffman_table[rl[0]]
            bit_len = get_bit_len(rl[1])
            len_binary = huffman_table[bit_len]
            amplitude_binary = to_twos_complement(rl[1], bit_len)
            ac_bits += bitarray(zeros_binary + len_binary +
                                amplitude_binary, endian="little")
        ac_bits += bitarray("00000000", endian="little")
    return ac_bits


def to_run_len(ac):
    run_len = []
    zeros = 0
    i = 0
    while i < len(ac):
        if ac[i] == 0:
            zeros += 1
            if zeros == 16:
                run_len.append((15, 0))
                zeros = 0
        else:
            run_len.append((zeros, ac[i]))
            zeros = 0
        i += 1
    return run_len


def encode(zigzags):
    DC = [zigzags[i][0] for i in range(len(zigzags))]
    DC_bits = DC_to_binary(DC)
    AC = [zigzags[i][1:] for i in range(len(zigzags))]
    AC_bits = AC_to_binary(AC)
    return DC_bits + AC_bits


def jpg(m, N):
    centered = m - 128
    dct = DCT_Matrix(centered, N)
    quantized = quantize(dct, N)
    return toZigZag(quantized, N)

    # from subprocess import Popen, PIPE
    # gzip_output_file = open(output_file_name, 'wb', 0)

    # If gzip is supported
    # output_stream = Popen(["compress"], stdin=PIPE, stdout=gzip_output_file)

    # for line in data:
    #     output_stream.stdin.write(str.encode(line + '\n'))

    # output_stream.stdin.close()
    # output_stream.wait()
    # gzip_output_file.close()


def compress(f, N=8, to_stdout=False, output_file_name="compressed.bin"):
    img = Image.open(f)
    img = img.convert("L")
    m = np.asarray(img, dtype=np.int16)
    (h, w) = N * ceil(m.shape[0] / N), N * ceil(m.shape[1] / N)

    padding = np.zeros((h, w))
    padding[:m.shape[0], :m.shape[1]] = m
    m = padding

    res = []
    for i in range(0, h, N):
        for j in range(0, w, N):
            sub = m[i:i+N, j:j+N]
            res.append(jpg(sub, N))
    dimension_bytes = pack("II", h, w)
    bits = encode(res)
    if to_stdout:
        sys.stdout.buffer.write(dimension_bytes)
        sys.stdout.buffer.write(bits.tobytes())
    else:
        with open(output_file_name, "wb") as f:
            f.write(dimension_bytes)
            f.write(bits.tobytes())


if __name__ == "__main__":
    compress(sys.stdin.buffer, 8, to_stdout=True)
