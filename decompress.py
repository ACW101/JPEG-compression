from PIL import Image
from subprocess import Popen, PIPE
import numpy as np
from math import cos, sqrt, pi, radians, ceil
from struct import unpack
from bitarray import bitarray
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

huffman_table = {0: "0000", 1: "0001", 2: "0010", 3: "0011", 4: "0100", 5: "0101", 6: "0110", 7: "0111",
                 8: "1000", 9: "1001", 10: "1010", 11: "1011", 12: "1100", 13: "1101", 14: "1110", 15: "1111"}
rev_huffman_table = {value: key for (key, value) in huffman_table.items()}


def fromZigZag(zig_zag_in, N):
    m = np.zeros((8, 8), dtype=np.float32)
    # res = [0] * (N * N)
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

    m = np.zeros((8, 8), dtype=np.float32)
    for x in range(N):
        for y in range(N):
            temp = 0.0
            for i in range(N):
                for j in range(N):
                    temp += cosines[x][i] * cosines[y][j] * \
                        DCT[i][j] * coefficients[i][j]
            m[x][y] = np.rint(temp / sqrt(2 * N))
    return m


def iDCT_Matrix(DCT, N):
   # pre-calculate cosine terms
    C = np.ones((8, 8), dtype=np.float32)
    for j in range(N):
        C[0][j] = 1 / sqrt(N)
    for i in range(1, N):
        for j in range(N):
            C[i][j] = sqrt(2 / N) * cos((2 * j + 1) * i * pi / (2 * N))
    m = np.matmul(np.matmul(np.transpose(C), DCT), C)
    return m


def restore_jpg(data, N):
    m = fromZigZag(data, N)
    dequantized = dequantize(m, N)
    centered = iDCT_Matrix(dequantized, N)
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


def from_twos_complement(b, bit_len):
    if (b & (1 << (bit_len - 1))) != 0:
        return b - (1 << bit_len)
    return b


def binary_to_num(binary, bit_len):
    adjusted = from_twos_complement(int(binary.to01(), 2), bit_len)
    unadjusted = adjusted + pow(2, bit_len - 1) \
        if adjusted >= 0 \
        else adjusted - pow(2, bit_len - 1) + 1
    return unadjusted


def from_binary_to_DC(bits, offset, zigzags, blocks):
    cur_bits = bits[offset: offset + 4].to01()
    i = 0
    while i < blocks:
        bit_len = rev_huffman_table[cur_bits]
        if bit_len > 0:
            amp_binary = bits[offset + 4: offset + 4 + bit_len]
            dc = binary_to_num(bitarray(amp_binary, endian="little"), bit_len)
            zigzags[i][0] = dc
        offset += (4 + bit_len)
        cur_bits = bits[offset: offset + 4].to01()
        i += 1
    return offset + 4


def rev_run_length(rl, N):
    res = []
    for (zeros, val) in rl:
        if zeros == 15 and val == 0:
            res.extend([0] * 16)
        elif zeros == 0 and val == 0:
            res.extend([0] * (N * N - len(res)))
        else:
            res.extend([0] * zeros)
            res.append(val)


def from_binary_to_AC(bits, N, offset, zigzags, blocks):
    i = 0
    zeros_bits = bits[offset: offset + 4].to01()
    bit_len_bits = bits[offset + 4: offset + 8].to01()
    while i < blocks:
        j = 1
        while zeros_bits != "0000" or bit_len_bits != "0000":
            if zeros_bits == "1111" and bit_len_bits == "0000":
                j += 16
                offset += 8
                zeros_bits = bits[offset: offset + 4].to01()
                bit_len_bits = bits[offset + 4: offset + 8].to01()
                continue
            # handle preceding zeros
            prec_zeros = rev_huffman_table[zeros_bits]
            while prec_zeros > 0:
                zigzags[i][j] = 0
                j += 1
                prec_zeros -= 1

            # handle non-zero ac value
            bit_len = rev_huffman_table[bit_len_bits]
            amp_bits = bits[offset + 8: offset + 8 + bit_len]
            ac = binary_to_num(bitarray(amp_bits, endian="little"), bit_len)
            zigzags[i][j] = ac
            offset += (8 + bit_len)
            zeros_bits = bits[offset: offset + 4].to01()
            bit_len_bits = bits[offset + 4: offset + 8].to01()
            j += 1
        offset += 8
        zeros_bits = bits[offset: offset + 4].to01()
        bit_len_bits = bits[offset + 4: offset + 8].to01()
        i += 1


def decode(input, N):
    b = bitarray(endian="little")
    dimension = unpack("II", input.read(8))
    b.frombytes(input.read())
    (h, w) = dimension
    zigzags = np.zeros((h * w, N * N), dtype=np.float32)
    offset = 0
    blocks = h * w / N / N
    offset = from_binary_to_DC(b, offset, zigzags, blocks)
    from_binary_to_AC(b, 8, offset, zigzags, blocks)
    data_iterator = iter(zigzags)
    h_elems = []
    for _ in range(h // 8):
        w_elems = []
        for _ in range(w // 8):
            w_elems.append(restore_jpg(next(data_iterator), N))
        h_elems.append(np.concatenate(w_elems, axis=1))
    restored_matrix = np.concatenate(h_elems, axis=0)
    img = Image.fromarray(restored_matrix, "L")
    img.save(sys.stdout.buffer, "bmp")


if __name__ == "__main__":
    decode(sys.stdin.buffer, 8)
