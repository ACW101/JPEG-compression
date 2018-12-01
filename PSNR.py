from PIL import Image
import numpy as np
from math import cos, sqrt, pi, radians, ceil


def PSNR(original, noisy):
    (h, w) = original.shape[0], original.shape[1]
    diff = np.subtract(original, noisy[0: h, 0: w])
    print(diff)
    squared = np.square(diff, dtype=np.int32)
    print(squared)
    MSE = np.sum(np.sum(squared, axis=1), axis=0) / (h * w)
    PSNR = 20 * log(255 / sqrt(MSE), 10)
    return PSNR
