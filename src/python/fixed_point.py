import numpy as np


def q31_multiplier(input_scale, weight_scale, output_scale):
    """Calculate the signed Q31 requantization multiplier."""
    real_multiplier = input_scale * weight_scale / output_scale
    return int(np.rint(real_multiplier * (2**31)))


def requantize_q31(accumulator, multiplier, minimum=0, maximum=127):
    """Integer Q31 requantization using round-to-nearest, ties-to-even."""
    acc = np.asarray(accumulator, dtype=np.int64)
    product = acc * np.int64(multiplier)

    quotient = product >> 31
    remainder = product & ((1 << 31) - 1)
    half = 1 << 30

    increment = (remainder > half) | (
        (remainder == half) & ((quotient & 1) != 0)
    )

    rounded = quotient + increment.astype(np.int64)
    return np.clip(rounded, minimum, maximum).astype(np.int32)
