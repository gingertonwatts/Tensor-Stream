"""Backward-compatible utility wrappers.

Prefer importing from ``tensor_stream``.
"""

import numpy as np

from tensor_stream.indexing import bit_flip, interleaved_bits_to_indices
from tensor_stream.piecewise import fit_bicubic_to_image, reconstruct_image_from_coefficients
from tensor_stream.video_mps import reconstruct_from_xy


def convert_to_indices(bits):
    return interleaved_bits_to_indices(bits)


def break_up_list(lst):
    n = len(lst) // 2
    return [(lst[2 * i], lst[2 * i + 1]) for i in range(n - 1, -1, -1)]


def generate_bit_strings(n):
    width = len(bin(n)) - 2
    return [bin(i)[2:].zfill(width) for i in range(n + 1)]


def unfold_tensor_from_vector(tensor, k, n):
    return np.reshape(tensor, (2**k, 2 ** (n - k)))


def compute_rank(matrix, epsilon):
    _, s, _ = np.linalg.svd(matrix)
    return int(np.sum(s > epsilon))
