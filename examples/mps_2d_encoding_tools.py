"""Backward-compatible utility wrappers.

Prefer importing from ``tensor_stream``.
"""

import numpy as np

from tensor_stream.indexing import interleaved_bits_to_indices


def reconstruct_from_xy(xs, ys):
    xs = list(xs)[::-1]
    ys = list(ys)[::-1]
    out = [None] * (len(xs) + len(ys))
    out[::2], out[1::2] = xs, ys
    return out


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
