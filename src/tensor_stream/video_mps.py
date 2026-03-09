"""Helpers for webcam/video piecewise MPS assembly workflows."""

from __future__ import annotations

import numpy as np
import quimb.tensor as qtn

from .indexing import bit_flip
from .polynomial_tt import generate_mps


def preprocess_frames(frames_bgr: list[np.ndarray], target_size: int) -> np.ndarray:
    """Convert BGR frames to normalized grayscale tensors of shape (T, H, W)."""
    import cv2

    out = []
    for frame in frames_bgr:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
        out.append(gray.astype(np.float32) / 255.0)
    return np.stack(out, axis=0)


def bits_from_ij(i: int, j: int, k: int) -> list[int]:
    """Return interleaved bits [x0,y0,x1,y1,...] with MSB-to-LSB ordering."""
    xb = [(i >> (k - 1 - b)) & 1 for b in range(k)]
    yb = [(j >> (k - 1 - b)) & 1 for b in range(k)]
    bits: list[int] = []
    for b in range(k):
        bits.extend([xb[b], yb[b]])
    return bits


def split_xy_cores_from_generate_mps(mps: qtn.MatrixProductState) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split interleaved x/y core data from a generated MPS."""
    pairs = []
    arr = [core.data for core in mps]
    n = len(arr) // 2
    for idx in range(n - 1, -1, -1):
        pairs.append((arr[2 * idx], arr[2 * idx + 1]))

    x_cores, y_cores = [], []
    for xc, yc in pairs:
        x_cores.append(xc)
        y_cores.append(yc)
    return x_cores, y_cores


def reconstruct_from_xy(xs: list[np.ndarray], ys: list[np.ndarray]) -> list[np.ndarray]:
    """Re-interleave split x and y core lists in original ordering."""
    xs, ys = list(xs)[::-1], list(ys)[::-1]
    out = [None] * (len(xs) + len(ys))
    out[::2], out[1::2] = xs, ys
    return out


def get_2d_mps_from_patch(A: np.ndarray, n: int, k: int, c: float, d: float, bits: list[int]) -> qtn.MatrixProductState:
    """Build and gate an MPS for a single bicubic patch polynomial."""
    x_bits = [bit_flip(x) for x in bits[0::2]]
    y_bits = [bit_flip(y) for y in bits[1::2]]

    mps = generate_mps(A, n, c, d)
    x_cores, y_cores = split_xy_cores_from_generate_mps(mps)

    x_cores[0][:, :, x_bits[0]] = 0.0
    y_cores[0][:, y_bits[0]] = 0.0

    for idx in range(1, k):
        x_cores[idx][:, :, x_bits[idx]] = 0.0
        y_cores[idx][:, :, y_bits[idx]] = 0.0

    return qtn.MatrixProductState(reconstruct_from_xy(x_cores, y_cores))


def add_2d_mps(mps_list: list[qtn.MatrixProductState], cutoff: float = 1e-8) -> qtn.MatrixProductState:
    """Add many MPS together using iterative compressed summation."""
    total = mps_list[0]
    for mps in mps_list[1:]:
        total.add_MPS(mps, inplace=True, compress=True, cutoff=cutoff)
    return total
