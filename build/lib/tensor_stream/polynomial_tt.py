"""Constructive TT/MPS utilities for bivariate polynomial models."""

from __future__ import annotations

import numpy as np
from scipy.special import comb
import quimb.tensor as qtn


def monomial_order(p: int) -> list[tuple[int, int]]:
    """Return lexicographic monomial index ordering up to degree ``p-1``."""
    return [(i, j) for i in range(p) for j in range(p)]


def digit_contribution(k: int, bit: int, lo: float, hi: float, n_bits: int) -> float:
    """Grid decomposition term t_k = lo*delta(k,1) + 2^(k-1)*bit*h."""
    h = (hi - lo) / (2**n_bits - 1)
    return (lo if k == 1 else 0.0) + (2 ** (k - 1)) * int(bit) * h


def coordinates_from_bits(x_bits: list[int], y_bits: list[int], lo: float, hi: float) -> tuple[float, float]:
    """Map binary coordinate digits to ``(x, y)`` values on the uniform grid."""
    n = len(x_bits)
    x = sum(digit_contribution(k + 1, x_bits[k], lo, hi, n) for k in range(n))
    y = sum(digit_contribution(k + 1, y_bits[k], lo, hi, n) for k in range(n))
    return x, y


def phi(coeffs: np.ndarray, x: float, y: float, k: int, ell: int) -> float:
    """Auxiliary function φ_{k,ell}(x,y) from the constructive derivation."""
    p = coeffs.shape[0] - 1
    q = coeffs.shape[1] - 1
    value = 0.0
    for n in range(k, p + 1):
        for m in range(ell, q + 1):
            value += coeffs[n, m] * comb(n, k) * comb(m, ell) * x ** (n - k) * y ** (m - ell)
    return float(value)


def evaluate_polynomial(coeffs: np.ndarray, x: float, y: float) -> float:
    """Evaluate ``P(x,y)=sum_{n,m} a[n,m] x^n y^m``."""
    p, q = coeffs.shape
    value = 0.0
    for n in range(p):
        for m in range(q):
            value += coeffs[n, m] * x**n * y**m
    return float(value)


def g_first(coeffs: np.ndarray, tx: float, ty: float) -> np.ndarray:
    p, q = coeffs.shape
    out = [phi(coeffs, tx, ty, k, ell) for k in range(p) for ell in range(q)]
    return np.asarray(out)


def g_middle(p: int, q: int, tx: float, ty: float) -> np.ndarray:
    order = [(i, j) for i in range(p) for j in range(q)]
    rows = []
    for k, ell in order:
        row = []
        for i, j in order:
            if i <= k and j <= ell:
                row.append(comb(k, k - i) * comb(ell, ell - j) * tx ** (k - i) * ty ** (ell - j))
            else:
                row.append(0.0)
        rows.append(row)
    return np.asarray(rows)


def g_last(p: int, q: int, tx: float, ty: float) -> np.ndarray:
    return np.asarray([tx**k * ty**ell for k in range(p) for ell in range(q)])


def mps_eval(coeffs: np.ndarray, x_bits: list[int], y_bits: list[int], lo: float, hi: float) -> float:
    """Evaluate polynomial via constructive TT contraction on bit digits."""
    p, q = coeffs.shape
    n = len(x_bits)
    prod = g_last(p, q, digit_contribution(n, x_bits[-1], lo, hi, n), digit_contribution(n, y_bits[-1], lo, hi, n))
    for k in range(n - 1, 1, -1):
        tx = digit_contribution(k, x_bits[k - 1], lo, hi, n)
        ty = digit_contribution(k, y_bits[k - 1], lo, hi, n)
        prod = g_middle(p, q, tx, ty) @ prod
    t1x = digit_contribution(1, x_bits[0], lo, hi, n)
    t1y = digit_contribution(1, y_bits[0], lo, hi, n)
    return float(g_first(coeffs, t1x, t1y) @ prod)


def generate_mps(coeffs: np.ndarray, n_bits: int, lo: float, hi: float) -> qtn.MatrixProductState:
    """Build a quimb MPS matching interleaved x/y bit ordering."""
    p, q = coeffs.shape
    rank = p * q
    tensors = []

    first = np.zeros((2, 2, rank))
    for iy in range(2):
        for ix in range(2):
            first[ix, iy, :] = g_first(coeffs, digit_contribution(1, ix, lo, hi, n_bits), digit_contribution(1, iy, lo, hi, n_bits))
    tensors.append(qtn.Tensor(first, inds=("s0", "s1", "i0"), tags=["T0"]))

    spin_idx, bond_idx = 2, 0
    for k in range(2, n_bits):
        core = np.zeros((rank, 2, 2, rank))
        for iy in range(2):
            for ix in range(2):
                core[:, ix, iy, :] = g_middle(p, q, digit_contribution(k, ix, lo, hi, n_bits), digit_contribution(k, iy, lo, hi, n_bits))
        tensors.append(
            qtn.Tensor(
                core,
                inds=(f"i{bond_idx}", f"s{spin_idx}", f"s{spin_idx+1}", f"i{bond_idx+1}"),
                tags=[f"T{k-1}"],
            )
        )
        spin_idx += 2
        bond_idx += 1

    last = np.zeros((rank, 2, 2))
    for iy in range(2):
        for ix in range(2):
            last[:, ix, iy] = g_last(p, q, digit_contribution(n_bits, ix, lo, hi, n_bits), digit_contribution(n_bits, iy, lo, hi, n_bits))
    tensors.append(qtn.Tensor(last, inds=(f"i{bond_idx}", f"s{spin_idx}", f"s{spin_idx+1}"), tags=[f"T{n_bits-1}"]))

    tn = qtn.TensorNetwork(tensors)
    tn.split_tensor(tags="T0", left_inds=["s0"])
    spin_idx = 2
    for i in range(1, n_bits - 1):
        tn.split_tensor(tags=f"T{i}", left_inds=[f"i{i-1}", f"s{spin_idx}"])
        spin_idx += 2
    tn.split_tensor(tags=f"T{n_bits-1}", left_inds=[f"i{n_bits-2}", f"s{2*(n_bits-1)}"])

    cores = []
    first_core = True
    for tensor in tn:
        shp = tensor.shape
        if len(shp) == 2 and first_core:
            cores.append(np.transpose(tensor.data, (1, 0)))
            first_core = False
        elif len(shp) == 3:
            cores.append(np.transpose(tensor.data, (0, 2, 1)))
        else:
            cores.append(tensor.data)
    return qtn.MatrixProductState(cores)
