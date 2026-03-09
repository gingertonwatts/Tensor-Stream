"""Index and bit-interleaving utilities for multidimensional tensor encodings."""

from __future__ import annotations

from typing import Iterable, Sequence


def interleave_bits(x_bits: Sequence[int], y_bits: Sequence[int]) -> list[int]:
    """Interleave equal-length bit sequences as ``[x0, y0, x1, y1, ...]``."""
    if len(x_bits) != len(y_bits):
        raise ValueError("x_bits and y_bits must have the same length.")

    out: list[int] = []
    for xb, yb in zip(x_bits, y_bits):
        out.extend((int(xb), int(yb)))
    return out


def split_interleaved_bits(bits: Sequence[int]) -> tuple[list[int], list[int]]:
    """Split ``[x0,y0,x1,y1,...]`` into ``(x_bits, y_bits)``."""
    if len(bits) % 2 != 0:
        raise ValueError("Interleaved bit array must have even length.")
    return list(bits[0::2]), list(bits[1::2])


def bits_to_int(bits: Iterable[int]) -> int:
    """Convert an MSB-first bit iterable to an integer."""
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def int_to_bits(value: int, width: int) -> list[int]:
    """Convert integer to a fixed-width MSB-first bit list."""
    if value < 0:
        raise ValueError("value must be non-negative")
    if width < 1:
        raise ValueError("width must be >= 1")
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


def interleaved_bits_to_indices(bits: Sequence[int]) -> tuple[int, int]:
    """Convert interleaved bit string into integer coordinates ``(i, j)``."""
    x_bits, y_bits = split_interleaved_bits(bits)
    return bits_to_int(x_bits), bits_to_int(y_bits)


def morton_code(i: int, j: int, d: int) -> int:
    """Compute 2D Morton / Z-order index with ``d`` bits per coordinate."""
    if d < 1:
        raise ValueError("d must be >= 1")
    xb = int_to_bits(i, d)[::-1]
    yb = int_to_bits(j, d)[::-1]
    code = 0
    for layer, (xk, yk) in enumerate(zip(xb, yb)):
        code |= (xk << (2 * layer))
        code |= (yk << (2 * layer + 1))
    return code


def bit_flip(bit: int) -> int:
    """Flip a binary value 0 <-> 1."""
    b = int(bit)
    if b not in (0, 1):
        raise ValueError("bit must be 0 or 1")
    return 1 - b
