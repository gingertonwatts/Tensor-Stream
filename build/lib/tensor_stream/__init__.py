"""Tensor Stream: tensor-network encodings for images and videos."""

from .indexing import (
    bit_flip,
    bits_to_int,
    interleave_bits,
    interleaved_bits_to_indices,
    int_to_bits,
    morton_code,
    split_interleaved_bits,
)
from .piecewise import fit_bicubic_to_image, reconstruct_image_from_coefficients
from .polynomial_tt import (
    coordinates_from_bits,
    evaluate_polynomial,
    generate_mps,
    mps_eval,
)

__all__ = [
    "bit_flip",
    "bits_to_int",
    "coordinates_from_bits",
    "evaluate_polynomial",
    "fit_bicubic_to_image",
    "generate_mps",
    "interleave_bits",
    "interleaved_bits_to_indices",
    "int_to_bits",
    "morton_code",
    "mps_eval",
    "reconstruct_image_from_coefficients",
    "split_interleaved_bits",
]
