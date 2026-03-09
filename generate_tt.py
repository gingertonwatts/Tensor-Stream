"""Backward-compatible wrappers for legacy imports.

Prefer importing from ``tensor_stream.polynomial_tt``.
"""

from tensor_stream.polynomial_tt import (  # noqa: F401
    coordinates_from_bits as xy,
    digit_contribution as t,
    evaluate_polynomial,
    generate_mps,
    g_first as G_first,
    g_last as G_last,
    g_middle as G,
    monomial_order,
    mps_eval as MPS_eval,
    phi,
)
