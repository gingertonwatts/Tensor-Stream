import numpy as np

from tensor_stream.polynomial_tt import (
    coordinates_from_bits,
    evaluate_polynomial,
    mps_eval,
)


def test_mps_eval_matches_direct_polynomial():
    coeffs = np.array(
        [
            [1.0, 0.5, 0.0],
            [2.0, -1.0, 0.0],
            [0.0, 0.0, 0.25],
        ]
    )
    x_bits = [1, 0, 1]
    y_bits = [0, 1, 1]
    x, y = coordinates_from_bits(x_bits, y_bits, 0.0, 1.0)
    direct = evaluate_polynomial(coeffs, x, y)
    tt_val = mps_eval(coeffs, x_bits, y_bits, 0.0, 1.0)
    assert np.isclose(tt_val, direct, atol=1e-10)
