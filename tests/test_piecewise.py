import numpy as np

from tensor_stream.piecewise import fit_bicubic_to_image, reconstruct_image_from_coefficients


def test_bicubic_fit_reconstructs_grid_samples():
    rng = np.random.default_rng(0)
    image = rng.random((8, 8))
    coeffs, sigmas = fit_bicubic_to_image(image, k=1)
    rec, mapping = reconstruct_image_from_coefficients(coeffs)

    assert rec.shape == (8, 8)
    assert sigmas.shape == (4, 4)
    assert len(mapping) == 4
    assert np.allclose(rec, image, atol=1e-8)
