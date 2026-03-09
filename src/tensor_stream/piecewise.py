"""Piecewise bicubic image fitting utilities."""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize


def fit_bicubic_to_image(image: np.ndarray, k: int, show_plot: bool = False) -> tuple[list[list[np.ndarray]], np.ndarray]:
    """Fit bicubic polynomials on a ``2^k x 2^k`` patch grid using 4x4 stencils."""
    if image.ndim == 3:
        image = rgb2gray(image)
    image = np.asarray(image, dtype=float)

    n_blocks = 2**k
    patch_size = 4
    img = resize(image, (n_blocks * patch_size, n_blocks * patch_size), anti_aliasing=True)

    coeffs: list[list[np.ndarray]] = [[None for _ in range(n_blocks)] for _ in range(n_blocks)]
    sigmas: list[np.ndarray] = []

    u = np.linspace(0, 1, patch_size)
    v = np.linspace(0, 1, patch_size)
    uu, vv = np.meshgrid(u, v)
    samples = np.column_stack([uu.ravel(), vv.ravel()])

    phi = np.zeros((16, 16))
    for row, (ux, vy) in enumerate(samples):
        for col in range(16):
            i = col % 4
            j = col // 4
            phi[row, col] = ux**i * vy**j

    for bi in range(n_blocks):
        for bj in range(n_blocks):
            patch = img[bi * 4 : (bi + 1) * 4, bj * 4 : (bj + 1) * 4]
            a = np.linalg.lstsq(phi, patch.ravel(), rcond=None)[0]
            a_mat = a.reshape((4, 4), order="F")
            coeffs[bi][bj] = a_mat
            _, s, _ = np.linalg.svd(a_mat, full_matrices=False)
            sigmas.append(s)

    sigma_arr = np.asarray(sigmas)
    if show_plot:
        import matplotlib.pyplot as plt

        dom = sigma_arr[:, 0].reshape(n_blocks, n_blocks)
        plt.figure(figsize=(8, 6))
        plt.imshow(dom, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Dominant Singular Value")
        plt.title(f"Dominant Singular Values for k={k}")
        plt.xlabel("Patch Index j")
        plt.ylabel("Patch Index i")
        plt.show()

    return coeffs, sigma_arr


def reconstruct_image_from_coefficients(
    coefficients: list[list[np.ndarray]],
    k: int | None = None,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    """Reconstruct image sampled on 4x4 points from bicubic coefficient blocks."""
    n_blocks = len(coefficients)
    if k is not None and 2**k != n_blocks:
        raise ValueError("k does not match coefficient grid size")

    patch_size = 4
    img = np.zeros((n_blocks * patch_size, n_blocks * patch_size), dtype=float)

    u = np.linspace(0, 1, patch_size)
    v = np.linspace(0, 1, patch_size)
    uu, vv = np.meshgrid(u, v)

    mapping: dict[tuple[int, int], np.ndarray] = {}
    for bi in range(n_blocks):
        for bj in range(n_blocks):
            a = coefficients[bi][bj]
            mapping[(bi, bj)] = a
            patch = np.zeros((4, 4), dtype=float)
            for m in range(4):
                for n in range(4):
                    patch += a[m, n] * (uu**m) * (vv**n)
            img[bi * 4 : (bi + 1) * 4, bj * 4 : (bj + 1) * 4] = patch

    return img, mapping
