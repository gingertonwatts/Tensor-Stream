from generate_tt import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from skimage import data as img_data
from skimage.color import rgb2gray
from skimage.transform import resize
from itertools import product


def reconstruct_from_xy(xs, ys):
    # Reverse both lists to restore original order
    xs.reverse()
    ys.reverse()

    # Interleave the elements from xs and ys
    original_list = [None] * (len(xs) + len(ys))
    original_list[::2] = xs
    original_list[1::2] = ys

    return original_list


def f(a, x, y):
    p = np.shape(a)[0] - 1

    poly_val = 0.
    for n in range(p + 1):
        for m in range(p + 1):
            poly_val += a[n][m] * x ** n * y ** m

    return poly_val


"""
   TT Encoding
"""


def generate_bit_strings(n):
    """
    Generates all bit strings from 0 to n (inclusive).

    :param n: An integer up to which bit strings are to be generated.
    :return: A list of bit strings.
    """
    # Calculate the number of bits needed to represent 'n' in binary
    num_bits = len(bin(n)) - 2

    # Generate bit strings
    bit_strings = [bin(i)[2:].zfill(num_bits) for i in range(n + 1)]

    return bit_strings


def bit_string_to_xy(arr, bit_string, c, d):
    i = int(bit_string, 2)
    i_kx = [int(bit) for bit in bit_string[0::2]]  # Take every alternate bit starting from the first
    i_ky = [int(bit) for bit in bit_string[1::2]]  # Take every alternate bit starting from the second

    x, y = xy(i_kx, i_ky, c, d)

    return arr[i][0], x, y


# Function to compute the unfolding matrix for a given k
def unfold_tensor_from_vector(tensor, k, n):
    # Reshape tensor to a 2^k x 2^(n-k) matrix
    unfolding_matrix = np.reshape(tensor, (2 ** k, 2 ** (n - k)))
    return unfolding_matrix


# Function to compute the rank with tolerance epsilon
def compute_rank(matrix, epsilon):
    # Perform singular value decomposition (SVD)
    _, s, _ = np.linalg.svd(matrix)
    # Count singular values larger than epsilon
    rank = np.sum(s > epsilon)
    return rank


def generate_outside_bitstrings(n, k, target_i, target_j):
    outside_bitstrings = []
    domain_size = 2 ** (n - k)  # Approximate domain size for each x and y partition

    # Loop over all possible i and j indices for each domain
    for i in range(2 ** k):
        for j in range(2 ** k):
            # Skip if this is the target domain D_ij
            if i == target_i and j == target_j:
                continue

            # Calculate bit string ranges for x and y in this domain
            x_range_start, x_range_end = i * domain_size, (i + 1) * domain_size
            y_range_start, y_range_end = j * domain_size, (j + 1) * domain_size

            # Generate bit strings for all (x, y) pairs in the current domain
            for x in range(x_range_start, x_range_end):
                for y in range(y_range_start, y_range_end):
                    # Convert x and y to bit strings of length n
                    x_bitstring = f"{x:0{n}b}"
                    y_bitstring = f"{y:0{n}b}"
                    outside_bitstrings.append((x_bitstring, y_bitstring))

    return outside_bitstrings



# Assuming 'lst' is your list of length 2*n
def break_up_list(lst):
    n = len(lst) // 2
    pairs = []

    for i in range(n - 1, -1, -1):
        pairs.append((lst[2 * i], lst[2 * i + 1]))

    return pairs



def fit_bicubic_to_image(image, k, show_plot=True):
    """
    Fits bicubic polynomials to patches of an image and performs SVD on the coefficient matrices.
    """
    # Ensure the image is a numpy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Convert image to grayscale if it's not already
    if image.ndim == 3:
        image = rgb2gray(image)
    
    
    # Resize image to a square for simplicity
    N = 2 ** k
    image_resized = resize(image, (N * 4, N * 4), anti_aliasing=True)

    # Initialize list to store coefficient matrices and singular values
    coefficients = [[None for _ in range(N)] for _ in range(N)]
    singular_values_list = []

    # Patch size
    patch_size = 4

    # Loop over patches
    for i in range(N):
        for j in range(N):
            # Extract patch
            x_start = i * patch_size
            x_end = x_start + patch_size
            y_start = j * patch_size
            y_end = y_start + patch_size
            patch = image_resized[x_start:x_end, y_start:y_end]

            # Coordinates in local [0,1] domain
            u = np.linspace(0, 1, patch_size)
            v = np.linspace(0, 1, patch_size)
            u_grid, v_grid = np.meshgrid(u, v)
            u_flat = u_grid.flatten()
            v_flat = v_grid.flatten()
            z_flat = patch.flatten()

            # Construct the Phi matrix
            Phi = np.zeros((16, 16))
            for n in range(16):
                for m in range(16):
                    i_m = m % 4
                    j_m = m // 4
                    Phi[n, m] = u_flat[n] ** i_m * v_flat[n] ** j_m

            # Solve for the coefficients
            a = np.linalg.lstsq(Phi, z_flat, rcond=None)[0]

            # Reshape coefficients into a 4x4 matrix
            A = a.reshape((4, 4), order='F')
            coefficients[i][j] = A

            # Perform SVD on A
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            singular_values_list.append(S)

            # Optional: Replace A with its rank-1 approximation
            # sigma1 = S[0]
            # u1 = U[:, 0]
            # v1 = Vt[0, :]
            # A_rank1 = sigma1 * np.outer(u1, v1)
            # coefficients[i][j] = A_rank1

    # Prepare data for plotting singular values
    singular_values_array = np.array(singular_values_list)
    dominant_singular_values = singular_values_array[:, 0]
    dominant_singular_values = dominant_singular_values.reshape(N, N)

    if show_plot:
        # Plot the dominant singular values as an image
        plt.figure(figsize=(8, 6))
        plt.imshow(dominant_singular_values, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Dominant Singular Value')
        plt.title(f'Dominant Singular Values for k={k}')
        plt.xlabel('Patch Index j')
        plt.ylabel('Patch Index i')
        plt.show()

    return coefficients, singular_values_array

def reconstruct_image_from_coefficients(coefficients, k):
    """
    Reconstructs the image from the fitted bicubic polynomials.
    """
    N = 2 ** k
    patch_size = 4
    image_size = N * patch_size
    reconstructed_image = np.zeros((image_size, image_size))

    u = np.linspace(0, 1, patch_size)
    v = np.linspace(0, 1, patch_size)
    u_grid, v_grid = np.meshgrid(u, v)
    
    As = {}
    for i in range(N):
        for j in range(N):
            A = coefficients[i][j]

            As[(i,j)] = A
            patch = np.zeros((patch_size, patch_size))
            for m in range(4):
                for n in range(4):
                    patch += A[m, n] * (u_grid ** m) * (v_grid ** n)
            x_start = i * patch_size
            x_end = x_start + patch_size
            y_start = j * patch_size
            y_end = y_start + patch_size
            reconstructed_image[x_start:x_end, y_start:y_end] = patch

    return reconstructed_image, As

def convert_to_indices(bits):
    """
    Converts a list of bits into two indices, `i` and `j`, 
    based on alternating bit positions for an arbitrary number of bits.
    
    Args:
        bits (list[int]): A list of bits (0 or 1) of even length.
    
    Returns:
        tuple: Two integers `i` and `j` representing the binary indices.
    """
    # Ensure the number of bits is even
    if len(bits) % 2 != 0:
        raise ValueError("The number of bits must be even.")
    
    # Split bits into x_bits and y_bits
    x_bits = bits[0::2]  # Extract bits at even indices
    y_bits = bits[1::2]  # Extract bits at odd indices
    
    # Convert bit lists to integers
    i = sum(bit * (2 ** idx) for idx, bit in enumerate(reversed(x_bits)))
    j = sum(bit * (2 ** idx) for idx, bit in enumerate(reversed(y_bits)))
    
    return i, j


def bit_flip(x):
    return abs(1-x)
