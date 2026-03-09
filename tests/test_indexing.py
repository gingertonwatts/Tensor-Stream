from tensor_stream.indexing import (
    bit_flip,
    interleave_bits,
    interleaved_bits_to_indices,
    int_to_bits,
    morton_code,
)


def test_interleave_and_inverse_indices():
    x_bits = [1, 0, 1]
    y_bits = [0, 1, 1]
    bits = interleave_bits(x_bits, y_bits)
    i, j = interleaved_bits_to_indices(bits)
    assert i == 0b101
    assert j == 0b011


def test_morton_code_small_case():
    assert morton_code(1, 2, 3) == 0b1001


def test_bit_flip_and_bit_creation():
    assert bit_flip(0) == 1
    assert bit_flip(1) == 0
    assert int_to_bits(5, 3) == [1, 0, 1]
