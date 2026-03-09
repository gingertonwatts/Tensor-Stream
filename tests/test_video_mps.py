from tensor_stream.video_mps import bits_from_ij


def test_bits_from_ij_interleaving():
    bits = bits_from_ij(0b101, 0b011, 3)
    assert bits == [1, 0, 0, 1, 1, 1]
