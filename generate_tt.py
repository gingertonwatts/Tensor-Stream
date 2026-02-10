import numpy as np
from scipy.special import comb
import quimb.tensor as qtn


def monomial_order(p):
    order = []
    for i in range(p):
        for j in range(p):
            order.append((i, j))

    return order

def t(k, i_k, c, d, n):
    """
    k = 1, ..., n
    i_k = 0, 1
    [c,d] discretized into 2**n points
    t_k = aδ1k + 2^k-1 i_k h
    """
    t_k = 0
    if k == 1:
        t_k += c
    h = (d - c) / (2 ** n - 1)  # changed h here
    t_k += 2 ** (k - 1) * i_k * h
    return t_k


def xy(i_kx, i_ky, c, d):
    n = len(i_kx)
    x, y = 0., 0.

    for k in range(1, n + 1):
        x += t(k, i_kx[k - 1], c, d, n)
        y += t(k, i_ky[k - 1], c, d, n)

    return x, y


def phi(a, x, y, k, l):
    p = np.shape(a)[0] - 1

    phi_val = 0.
    for n in range(k, p + 1):
        for m in range(l, p + 1):
            ckn, clm = comb(n, k), comb(m, l)
            phi_val += a[n][m] * ckn * clm * x ** (n - k) * y ** (m - l)

    return phi_val


def G_first(a, tx, ty):
    p = np.shape(a)[0] - 1
    order = monomial_order(p + 1)

    arr = []
    for k, l in order:
        phi_val = phi(a, tx, ty, k, l)
        arr.append(phi_val)

    return np.array(arr)


def G(p, tx, ty):
    order = monomial_order(p + 1)
    arr = []
    for k, l in order:
        row = []
        for i, j in order:
            if i <= k and j <= l:
                cki, clj = comb(k, k - i), comb(l, l - j)
                row.append(cki * clj * tx ** (k - i) * ty ** (l - j))
            else:
                row.append(0.0)
        arr.append(row)

    return np.array(arr)


def G_last(p, tx, ty):
    order = monomial_order(p + 1)
    arr = []
    for k, l in order:
        arr.append(tx ** k * ty ** l)

    return np.array(arr)


def MPS_eval(a, i_kx, i_ky, c, d):
    p = np.shape(a)[0] - 1
    n = len(i_kx)

    tnx = t(n, i_kx[-1], c, d, n)
    tny = t(n, i_ky[-1], c, d, n)

    prod = G_last(p, tnx, tny)

    for k in range(n - 1, 1, -1):
        tkx = t(k, i_kx[k - 1], c, d, n)
        tky = t(k, i_ky[k - 1], c, d, n)
        prod = G(p, tkx, tky) @ prod

    t1x = t(1, i_kx[0], c, d, n)
    t1y = t(1, i_ky[0], c, d, n)

    prod = np.dot(G_first(a, t1x, t1y), prod)

    return prod


def generate_tt(a, n, c, d):
    """
     for spin index use sj
     for array index use ij
     for tags use Tj

    """
    p = np.shape(a)[0] - 1

    tensors = []

    core1 = np.zeros((2, 2, (p + 1) ** 2))

    for iy in range(2):
        for ix in range(2):
            t1x = t(1, ix, c, d, n)
            t1y = t(1, iy, c, d, n)

            core1[ix, iy, :] = G_first(a, t1x, t1y)

    tensor1 = qtn.Tensor(core1, inds=('s0', 's1', 'i0'), tags=['T0'])
    tensors.append(tensor1)

    tag_idx = 1
    spin_idx = 2
    array_idx = 0

    for k in range(2, n):

        core2 = np.zeros(((p + 1) ** 2, 2, 2, (p + 1) ** 2))  # need to change this to correct shape for degree p
        for iy in range(2):
            for ix in range(2):
                tkx = t(k, ix, c, d, n)
                tky = t(k, iy, c, d, n)
                core2[:, ix, iy, :] = G(p, tkx, tky)

        tensor2 = qtn.Tensor(core2,
                             inds=('i' + str(array_idx), 's' + str(spin_idx), 's' + str(spin_idx + 1),
                                   'i' + str(array_idx + 1)),
                             tags=['T' + str(tag_idx)])
        tensors.append(tensor2)

        tag_idx += 1
        spin_idx += 2
        array_idx += 1

    core3 = np.zeros(((p + 1) ** 2, 2, 2))

    for iy in range(2):
        for ix in range(2):
            tnx = t(n, ix, c, d, n)
            tny = t(n, iy, c, d, n)

            core3[:, ix, iy] = G_last(p, tnx, tny)

    tensor3 = qtn.Tensor(core3,
                         inds=('i' + str(array_idx), 's' + str(spin_idx), 's' + str(spin_idx + 1)),
                         tags=['T' + str(tag_idx)])
    tensors.append(tensor3)

    TN = qtn.TensorNetwork(tensors)

    return TN


def generate_mps(a, n, c, d):
    tt = generate_tt(a, n, c, d)

    tt.split_tensor(tags='T0', left_inds=['s0'], right_inds=None, rtags=None)

    spin_idx = 2
    for i in range(1, n - 1):
        tt.split_tensor(tags='T' + str(i), left_inds=['i' + str(i - 1), 's' + str(spin_idx)], right_inds=None,
                        rtags=None)
        spin_idx += 2

    tt.split_tensor(tags='T' + str(n - 1), left_inds=['i' + str(n - 2), 's' + str(2 * (n - 1))], right_inds=None,
                    rtags=None)

    cores = []

    first_core = True
    for core in tt:
        core_data = core.data
        core_shape = np.shape(core_data)

        if len(core_shape) == 2 and first_core:
            cores.append(np.transpose(core_data, (1, 0)))
            first_core = False

        elif len(core_shape) == 3:
            cores.append(np.transpose(core_data, (0, 2, 1)))

        elif len(core_shape) == 2 and not first_core:
            cores.append(core_data)

    mps = qtn.MatrixProductState(cores)

    return mps
