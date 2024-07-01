from string import ascii_lowercase
from functools import reduce
from operator import add
import numpy as np

__all__ = ["nkron"]


def nkron(*mats):
    """Computes chain Kronecker product of n matrices.

    The arguments must be provided in the reversed order. For example, to compute
    the Kronecker product C ⨂ (B ⨂ A), call `nkron(A, B, C)'.

    Parameters
    ----------
    mats: numpy.ndarrays
        Input matrices in the reversed order.

    Returns
    -------
    2D numpy.ndarray of a shape (m1 * m2 * ... * mN) x (n1 * n2 * ... * nN)
    where mi and ni are a number of rows and a number of columns in i-th input
    array, respectively.
    """
    nrow, ncol = np.prod([mat.shape[0] for mat in mats]), np.prod(
        [mat.shape[1] for mat in mats]
    )
    ndim = len(mats)

    mats = mats[::-1]

    # Construct subscripts string for einsum, i.e. "ab,cd->acbd" for 2D grid,
    # "ab,cd,ef->acegbdfh" - for 3D grid etc.
    subscripts = (
        ",".join(ascii_lowercase[i : i + 2] for i in range(0, ndim * 2, 2)) + "->"
    )

    ranges = [ascii_lowercase[i : i + 2] for i in range(0, ndim * 2, 2)]
    for ijk in zip(*ranges):
        subscripts = reduce(add, ijk, subscripts)

    return np.einsum(subscripts, *mats).reshape(nrow, ncol)
