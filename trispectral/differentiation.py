"""
Nth-order differential matrices.

Default discretization:
* "chebyshev" for Chebyshev spectral differentiation matrices
* "periodic" for Fourier spectral differentiation matrices
* "uniform" for finite-differences differentiation matrices (NotImplemented)

By design, differentiation matrices should be created by passing the numerical
grid and derivative's parameters to the Python class DifferentiatonMatrix.
Alternatively, differentiation matrices may be created using their constructors.
For example, the constructor for a Fourier differentiation matrix is `four_mat'.
In this case, must pass suitable grid points to the constructor function as a
parameter.

This module contains the following (public) routines:
* cheb_mat(x, order, **kwargs) -> np.ndarray
* bary_cheb_mat(x, order, **kwargs) -> np.ndarray
* four_mat(x, order, **kwargs) -> np.ndarray

And the Python class:
* DifferentiationMatrix

DifferentiationMatrix is the main class for working with differentiation
matrices. Get an instance of a matrix by calling on of the two methods
(see the documentation below for details):
* DifferentiationMatrix(
    grid: Grid,
    axis: int = 0,
    order: int = 1,
    accuracy: Union[None, int] = None,
    parity: Union[None, int] = None,
    symmetry: Union[None, str] = None,
    **kwargs,
)
* DifferentiationMatrix.register(
    grid: Grid,
    axis: int = 0,
    order: int = 1,
    accuracy: Union[None, int] = None,
    parity: Union[None, int] = None,
    symmetry: Union[None, str] = None,
    num: Union[None, int] = None,
    override: bool = False,
    **kwargs,
)
"""

import warnings
from typing import Union
from operator import index
from copy import deepcopy
from numbers import Number
import numpy as np
from scipy.linalg import toeplitz
from .manager import ObjectManager
from .math_utils import nkron
from .grid import cheb_bary_weights, Grid

__all__ = [
    "cheb_mat",
    "bary_cheb_mat",
    "four_mat",
    "DifferentialMatrix",
    "gradient_operator",
    "divergence_operator",
    "scalar_laplacian_operator",
    "vector_laplacian_operator",
    "directional_derivative_operator",
    "curl_operator",
]


def _dmat_manager(_dmat_manager_instance=ObjectManager()):
    """Getter for the DmatManager's one and only instance."""
    return _dmat_manager_instance


def cheb_mat(x, order, **kwargs) -> np.ndarray:
    """Computes 1D Chebyshev differential matrix [1].

    References
    ----------
    [1] L.N. Trefethen, "Spectral Methods in MATLAB". SIAM, Philadelphia, 2000.
    """
    npts = len(x)
    x = x.reshape((npts, 1))
    n = np.arange(npts)

    c = (np.hstack(([2.0], np.ones(npts - 2), [2.0])) * (-1) ** n).reshape(
        npts, 1
    )
    X = np.tile(x, (1, npts))
    dX = X - X.T
    # The np.eye term is to avoid division by zero on the diagonal; the diagonal
    # part of D is properly computed in the subsequent line.
    mat = c * (1.0 / c.T) / (dX + np.eye(npts))
    mat -= np.diag(np.sum(mat, axis=1))

    return np.linalg.matrix_power(mat, order)


def bary_cheb_mat(x, order, **kwargs) -> np.ndarray:
    """Computes Chebyshev differential matrix using method described in [1].

    References
    ----------
    [1] R. Baltensperger and M. A. Trummer, "Spectral Differencing with a Twist",
        SIAM J. Sci. Comput., 2003.
    """
    nx = len(x)
    w = cheb_bary_weights(nx)

    a = np.arange(nx)[::-1] * np.pi / (nx - 1)
    a = np.repeat(a[::-1], nx).reshape(nx, nx)

    args = np.triu(np.ones(nx, dtype=int), k=1).T[:, ::-1]

    dx = 2 * np.sin(a / 2 + a.T / 2) * np.sin(a / 2 - a.T / 2)
    dx[args] = -dx[::-1, ::-1][args]
    np.fill_diagonal(dx, 1)
    dx = 1 / dx

    w = np.repeat(w, nx).reshape(nx, nx)
    dw = w.T / w
    np.fill_diagonal(dw, 0)

    d = dx * dw
    np.fill_diagonal(d, 0)
    np.fill_diagonal(d, -np.sum(d, axis=1))

    k = np.diagonal(d).copy()
    k[: nx - int(nx / 2) - 1 : -1] = -k.copy()[: int(nx / 2)]
    np.fill_diagonal(d, k)

    if order == 2:
        d *= 2 * (np.repeat(k, nx).reshape(nx, nx) - dx)
        np.fill_diagonal(d, 0)
        np.fill_diagonal(d, -np.sum(d, axis=1))

    if order > 2:
        raise NotImplementedError(
            f"Chebyshev matrix of order {order} not implemented"
        )

    return d


def four_mat(x, order, **kwargs) -> np.ndarray:
    """Computes 1D Fourier differential matrix [1].

    References
    ----------
    [1] L.N. Trefethen, "Spectral Methods in MATLAB". SIAM, Philadelphia, 2000.
    """
    nx = len(x)
    dx = 2 * np.pi / nx
    if order == 1:
        if nx % 2 == 0:
            col = np.hstack([0, 0.5 / np.tan(np.arange(1, nx) * 0.5 * dx)])
        else:
            raise NotImplementedError(
                "Fourier matrix not implemented for odd number of points"
            )
        col[1::2] *= -1
        row = col[[0, *np.arange(nx - 1, 0, -1)]]
        return toeplitz(col, row)
    elif order == 2:
        if nx % 2 == 0:
            col = np.hstack(
                [
                    np.pi**2 / 3 / dx**2 + 1 / 6,
                    0.5 / np.sin(np.arange(1, nx) * 0.5 * dx) ** 2,
                ]
            )
        else:
            raise NotImplementedError(
                "Fourier matrix not implemented for odd number of points"
            )
        col[::2] *= -1
        return toeplitz(col)
    else:
        raise NotImplementedError(
            "Fourier matrix not implemented for derivatives of order higher than 2"
        )


_DMAT_CONSTRUCTORS = {
    "chebyshev": cheb_mat,
    "periodic": four_mat,
    "uniform": (
        lambda: (_ for _ in ()).throw(
            NotImplementedError(
                "Differential matrix for uniform grid not implemented"
            )
        )
    ),
}


class DifferentialMatrix(np.ndarray):
    def __new__(
        cls,
        grid: Grid,
        axis: int = 0,
        order: int = 1,
        accuracy: Union[None, int] = None,
        parity: Union[None, int] = None,
        symmetry: Union[None, str] = None,
        **kwargs,
    ):
        mats = [np.eye(grid.npts[axis]) for axis in range(grid.num_dim)]

        if order > 0:  # if 1, identity matrix is returned
            disc = grid.discs[axis]
            # If the user specified the accuracy, pass it as a keyword argument.
            if accuracy is not None:
                kwargs["accuracy"] = accuracy

            s = 1  # symmetry coefficient
            if symmetry is not None:
                if "anti-pole" in symmetry:
                    s = -1

            mats[axis] = _DMAT_CONSTRUCTORS[disc](
                grid.coordinate_array(axis, ignore_geom=True), order, **kwargs
            )

        match grid.geom:
            case "cart":
                if grid.num_dim == 1:
                    obj = mats[0]
                else:
                    obj = nkron(*mats)
            case "radial":
                n = kwargs["azimuthal_wavenumber"]
                nr = int(grid.npts[0] / 2)

                obj = (
                    mats[0][nr:, nr:]
                    + parity * (-1) ** n * mats[0][nr:, nr - 1 :: -1]
                )
            case "polar":
                nphi, nr = grid.npts
                nr = p = int(nr / 2)

                if axis == 0:  # derivative in ϕ
                    # No special care needed for the radial matrix except that only
                    # "positive" quarter is taken into into account.
                    mats[1] = mats[1][p:, p:]

                    if symmetry is not None:
                        if "pole" in symmetry:
                            nphi = m = int(nphi / 2)

                            # Add contribution from the lower half-disk.
                            mats[0][m:, m:] += s * mats[0][m:, :m]
                            mats[0] = mats[0][m:, m:]

                    obj = nkron(*mats)
                elif axis == 1:
                    if parity is None:
                        warnings.warn(
                            "The parity of radial derivatives not specified "
                            "(see documentation) - using default value 1",
                            RuntimeWarning,
                        )
                        parity = 1

                    # pmats contain a slice of a radial differential matrix related to a
                    # "positive" half of an extended radial domain.
                    pmats, nmats = deepcopy(mats), deepcopy(mats)

                    pmats[1] = pmats[1][p:, p:]
                    nmats[1] = nmats[1][p:, p - 1 :: -1]

                    if symmetry is None:
                        nmats[0] = np.roll(nmats[0], int(nphi / 2), axis=1)
                    else:
                        nphi = m = int(nphi / 2)

                        pmats[0] = pmats[0][m:, m:]
                        nmats[0] = s * nmats[0][m:, m:]

                    obj = nkron(*pmats) + parity * nkron(*nmats)

        obj = obj.view(cls)
        obj._num = None  # 'register' differential operator using dmat.
        obj.axis = axis
        obj.order = order

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._num = None
        self.axis = None
        self.order = None

    @property
    def num(self):
        return self._num

    @classmethod
    def register(
        cls,
        grid: Grid,
        axis: int = 0,
        order: int = 1,
        accuracy: Union[None, int] = None,
        parity: Union[None, int] = None,
        symmetry: Union[None, str] = None,
        num: Union[None, int] = None,
        override: bool = False,
        **kwargs,
    ):
        dmat_manager = _dmat_manager()
        if num is None:
            nums = dmat_manager.nums()
            num = max(nums) + 1 if nums else 0
        else:
            try:
                index(num)
            except TypeError as e:
                raise TypeError(
                    "Unique identifier num must be an integer"
                ) from e

        dmat = getattr(dmat_manager, str(num), None)
        if dmat is None or override:
            dmat = cls(grid, axis, order, accuracy, parity, symmetry, **kwargs)
            dmat._num = num
            setattr(dmat_manager, str(num), dmat)

        return dmat

    def unregister(self):
        dmat_manager = _dmat_manager()
        dmat_manager.drop(num=self.num)


def gradient_operator(
    grid: Grid,
    accuracy: Union[None, int] = None,
    symmetry: Union[None, str] = None,
    parities: list = [1, 1, 1],
    wavevector: Union[None, tuple] = None,
    stack: bool = True,
):
    """
    Construct the gradient operator for a given grid.

    Parameters
    ----------
    grid: trispectral.Grid
        Numerical grid.
    accuracy: Union[None, int], default = None
        Accuracy of finite difference method if relevant.
    symmetry: Union[None, str], default = None
        Polar symmetry of the scalar field (only relevant for grid geometries
        'radial' and 'polar').
    parities: list, default = [1, 1, 1]
        Parities of vector components in circular geometry. Since parity of a
        scalar field is 1 (see ...), the default values should always be used
        to construct the gradient operator. However, if the components of the
        nabla operator are seeked instead, different parities can be imposed.
    wavevector: Union[None, tuple], default = None
        Components of 3D wavevector in the case of one or more homogeneous
        directions, such that f  ̴ exp(ikx), where f is an arbitrary grid
        function, k is the wavevector, and x is the position vector. The
        wavevector, is specified, must be passed as an array-like containing
        three elements, corresponding to each dimension. The elements
        corresponding to inhomogeneous dimensions must take form of arrays-like
        with two elements - the index of the respectivegrid axis and None (see
        Examples).
    stack: bool, default = True
        Whether to return the gradient operator as a 3N⨉N array of type
        DifferentialMatrix (if True) or as a list containing three N⨉N arrays
        of type DifferentialMatrix.

    Returns
    -------
    Gradient operator as a 3N⨉N DifferentialMatrix (is stack=True) or a list
    containing three N⨉N DifferentialMatrix (if stack=False).

    Examples
    --------
    """
    if wavevector is None:
        mats = [
            DifferentialMatrix(
                grid,
                axis=axis,
                order=1,
                accuracy=accuracy,
                parity=parity,
                symmetry=symmetry,
            )
            for axis, parity in zip(range(grid.num_dim), parities)
        ]
    else:
        wavevector = [
            (None, element) if isinstance(element, Number)
            else element for element in wavevector
        ]

        mats = []
        for (axis, k), parity in zip(wavevector, parities):
            if axis is not None:
                if axis >= grid.num_dim:
                    raise ValueError(
                        f"Axis {axis} is out of bounds for grid "
                        f"with {grid.num_dim} dimension(s)"
                    )
                if k is not None:
                    raise ValueError(
                        f"Could not impose periodic direction for axis {axis} "
                        f" and grid with {grid.num_dim} dimension(s)"
                    )
                mat = DifferentialMatrix(
                    grid,
                    axis=axis,
                    order=1,
                    accuracy=accuracy,
                    parity=parity,
                    symmetry=symmetry,
                )
            else:
                mat = (
                    DifferentialMatrix(  # to make sure, symmetry is respected
                        grid,
                        order=0,
                        accuracy=accuracy,
                        parity=parity,
                        symmetry=symmetry,
                    )
                    * 1j
                    * k
                )

            mats.append(mat)

    if "radial" in grid.geom or "polar" in grid.geom:
        ri = 1 / (grid[grid.num_dim - 1][grid[grid.num_dim - 1] > 0])

        mats[0] = ri[:, np.newaxis] * mats[0]

    if stack:
        mats = np.vstack(mats).view(DifferentialMatrix)
        mats.order = 1

    return mats


def divergence_operator(
    grid: Grid,
    accuracy: Union[None, int] = None,
    symmetry: Union[None, str] = None,
    parities: list = [-1, -1, 1],
    wavevector: Union[None, tuple] = None,
    stack: bool = True,
):
    mats = gradient_operator(
        grid, accuracy, symmetry, parities, wavevector, stack=False
    )

    if "radial" in grid.geom or "polar" in grid.geom:
        ri = 1 / (grid[grid.num_dim - 1][grid[grid.num_dim - 1] > 0])

        mats[1] = mats[1] + ri[:, np.newaxis] * np.identity(mats[1].shape[0])

    if stack:
        mats = np.hstack(mats).view(DifferentialMatrix)
        mats.order = 1

    return mats


def scalar_laplacian_operator(
    grid: Grid,
    accuracy: Union[None, int] = None,
    symmetry: Union[None, str] = None,
    parity: int = 1,
    wavevector: Union[None, tuple] = None,
):
    if wavevector is None:
        mats = [
            DifferentialMatrix(
                grid,
                axis=axis,
                order=2,
                accuracy=accuracy,
                parity=parity,
                symmetry=symmetry,
            )
            for axis in range(grid.num_dim)
        ]
    else:
        wavevector = [
            (None, element) if isinstance(element, Number)
            else element for element in wavevector
        ]

        mats = []
        for axis, k in wavevector:
            if axis is not None:
                if axis >= grid.num_dim:
                    raise ValueError(
                        f"Axis {axis} is out of bounds for grid "
                        f"with {grid.num_dim} dimension(s)"
                    )
                if k is not None:
                    raise ValueError(
                        f"Could not impose periodic direction for axis {axis} "
                        f" and grid with {grid.num_dim} dimension(s)"
                    )
                mat = DifferentialMatrix(
                    grid,
                    axis=axis,
                    order=2,
                    accuracy=accuracy,
                    parity=parity,
                    symmetry=symmetry,
                )
            else:
                mat = (
                    -DifferentialMatrix(  # to make sure, symmetry is respected
                        grid,
                        order=0,
                        accuracy=accuracy,
                        parity=parity,
                        symmetry=symmetry,
                    )
                    * k**2
                )

            mats.append(mat)

    if "radial" in grid.geom or "polar" in grid.geom:
        ri = 1 / (grid[grid.num_dim - 1][grid[grid.num_dim - 1] > 0])

        mats[1] = mats[1] + ri[:, np.newaxis] * DifferentialMatrix(
            grid,
            axis=(grid.num_dim - 1),
            order=1,
            accuracy=accuracy,
            parity=parity,
            symmetry=symmetry,
        )
        mats[0] = (ri**2)[:, np.newaxis] * mats[0]

    mats = np.sum(mats, axis=0).view(DifferentialMatrix)
    mats.order = 2

    return mats


def vector_laplacian_operator(
    grid: Grid,
    accuracy: Union[None, int] = None,
    symmetry: Union[None, str] = None,
    parities: list = [-1, -1, 1],
    wavevector: Union[None, tuple] = None,
):
    npts = np.prod(grid.npts)
    if "radial" in grid.geom or "polar" in grid.geom:
        npts = npts / 2

    ndim = grid.num_dim if wavevector is None else 3

    mats = np.zeros([ndim * npts, ndim * npts])

    for axis in range(ndim):
        mats[
            axis * npts : npts + axis * npts,
            axis * npts : npts + axis * npts,
        ] = scalar_laplacian_operator(
            grid, accuracy, symmetry, parities[axis], wavevector
        )

    if "radial" in grid.geom or "polar" in grid.geom:
        ri = 1 / (grid[grid.num_dim - 1][grid[grid.num_dim - 1] > 0])

        mats[:npts, :npts] -= ri[:, np.newaxis] * np.identity(npts)
        mats[npts : 2 * npts, npts : 2 * npts] -= ri[
            :, np.newaxis
        ] * np.identity(npts)

        if wavevector[0][1] is None:
            d = DifferentialMatrix(
                grid,
                axis=0,
                order=1,
                accuracy=accuracy,
                parity=-1,
                symmetry=symmetry,
            )
        else:
            d = (
                DifferentialMatrix(
                    grid,
                    order=0,
                    parity=-1,
                    symmetry=symmetry,
                )
                * 1j
                * wavevector[0][1]
            )

        mats[:npts, npts : 2 * npts] = 2 * (ri**2)[:, np.newaxis] * d
        mats[npts : 2 * npts, :npts] = -2 * (ri**2)[:, np.newaxis] * d

    return mats.view(DifferentialMatrix)


def _directional_derivative_operator_from_a(
    grid: Grid,
    a: np.ndarray,
    accuracy: Union[None, int] = None,
    symmetry: Union[None, str] = None,
    parities: list = [-1, -1, 1],
    wavevector: Union[None, tuple] = None,
):
    npts = np.prod(grid.npts)
    if "radial" in grid.geom or "polar" in grid.geom:
        npts = npts / 2

    ndim = grid.num_dim if wavevector is None else 3

    mats = np.zeros([ndim * npts, ndim * npts])

    for axis in range(ndim):
        g = gradient_operator(
            grid,
            accuracy,
            symmetry,
            parities=3 * [parities[axis]],
            wavevector=wavevector,
            stack=False,
        )
        g = [
            a[
                0 + j * npts : npts + j * npts
            ][:, np.newaxis] * g[j] for j in range(ndim)
        ]
        g = np.sum(g, axis=0)

        mats = mats.astype(g.dtype)

        mats[
            0 + axis * npts : npts + axis * npts,
            0 + axis * npts : npts + axis * npts,
        ] = g

    if "radial" in grid.geom or "polar" in grid.geom:
        ri = 1 / (grid[grid.num_dim - 1][grid[grid.num_dim - 1] > 0])

        mats[:npts, npts : 2 * npts] = (ri * a[0])[:, np.newaxis] * np.identity(
            npts
        )
        mats[npts : 2 * npts, :npts] = -(ri * a[0])[
            :, np.newaxis
        ] * np.identity(npts)

    return mats.view(DifferentialMatrix)


def _directional_derivative_operator_from_b(
    grid: Grid,
    b: np.ndarray,
    accuracy: Union[None, int] = None,
    symmetry: Union[None, str] = None,
    parities: list = [-1, -1, 1],
    wavevector: Union[None, tuple] = None,
):
    npts = np.prod(grid.npts)
    if "radial" in grid.geom or "polar" in grid.geom:
        npts = npts / 2

    ndim = grid.num_dim if wavevector is None else 3

    mats = np.zeros([ndim * npts, ndim * npts])

    for axis in range(ndim):
        g = gradient_operator(
            grid,
            accuracy,
            symmetry,
            parities=3 * [parities[axis]],
            wavevector=wavevector,
            stack=False,
        )
        g = np.hstack(
            [
                np.diag(g[j] @ b[0 + axis * npts : npts + axis * npts])
                for j in range(ndim)
            ]
        )

        mats = mats.astype(g.dtype)
        mats[0 + axis * npts : npts + axis * npts] = g

    if "radial" in grid.geom or "polar" in grid.geom:
        ri = 1 / (grid[grid.num_dim - 1][grid[grid.num_dim - 1] > 0])

        mats[:npts, :npts] += (ri * b[1])[:, np.newaxis] * np.identity(npts)
        mats[npts : 2 * npts, :npts] -= (ri * b[0])[
            :, np.newaxis
        ] * np.identity(npts)

    return mats.view(DifferentialMatrix)


def directional_derivative_operator(
    grid: Grid,
    a: Union[None, np.ndarray] = None,
    b: Union[None, np.ndarray] = None,
    accuracy: Union[None, int] = None,
    symmetry: Union[None, str] = None,
    parities: list = [-1, -1, 1],
    wavevector: Union[None, tuple] = None,
):
    if (a is None and b is None) or (a is not None and b is not None):
        raise ValueError("Either only a or only b must be specified")
    elif a is not None:
        return _directional_derivative_operator_from_a(
            grid, a, accuracy, symmetry, parities, wavevector
        )
    else:
        return _directional_derivative_operator_from_b(
            grid, b, accuracy, symmetry, parities, wavevector
        )


def _curl_line(
    grid: Grid,
    axis: int,
    accuracy: Union[None, int] = None,
    symmetry: Union[None, str] = None,
    parities: list = [-1, -1, 1],
    wavevector: Union[None, tuple] = None,
):
    npts = np.prod(grid.npts)
    if "radial" in grid.geom or "polar" in grid.geom:
        npts = npts / 2

    axes = 0, 1, 2

    if wavevector is None:
        wavevector = [(axis, None) for axis in axes]
    wavevector = [
        (None, element) if isinstance(element, Number)
        else element for element in wavevector
    ]

    if wavevector[(axis - 1) % 3][1] is None:
        a = -DifferentialMatrix(
            grid,
            axis=axes[(axis - 1) % 3],
            order=1,
            accuracy=accuracy,
            parity=parities[(axis + 1) % 3],
            symmetry=symmetry,
        )
    else:
        a = -DifferentialMatrix(
            grid,
            order=0,
            accuracy=accuracy,
            parity=parities[(axis + 1) % 3],
            symmetry=symmetry,
        ) * 1j * wavevector[(axis - 1) % 3][1]

    if wavevector[(axis + 1) % 3][1] is None:
        b = DifferentialMatrix(
            grid,
            axis=axes[(axis + 1) % 3],
            order=1,
            accuracy=accuracy,
            parity=parities[(axis + 2) % 3],
            symmetry=symmetry,
        )
    else:
        b = DifferentialMatrix(
            grid,
            order=0,
            accuracy=accuracy,
            parity=parities[(axis + 2) % 3],
            symmetry=symmetry,
        ) * 1j * wavevector[(axis + 1) % 3][1]

    return np.roll(
        np.hstack([np.zeros([npts, npts]), a, b]), axis * npts, axis=1
    )


def curl_operator(
    grid: Grid,
    accuracy: Union[None, int] = None,
    symmetry: Union[None, str] = None,
    parities: list = [-1, -1, 1],
    wavevector: Union[None, tuple] = None,
):
    npts = np.prod(grid.npts)
    if "radial" in grid.geom or "polar" in grid.geom:
        npts = npts / 2

    ndim = grid.num_dim if wavevector is None else 3
    if ndim < 3:
        raise ValueError("Curl operator is defined uniquely in 3D space")

    mats = np.empty([ndim * npts, ndim * npts])

    for axis in range(ndim):
        mats[
            axis * npts : npts + axis * npts,
        ] = _curl_line(
            grid, axis, accuracy, symmetry, parities, wavevector
        )

    return mats.view(DifferentialMatrix)