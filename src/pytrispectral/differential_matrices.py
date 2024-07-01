import warnings
from typing import Union
from dataclasses import dataclass, field
from operator import index
from copy import deepcopy
import numpy as np
from scipy.linalg import toeplitz
from .manager import ObjectManager
from .math_utils import nkron
from .grid import cheb_bary_weights, Grid

__all__ = [
    "cheb_mat",
    "bary_cheb_mat",
    "four_mat",
    "Derivative",
    "DifferentialMatrix",
]

def _dmat_manager(_dmat_manager_instance=ObjectManager()):
    """Getter for the DmatManager's one and only instance."""
    return _dmat_manager_instance


def cheb_mat(x, order, **kwargs):
    """Computes 1D Chebyshev differential matrix [1].

    References
    ----------
    [1] L.N. Trefethen, "Spectral Methods in MATLAB". SIAM, Philadelphia, 2000.
    """
    npts = len(x)
    x = x.reshape((npts, 1))
    n = np.arange(npts)

    c = (np.hstack(([2.0], np.ones(npts - 2), [2.0])) * (-1) ** n).reshape(npts, 1)
    X = np.tile(x, (1, npts))
    dX = X - X.T
    # The np.eye term is to avoid division by zero on the diagonal; the diagonal
    # part of D is properly computed in the subsequent line.
    mat = c * (1.0 / c.T) / (dX + np.eye(npts))
    mat -= np.diag(np.sum(mat, axis=1))

    return np.linalg.matrix_power(mat, order)


def bary_cheb_mat(x, order, **kwargs):
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
    k[:nx - int(nx / 2) - 1:-1] = -k.copy()[:int(nx / 2)]
    np.fill_diagonal(d, k)

    if order == 2:
        d *= 2 * (np.repeat(k, nx).reshape(nx, nx) - dx)
        np.fill_diagonal(d, 0)
        np.fill_diagonal(d, -np.sum(d, axis=1))

    if order > 2:
        raise NotImplementedError(f"Chebyshev matrix of order {order} not implemented")

    return d


def four_mat(x, order, **kwargs):
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


@dataclass
class Derivative:
    """Helper class to isolate derivative's properties and post-init checks"""
    axis: int
    order: int
    accuracy: Union[None, int] = None
    parity: Union[None, int] = None
    symmetry: Union[None, str] = None
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        # TODO: check for symmetry
        try:
            index(self.order)
        except TypeError as e:
            raise TypeError("Derivatives' order must be an integer.") from e
        if self.order < 0:
            raise ValueError("Derivative's order must be a positive number or 0.")


class DifferentialMatrix(np.ndarray):
    def __new__(cls, grid, d: Derivative):
        mats = [np.eye(grid.npts[axis]) for axis in range(grid.num_dim)]

        order = d.order
        if order > 0:
            axis, accuracy, parity, symmetry, kwargs = (
                d.axis, d.accuracy, d.parity, d.symmetry, d.kwargs
            )
            disc = grid.discs[axis]
            # If the user specified the accuracy, pass it as a keyword argument.
            if accuracy is not None:
                kwargs["accuracy"] = accuracy

            s = 1 # symmetry coefficient
            if symmetry is not None:
                if "anti-pole" in symmetry:
                    s = -1

            mats[axis] = _DMAT_CONSTRUCTORS[disc](
                grid.coordinate_array(axis, ignore_geom=True), order, **kwargs
            )
        else:
            warnings.warn(
                f"Requested a 0-th-order derivative along axis {axis} "
                f"- the identity matrix will be outputted",
                RuntimeWarning,
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
                    + parity * (-1)**n * mats[0][nr:, nr - 1 :: -1]
                )
            case "polar":
                nphi, nr = grid.npts
                nr = p = int(nr / 2)

                if axis == 0: # derivative in ϕ
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
        self.orders = None
        self.discs = None

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
                raise TypeError("Unique identifier num must be an integer") from e


        dmat = getattr(dmat_manager, str(num), None)
        if dmat is None or override:
            deriv = Derivative(axis, order, accuracy, parity, symmetry, kwargs)
            dmat = cls(grid, deriv)
            dmat._num = num
            setattr(dmat_manager, str(num), dmat)

        return dmat

    def unregister(self):
        dmat_manager = _dmat_manager()
        dmat_manager.drop(num=self.num)

