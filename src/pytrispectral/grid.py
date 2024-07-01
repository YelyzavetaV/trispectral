"""
Tools to produce N-dimensional grids.

Default discretization:
* "chebyshev" for Chebyshev points of second kind
* "periodic" for uniform periodic grid
* "uniform" for uniform grid

This module contains the following (public) routines:
* cheb_pts(xmin=-1., xmax=1., npts=30) -> np.ndarray
* clenshaw_curtis_weights(nx: int) -> np.ndarray
* cheb_bary_weights(nx: int) -> np.ndarray

By design, the functionality of the module should be accessed via
the Python class Grid. An instance of a grid can be created using
one of the following class methods:
* Grid(arrs, *, discs=[], geom="cart")
* Grid.from_bounds(*bounds, discs=[], geom="cart")
* Grid.from_arrays(*arrs, discs=[], geom="cart")
* Grid.register(
    *args,
    from_bounds=False,
    discs=[],
    geom="cart",
    num=None,
    override=False,
)
* Grid.radial(nr, rdisc="chebyshev", register=True, num=None, override=False)
* Grid.polar(
    nphi, nr, rdisc="chebyshev", register=True, num=None, override=False
)
"""
from typing import Union
import operator
import warnings
import numpy as np
from .manager import ObjectManager
from .math_utils import nkron

__all__ = [
    "cheb_pts",
    "clenshaw_curtis_weights",
    "cheb_bary_weights",
    "Grid",

]

def _grid_manager(_grid_manager_instance=ObjectManager()):
    """Getter for the GridManager's one and only instance."""
    return _grid_manager_instance


def cheb_pts(xmin=-1., xmax=1., npts=30) -> np.ndarray:
    """
    Returns (mapped) Chebyshev points of second kind.

    Chebyshev points are defined in the interval [-1, 1] as follows:
        xj = cos(pi * j / (N - 1)),
    where j = 0, ..., N - 1. Can be mapped to an arbitrary interval [xmin, xmax].

    Parameters
    ----------
    xmin: float, default = -1.0
        Lower bound.
    xmax: float, default = 1.0
        Upper bound.
    npts: int, default = 30
        Number of points.

    Returns
    -------
    1D numpy.ndarray containing (mapped) Chebyshev points of second kind.

    Raises
    ------
    TypeError
        In the case npts is not an integer.
    ValueError
        In the case npts is a negative number.
    """
    try:
        npts = operator.index(npts)
    except TypeError as e:
        raise TypeError("npts must be an integer.") from e
    if npts < 0:
        raise ValueError(f"Number of grid points, {npts}, must be a positive integer.")

    x = np.sin(np.arange(-npts + 1, npts, 2) * np.pi / 2 / (npts - 1))
    return (x + 1) / 2.0 * (xmax - xmin) + xmin


_PTS_CONSTRUCTORS = {
    "chebyshev": cheb_pts,
    "periodic": np.linspace,
    "uniform": np.linspace,
}


def clenshaw_curtis_weights(nx: int) -> np.ndarray:
    """
    Clenshaw-Curtis quadrature weights for Chebyshev points of second kind.

    Parameters
    ----------
    nx: int
        Number of Chebyshev points.

    Returns
    -------
    1D np.ndarray containing Clenshaw-Curtis quadrature weights.
    """
    c = 2 / np.hstack([1, 1 - np.arange(2, nx, 2) ** 2])
    c = np.hstack([c, np.take(c, np.arange(np.floor(nx / 2) - 1, 0, -1, dtype=int))])

    w = np.fft.ifft(c)
    w[0] /= 2
    return np.append(w, w[0]).real


def cheb_bary_weights(nx: int) -> np.ndarray:
    """Barycentric weights for Chebyshev points [1].

    Parameters
    ----------
    nx: int
        Number of Chebyshev points.

    Returns
    -------
    1D np.ndarray containing Chebyshev barycentric weights.

    References
    ----------
    [1] R. Baltensperger and M. A. Trummer, "Spectral Differencing with a Twist",
        SIAM J. Sci. Comput., 2003.
    """
    w = np.ones(nx)
    w[-1] = 0.5
    w[-2::-2] *= -1
    w[0] *= 0.5
    return w


class Grid(np.ndarray):
    """
    N-dimensional grid.

    Parameters
    ----------
    arrs: array-like
        Coordinate arrays.
    discs: list, default = []
        Discretization types for coordinate arrays.
    geom: str, default = "cart"
        Grid geometry. Default values are the following:
        * "cart" - Cartesian geometry;
        * "radial" - radial geometry (for the case of 1D pipe geometry);
        * "polar" - polar disk geometry;
        Custom geometries are also possible.

    Attributes
    ----------
    num: int, default = None
        Unique identifier of the grid.
    num_dim: int
        Number of dimensions in the grid.
    npts: np.ndarray[int]
        Number of grid points along each axis.
    discs: list[str]
        Discretization type along each axis.
    geom: str
        Grid geometry.

    Methods
    -------
    from_bounds: class method
        Get an instance of Grid from grid bounds and number of points along
        each axis.
    from_arrs: class method
        Get an instance of Grid from coordinate arrays along each axis.
    register: class method
        Get an instance of Grid from bounds or from coordinate arrays and
        `register' it with GridManager, so that it's accessible at runtime
        using unique identifier. If an instance of Grid has been previously
        registered, call this method with its unique identifier as the only
        parameter.
    radial: class method
        Get a radial grid with or without `registering' it.
    polar: class method
        Get a polar grid with or without `registering' it.
    unregister: instance method
        Remove the instance from GridManager.
    coordinate_array: instance method
        Get a coordinate array along specific axis.
    weights: instance method
        Get quadrature weights along specific axis.
    weight_matrix: instance method
        Get the weight matrix of the grid.

    Notes
    -----
    This class inherits from np.ndarray and therefore possesses all the
    attrributes and methods of a numpy array.
    """
    def __new__(cls, arrs, *, discs=[], geom="cart"):
        mgrids = np.meshgrid(*arrs, indexing="ij")

        # Linearize grid representation. Column-major order is adopted. Each coordinate
        # array is stored in the i-th row of the object where i is the serial number
        # of the corresponding axis.
        obj = np.array([mgrid.flatten(order="F") for mgrid in mgrids])

        obj = obj.view(cls)

        obj._num = None  # Unique identifier of the grid

        obj.num_dim = obj.shape[0]  # Number of grid dimensions
        obj.npts = np.array([len(arr) for arr in arrs])
        obj.discs = discs
        obj.geom = geom

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self._num = None
        # Grid is designed in a way that the coordinate arrays are stored in a column-
        # major order. If the Grid instance is created through slicing or view-casting,
        # it becomes the responsibility of the user to ensure correct shaping and set
        # the attributes that are dependent on the ordering.
        self.num_dim = None
        self.npts = None
        self.discs = None
        self.geom = getattr(obj, "geom", None)

    @property
    def num(self):
        return self._num

    # Grid constructors.
    @classmethod
    def from_bounds(cls, *bounds, discs=[], geom="cart"):
        """
        Create a grid from axes bounds.

        Parameters
        ----------
        bound0, bound1,..., boundN: array-like
            Bounds for each dimension. Each bound must be an iterable containing
            three elements in the following order: lower bound, upper bound,
            number of points along the given direction. Number of points must be
            an integer.
        discs: array-like of str, default = []
            Discretization along each axis.
        geom: str, default = "cart"
            Geometry of the grid.

        Returns
        -------
        Instance of Grid.

        Raises
        ------
        ValueError
            * If shape of any bound is not (3,).
        """
        for bound in bounds:
            if np.asarray(bound).shape != (3,):
                raise ValueError(
                    "Wrong bound format: each bound array must have a single dimension"
                    " and contain three elements in the following order: lower bound, "
                    "upper bound and number of grid points."
                )

        num_dim = len(bounds)
        if not discs:
            discs = ["uniform"] * num_dim

        arrs = [_PTS_CONSTRUCTORS[disc](*bound) for bound, disc in zip(bounds, discs)]

        return cls(arrs, discs=discs, geom=geom)

    @classmethod
    def from_arrays(cls, *arrs, discs=[], geom="cart"):
        """
        Create a grid from coordinate arrays.

        Parameters
        ----------
        arr0, arr1,..., arrN: array_like
            1D coordinate arrays.
        discs: array-like of str, default = []
            Discretization along each axis.
        geom: str, default = "cart"
            Geometry of the grid.

        Returns
        -------
        Instance of Grid.

        Raises
        ------
        ValueError
            In the case if any of the arrs are not one-dimensional array-like.
        """
        for arr in arrs:
            if np.asarray(arr).ndim != 1:
                raise ValueError(
                    "Coordinate arrays must be one-dimensional array-like."
                )

        ndim = len(arrs)
        if not discs:
            warnings.warn("Discretization of every coordinate array must be specified.")

            discs = ["uniform"] * ndim

        return cls(arrs, geom=geom)

    @classmethod
    def register(
        cls,
        *args,
        from_bounds=False,
        discs=[],
        geom="cart",
        num=None,
        override=False,
    ):
        """
        (i) Create an instance of Grid from bounds or from coordinate arrays and
        `register' it with GridManager; (ii) get an existing withing GridManager
        instance of Grid using its unique identifier.

        Parameters
        ----------
        arg1,arg2,...,argN: array-like
            Bounds for each direction or coordinate arrays (see from_bounds and
            from_arrays for details).
        from_bounds: bool, default = False
            Whether to create a grid from bounds or coordinate arrays.
        discs: array-like of str, default = []
            Discretization along each axis.
        geom: str, default = "cart"
            Geometry of the grid.
        num: int, default = None
            Unique identifier of the grid. If not specified, l + 1 will be
            assigned where l is the identifier of the last registered grid.
        override: bool, default = False
            Whether to vverride a registered grid with identifier num.

        Returns
        -------
        Instance of Grid.

        Raises
        ------
        TypeError
            If num is not integer.
        ValueError
            If no grid data provided.
        """
        grid_manager = _grid_manager()

        if num is None:
            nums = grid_manager.nums()
            num = max(nums) + 1 if nums else 0
        else:
            try:
                operator.index(num)
            except TypeError as e:
                raise TypeError("Unique identifier num must be an integer") from e

        grid = getattr(grid_manager, str(num), None)

        if grid is None or override:
            if not args:
                raise ValueError(
                    "Could not create grid: specify grid data"
                    " (bounds or coordinate arrays)."
                )

        if from_bounds:
            grid = cls.from_bounds(*args, discs=discs, geom=geom)
        else:
            grid = cls.from_arrays(*args, discs=discs, geom=geom)

        grid._num = num
        setattr(grid_manager, str(num), grid)

        return grid

    @classmethod
    def radial(
        cls, nr, rdisc="chebyshev", register=True, num=None, override=False
    ):
        """
        Create an instance of radial grid. This should be viewed as convenience
        method for working in 1D pipe geometry.

        Parameters
        ----------
        nr: int
            Number of points.
        rdisc: str, default = "chebyshev"
            Discretization type.
        register: bool, default = True
            Whether to register grid with GridManager.
        num: int, default = None
            Unique identifier of the grid.
        override: bool, default = False
            Whether to vverride a registered grid with identifier num.

        Returns
        -------
        Instance of Grid.
        """
        if register:
            return cls.register(
                (-1., 1., 2 * nr),
                from_bounds=True,
                discs=(rdisc,),
                geom="radial",
                num=num,
                override=override,
            )
        else:
            return cls.from_bounds(
                (-1., 1., 2 * nr), discs=(rdisc,), geom="radial"
            )

    @classmethod
    def polar(
        cls, nphi, nr, rdisc="chebyshev", register=True, num=None, override=False
    ):
        """
        Create an instance of polar grid.

        Parameters
        ----------
        nphi: int
            Number of azimuthal grid points. Must be divisible by 4.
        nr: int
            Number of radial grid points.
        rdisc: str, default = "chebyshev"
            Type of radial discretization.
        register: bool, default = True
            Whether to register grid with GridManager.
        num: int, default = None
            Unique identifier of the grid.
        override: bool, default = False
            Whether to vverride a registered grid with identifier num.

        Returns
        -------
        Instance of Grid.

        Raises
        ------
        ValueError
            If nphi is not divisible by 4.
        """
        if nphi % 4 != 0:
            raise ValueError("nphi % 4 must be 0")

        if register:
            return cls.register(
                (-np.pi, np.pi - 2.0 * np.pi / nphi, nphi),
                (-1., 1., 2 * nr),
                from_bounds=True,
                discs=("periodic", rdisc),
                geom="polar",
                num=num,
                override=override,
            )
        else:
            return cls.from_bounds(
                (-np.pi, np.pi - 2.0 * np.pi / nphi, nphi),
                (-1., 1., 2 * nr),
                discs=("periodic", rdisc),
                geom="polar",
            )

    def unregister(self):
        """Remove the grid from GridManager"""
        grid_manager = _grid_manager()
        grid_manager.drop(num=self.num)

    def coordinate_array(self, axis: int, ignore_geom: bool = False):
        """
        Get coordinates for a given axis.

        Parameters
        ----------
        axis: int
            Index of the axis to get coordinates along.
        ignore_geom: bool, default = False
            Relevant for geometries of types "radial" and "polar". If set to
            True, the redundant domain won't be included.

        Returns
        -------
        1D numpy.ndarray containing coordinates along a given axis.

        Raises
        ------
        TypeError
            If axis is not integer.
        IndexError
            If axis out of bounds.
        """
        try:
            operator.index(axis)
        except TypeError as e:
            raise TypeError("Axis's index axis must be an integer.") from e
        if (axis > self.num_dim - 1) | (axis < -self.num_dim):
            raise IndexError(
                f"Axis' index {axis} out of bounds for the grid with {self.num_dim} "
                f"dimensions in axes. Allowed interval is from {-self.num_dim} to "
                f"{self.num_dim - 1}."
            )
        if axis < 0:
            axis += self.num_dim

        imax = int(np.prod(self.npts[: axis + 1]))
        step = int(imax / self.npts[axis])

        pts = np.array(self[axis][:imax:step])

        if "polar" in self.geom or "radial" in self.geom:
            if self.num_dim - 1 == axis and not ignore_geom:
                pts = pts[int(self.npts[axis] / 2):]

        return pts

    def weights(
        self,
        axis: int = 0,
        symmetry: Union[None, str] = None,
        fun: callable = lambda x: x,
    ):
        """
        Get quadrature weights for a given axis.

        Parameters
        ----------
        axis: int
            Index of the axis to get coordinates along.
        symmetry: Union[None, str], default = None
            Type of symmetry applied for the problem. Currently implemented
            for radial and polar geometries. Accepted values are "pole" for
            polar symmetry, "anti-pole" for polar asymmetry and None for no
            symmetry.
        fun: callable, default = lambda x: x
            A function to apply to quadrature weights.

        Returns
        -------
        1D numpy.ndarray containing quadrature weights along a given axis.

        Raises
        ------
        ValueError
            If unknown discretization.
        """
        npts = self.npts[axis]

        match self.discs[axis]:
            case "chebyshev":
                w = clenshaw_curtis_weights(npts)
            case "periodic" | "uniform":
                w = np.ones(npts)
            case _: # shouldn't happen
                raise ValueError(f"Unknown discretization {self.discs[axis]}")

        if "polar" in self.geom or "radial" in self.geom:
            if self.num_dim - 1 == axis:
                w = w[int(npts / 2):] * self.coordinate_array(axis)
            else:
                if symmetry is not None:
                    if "pole" in symmetry:
                        w = w[int(npts / 2) :]

        return fun(w)

    def weight_matrix(
        self, symmetry: Union[None, str] = None, fun: callable = lambda x: x
    ):
        """
        Get a weight matrix.

        Parameters
        ----------
        symmetry: Union[None, str], default = None
            Type of symmetry applied for the problem. Currently implemented
            for radial and polar geometries. Accepted values are "pole" for
            polar symmetry, "anti-pole" for polar asymmetry and None for no
            symmetry.
        fun: callable, default = lambda x: x
            A function to apply to quadrature weights.

        Returns
        -------
        2D numpy.ndarray contaning the weight matrix.
        """
        weights = [
            np.diag(self.weights(axis, symmetry, fun)) for axis in range(self.num_dim)
        ]
        if self.num_dim == 1:
            return weights[0]
        else:
            return nkron(*weights)
