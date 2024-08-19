from typing import Union
from numbers import Number
import numpy as np
from scipy.linalg import inv, qr
from ..grid import Grid
from ..differentiation import (
    DifferentialMatrix,
    gradient_operator,
    divergence_operator,
    vector_laplacian_operator,
    directional_derivative_operator,
)

__all__ = ["reduced_linear_operator"]

def reduced_linear_operator(
    grid: Grid,
    flow: np.ndarray,
    wavevector: Union[None, tuple] = None,
    reynolds: float = 2000.,
    accuracy: Union[None, int] = None,
    symmetry: Union[None, str] = None,
    parities: list = [-1, -1, 1],
    bc_vals: float = -1000.,
    return_transform_matrix: bool = False,
):
    gradient = gradient_operator(
        grid, accuracy=accuracy, symmetry=symmetry, wavevector=wavevector
    )
    divergence = divergence_operator(
        grid,
        accuracy=accuracy,
        symmetry=symmetry,
        parities=parities,
        wavevector=wavevector,
    )
    laplacian = vector_laplacian_operator(
        grid,
        accuracy=accuracy,
        symmetry=symmetry,
        parities=parities,
        wavevector=wavevector,
    )

    basevector = [
        0 if isinstance(element, Number) else element for element in wavevector
    ] if wavevector is not None else wavevector
    convection = directional_derivative_operator(
        grid,
        a=flow,
        accuracy=accuracy,
        symmetry=symmetry,
        parities=parities,
        wavevector=wavevector,
    ) + directional_derivative_operator(
        grid,
        b=flow,
        accuracy=accuracy,
        symmetry=symmetry,
        parities=parities,
        wavevector=basevector
    )

    velocity_operator = convection + 1 / reynolds * laplacian

    bnd = grid.argbnd() # find boundary nodes

    # Assume the same BC for all boundaries.
    bnd = np.concatenate(bnd)

    npts = np.prod(grid.npts)
    if "radial" in grid.geom or "polar" in grid.geom:
        npts = npts / 2

    O, I = np.zeros([npts, npts]), np.identity(npts)

    # No-slip everywhere.
    velocity_operator[bnd] = bc_vals * np.hstack([I[bnd], O[bnd], O[bnd]])
    velocity_operator[bnd + npts] = bc_vals * np.hstack(
        [O[bnd], I[bnd], O[bnd]]
    )
    velocity_operator[bnd + 2 * npts] = bc_vals * np.hstack(
        [O[bnd], O[bnd], I[bnd]]
    )
    gradient[bnd] = gradient[bnd + npts] = gradient[bnd + 2 * npts] = 0

    u, _ = qr(gradient)
    v, _ = qr(np.conj(divergence).T)
    u, v = u[:, npts:], v[:, npts:]

    velocity_operator = np.conj(u).T @ (velocity_operator @ v)
    reduced_operator = inv(np.conj(u).T @ v) @ velocity_operator

    if return_transform_matrix:
        return reduced_operator, v
    else:
        return reduced_operator
