from typing import Union
from numbers import Number
import numpy as np
from scipy.linalg import inv, qr
from ..grid import Grid
from ..differentiation import (
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
    reynolds_number: float = 2000.,
    external_force_operator: Union[None, np.ndarray] = None,
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

    velocity_operator = -convection + 1 / reynolds_number * laplacian

    if external_force_operator is not None:
        velocity_operator += external_force_operator

    bnds = grid.boundary_indices() # find boundary nodes

    match grid.geom:
        case "cart":
            # Assume the same BC for all boundaries.
            bnds = np.concatenate([np.concatenate(bnd) for bnd in bnds])
        case "polar":
            bnds = bnds[1]
        case _:
            raise ValueError(f"Grid geometry {grid.geom} not supported")

    npts = np.prod(grid.npts)
    if "radial" in grid.geom or "polar" in grid.geom:
        npts = int(npts / 2)

    O, I = np.zeros([npts, npts]), np.identity(npts)

    # No-slip everywhere.
    velocity_operator[bnds] = bc_vals * np.hstack([I[bnds], O[bnds], O[bnds]])
    velocity_operator[bnds + npts] = bc_vals * np.hstack(
        [O[bnds], I[bnds], O[bnds]]
    )
    velocity_operator[bnds + 2 * npts] = bc_vals * np.hstack(
        [O[bnds], O[bnds], I[bnds]]
    )
    gradient[bnds] = gradient[bnds + npts] = gradient[bnds + 2 * npts] = 0

    u, _ = qr(-gradient)
    v, _ = qr(np.conj(divergence).T)
    u, v = u[:, npts:], v[:, npts:]

    velocity_operator = np.conj(u).T @ (velocity_operator @ v)
    reduced_operator = inv(np.conj(u).T @ v) @ velocity_operator

    if return_transform_matrix:
        return reduced_operator, v
    else:
        return reduced_operator
