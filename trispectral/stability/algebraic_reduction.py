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

    # Apply BC.
    match grid.geom:
        case "cart":
            if grid.num_dim > 3:
                raise NotImplementedError(
                    "Case with grid.num_dim > 3 not implemented"
                )

            bnd = [
                np.argwhere(grid[axis] == grid[axis][0]).ravel()
                for axis in range(grid.num_dim)
            ] + [
                np.argwhere(grid[axis] == grid[axis][-1]).ravel()
                for axis in range(grid.num_dim)
            ]

            # Take care of the corner nodes: xy-corners belong to y-boundaries,
            # xz-corners -- to z-boundaries, and yz-corners -- to z-boundaries.
            if grid.num_dim > 1:
                for i in range(grid.num_dim - 1):
                    for j in range(1, grid.num_dim):
                        if i == j: # 'degenerate' case
                            continue
                        bnd[i] = bnd[i][~np.isin(bnd[i], bnd[j])]
                        bnd[i] = bnd[i][~np.isin(bnd[i], bnd[grid.num_dim + j])]
                        bnd[grid.num_dim + i] = bnd[grid.num_dim + i][
                            ~np.isin(bnd[grid.num_dim + i], bnd[j])
                        ]
                        bnd[grid.num_dim + i] = bnd[grid.num_dim + i][
                            ~np.isin(bnd[grid.num_dim + i], bnd[grid.num_dim + j])
                        ]

            # Assume the same BC for all boundaries.
            bnd = np.concatenate(bnd)
        case _:
            raise ValueError(f"Unknown geometry {grid.geom}")

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
