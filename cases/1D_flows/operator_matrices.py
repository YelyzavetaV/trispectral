from typing import Union
import numpy as np
import pytrispectral as ts

__all__ = ["oss_cartesian_matrices", "oss_cylindrical_matrices"]

def oss_cartesian_matrices(
    flow: str,
    grid: ts.Grid,
    Re: float = 5000.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    accuracy: Union[None, int] = None,
):
    """
    Matrices of classical Orr-Sommerfeld-Squire operator for 1D shear flows in
    Cartesian geometry.

    Parameters
    ----------
    flow: str
        Flow type ("poiseuille" or "couette").
    grid: pytrispectral.Grid
        1D Cartesian grid.
    Re: float, default = 5000.0
        Reynolds number.
    alpha: float, default = 1.0
        Streamwise wavenumber of periodicity.
    beta: float, default = 1.0
        Spanwise wavenumber of periodicity.
    accuracy: Union[None, int], default = None
        Accuracy of finite difference method if relevant.

    Returns
    -------
    Two 2D numpy arrays corresponding to the LHS and the RHS matrices of OSS
    operator.

    Raises
    ------
    ValueError
        If unknown flow type.
    """
    x, nx = grid[0], grid.npts[0]
    k = alpha**2 + beta**2

    match flow:
        case "poiseuille":
            u, d1u, d2u = 1.0 - x**2, -2 * x, np.full_like(x, -2.0)
        case "couette":
            u, d1u, d2u = x.copy(), np.ones_like(x), np.zeros_like(x)
        case _:
            raise ValueError(f"Unknown flow type {flow}")

    O, I = np.zeros([nx, nx]), np.identity(nx)
    d1 = ts.DifferentialMatrix(grid, order=1, accuracy=accuracy)
    d2 = ts.DifferentialMatrix(grid, order=2, accuracy=accuracy)
    d4 = ts.DifferentialMatrix(grid, order=4, accuracy=accuracy)

    # Assemble OSS matrices.
    A = (
        alpha * u[:, np.newaxis] * (d2 - k * I)
        - alpha * d2u[:, np.newaxis] * I
        + 1j / Re * (d4 - 2 * k * d2 + k**2 * I)
    )
    C = beta * d1u[:, np.newaxis] * I
    D = (
        alpha * u[:, np.newaxis] * I
        + 1j / Re * (d2 - k * I)
    )

    lmat = np.vstack([np.hstack([A, O]), np.hstack([C, D])])
    rmat = np.vstack([np.hstack([d2 - k * I, O]), np.hstack([O, I])])
    rmat = rmat.astype(complex)

    # Apply the boundary conditions v = v' = η = 0 at x = ±1. The eigevalues
    # Associated with boundary conditions are explicitly set to -100j.
    lmat[0] = -100j * np.hstack([I[0], O[0]])
    rmat[0] = np.hstack([I[0], O[0]])
    lmat[nx - 1] = -100j * np.hstack([I[-1], O[-1]])
    rmat[nx - 1] = np.hstack([I[-1], O[-1]])
    lmat[1] = -100j * np.hstack([d1[0], O[0]])
    rmat[1] = np.hstack([d1[0], O[0]])
    lmat[nx - 2] = -100j * np.hstack([d1[-1], O[-1]])
    rmat[nx - 2] = np.hstack([d1[-1], O[-1]])
    lmat[nx] = -100j * np.hstack([O[0], I[0]])
    rmat[nx] = np.hstack([O[0], I[0]])
    lmat[-1] = -100j * np.hstack([O[-1], I[-1]])
    rmat[-1] = np.hstack([O[-1], I[-1]])

    return lmat, rmat


def oss_cylindrical_matrices(
    grid: ts.Grid,
    Re: float = 2000.0,
    alpha: float = 1.0,
    n: int = 1,
    accuracy: Union[None, int] = None,
):
    """
    Matrices of Burridge-Drazin operator for 1D pipe Poiseuille flow in
    cylindrical geometry [1].

    Parameters
    ----------
    grid: pytrispectral.Grid
        1D radial grid. For the sake of testing, here, we require that the grid
        contains the centerline point r = 0.
    Re: float, default = 2000.0
        Reynolds number.
    alpha: float, default = 1.0
        Streamwise wavenumber of periodicity.
    n: int, default = 1
        Azimuthal wavenumber of periodicity.
    accuracy: Union[None, int], default = None
        Accuracy of finite difference method if relevant.

    Returns
    -------
    Two 2D numpy arrays corresponding to the LHS and the RHS matrices of OSS
    operator.

    Raises
    ------
    ValueError
        If r = 0 is not included in the grid.

    References
    ----------
    [1] D. M. Burridge and P. G. Drazin, "Comments on 'Stability of Pipe
        Poiseuille Flow'", Phys. Fluids, 1969.
    """
    r, nr = grid[0], grid.npts[0]
    if 0 not in r:
        raise ValueError("Here, we require that the grid contains r = 0")
    r = np.ma.masked_array(r, mask=(r == 0)) # mask singularity

    O, I = np.zeros([nr, nr]), np.identity(nr)
    d1 = ts.DifferentialMatrix(grid, order=1, accuracy=accuracy)
    d2 = ts.DifferentialMatrix(grid, order=2, accuracy=accuracy)
    d3 = ts.DifferentialMatrix(grid, order=3, accuracy=accuracy)
    d4 = ts.DifferentialMatrix(grid, order=4, accuracy=accuracy)

    # Burridge-Drazin coefficients.
    m = alpha**2 * r**2 + n**2
    dm = 2 * alpha**2 * r
    d2m = 2 * alpha**2
    u = -m / r**2
    du = -dm / r**2 + 2 * m / r**3
    d2u = -d2m / r**2 + 4 * dm / r**3 - 6 * m / r**4
    g = 1 / r - dm / m
    dg = -1 / r**2 - d2m / m + dm**2 / m**2
    d2g = 2 / r**3 + 3 * d2m * dm / m**2 - 2 * dm**3 / m**3

    c_T_0 = c_S_0 = u
    c_T_1 = g
    c_S_1 = 1 / r + dm / m
    c_H_0 = d2u + g * du + u**2
    c_H_1 = d2g + dg * g + 2 * du + 2 * u * g
    c_H_2 = 2 * dg + g**2 + 2 * u
    c_H_3 = 2 * g

    T = (
        c_T_0[:, np.newaxis] * I
        + c_T_1[:, np.newaxis] * d1
        + d2
    )
    S = (
        c_S_0[:, np.newaxis] * I
        + c_S_1[:, np.newaxis] * d1
        + d2
    )
    H = (
        c_H_0[:, np.newaxis] * I
        + c_H_1[:, np.newaxis] * d1
        + c_H_2[:, np.newaxis] * d2
        + c_H_3[:, np.newaxis] * d3
        + d4
    )

    # Operator acting on Φ in 'Orr-Sommerfeld' equation.
    A = (
        1j / Re * H
        + alpha * (1 - r**2)[:, np.newaxis] * T
        + 2 * alpha * (1 + r * g)[:, np.newaxis] * I
    )
    # Operator acting on Ω in 'Orr-Sommerfeld' equation.
    B = -2j / Re * alpha * n * T
    # Operator acting on Φ in 'Squire' equation.
    C = (
        2j / Re * alpha * n / (m**2)[:, np.newaxis] * T
        + 2 * n / m[:, np.newaxis] * I
    )
    # Operator acting on Ω in 'Squire' equation.
    D = (
        1j / Re * S
        + alpha * (1 - r**2)[:, np.newaxis] * I
    )

    if n != 0:
        lmat = np.vstack([np.hstack([A, B]), np.hstack([C, D])])
        rmat = np.vstack([np.hstack([T, O]), np.hstack([O, I])])
        rmat = rmat.astype(complex)

        # Apply Φ = Φ' = Ω = 0 at r = 1. The eigenvalues associated with the BC
        # are set to -100j.
        lmat[0] = -100j * np.hstack([I[0], O[0]])
        rmat[0] = np.hstack([I[0], O[0]])
        lmat[1] = -100j * np.hstack([d1[0], O[0]])
        rmat[1] = np.hstack([d1[0], O[0]])
        lmat[nr] = -100j * np.hstack([O[0], I[0]])
        rmat[nr] = np.hstack([O[0], I[0]])

        # Apply pole conditions.
        lmat[nr - 1] = -100j * np.hstack([I[nr - 1], O[nr - 1]])
        rmat[nr - 1] = np.hstack([I[nr - 1], O[nr - 1]])
        if n != 1:
            lmat[nr - 2] = -100j * np.hstack([d1[nr - 1], O[nr - 1]])
            rmat[nr - 2] = np.hstack([d1[nr - 1], O[nr - 1]])
        lmat[2 * nr - 1] = -100j * np.hstack([O[nr - 1], I[nr - 1]])
        rmat[2 * nr - 1] = np.hstack([O[nr - 1], I[nr - 1]])
    else: # if n = 0, the OS and the Squire equations decouple
        lmat, rmat = A, T

        # Apply Φ = Φ' = 0 at r = 1.
        lmat[0] = -100j * I[0]
        rmat[0] = I[0]
        lmat[1] = -100j * d1[0]
        rmat[1] = d1[0]

        # Apply pole conditions.
        lmat[-1] = -100j * I[-1]
        rmat[-1] = I[-1]
        lmat[-2] = -100j * d1[-1]
        rmat[-2] = d1[-1]

    return lmat, rmat
