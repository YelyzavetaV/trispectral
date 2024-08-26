import numpy as np
import pytest
from trispectral import (
    Grid,
    DifferentialMatrix,
    gradient_operator,
    divergence_operator,
    scalar_laplacian_operator,
    vector_laplacian_operator,
    directional_derivative_operator,
)


grid_2D_cheb = Grid.from_bounds(
    [-2., 2., 41], [-2., 2., 41], discs=["chebyshev"] * 2
)
x_2D_cheb, y_2D_cheb = grid_2D_cheb

f_2D_cheb = x_2D_cheb * np.exp(-x_2D_cheb**2 - y_2D_cheb**2)
g_2D_cheb = np.concatenate(
    [
        np.exp(-x_2D_cheb**2 - y_2D_cheb**2) - 2 * x_2D_cheb * f_2D_cheb,
        -2 * y_2D_cheb * f_2D_cheb,
    ]
)
u_2D_cheb = np.concatenate(
    [
        x_2D_cheb + 3 * y_2D_cheb,
        -y_2D_cheb - 2 * x_2D_cheb,
    ]
)
d_2D_cheb = np.zeros_like(x_2D_cheb)

grid_1D_cheb_wave = Grid.from_bounds([-1., 1., 41], discs=["chebyshev"])
y_1D_cheb_wave = grid_1D_cheb_wave[0]

k_1D_cheb_wave = 1., [0, None], 2.
f_1D_cheb_wave = 1 - y_1D_cheb_wave**2
g_1D_cheb_wave = np.concatenate(
    [
        1j * k_1D_cheb_wave[0] * f_1D_cheb_wave,
        -2 * y_1D_cheb_wave,
        1j * k_1D_cheb_wave[2] * f_1D_cheb_wave,
    ]
)


GRIDS = (grid_2D_cheb, grid_1D_cheb_wave)
SCALARS = (f_2D_cheb, f_1D_cheb_wave)
VECTORS = (u_2D_cheb,)
GRADS = (g_2D_cheb, g_1D_cheb_wave)
DIVS = (d_2D_cheb,)
WAVEVECTORS = (None, k_1D_cheb_wave)


@pytest.mark.parametrize(
    "grid,field,exact,wavevector",
    [
        (GRIDS[0], SCALARS[0], GRADS[0], WAVEVECTORS[0]),
        (GRIDS[1], SCALARS[1], GRADS[1], WAVEVECTORS[1]),
    ]
)
def test_gradient_operator(grid, field, exact, wavevector):
    g = gradient_operator(grid, wavevector=wavevector) @ field
    assert np.allclose(g, exact)


@pytest.mark.parametrize(
    "grid,field,exact,wavevector",
    [
        (GRIDS[0], VECTORS[0], DIVS[0], WAVEVECTORS[0]),
    ]
)
def test_divergence_operator(grid, field, exact, wavevector):
    g = divergence_operator(grid, wavevector=wavevector) @ field
    assert np.allclose(g, exact)
