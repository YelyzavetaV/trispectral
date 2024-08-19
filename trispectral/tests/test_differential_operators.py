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


GRID_2D_CART = Grid.from_bounds(
    [-2., 2., 41], [-2., 2., 41], discs=["chebyshev"] * 2
)
X_2D_CART, Y_2D_CART = GRID_2D_CART

F_2D_CART = X_2D_CART * np.exp(-X_2D_CART**2 - Y_2D_CART**2)
G_2D_CART = np.concatenate(
    [
        np.exp(-X_2D_CART**2 - Y_2D_CART**2) - 2 * X_2D_CART * F_2D_CART,
        -2 * Y_2D_CART * F_2D_CART,
    ]
)
U_2D_CART = np.concatenate(
    [
        X_2D_CART + 3 * Y_2D_CART,
        -Y_2D_CART - 2 * X_2D_CART,
    ]
)
D_2D_CART = np.zeros_like(X_2D_CART)

GRID_3D_CART = Grid.from_bounds(
    [0., 1., 21], [-1., 1., 21], [-np.pi, np.pi, 21], discs=3 * ["chebyshev"]
)
X_3D_CART, Y_3D_CART, Z_3D_CART = GRID_3D_CART

F_3D_CART = 2 * X_3D_CART + 3 * Y_3D_CART**2 - np.sin(Z_3D_CART)
G_3D_CART = np.concatenate(
    [
        np.full_like(X_3D_CART, 2.),
        6 * Y_3D_CART,
        -np.cos(Z_3D_CART),
    ]
)

GRID_1D_CART_WAVE = Grid.from_bounds([-1., 1., 41], discs=["chebyshev"])
Y_1D_CART_WAVE = GRID_1D_CART_WAVE[0]

K_1D_CART = 1., [0, None], 2.
F_1D_CART_WAVE = 1 - Y_1D_CART_WAVE**2
G_1D_CART_WAVE = np.concatenate(
    [
        1j * K_1D_CART[0] * F_1D_CART_WAVE,
        -2 * Y_1D_CART_WAVE,
        1j * K_1D_CART[2] * F_1D_CART_WAVE,
    ]
)


@pytest.mark.parametrize(
    "grid,field,exact,wavevector",
    [
        (GRID_2D_CART, F_2D_CART, G_2D_CART, None),
        (GRID_3D_CART, F_3D_CART, G_3D_CART, None),
        (GRID_1D_CART_WAVE, F_1D_CART_WAVE, G_1D_CART_WAVE, K_1D_CART),
    ]
)
def test_gradient_operator(grid, field, exact, wavevector):
    g = gradient_operator(grid, wavevector=wavevector) @ field
    assert np.allclose(g, exact)


@pytest.mark.parametrize(
    "grid,field,exact,wavevector",
    [
        (GRID_2D_CART, U_2D_CART, D_2D_CART, None),
    ]
)
def test_divergence_operator(grid, field, exact, wavevector):
    g = divergence_operator(grid, wavevector=wavevector) @ field
    assert np.allclose(g, exact)
