import numpy as np
import pytest
from trispectral import (
    Grid,
    gradient_operator,
    divergence_operator,
    scalar_laplacian_operator,
    vector_laplacian_operator,
    directional_derivative_operator,
    curl_operator,
)


grids = (
    Grid.box([-1.0, 1.0, 17], discs=["chebyshev"]),
    Grid.box(  # 1D Chebyshev grid
        [-2.0, 2.0, 31], [0.0, 1.0, 17], discs=["chebyshev"] * 2
    ),  # 2D Chebyshev grid
    Grid.box(  # 1D Chebyshev grid
        [0.0, 0.5, 11], [0.0, 0.5, 11], [0.0, 0.5, 11], discs=["chebyshev"] * 3
    ),  # 3D Chebyshev grid
)

funcs = (
    (lambda x: np.sin(np.pi * x) - x * x,),  # test on 1D Chebyshev grid
    (  # test on 2D Chebyshev grid
        lambda x, y: x * np.exp(-x * x - y * y),
        lambda x, y: np.concatenate(
            [
                x + 3 * y,
                -y - 2 * x,
            ]
        ),
        lambda x, y: np.concatenate(
            [
                np.cos(x * x + y * y),
                np.sin(x * x + y * y),
            ]
        ),
    ),
    ( # test on 3D Chebyshev grid
        lambda x, y, z: np.concatenate(
            [
                -np.sin(x + y),
                -np.sin(x + y) + np.cos(z - y),
                -np.cos(z - y),
            ]
        ),
    ),
)

grads = (
    (lambda x: np.pi * np.cos(np.pi * x) - 2 * x,),  # test on 1D Chebyshev grid
    (  # test on 2D Chebyshev grid
        lambda x, y: np.concatenate(
            [
                np.exp(-x * x - y * y) - 2 * x * x * np.exp(-x * x - y * y),
                -2 * x * y * np.exp(-x * x - y * y),
            ]
        ),
        None,
    ),
    (None,),  # test on 3D Chebyshev grid
)

divs = (
    (lambda x: np.pi * np.cos(np.pi * x) - 2 * x,),  # test on 1D Chebyshev grid
    (  # test on 2D Chebyshev grid
        None,
        lambda x, y: np.zeros_like(x),
    ),
    (None,),  # test on 3D Chebyshev grid
)

slaps = (
    (lambda x: -np.pi**2 * np.sin(np.pi * x) - 2,),  # test on 1D Chebyshev grid
    (  # test on 2D Chebyshev grid
        lambda x, y: 4 * x * np.exp(-x * x - y * y) * (x * x + y * y - 2),
        None,
    ),
    (None,),  # test on 3D Chebyshev grid
)

vlaps = ((None,), (None, lambda x, y: np.zeros(x.shape[0] * 2)), (None,))

dir_derivs = (
    (None,),
    (
        lambda x, y: np.concatenate(
            [
                -2 * np.sin(x * x + y * y) * (x * x - y * y + x * y),
                2 * np.cos(x * x + y * y) * (x * x - y * y + x * y),
            ]
        ),
    ),
    (None,),  # test on 3D Chebyshev grid
)

curls = (
    (None,), (None, None), (lambda x, y, z: np.zeros(x.shape[0] * 3),)
)


@pytest.mark.parametrize(
    "grid,func,grad",
    [
        (grids[0], funcs[0][0], grads[0][0]),
        (grids[1], funcs[1][0], grads[1][0]),
    ],
)
def test_gradient_operator(grid, func, grad):
    field, ge = func(*grid), grad(*grid)
    g = gradient_operator(grid) @ field
    assert np.allclose(g, ge)


@pytest.mark.parametrize(
    "grid,func,div",
    [
        (grids[1], funcs[1][1], divs[1][1]),
    ],
)
def test_divergence_operator(grid, func, div):
    field, de = func(*grid), div(*grid)
    d = divergence_operator(grid) @ field
    assert np.allclose(d, de)


@pytest.mark.parametrize(
    "grid,func,slap",
    [
        (grids[0], funcs[0][0], slaps[0][0]),
        (grids[1], funcs[1][0], slaps[1][0]),
    ],
)
def test_scalar_laplacian_operator(grid, func, slap):
    field, sle = func(*grid), slap(*grid)
    sl = scalar_laplacian_operator(grid) @ field
    assert np.allclose(sl, sle)


@pytest.mark.parametrize(
    "grid,func,vlap",
    [
        (grids[1], funcs[1][1], vlaps[1][1]),
    ],
)
def test_vector_laplacian_operator(grid, func, vlap):
    field, vle = func(*grid), vlap(*grid)
    vl = vector_laplacian_operator(grid) @ field
    assert np.allclose(vl, vle)


@pytest.mark.parametrize(
    "grid,a,b,dir_deriv",
    [
        (grids[1], funcs[1][1], funcs[1][2], dir_derivs[1][0]),
    ],
)
def test_directional_derivative_operator(grid, a, b, dir_deriv):
    a, b = a(*grid), b(*grid)
    dde = dir_deriv(*grid)
    dda = directional_derivative_operator(grid, a=a) @ b
    ddb = directional_derivative_operator(grid, b=b) @ a
    assert np.allclose(dda, ddb) and np.allclose(dda, dde)


@pytest.mark.parametrize(
    "grid,func,curl",
    [
        (grids[2], funcs[2][0], curls[2][0]),
    ],
)
def test_curl_operator(grid, func, curl):
    field, ce = func(*grid), curl(*grid)
    c = curl_operator(grid) @ field
    assert np.allclose(c, ce)