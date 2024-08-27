# Trispectral

Trispectral is a NumPy-based compact Python package for numerical differentiation using spectral collocation methods.

For example, consider the function $f(x,y) = xe^{-(x^2 + y^2)}$ for $-1 \le x, y \le 1$. Using Trispectral we can compute the gradient of $f$ as follows:
```python
grid = Grid.from_bounds(
    [-1., 1., 41], [-1., 1., 41], discs=["chebyshev"] * 2
)
x, y = grid

f = x * numpy.exp(-x**2 - y**2)
grad = gradient_operator(grid) @ f
```
Here, the object `grid` represents a $41\times 41$ Chebyshev-Chebyshev grid.

Combined with SciPy, Trispectral also provides tools for solving linear boundary-value problems and linear eigenvalue problems (including the problems of hydrodynamic linear stability). See the [Trispectral tutorials](https://github.com/YelyzavetaV/trispectral/tree/main/tutorials) for more information.