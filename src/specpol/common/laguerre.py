"""Generalised Laguerre polynomials."""
import warnings
from functools import lru_cache
from typing import Callable

import mpmath as mp
import numpy as np
from scipy.optimize import brentq


@lru_cache(maxsize=None)
def laguerre(n: int, a: float, x: complex) -> complex:
    """Calculate the associated Laguerre polynomial L_n^a(x).

    Parameters
    ----------
    n: int
        the order of the Laguerre polynomial.
    a: float
        the generalised alpha-parameter of the Laguerre polynomial.
    x: complex
        the value at which the polynomial is evaluated.

    Returns
    -------
    complex
        The Laguerre polynomial L_n^a evaluated at x.
    """
    if n == 0:
        return 1
    if n == 1:
        return -x + a + 1
    if n == 2:
        return x**2 / 2 - (a + 2) * x + (a + 1) * (a + 2) / 2
    return (2 + (a - 1 - x) / n) * laguerre(n - 1, a, x) - (1 + (a - 1) / n) * laguerre(n - 2, a, x)


def lagdiff(n: int, a: float, x: complex, *, degree: int) -> complex:
    """Calculate the n'th derivative of the associated Laguerre polynomial L_n^a(x).

    Parameters
    ----------
    n: int
        the order of the Laguerre polynomial.
    a: float
        the generalised alpha-parameter of the Laguerre polynomial.
    x: complex
        the value at which the polynomial is evaluated.
    degree: int
        the degree of the derivative taken.

    Returns
    -------
    complex
        The Laguerre polynomial D_{degree} L_n^a evaluated at x.
    """
    if n - degree >= 0:
        return laguerre(n - degree, a + degree, x)
    return 0


def lagquad(n: int) -> float:
    """
    Calculate the sample points and weights for Gauss-Laguerre quadrature.

    Parameters
    ----------
    n: int
        The number of sample points to use.

    Returns
    -------
    float
        The result of the integral.
    """

    # we need arbitrary precision or the quadrature falls apart
    # at high n; use mpmath arbitrary-precision Laguerre polys
    def objective(x: float) -> float:
        return mp.laguerre(n, 0, x)

    def sign(x: float) -> float:
        return -1 if x < 0 else 1

    # first we need to bound each root;
    # find subdivisions where the objective func changes sign
    linspace = np.linspace(0, n + (n-1*np.sqrt(n)), n*20)
    mesh = (objective(x) for x in linspace)

    prev_point = next(mesh)
    root_bounds = []
    for i, point in enumerate(mesh):
        if sign(prev_point) != sign(point):
            root_bounds.append((linspace[i], linspace[i+1]))
        prev_point = point

    if len(root_bounds) < n:
        warnings.warn("lagquad failed to bound all roots for the polynomial.")

    roots = []
    for a, b in root_bounds:
        root, result = brentq(objective, a, b, full_output=True)
        if not result.converged:
            warnings.warn(f"lagquad failed to find root between {a} and {b}. "
                          "even though there is a sign change on that interval!")
        roots.append(root)

    weights = [(x / ((n + 1) * mp.laguerre(n + 1, 0, x)) ** 2) for x in roots]

    return roots, weights