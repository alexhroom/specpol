"""Generalised Laguerre polynomials."""
from functools import lru_cache

import mpmath as mp
from scipy.optimize import newton
from scipy.special import jn_zeros


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
    if n == 2:  # noqa: PLR2004
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
    """Calculate the sample points and weights for Gauss-Laguerre quadrature.

    Parameters
    ----------
    n: int
        The number of sample points to use.

    Returns
    -------
    float
        The result of the integral.
    """

    # the function and its derivatives for Halley's method
    # we use arbitrary precision as regular isn't good enough here!
    def objective(x):
        return float(mp.laguerre(n, 0, x))

    def fprime(x):
        return -float(mp.laguerre(n - 1, 1, x))

    def fprime2(x):
        return float(mp.laguerre(n - 2, 2, x))

    # the zeroes of the Laguerre polynomial can be approximated by the zeroes
    # of the Bessel function, as accurate as 5sf
    # (see e.g Abramowitz and Stegun §22.16)
    bessel_zeroes = jn_zeros(0, n)

    def approx_root(m):
        """Approximate the m'th zero of L_n."""
        return (bessel_zeroes[m] ** 2 / (4 * (n + 1 / 2))) * (
            1 + (-2 + bessel_zeroes[m] ** 2) / (48 * (n + 1 / 2) ** 2)
        )

    roots = []
    # apply Halley's method
    for m in range(1, n):
        x0 = approx_root(m)
        roots.append(newton(objective, x0, fprime=fprime, fprime2=fprime2, maxiter=n**2, tol=1e-12))

    weights = [float(x / ((n + 1) * mp.laguerre(n + 1, 0, x)) ** 2) for x in roots]

    return roots, weights
