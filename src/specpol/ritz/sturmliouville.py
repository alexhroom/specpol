"""Ritz methods for Sturm-Liouville operators."""
from functools import lru_cache
from typing import Callable, Tuple

import numpy as np
from pyfilon import filon_fun_iexp

from specpol.common import Eigenpairs, generate_matrix, lagdiff, lagquad, laguerre


def sturm_liouville_bdd(
    potential: Callable,
    boundaries: float,
    matrix_size: int,
    quad_mesh_size: int,
    boundary_angles: Tuple[float, float],
    **settings: dict,
) -> np.array:
    r"""Approximate the spectrum of a bounded Sturm-Liouville operator with potential Q.

    The Sturm-Liouville operator is defined
    Ly = -y'' + Q(x)y
    with boundary conditions
    cos(a)*y(a) + sin(a)*y'(a) = 0
    cos(b)*y(b) + sin(b)*y'(b) = 0

    Parameters
    ----------
    potential: Callable
        The function Q in the operator definition.
    boundaries: float
        The boundaries of the domain (a, b)
    matrix_size: int
        The size of the Ritz matrix used for approximation.
    quad_mesh_size: int
        The number of interpolation points used for quadrature.
    boundary_angles: Tuple[float, float]
        The angles a and b at the zero and domain_len boundaries respectively.
    settings:
        dbm: Callable
            Add a dissipative barrier to the operator; this is a function
            of the form $i*\gamma*f(x)$ where f is a function with compact
            support.
        vectors: bool, default True
            If true, add
    """
    a, b = boundaries
    domain_len = b - a

    @lru_cache(maxsize=matrix_size)
    def onb_func(n: int) -> Callable:
        return lambda x: 1 / np.sqrt(domain_len) * np.exp(2j * np.pi * n * x / domain_len)

    dbm = settings.get("dbm", lambda x: 0)

    # exp(-2i * j * pi * x) is factored out and implicit in the quadrature
    def integrand(i: int, j: int) -> Callable:
        def integ(x: float) -> float:
            derivative = 4 * i * j * np.pi**2 * onb_func(i)(x) / domain_len**2
            barrier = 1j * dbm(x) * onb_func(i)(x)
            return 1 / np.sqrt(domain_len) * (derivative + potential(x) * onb_func(i)(x) + barrier)

        return integ

    alpha, beta = boundary_angles
    cot_alpha = 1 / np.tan(alpha)
    cot_beta = 1 / np.tan(beta)

    def entry_func(i: int, j: int) -> complex:
        return (
            - cot_alpha
            + cot_beta
            + filon_fun_iexp(integrand(i, j), a, b, -2 * j * np.pi / domain_len, quad_mesh_size)
        )

    ritz_matrix = generate_matrix(
        entry_func,
        matrix_size,
        start_index=0,
        doubleinf=True,
    )

    return np.linalg.eigvals(ritz_matrix)


def sturm_liouville_halfline(
    potential: Callable,
    matrix_size: int,
    quad_mesh_size: int,
    alpha: float,
    **settings,
) -> np.array:
    r"""Ritz method for a Sturm-Liouville operator on the half-line [0, \infty).

    This is an operator with the form
    $-y'' + Qy
    and boundary condition $cos(\alpha)*y(0) + sin(\alpha)*y'(0) = 0$
    (where $\alpha$ is not equal to $n\pi$, i.e. the BC is not Dirichlet)

    Parameters
    ----------
    potential: Callable
        The function Q in the operator definition.
    matrix_size: int
        The size of the Ritz matrix used for approximation.
    quad_mesh_size: int
        The number of interpolation points used for quadrature.
    alpha: float
        The alpha-value used in the boundary condition.
    settings:
        dbm: Callable
            Add a dissipative barrier to the operator; this is a function
            of the form $i*\gamma*f(x)$ where f is a function with compact
            support.
        exclusive: bool
            If True, exclude the 0 endpoint (e.g. if there is a singularity at 0)
        returns: str
            Alternate return values for the calculation. Can be:
            'matrix': return the Ritz matrix for the operator.
            'vectors': return eigenvectors as well as eigenvalues.
            for any other input, returns just the eigenvalues.
    """
    # the weighted Laguerre polynomials L_n * exp(-x/2) form
    # an orthonormal basis for the half-line
    # the BC's are not Dirichlet so we can use any ONB

    epsilon = settings.get("exclusive", False) * 0.001

    @lru_cache(maxsize=matrix_size**2)
    def integrand(i: int, j: int) -> Callable:
        def integ(x: float) -> complex:
            derivative = (lagdiff(i, 0, x, degree=1) + laguerre(i, 0, x) / 2) * (
                lagdiff(j, 0, x, degree=1) + laguerre(j, 0, x) / 2
            )
            barrier = (
                1j * settings.get("dbm", lambda x: 0)(x) * laguerre(i, 0, x) * laguerre(j, 0, x)
            )
            return (
                derivative
                + potential(x + epsilon) * laguerre(i, 0, x) * laguerre(j, 0, x)
                + barrier
            )

        return np.vectorize(integ)

    # we define entries by the weak formulation of the BVP:
    # $B[u, v] = \cot(\alpha)*u(0)*v(0) + \int_0^\infty u'v' + quv$
    # and we make the change of variables y = e^-x to simplify the quadrature
    sample_points, weights = lagquad(quad_mesh_size)
    cot_alpha = 1 / np.tan(alpha)

    def entry_func(i: int, j: int) -> complex:
        return - cot_alpha * laguerre(i, 0, 0) * laguerre(j, 0, 0) + sum(
            integrand(i, j)(sample_points) * weights
        )

    # TODO: consider
    # "return cot_alpha * laguerre(i, 0, 0) * laguerre(j, 0, 0) + quad(
    #        lambda x: integrand(i, j)(-np.log(x)), 0, 1, complex_func = True
    #    )"

    ritz_matrix = generate_matrix(entry_func, matrix_size, start_index=0)
    if settings.get("returns") == "matrix":
        return ritz_matrix

    eigvals, eigvecs = np.linalg.eig(ritz_matrix)

    if settings.get("returns") == "vectors":
        return Eigenpairs(eigvals, eigvecs)
    return eigvals
