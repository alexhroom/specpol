"""Ritz methods for Sturm-Liouville operators."""
from functools import lru_cache
from typing import Callable, Tuple

import numpy as np
from scipy.special import roots_laguerre
from pyfilon import filon_fun_iexp

from specpol.common import generate_matrix, lagdiff, laguerre, lagquad


def ritz_sturm_liouville(
    potential: Callable,
    domain_len: float,
    matrix_size: int,
    quad_mesh_size: int,
    boundary_angles: Tuple[float, float],
    *,
    dbm: bool = False,
) -> np.array:
    r"""Approximate the spectrum of a Sturm-Liouville operator with potential Q.

    The Sturm-Liouville operator is defined
    Ly = -y'' + Q(x)y
    with boundary conditions
    cos(a)*y(0) + sin(a)*y(0) = 0
    cos(b)*y(L) + sin(b)*y(L) = 0

    Parameters
    ----------
    potential: Callable
        The function Q in the operator definition.
    domain_len: float
        The length of the domain (0, `domain_len`)
    matrix_size: int
        The size of the Ritz matrix used for approximation.
    quad_mesh_size: int
        The number of interpolation points used for quadrature.
    boundary_angles: Tuple[float, float]
        The angles a and b at the zero and domain_len boundaries respectively.
    dbm: bool, default False
        Whether to add a dissipative barrier to the operator.
    """

    @lru_cache(maxsize=matrix_size)
    def onb_func(n: int) -> Callable:
        return lambda x: 1 / np.sqrt(domain_len) * np.exp(2j * np.pi * n * x / domain_len)

    # exp(-2i * j * pi * x) is factored out and implicit in the quadrature
    def integrand(i: int, j: int) -> Callable:
        def integ(x: float) -> float:
            barrier = 0
            derivative = 4 * i * j * np.pi**2 * onb_func(i)(x) / domain_len**2
            if dbm:
                barrier = 1j * onb_func(i)(x) * (x <= matrix_size // 2)
            return 1/np.sqrt(domain_len) * (derivative + potential(x) * onb_func(i)(x) + barrier)

        return integ

    alpha, beta = boundary_angles
    cot_alpha = 1 / np.tan(alpha)
    cot_beta = 1 / np.tan(beta)

    def entry_func(i: int, j: int) -> complex:
        return (
            cot_alpha
            - cot_beta
            + filon_fun_iexp(integrand(i, j), 0, domain_len, -2 * j * np.pi / domain_len, quad_mesh_size)
        )

    ritz_matrix = generate_matrix(
        entry_func,
        matrix_size,
        start_index=0,
        doubleinf=True,
    )

    return np.linalg.eigvals(ritz_matrix)


def ritz_unbounded_sturm_liouville(
    potential: Callable,
    matrix_size: int,
    quad_mesh_size: int,
    alpha: float,
    *,
    dbm: bool = False,
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
    dbm: bool, default False
        Whether to add a dissipative barrier to the operator.
    """

    # the weighted Laguerre polynomials L_n * exp(-x/2) form
    # an orthonormal basis for the half-line
    # the BC's are not Dirichlet so we can use any ONB

    @lru_cache(maxsize=matrix_size**2)
    def integrand(i: int, j: int) -> Callable:
        def integ(x: float) -> complex:
            barrier = 0
            derivative = (lagdiff(i, 0, x, degree=1) + laguerre(i, 0, x) / 2) * (
                lagdiff(j, 0, x, degree=1) + laguerre(j, 0, x) / 2
            )
            if dbm:
                barrier = 1j * laguerre(i, 0, x) * laguerre(j, 0, x) * (x <= matrix_size//2)
            return derivative + potential(x) * laguerre(i, 0, x) * laguerre(j, 0, x) + barrier

        return np.vectorize(integ)

    # we define entries by the weak formulation of the BVP:
    # $B[u, v] = \cot(\alpha)*u(0)*v(0) + \int_0^\infty u'v' + quv$
    sample_points, weights = lagquad(quad_mesh_size)
    cot_alpha = 1/np.tan(alpha)

    def entry_func(i: int, j: int) -> complex:
        return cot_alpha * laguerre(i, 0, 0) * laguerre(j, 0, 0) + complex(sum(
            integrand(i,j)(x) * w for x, w in zip(sample_points, weights)
        ))

    ritz_matrix = generate_matrix(entry_func, matrix_size, start_index=0)

    return np.linalg.eigvals(ritz_matrix)
