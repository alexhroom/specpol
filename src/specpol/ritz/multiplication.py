"""Ritz method for various types of operator."""
from functools import lru_cache
from typing import Callable

import numpy as np
from pyfilon import filon_fun_iexp

from specpol.common import generate_matrix
from specpol.operators import Operator


# pylint: disable=invalid-name
def ritz_bounded_L2(
    operator: Operator,
    b: float,
    matrix_size: int,
    quad_mesh_size: int,
) -> np.array:
    """Ritz approximation on L2(0, b).

    Parameters
    ----------
    operator: Operator
        The operator to approximate the spectrum of.
    b: float
        The lower and upper limits of the domain, respectively.
    matrix_size: int
        The size of the square Ritz matrix.
    quad_mesh_size: int
        The size of the mesh used for quadrature.
        Must be an odd integer greater than 1.

    Returns
    -------
    np.array
        A one-dimensional array listing the estimated spectral points of the operator.
    """

    # we have a factor of 2 in the integrand to account for the
    # sqrt(2) attached to each sin(n pi x)
    def onb_func(n: int) -> Callable:
        return lambda x: 1 / np.sqrt(b) * np.exp(2j * np.pi * n * x / b)

    @lru_cache(maxsize=matrix_size)
    def integrand(n: int) -> Callable:
        return lambda x: operator(onb_func(n))(x)

    def entry_func(i: int, j: int) -> complex:
        """Calculate entry i,j of the Ritz matrix for the multiplication operator.

        These are the scalar products (M_f e_i, e_j)
        where e_k = exp(2*i*pi*k*x)
        """
        # the Filon quadrature has the second iexp as implicit
        return (
            1 / np.sqrt(b) * filon_fun_iexp(integrand(i), 0, b, -2 * j * np.pi / b, quad_mesh_size)
        )

    ritz_matrix = generate_matrix(
        entry_func,
        matrix_size,
        start_index=0,
        doubleinf=True,
    )

    return np.linalg.eigvals(ritz_matrix)


def ptb_ritz(
    func: Callable,
    matrix_size: int,
    quad_mesh_size: int,
    *,
    dbm: bool,
) -> np.array:
    r"""Approximate the spectrum of a perturbed multiplication operator on L^2(0, 1).

    This is a multiplication operator with rank-one perturbation (\phi, u)*\phi.

    Parameters
    ----------
    func: Callable
        The symbol f of the operator M_f.
    matrix_size: int
        The size of the square Ritz matrix.
    quad_mesh_size: int
        The size of the mesh used for quadrature.
        Must be an odd integer greater than 1.
    dbm: bool
        Whether to add a dissipative barrier to show that the eigenvalue
        created by the perturbation is indeed a real eigenvalue.

    Returns
    -------
    np.array
        A one-dimensional array listing the estimated spectral points of the operator.
    """
    # TODO: GENERALISE
    # currently only works for lambda = 0.7

    # phi can be any function which satisfies the normalisation condition.
    # we just choose the constant function!
    phi_squared = 1 / (np.log(3) - 3 * np.log(2) + np.log(5) + np.log(7) - np.log(10))

    # we have a factor of 2 in the integrand to account for the
    # sqrt(2) attached to each sin(n pi x)
    # the 1j term is a dissipative barrier
    @lru_cache(maxsize=matrix_size)
    def integrand(n: int) -> Callable:
        if n == 0:  # need to separate out this case to avoid a divide by 0
            return lambda x: func(x) + phi_squared + 1j * (dbm is True and abs(n) <= 25)
        return lambda x: (
            func(x) * np.exp(2j * n * np.pi * x)
            + -1j * (-1 + np.exp(2j * n * np.pi)) / (2 * np.pi * n) * phi_squared
            + 1j * (dbm is True and abs(n) <= 25) * np.exp(2j * n * np.pi * x)
        )

    def entry_func(i: int, j: int) -> complex:
        """Calculate entry i,j of the Ritz matrix for the multiplication operator.

        These are the scalar products (M_f e_i, e_j)
        where e_k = sin(k*pi*x).
        """
        # the Filon quadrature has the second sin as implicit
        return filon_fun_iexp(integrand(i), 0, 1, -2 * j * np.pi, quad_mesh_size)

    ritz_matrix = generate_matrix(
        entry_func,
        matrix_size,
        start_index=0,
        doubleinf=True,
    )

    return np.linalg.eigvals(ritz_matrix)
