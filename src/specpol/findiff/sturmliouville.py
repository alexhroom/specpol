"""Calculate the spectrum of a Sturm-Liouville operator via finite differences."""
from typing import Callable

import numpy as np


def sturm_liouville(
    potential: Callable,
    discret_const: float,
    alpha: float,
    matrix_size: int,
    **settings,
) -> np.array:
    r"""Approximate the spectrum of a Sturm-Liouville problem with a finite-difference scheme.

    The problem is a boundary value problem on $[0, \infty)$ of the form

    \[
    \begin{equation}
    -y'' + qy = \lambda y,
    cos \alpha y(0) + sin \alpha y'(0) = 0.
    \end{equation}
    \]


    Parameters
    ----------
    potential: Callable
        The potential function q for the Sturm-Liouville problem.
    discret_const: float
        The value of the discretisation constant, $h$ or $\Delta x$.
    alpha: float
        The value of alpha in the boundary condition.
    matrix_size: int
        The size of the tridiagonal matrix created by the finite-difference scheme.
    settings:
        dbm: Callable, default 0
            Add a dissipative barrier to the operator; this is a function
            of the form $i*\gamma*f(x)$ where f is a function with compact
            support.
        vectors: bool, default False
            Whether to return the eigenvectors as well as the eigenvalues.
        matrix: bool, default False
            If true, returns the finite difference matrix instead of the
            eigenvalues and vectors.

    Returns
    -------
    np.array
        An array of eigenvalues for the Sturm-Liouville problem.
    """
    h = discret_const

    potential_mesh = np.array([potential(x * h) for x in range(0, matrix_size)])
    dissipative_barrier = np.array(
        [settings.get("dbm", lambda x: 0)(x * h) for x in range(0, matrix_size)]
    )

    # the matrix will be symmetrised
    potential_mesh[0] /= 2
    off_diagonal = np.repeat([-1 / h**2], matrix_size - 1)
    diagonal = (
        np.array([1 / (2 * h**2) * 1 / np.tan(alpha) + 1] + [2 / h**2] * (matrix_size - 1))
        + potential_mesh
        + dissipative_barrier
    )
    scheme_matrix = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)

    if settings.get("matrix", False):
        return scheme_matrix

    eigenvalues, eigenvectors = np.linalg.eig(scheme_matrix)

    if settings.get("vectors", False):
        return eigenvalues, eigenvectors
    return eigenvalues
