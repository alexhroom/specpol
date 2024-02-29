"""Ritz and supercell methods for discrete operators."""
from copy import copy
from typing import Any, Callable

import numpy as np


def ritz_tridiag(
    subdiag: Callable | complex,
    diag: Callable | complex,
    supradiag: Callable | complex,
    matrix_size: int,
    **settings,
):
    """
    Approximate the spectrum of a tridiagonal matrix operator via the Ritz method.
    """

    def make_const_callable(x: Any):
        """If x is a constant value, make it a callable returning that value; else, leave it alone."""
        if callable(x):
            return x
        return lambda z: x

    subdiag = make_const_callable(subdiag)
    diag = make_const_callable(diag)
    supradiag = make_const_callable(supradiag)

    supradiagonal = np.array([supradiag(n) for n in range(-matrix_size // 2, matrix_size // 2 - 1)])
    diagonal = np.array([diag(n) for n in range(-matrix_size // 2, matrix_size // 2)])
    subdiagonal = np.array([subdiag(n) for n in range(-matrix_size // 2, matrix_size // 2 - 1)])
    ritz_matrix = np.diag(diagonal) + np.diag(supradiagonal, 1) + np.diag(subdiagonal, -1)

    if settings.get("matrix", False):
        return ritz_matrix

    eigenvalues, eigenvectors = np.linalg.eig(ritz_matrix)

    if settings.get("vectors", False):
        return eigenvalues, eigenvectors
    return eigenvalues


def supercell(
    subdiag: Callable | complex,
    diag: Callable | complex,
    supradiag: Callable | complex,
    matrix_size: int,
    alpha_samples: int = 50,
    **settings,
):
    """
    Approximate the spectrum of a tridiagonal matrix operator via the supercell method.
    """

    def make_const_callable(x: Any):
        """If x is a constant value, make it a callable returning that value; else, leave it alone."""
        if callable(x):
            return x
        return lambda z: x

    subdiag = make_const_callable(subdiag)
    diag = make_const_callable(diag)
    supradiag = make_const_callable(supradiag)

    supradiagonal = np.array([supradiag(n) for n in range(-matrix_size // 2, matrix_size // 2 - 1)])
    diagonal = np.array([diag(n) for n in range(-matrix_size // 2, matrix_size // 2)])
    subdiagonal = np.array([subdiag(n) for n in range(-matrix_size // 2, matrix_size // 2 - 1)])
    ritz_matrix = np.diag(diagonal) + np.diag(supradiagonal, 1) + np.diag(subdiagonal, -1)

    # add periodic entries
    def supercell_mat(theta):
        alpha = np.exp(1j * theta)
        supercell_matrix = copy(ritz_matrix)
        supercell_matrix[-1][0] = 1 / alpha
        supercell_matrix[0][-1] = subdiag(matrix_size // 2) * alpha

        return supercell_matrix

    eigenvalues = {}
    eigenvectors = {}
    for theta in np.linspace(0, 2 * np.pi, alpha_samples):
        eigenvalues[theta], eigenvectors[theta] = np.linalg.eig(supercell_mat(theta))

    if settings.get("vectors", False):
        return eigenvalues, eigenvectors
    return eigenvalues
