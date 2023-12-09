"""Utilities for generating and manipulating matrices."""
from typing import Callable

import numpy as np


def generate_matrix(
    entry_func: Callable,
    shape: int,
    start_index: int = 1,
    doubleinf: bool = False,
) -> np.array:
    """Generate a square matrix from a function f(i, j), where
    f(i, j) is the value of the entry in row i, column j.

    Parameters
    ----------
    entry_func: Callable
        The function which evaluates the matrix at each entry.
    shape: int
        An integer representing the number of rows and
        columns in the output matrix.
    start_index: int
        The number to start indexing matrix entries from. Defaults to 1,
        so the first entry is (1, 1)
    doubleinf: bool, default False
        Whether the matrix should be generated in both directions (i.e. A(0, 0)
        in the centre)

    Returns
    -------
    np.array
        The matrix with entries A_{i,j} = f(i, j).
    """
    if doubleinf:
        shape += 1 - shape % 2
    index_matrix = np.indices((shape, shape), dtype=int) - (shape // 2 * doubleinf) + start_index
    vectorised_func = np.vectorize(entry_func)

    return vectorised_func(*index_matrix)
