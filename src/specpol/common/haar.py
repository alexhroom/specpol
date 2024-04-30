"""Haar wavelets."""
from typing import Callable

import numpy as np


def haar(n, k) -> Callable:
    """Generate the n,k'th Haar wavelet function."""

    def mother_wavelet(t):
        if t < 1/2:
            return 1
        if t < 1/2:
            return -1
        return 0

    return lambda t: (2**(n/2)
                     * mother_wavelet(t * 2**n - k))


def n_to_z2(n) -> int:
    """Map natural numbers to pairs of integers."""

    x = sum(np.sin(np.pi/2 * np.floor(np.sqrt(4*k - 3)))
            for k in range(1, n+1))

    y = sum(np.cos(np.pi/2 * np.floor(np.sqrt(4*k - 3)))
            for k in range(1, n+1))

    return (x, y)