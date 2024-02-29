"""Plotting of approximations."""
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from specpol.algebra import OrthonormalBasis


def plot_ritz(
    ritz_results: Dict[int, np.array],
    *,
    dbm: int | None = None,
) -> (plt.Figure, plt.Axes, plt.Axes):
    """Plot a Ritz approximation.

    Parameters
    ----------
    ritz_results: Dict[int, np.array]
        A dictionary with keys corresponding to Ritz matrix size, and
        values corresponding to the eigenvalues of the matrix of that size.
    dbm: int or None, default None
        If not None, removes all datapoints with imaginary part smaller
        than `dbm`.

    Returns
    -------
    Figure, Axes, Axes
        Returns the figure and its two subplots for further modification if desired.
    """
    if dbm is not None:
        specs = {
            key: np.array([v for v in ritz_results[key] if v.imag > dbm]) for key in ritz_results
        }
    else:
        specs = ritz_results

    viridis = colormaps["viridis"].resampled(len(specs))

    fig = plt.figure(figsize=(13, 5))

    ax1 = fig.add_subplot(1, 2, 1, adjustable="box")
    ax1.set_prop_cycle(color=viridis.colors)

    for i in specs:
        ax1.scatter([i] * len(specs[i]), specs[i].real, s=8)

    ax1.set_xlabel("size of Ritz matrix (number of rows/columns)")
    ax1.set_ylabel("real part of eigenvalues of the Ritz matrix")

    ax2 = fig.add_subplot(1, 2, 2, adjustable="box")
    ax2.set_prop_cycle(color=viridis.colors)
    ax2.set_xlabel("real part of eigenvalues of the Ritz matrix")
    ax2.set_ylabel("imaginary part of eigenvalues of the Ritz matrix")

    for i in specs:
        ax2.scatter(specs[i].real, specs[i].imag, s=8)

    norm = plt.Normalize(min(specs), max(specs))
    cmap = plt.get_cmap("viridis")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax2, ticks=list(specs.keys()))
    cb.set_label(
        "size of Ritz matrix (number of rows/columns)",
        rotation=270,
        labelpad=15,
    )

    return fig, ax1, ax2


def plot_eigenfunction(func: Callable, val: complex, onb: OrthonormalBasis):
    """Plot an eigenfunction given an orthonormal basis."""
    vector = [func(x) for x in np.linspace]
    plt.plot(vector, label=f"{val}")
    plt.legend()
    plt.show()
