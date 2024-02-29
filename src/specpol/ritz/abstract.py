"""Abstract base class & factory for Ritz methods."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from specpol.common import generate_matrix
from specpol.onb import OrthonormalBasis

if TYPE_CHECKING:
    import numpy as np


class Ritz(ABC):
    """
    Handle the approximation of an operator by the Ritz-Galerkin method.
    """

    def __init__(self: "Ritz", onb: OrthonormalBasis):
        """Class to handle Ritz approximation of an operator."""
        self.onb = onb

    @abstractmethod
    def entry_func(self: "Ritz", i: int, j: int) -> complex:
        """
        Calculate an entry of the Ritz matrix.

        Parameters
        ----------
        i: int
            The row of the entry.
        j: int
            The column of the entry.
        """
        raise NotImplementedError

    def ritz_matrix(self: "Ritz", size: int) -> "np.array":
        """Calculate the Ritz matrix of a given size for the operator."""
        return generate_matrix(
            self.entry_func,
            size,
            start_index=(0 if self.onb.index_set.doubleinf else self.onb.index_set[0]),
            doubleinf=self.onb.index_set.doubleinf,
        )
