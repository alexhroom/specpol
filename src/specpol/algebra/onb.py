"""Orthonormal bases for function spaces."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Tuple


class Domain(Enum):
    """Enum class for various subsets of the real numbers."""

    R = (-float("inf"), float("inf"))
    R_plus = (0, float("inf"))
    Z = (-float("inf"), float("inf"))
    N = (1, float("inf"))
    N_0 = (0, float("inf"))

    def doubleinf(self) -> bool:
        """Return whether the domain is doubly infinite."""
        return self[0] == -float("inf")


class OrthonormalBasis(ABC):
    """
    Class for an orthonormal basis (ONB).
    """

    @abstractmethod
    def __call__(self, n: int) -> Callable:
        """Return the n'th orthonormal basis function."""

        raise NotImplementedError

    @abstractmethod
    def diff(self, j: int, n: int) -> Callable:
        """Return the j'th derivative of the n'th ONB function."""

        raise NotImplementedError

    @property
    @abstractmethod
    def index_set(self) -> Domain:
        """The set over which the ONB is indexed."""

        raise NotImplementedError

    @property
    @abstractmethod
    def domain(self) -> Tuple[float, float]:
        """The domain that the ONB is an ONB for."""

        raise NotImplementedError

    @property
    def bounded(self) -> bool:
        """Whether the domain is bounded."""

        return self.domain[0] > -float("inf") and self.domain[1] < float("inf")

    @property
    def semibounded(self) -> bool:
        """Whether the domain is semibounded."""

        return self.domain[0] > -float("inf") or self.domain[1] < float("inf")
