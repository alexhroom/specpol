"""Dataclasses."""
from typing import Callable

import numpy as np


class Eigenpairs:
    """Dataclass for eigenvalues and vectors."""

    def __init__(self, vals: np.array, vecs: np.array):
        """Initialise an Eigenpairs dataclass from eigenvalues and vectors."""
        self.data = {val: vec for val, vec in zip(vals, vecs.T)}

    def _init_from_dict(self, pairs: dict) -> "Eigenpairs":
        """Create an Eigenpairs object with the dict as its data."""
        new_obj = Eigenpairs([], np.empty(1))
        new_obj.data = pairs
        return new_obj

    def filter(self, predicate: Callable[[complex], bool]) -> dict:  # noqa: A003
        """Filter eigenpairs by a predicate on the eigenvalues."""
        return self._init_from_dict({val: vec for val, vec in self.data.items() if predicate(val)})

    def plot_vector(self, key: complex, onb: Callable) -> None:
        """Plot an eigenvector, given its value and an orthonormal basis."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Show data when converted to string."""
        return str(self.data)

    def __getitem__(self, key: complex) -> np.array:
        """Get a vector from its eigenvalue."""
        return self.data[key]

