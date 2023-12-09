"""Module for Ritz approximation of operators."""
from specpol.ritz.multiplication import ptb_ritz, ritz_bounded_L2
from specpol.ritz.sturmliouville import (ritz_sturm_liouville,
                                         ritz_unbounded_sturm_liouville)

__all__ = [
    "ritz_bounded_L2",
    "ptb_ritz",
    "ritz_sturm_liouville",
    "ritz_unbounded_sturm_liouville",
]
