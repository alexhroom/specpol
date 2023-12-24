"""Module for Ritz approximation of operators."""
from specpol.ritz.multiplication import ptb_ritz, ritz_bounded_L2
from specpol.ritz.sturmliouville import sturm_liouville_bdd, sturm_liouville_halfline

__all__ = [
    "ritz_bounded_L2",
    "ptb_ritz",
    "sturm_liouville_bdd",
    "sturm_liouville_halfline",
]
