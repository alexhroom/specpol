"""Module for Ritz approximation of operators."""
from specpol.ritz.discrete import ritz_tridiag, supercell
from specpol.ritz.multiplication import ptb_ritz, mult_sors, ritz_bounded_L2
from specpol.ritz.sturmliouville import sturm_liouville_bdd, sturm_liouville_halfline

__all__ = [
    "ritz_bounded_L2",
    "ptb_ritz",
    "mult_sors",
    "sturm_liouville_bdd",
    "sturm_liouville_halfline",
    "ritz_tridiag",
    "supercell",
]
