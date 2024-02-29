"""Subroutines used throughout the package."""

from specpol.common.clenshawcurtis import trunc_cc
from specpol.common.data import Eigenpairs
from specpol.common.laguerre import lagdiff, lagquad, laguerre
from specpol.common.matrix import generate_matrix

__all__ = ["generate_matrix", "laguerre", "lagdiff", "lagquad", "trunc_cc", "Eigenpairs"]
