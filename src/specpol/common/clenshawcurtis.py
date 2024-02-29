"""Truncated Clenshaw-Curtis quadrature for semi-infinite intervals."""
from typing import Callable

import numpy as np


def trunc_cc(func: Callable, tol: float = 1e-15) -> float:
    """
    Calculate the integral of a function on (0, \infty)
    via truncated Clenshaw-Curtis quadrature.

    CITATION:
        A truncated Clenshaw-Curtis formula approximates
        integrals over a semi-inﬁnite interval
        Hiroshi Sugiura, Takemitsu Hasegawa
        Numerical Algorithms (2021) 86:659-674
        DOI 10.1007/s11075-020-00905-w

    Parameters
    ----------
    func: Callable
        The function to integrate.
    tol: float, default 1e-15
        The tolerance of the approximation's accuracy.

    Returns
    -------
    float
        An approximation of the integral of `func`
        from 0 to \infty.
    """

    # calculate truncation endpoint a
    a = -np.log10(tol)
    if abs(func(a)) > tol:
        while abs(func(a)) > tol:
            a += 2
        while abs(func(a)) < tol:
            a -= 0.2
    else:
        while abs(func(a)) > tol:
            a += 0.1
        while abs(func(a)) < tol:
            a -= 1

    epsilon = max(abs(func(a)), tol)

    def error_estimate(mesh, nodes, n):
        """Estimate the error from integration."""
        sum_f_even = sum(mesh[0::2])
        sum_f_odd = sum(mesh[1::2])
        err_est_CC1 = a * abs(sum_f_even - sum_f_odd) / n
        sum_even = mesh[0::2] * nodes[0::2] - mesh[0] * nodes[0] / 2
        sum_odd = mesh[0::2] * nodes[0::2]
        err_est_CC2 = 2 * a * abs(sum_even - sum_odd) / n

        return (err_est_CC1 + err_est_CC2) * 8 * n / ((n**2 - 9) * (n**2 - 1))

    fo = None
    for k in range(2, 11):
        n = 2**k
        weights, nodes = clenshaw_curtis(n * 2)

        nodes = nodes[:n]  # take first n+1 nodes

        if k == 2:
            xo = a * (nodes + 1)
            mesh = np.array([func(x) for x in xo])
        else:
            mesh[0:n:2] = fo
            xe = nodes[1::2]
            xe = a * (xe + 1)
            mesh[1:n:2] = [func(x) for x in xe]

        result = mesh * weights * a

        err_est = error_estimate(mesh, nodes, n)
        if err_est <= epsilon:
            break
        fo = mesh

    return result, err_est


def clenshaw_curtis(n: int):
    """Calculate Clenshaw-Curtis weights and nodes."""
    nodes = np.arange(3, n + 1, 2).T
    v = np.concatenate((np.array([2]), -4 / nodes / (nodes - 2)))
    v[-1] = v[-1] / (2 - n % 2)
    weights = np.fft.ifft(v, n).real
    weights[0] = weights[0] / 2
    weights = np.concatenate((weights, [weights[0]]))
    mesh = np.linspace(0, np.pi, n + 1)
    nodes = [-np.cos(x) for x in mesh]

    return weights, nodes
