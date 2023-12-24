"""Classes for operator theoretic objects."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Tuple, TypeAlias

OperatorAction: TypeAlias = Callable[[Callable], Callable]


class Operator:
    """A class to represent an operator.

    Parameters
    ----------
    action: OperatorAction (Callable[[Callable], Callable])
        A function-valued function representing the application
        of the operator to
    """

    def __init__(self: "Operator", action: OperatorAction) -> None:
        """Set the action for the operator."""
        self.action = action

    def __call__(self: "Operator", func: Callable) -> Callable:
        """Apply the operator with a function call.

        Parameters
        ----------
        func: Callable
            The Callable to which the operator is applied.
        """
        return self.action(func)


def create_m_op(func: Callable) -> Operator:
    """Instantiate the multiplication operator M_f with action (M_f u)(x) = f(x)u(x).

    Parameters
    ----------
    func: Callable
        The symbol of the multiplication operator.

    Returns
    -------
    Operator
        The multiplication operator with symbol `func`.
    """

    def mult_action(arg: Callable) -> Callable:
        def action(x: Any) -> Any:
            return func(x) * arg(x)

        return action

    return Operator(mult_action)
