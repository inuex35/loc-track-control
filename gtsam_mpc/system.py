"""Discrete-time linear time-invariant systems for MPC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearSystem:
    """A discrete-time linear time-invariant system ``x_{k+1} = A x_k + B u_k``.

    Attributes:
        A: State transition matrix, shape ``(n, n)``.
        B: Control input matrix, shape ``(n, m)``.
    """

    A: np.ndarray
    B: np.ndarray

    def __post_init__(self) -> None:
        self.A = np.atleast_2d(np.asarray(self.A, dtype=float))
        self.B = np.atleast_2d(np.asarray(self.B, dtype=float))
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A must be square, got {self.A.shape}")
        if self.B.shape[0] != self.A.shape[0]:
            raise ValueError(
                f"B must have {self.A.shape[0]} rows to match A, got {self.B.shape}"
            )

    @property
    def n_states(self) -> int:
        """Number of state variables ``n``."""
        return self.A.shape[0]

    @property
    def n_controls(self) -> int:
        """Number of control inputs ``m``."""
        return self.B.shape[1]

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Propagate one step: ``x_{k+1} = A x_k + B u_k``."""
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        return self.A @ x + self.B @ u


def double_integrator(dt: float = 0.1) -> LinearSystem:
    """A 1-D double integrator discretized at timestep ``dt``.

    State is ``[position, velocity]`` and the control is acceleration.
    Discretized with exact zero-order hold:

        A = [[1, dt], [0, 1]],   B = [[dt^2 / 2], [dt]]
    """
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt * dt], [dt]])
    return LinearSystem(A, B)
