"""Dynamics models used by the estimators and controllers.

Two linear systems (a 1-D and a 2-D double integrator) for the linear MPC, and
a nonlinear kinematic bicycle model (RK4-integrated, with analytic Jacobians)
for the nonlinear MPC and the localization demos.
"""

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
    """
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt * dt], [dt]])
    return LinearSystem(A, B)


def point_mass_2d(dt: float = 0.1) -> LinearSystem:
    """A 2-D point mass (decoupled double integrators) at timestep ``dt``.

    State is ``[px, py, vx, vy]`` and the control is acceleration ``[ax, ay]``.
    A goal ``[px, py, 0, 0]`` (any position, zero velocity) is an equilibrium.
    """
    A = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    half_dt2 = 0.5 * dt * dt
    B = np.array(
        [
            [half_dt2, 0.0],
            [0.0, half_dt2],
            [dt, 0.0],
            [0.0, dt],
        ]
    )
    return LinearSystem(A, B)


@dataclass
class BicycleModel:
    """Discrete-time kinematic bicycle model (classic RK4 integration).

    The continuous dynamics ``xdot = f(x, u)`` are integrated over ``dt`` with
    the 4th-order Runge-Kutta rule. RK4 is ``O(dt^5)`` accurate per step versus
    the ``O(dt^2)`` of forward Euler, which removes the systematic discretization
    bias Euler introduces on curved motion (e.g. a steady cross-track offset
    when tracking a circle), letting the MPC use a coarse prediction grid.

    For extra accuracy on a coarse grid the ``dt`` interval can be split into
    ``substeps`` equal RK4 sub-integrations (as acados does with
    ``sim_method_num_steps``); the control ``u`` is held constant across them.

    State is ``[x, y, theta, v]`` and control is ``[a, delta]``.

    Attributes:
        wheelbase: Distance between front and rear axles ``L`` [m].
        dt: Integration timestep [s].
        substeps: Number of equal RK4 sub-integrations per :meth:`step`.
    """

    wheelbase: float = 2.5
    dt: float = 0.1
    substeps: int = 1

    n_states: int = 4
    n_controls: int = 2

    def _f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous-time dynamics ``xdot = f(x, u)``."""
        _, _, theta, v = x
        a, delta = u
        L = self.wheelbase
        return np.array(
            [v * np.cos(theta), v * np.sin(theta), (v / L) * np.tan(delta), a]
        )

    def _f_jac(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Continuous Jacobians ``(df/dx, df/du)`` of :meth:`_f`."""
        _, _, theta, v = x
        _, delta = u
        L = self.wheelbase
        c, s = np.cos(theta), np.sin(theta)
        Ac = np.array(
            [
                [0.0, 0.0, -v * s, c],
                [0.0, 0.0, v * c, s],
                [0.0, 0.0, 0.0, np.tan(delta) / L],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        Bc = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, v / (L * np.cos(delta) ** 2)],
                [1.0, 0.0],
            ]
        )
        return Ac, Bc

    def _rk4(self, x: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
        """One classic RK4 sub-integration of size ``h``."""
        k1 = self._f(x, u)
        k2 = self._f(x + 0.5 * h * k1, u)
        k3 = self._f(x + 0.5 * h * k2, u)
        k4 = self._f(x + h * k3, u)
        return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _rk4_jac(
        self, x: np.ndarray, u: np.ndarray, h: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact ``(d/dx, d/du)`` of one RK4 sub-integration :meth:`_rk4`.

        Obtained by propagating the continuous Jacobians through the four RK4
        stages by the chain rule.
        """
        eye = np.eye(4)
        k1 = self._f(x, u)
        s2 = x + 0.5 * h * k1
        k2 = self._f(s2, u)
        s3 = x + 0.5 * h * k2
        k3 = self._f(s3, u)
        s4 = x + h * k3

        A1, B1 = self._f_jac(x, u)
        A2, B2 = self._f_jac(s2, u)
        A3, B3 = self._f_jac(s3, u)
        A4, B4 = self._f_jac(s4, u)

        # d k_i / dx, chained through each stage's state dependence on x.
        Dk1 = A1
        Dk2 = A2 @ (eye + 0.5 * h * Dk1)
        Dk3 = A3 @ (eye + 0.5 * h * Dk2)
        Dk4 = A4 @ (eye + h * Dk3)
        A = eye + (h / 6.0) * (Dk1 + 2.0 * Dk2 + 2.0 * Dk3 + Dk4)

        # d k_i / du, with each stage's state also depending on u via prior k's.
        Ek1 = B1
        Ek2 = A2 @ (0.5 * h * Ek1) + B2
        Ek3 = A3 @ (0.5 * h * Ek2) + B3
        Ek4 = A4 @ (h * Ek3) + B4
        B = (h / 6.0) * (Ek1 + 2.0 * Ek2 + 2.0 * Ek3 + Ek4)

        return A, B

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Propagate one ``dt`` with RK4, split into ``substeps`` sub-steps."""
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            x = self._rk4(x, u, h)
        return x

    def jacobians(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(df/dx, df/du)`` of the full :meth:`step` at ``(x, u)``.

        The per-sub-step Jacobians are composed across ``substeps`` by the chain
        rule, so they match a finite difference of :meth:`step` to machine
        precision regardless of how many sub-steps are used.
        """
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        h = self.dt / self.substeps
        Jx = np.eye(4)
        Ju = np.zeros((4, 2))
        for _ in range(self.substeps):
            A, B = self._rk4_jac(x, u, h)
            Ju = A @ Ju + B
            Jx = A @ Jx
            x = self._rk4(x, u, h)
        return Jx, Ju
