"""Linear MPC formulated as inference on a Gaussian factor graph.

A finite-horizon linear-quadratic optimal control problem

    minimize   sum_{k=0}^{N-1} (x_k^T Q x_k + u_k^T R u_k) + x_N^T Qf x_N
    subject to x_0 = x_init
               x_{k+1} = A x_k + B u_k

is exactly equivalent to finding the maximum-a-posteriori (MAP) estimate of a
Gaussian factor graph (Dellaert & Kaess, *Factor Graphs for Robot
Perception*). We build that graph with GTSAM and let GTSAM's linear solver
recover the optimal state and control trajectory. Eliminating the graph is
algebraically the same computation as the Riccati backward recursion.

Encoding:
    * The initial condition ``x_0 = x_init`` and each dynamics relation
      ``x_{k+1} - A x_k - B u_k = 0`` are *hard* equality constraints
      (``noiseModel.Constrained``).
    * Each running state cost ``x_k^T Q x_k`` and control cost ``u_k^T R u_k``
      is a quadratic factor whose information (inverse covariance) matrix is
      ``Q`` and ``R`` respectively. The terminal cost uses ``Qf``.
"""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.symbol_shorthand import U, X

from .system import LinearSystem


@dataclass
class MPCResult:
    """Optimal open-loop trajectory returned by :meth:`LinearMPC.solve`.

    Attributes:
        states: Optimal states, shape ``(horizon + 1, n_states)``.
        controls: Optimal controls, shape ``(horizon, n_controls)``.
    """

    states: np.ndarray
    controls: np.ndarray

    @property
    def u0(self) -> np.ndarray:
        """The first control input -- the one applied in receding-horizon MPC."""
        return self.controls[0]


def _require_pd(name: str, M: np.ndarray, dim: int) -> np.ndarray:
    """Validate that ``M`` is a symmetric positive-definite ``dim x dim`` matrix."""
    M = np.atleast_2d(np.asarray(M, dtype=float))
    if M.shape != (dim, dim):
        raise ValueError(f"{name} must have shape {(dim, dim)}, got {M.shape}")
    if not np.allclose(M, M.T):
        raise ValueError(f"{name} must be symmetric")
    # GTSAM's Gaussian information noise model requires a Cholesky factor, so the
    # weight matrices must be strictly positive definite.
    if np.any(np.linalg.eigvalsh(M) <= 0):
        raise ValueError(f"{name} must be positive definite")
    return M


class LinearMPC:
    """Finite-horizon linear MPC solved as a Gaussian factor graph.

    Args:
        system: The discrete-time linear system to control.
        Q: Running state cost matrix, shape ``(n, n)``, positive definite.
        R: Running control cost matrix, shape ``(m, m)``, positive definite.
        horizon: Prediction horizon ``N`` (number of control steps).
        Qf: Terminal state cost matrix, shape ``(n, n)``. Defaults to ``Q``.
    """

    def __init__(
        self,
        system: LinearSystem,
        Q: np.ndarray,
        R: np.ndarray,
        horizon: int,
        Qf: np.ndarray | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        n, m = system.n_states, system.n_controls
        self.system = system
        self.horizon = int(horizon)
        self.Q = _require_pd("Q", Q, n)
        self.R = _require_pd("R", R, m)
        self.Qf = self.Q if Qf is None else _require_pd("Qf", Qf, n)

    def _build_graph(self, x0: np.ndarray) -> gtsam.GaussianFactorGraph:
        n, m = self.system.n_states, self.system.n_controls
        A, B = self.system.A, self.system.B
        N = self.horizon

        cost_x = gtsam.noiseModel.Gaussian.Information(self.Q)
        cost_u = gtsam.noiseModel.Gaussian.Information(self.R)
        cost_xN = gtsam.noiseModel.Gaussian.Information(self.Qf)
        constraint_x = gtsam.noiseModel.Constrained.All(n)

        graph = gtsam.GaussianFactorGraph()

        # Hard constraint: x_0 = x0.
        graph.add(X(0), np.eye(n), np.asarray(x0, dtype=float), constraint_x)

        for k in range(N):
            # Hard constraint: x_{k+1} - A x_k - B u_k = 0.
            graph.add(
                X(k + 1), np.eye(n),
                X(k), -A,
                U(k), -B,
                np.zeros(n), constraint_x,
            )
            # Running costs pull states and controls toward the origin.
            graph.add(X(k), np.eye(n), np.zeros(n), cost_x)
            graph.add(U(k), np.eye(m), np.zeros(m), cost_u)

        # Terminal state cost.
        graph.add(X(N), np.eye(n), np.zeros(n), cost_xN)
        return graph

    def _reference(self, xref: np.ndarray | None) -> np.ndarray:
        n = self.system.n_states
        if xref is None:
            return np.zeros(n)
        return np.asarray(xref, dtype=float).reshape(n)

    def solve(self, x0: np.ndarray, xref: np.ndarray | None = None) -> MPCResult:
        """Solve the open-loop optimal-control problem from initial state ``x0``.

        Args:
            x0: Current state, shape ``(n,)``.
            xref: Target state to regulate toward, shape ``(n,)``. Defaults to
                the origin. The cost is applied to the tracking error
                ``x - xref``; tracking is exact when ``xref`` is an equilibrium
                of the system (``A xref = xref``), e.g. a zero-velocity goal for
                a double integrator.
        """
        n, m = self.system.n_states, self.system.n_controls
        x0 = np.asarray(x0, dtype=float).reshape(n)
        xref = self._reference(xref)

        # Regulate the error state e = x - xref toward zero, then shift back.
        solution = self._build_graph(x0 - xref).optimize()

        states = np.array(
            [solution.at(X(k)) + xref for k in range(self.horizon + 1)]
        )
        controls = np.array([solution.at(U(k)) for k in range(self.horizon)])
        return MPCResult(states=states, controls=controls.reshape(self.horizon, m))

    def control(self, x0: np.ndarray, xref: np.ndarray | None = None) -> np.ndarray:
        """Return the first optimal control for ``x0`` (one receding-horizon step)."""
        return self.solve(x0, xref).u0

    def simulate(
        self, x0: np.ndarray, steps: int, xref: np.ndarray | None = None
    ) -> MPCResult:
        """Run the receding-horizon controller in closed loop for ``steps`` steps.

        At each step the full horizon is re-optimized from the current state and
        only the first control is applied to the (true) system dynamics.

        Args:
            x0: Initial state.
            steps: Number of closed-loop steps to simulate.
            xref: Target state to track (see :meth:`solve`).

        Returns:
            An :class:`MPCResult` with the realized closed-loop ``states``
            (shape ``(steps + 1, n)``) and applied ``controls`` (shape
            ``(steps, m)``).
        """
        n, m = self.system.n_states, self.system.n_controls
        x = np.asarray(x0, dtype=float).reshape(n)

        states = [x.copy()]
        controls = []
        for _ in range(steps):
            u = self.control(x, xref)
            x = self.system.step(x, u)
            controls.append(u)
            states.append(x.copy())
        return MPCResult(
            states=np.array(states),
            controls=np.array(controls).reshape(steps, m),
        )
