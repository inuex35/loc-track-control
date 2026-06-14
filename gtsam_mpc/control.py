"""Model Predictive Control solvers, as factor-graph optimization.

* :class:`LinearMPC` -- finite-horizon linear-quadratic MPC. The whole problem
  is a *linear* Gaussian factor graph whose MAP estimate is the optimal
  trajectory (eliminating it is the Riccati recursion).
* :class:`BicycleMPC` -- nonlinear MPC for the kinematic bicycle. The graph is
  assembled from the shared :mod:`gtsam_mpc.factors` vocabulary; input/speed/
  obstacle inequalities are handled by a pluggable
  :mod:`gtsam_mpc.constraints` strategy (barrier / AL / slack).
"""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.symbol_shorthand import U, X

from . import factors
from .constraints import ConstraintStrategy, Inequality, make_strategy
from .models import BicycleModel, LinearSystem


@dataclass
class MPCResult:
    """Optimal open-loop trajectory returned by an MPC ``solve``.

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


def require_pd(name: str, M: np.ndarray, dim: int) -> np.ndarray:
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

    def __init__(self, system: LinearSystem, Q: np.ndarray, R: np.ndarray,
                 horizon: int, Qf: np.ndarray | None = None) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        n, m = system.n_states, system.n_controls
        self.system = system
        self.horizon = int(horizon)
        self.Q = require_pd("Q", Q, n)
        self.R = require_pd("R", R, m)
        self.Qf = self.Q if Qf is None else require_pd("Qf", Qf, n)

    def _build_graph(self, x0: np.ndarray) -> gtsam.GaussianFactorGraph:
        n, m = self.system.n_states, self.system.n_controls
        A, B = self.system.A, self.system.B
        N = self.horizon

        cost_x = gtsam.noiseModel.Gaussian.Information(self.Q)
        cost_u = gtsam.noiseModel.Gaussian.Information(self.R)
        cost_xN = gtsam.noiseModel.Gaussian.Information(self.Qf)
        constraint_x = gtsam.noiseModel.Constrained.All(n)

        graph = gtsam.GaussianFactorGraph()
        graph.add(X(0), np.eye(n), np.asarray(x0, dtype=float), constraint_x)
        for k in range(N):
            graph.add(X(k + 1), np.eye(n), X(k), -A, U(k), -B,
                      np.zeros(n), constraint_x)
            graph.add(X(k), np.eye(n), np.zeros(n), cost_x)
            graph.add(U(k), np.eye(m), np.zeros(m), cost_u)
        graph.add(X(N), np.eye(n), np.zeros(n), cost_xN)
        return graph

    def _reference(self, xref: np.ndarray | None) -> np.ndarray:
        n = self.system.n_states
        if xref is None:
            return np.zeros(n)
        return np.asarray(xref, dtype=float).reshape(n)

    def solve(self, x0: np.ndarray, xref: np.ndarray | None = None) -> MPCResult:
        """Solve the open-loop optimal-control problem from initial state ``x0``.

        ``xref`` is a target state to regulate toward (default origin); the cost
        is on the error ``x - xref``, exact when ``xref`` is an equilibrium.
        """
        n, m = self.system.n_states, self.system.n_controls
        x0 = np.asarray(x0, dtype=float).reshape(n)
        xref = self._reference(xref)

        solution = self._build_graph(x0 - xref).optimize()

        states = np.array([solution.at(X(k)) + xref for k in range(self.horizon + 1)])
        controls = np.array([solution.at(U(k)) for k in range(self.horizon)])
        return MPCResult(states=states, controls=controls.reshape(self.horizon, m))

    def control(self, x0: np.ndarray, xref: np.ndarray | None = None) -> np.ndarray:
        """Return the first optimal control for ``x0`` (one receding-horizon step)."""
        return self.solve(x0, xref).u0

    def simulate(self, x0: np.ndarray, steps: int,
                 xref: np.ndarray | None = None) -> MPCResult:
        """Receding-horizon closed loop for ``steps`` steps (re-solve, apply u0)."""
        n, m = self.system.n_states, self.system.n_controls
        x = np.asarray(x0, dtype=float).reshape(n)
        states = [x.copy()]
        controls = []
        for _ in range(steps):
            u = self.control(x, xref)
            x = self.system.step(x, u)
            controls.append(u)
            states.append(x.copy())
        return MPCResult(states=np.array(states),
                         controls=np.array(controls).reshape(steps, m))


class BicycleMPC:
    """Nonlinear receding-horizon MPC for a :class:`BicycleModel`.

    The base graph (hard dynamics + initial condition + quadratic tracking and
    control costs) is built from :mod:`gtsam_mpc.factors`. Input limits, speed
    limits and obstacle keep-outs are inequalities handled by a
    :class:`~gtsam_mpc.constraints.ConstraintStrategy`.

    Args:
        model: The bicycle model to control.
        Q, R, Qf: Running state, control and terminal cost matrices (PD). The
            heading component of the state cost uses the wrapped error.
        horizon: Prediction horizon ``N``.
        a_bounds, delta_bounds: ``(min, max)`` input limits.
        v_bounds: Optional ``(min, max)`` speed limits.
        constraints: A :class:`ConstraintStrategy`. If ``None`` one is built from
            ``constraint_mode`` and the ``barrier_weight`` / ``al_*`` knobs.
        constraint_mode: ``"barrier"`` (default), ``"al"`` or ``"slack"``.
        safety_radius: Keep-out radius around each obstacle.
        obstacle_weight: Barrier precision for keep-out (barrier mode).
        max_iterations: Max Levenberg-Marquardt iterations per (inner) solve.
    """

    def __init__(
        self,
        model: BicycleModel,
        Q: np.ndarray,
        R: np.ndarray,
        horizon: int,
        Qf: np.ndarray | None = None,
        a_bounds: tuple[float, float] = (-3.0, 3.0),
        delta_bounds: tuple[float, float] = (-0.5, 0.5),
        v_bounds: tuple[float, float] | None = None,
        constraints: ConstraintStrategy | None = None,
        constraint_mode: str = "barrier",
        barrier_weight: float = 300.0,
        safety_radius: float = 2.0,
        obstacle_weight: float = 150.0,
        al_iterations: int = 5,
        al_rho: float = 100.0,
        al_rho_scale: float = 10.0,
        al_rho_max: float = 1e6,
        al_tol: float = 1e-3,
        max_iterations: int = 100,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.model = model
        self.horizon = int(horizon)
        self.Q = require_pd("Q", Q, 4)
        self.R = require_pd("R", R, 2)
        self.Qf = self.Q if Qf is None else require_pd("Qf", Qf, 4)

        self.u_lower = np.array([a_bounds[0], delta_bounds[0]])
        self.u_upper = np.array([a_bounds[1], delta_bounds[1]])
        self.v_bounds = v_bounds
        self.safety_radius = float(safety_radius)
        self.obstacle_weight = float(obstacle_weight)
        # ``make_strategy`` validates the mode name.
        self.strategy = constraints if constraints is not None else make_strategy(
            constraint_mode, barrier_weight=barrier_weight,
            al_iterations=al_iterations, al_rho=al_rho, al_rho_scale=al_rho_scale,
            al_rho_max=al_rho_max, al_tol=al_tol)

        self._params = gtsam.LevenbergMarquardtParams()
        self._params.setMaxIterations(int(max_iterations))
        self._params.setRelativeErrorTol(1e-8)
        self._params.setAbsoluteErrorTol(1e-8)

        self._last_solution: gtsam.Values | None = None

    # --- normalization ---------------------------------------------------
    def _normalize_xref(self, xref) -> np.ndarray:
        arr = np.asarray(xref, dtype=float)
        if arr.shape == (4,):
            return np.tile(arr, (self.horizon + 1, 1))
        if arr.shape == (self.horizon + 1, 4):
            return arr
        raise ValueError(
            f"xref must be shape (4,) or ({self.horizon + 1}, 4), got {arr.shape}")

    def _normalize_obstacles(self, obstacles) -> list[np.ndarray]:
        out = []
        for obs in obstacles:
            arr = np.asarray(obs, dtype=float)
            if arr.shape == (2,):
                arr = np.tile(arr, (self.horizon + 1, 1))
            elif arr.shape != (self.horizon + 1, 2):
                raise ValueError(
                    f"obstacle must be shape (2,) or ({self.horizon + 1}, 2), "
                    f"got {arr.shape}")
            out.append(arr)
        return out

    # --- graph + inequalities --------------------------------------------
    def _base_graph(self, x0: np.ndarray, xref: np.ndarray) -> gtsam.NonlinearFactorGraph:
        N = self.model.n_states
        cost_x = factors.information(self.Q)
        cost_u = factors.information(self.R)
        cost_xf = factors.information(self.Qf)
        hard = factors.hard(4)

        graph = gtsam.NonlinearFactorGraph()
        graph.add(factors.prior(X(0), x0, hard))
        for k in range(self.horizon):
            graph.add(factors.dynamics(self.model, X(k), U(k), X(k + 1), hard))
            graph.add(factors.state_cost(X(k), xref[k], cost_x))
            graph.add(factors.zero_cost(U(k), 2, cost_u))
        graph.add(factors.state_cost(X(self.horizon), xref[self.horizon], cost_xf))
        return graph

    def _inequalities(self, obstacles: list[np.ndarray] | None) -> list[Inequality]:
        N = self.horizon
        u_lower, u_upper = self.u_lower, self.u_upper
        u_jac = np.vstack([np.eye(2), -np.eye(2)])
        specs: list[Inequality] = []

        def u_eval(u):
            return np.concatenate([u - u_upper, u_lower - u]), u_jac

        for k in range(N):
            specs.append(Inequality(U(k), 4, u_eval))

        if self.v_bounds is not None:
            vlo, vhi = self.v_bounds
            v_jac = np.zeros((2, 4))
            v_jac[0, 3], v_jac[1, 3] = 1.0, -1.0

            def v_eval(x):
                return np.array([x[3] - vhi, vlo - x[3]]), v_jac

            for k in range(1, N + 1):
                specs.append(Inequality(X(k), 2, v_eval))

        if obstacles:
            radius = self.safety_radius

            def make_obs_eval(p):
                def ev(x):
                    d = x[:2] - p
                    dist = float(np.hypot(d[0], d[1]))
                    J = np.zeros((1, 4))
                    if dist > 1e-6:
                        J[0, 0] = -d[0] / dist
                        J[0, 1] = -d[1] / dist
                    return np.array([radius - dist]), J
                return ev

            for obs in obstacles:
                for k in range(N + 1):
                    specs.append(Inequality(X(k), 1, make_obs_eval(obs[k]),
                                            weight=self.obstacle_weight))
        return specs

    def _rollout_init(self, x0: np.ndarray) -> gtsam.Values:
        values = gtsam.Values()
        x = x0.copy()
        values.insert(X(0), x)
        for k in range(self.horizon):
            u = np.zeros(2)
            values.insert(U(k), u)
            x = self.model.step(x, u)
            values.insert(X(k + 1), x)
        return values

    # --- public API ------------------------------------------------------
    def solve(self, x0: np.ndarray, xref: np.ndarray, obstacles=None,
              warm_start: bool = True) -> MPCResult:
        """Optimize the trajectory from ``x0`` toward the reference ``xref``.

        Args:
            x0: Current state ``[x, y, theta, v]``.
            xref: A single target state ``(4,)`` (set-point) or a full reference
                trajectory ``(horizon + 1, 4)`` (path tracking).
            obstacles: Optional iterable of obstacles to avoid, each a static
                centre ``(2,)`` or a per-step prediction ``(horizon + 1, 2)``.
            warm_start: Reuse the previous solution as the initial guess.
        """
        x0 = np.asarray(x0, dtype=float).reshape(4)
        xref = self._normalize_xref(xref)
        obs = self._normalize_obstacles(obstacles) if obstacles else None

        if warm_start and self._last_solution is not None:
            initial = self._last_solution
        else:
            initial = self._rollout_init(x0)

        inequalities = self._inequalities(obs)
        result = self.strategy.solve(
            lambda: self._base_graph(x0, xref), inequalities, initial, self._params)
        self._last_solution = result

        states = np.array([result.atVector(X(k)) for k in range(self.horizon + 1)])
        controls = np.array([result.atVector(U(k)) for k in range(self.horizon)])
        return MPCResult(states=states, controls=controls)

    def control(self, x0: np.ndarray, xref: np.ndarray, obstacles=None) -> np.ndarray:
        """Return the first optimal control for ``x0`` (one receding-horizon step)."""
        return self.solve(x0, xref, obstacles=obstacles).u0

    def reset(self) -> None:
        """Forget the cached warm-start solution."""
        self._last_solution = None

    def simulate(self, x0: np.ndarray, xref: np.ndarray, steps: int,
                 obstacles=None) -> MPCResult:
        """Closed-loop receding-horizon rollout for ``steps`` steps toward ``xref``."""
        x = np.asarray(x0, dtype=float).reshape(4)
        self.reset()
        states = [x.copy()]
        controls = []
        for _ in range(steps):
            u = self.control(x, xref, obstacles=obstacles)
            x = self.model.step(x, u)
            controls.append(u)
            states.append(x.copy())
        return MPCResult(states=np.array(states), controls=np.array(controls))
