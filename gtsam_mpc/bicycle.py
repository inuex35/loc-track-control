"""Nonlinear MPC for a kinematic bicycle model, solved with GTSAM.

The kinematic bicycle model (rear-axle reference) has state
``[x, y, theta, v]`` (position, heading, speed) and control ``[a, delta]``
(acceleration, front-wheel steering angle):

    x_dot     = v * cos(theta)
    y_dot     = v * sin(theta)
    theta_dot = (v / L) * tan(delta)
    v_dot     = a

where ``L`` is the wheelbase. This is nonlinear, so unlike the linear MPC in
:mod:`gtsam_mpc.mpc` we build a *nonlinear* factor graph: the discretized
dynamics and costs are :class:`gtsam.CustomFactor` factors with analytic
Jacobians, and GTSAM's Levenberg-Marquardt optimizer iterates to the MAP
trajectory.

Input limits (``|a| <= a_max``, ``|delta| <= delta_max``) and optional speed
limits are imposed as **soft barrier factors**: a one-sided quadratic penalty
that is zero inside the feasible box and grows with the violation. The
penalty stiffness is controlled by ``barrier_weight``.
"""

from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.symbol_shorthand import U, X

from .mpc import MPCResult, _require_pd


def _wrap(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class BicycleModel:
    """Discrete-time kinematic bicycle model (forward-Euler integration).

    Attributes:
        wheelbase: Distance between front and rear axles ``L`` [m].
        dt: Integration timestep [s].
    """

    wheelbase: float = 2.5
    dt: float = 0.1

    n_states: int = 4
    n_controls: int = 2

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Propagate one step with forward Euler."""
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        px, py, theta, v = x
        a, delta = u
        dt, L = self.dt, self.wheelbase
        return np.array(
            [
                px + dt * v * np.cos(theta),
                py + dt * v * np.sin(theta),
                theta + dt * (v / L) * np.tan(delta),
                v + dt * a,
            ]
        )

    def jacobians(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(df/dx, df/du)`` of :meth:`step` at ``(x, u)``."""
        _, _, theta, v = np.asarray(x, dtype=float)
        _, delta = np.asarray(u, dtype=float)
        dt, L = self.dt, self.wheelbase
        c, s = np.cos(theta), np.sin(theta)
        dfdx = np.array(
            [
                [1.0, 0.0, -dt * v * s, dt * c],
                [0.0, 1.0, dt * v * c, dt * s],
                [0.0, 0.0, 1.0, dt * np.tan(delta) / L],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        dfdu = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, dt * v / (L * np.cos(delta) ** 2)],
                [dt, 0.0],
            ]
        )
        return dfdx, dfdu


class BicycleMPC:
    """Nonlinear receding-horizon MPC for a :class:`BicycleModel`.

    Args:
        model: The bicycle model to control.
        Q: Running state cost, shape ``(4, 4)``, positive definite. The heading
            component is penalized on the wrapped error.
        R: Running control cost, shape ``(2, 2)``, positive definite.
        horizon: Prediction horizon ``N``.
        Qf: Terminal state cost, shape ``(4, 4)``. Defaults to ``Q``.
        a_bounds: ``(min, max)`` acceleration limits [m/s^2].
        delta_bounds: ``(min, max)`` steering-angle limits [rad].
        v_bounds: Optional ``(min, max)`` speed limits [m/s].
        barrier_weight: Stiffness of the soft box-constraint penalty (the
            precision of the barrier factors). Larger = closer to a hard limit.
        max_iterations: Max Levenberg-Marquardt iterations per solve.
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
        barrier_weight: float = 300.0,
        max_iterations: int = 100,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        self.model = model
        self.horizon = int(horizon)
        self.Q = _require_pd("Q", Q, 4)
        self.R = _require_pd("R", R, 2)
        self.Qf = self.Q if Qf is None else _require_pd("Qf", Qf, 4)

        self.u_lower = np.array([a_bounds[0], delta_bounds[0]])
        self.u_upper = np.array([a_bounds[1], delta_bounds[1]])
        self.v_bounds = v_bounds
        self.barrier_weight = float(barrier_weight)

        self._params = gtsam.LevenbergMarquardtParams()
        self._params.setMaxIterations(int(max_iterations))
        self._params.setRelativeErrorTol(1e-8)
        self._params.setAbsoluteErrorTol(1e-8)

        self._last_solution: gtsam.Values | None = None

    # --- factor error functions ------------------------------------------
    def _dynamics_error(self):
        model = self.model

        def error(this, values, H):
            k = this.keys()
            xk = values.atVector(k[0])
            uk = values.atVector(k[1])
            xk1 = values.atVector(k[2])
            if H is not None:
                dfdx, dfdu = model.jacobians(xk, uk)
                H[0] = -dfdx
                H[1] = -dfdu
                H[2] = np.eye(4)
            return xk1 - model.step(xk, uk)

        return error

    @staticmethod
    def _state_cost_error(xref: np.ndarray):
        def error(this, values, H):
            x = values.atVector(this.keys()[0])
            e = x - xref
            e[2] = _wrap(e[2])
            if H is not None:
                H[0] = np.eye(4)
            return e

        return error

    @staticmethod
    def _control_cost_error(this, values, H):
        u = values.atVector(this.keys()[0])
        if H is not None:
            H[0] = np.eye(2)
        return u

    @staticmethod
    def _box_error(lower: np.ndarray, upper: np.ndarray):
        def error(this, values, H):
            x = values.atVector(this.keys()[0])
            e = np.maximum(0.0, x - upper) - np.maximum(0.0, lower - x)
            if H is not None:
                H[0] = np.diag(((x > upper) | (x < lower)).astype(float))
            return e

        return error

    @staticmethod
    def _prior_error(x0: np.ndarray):
        def error(this, values, H):
            if H is not None:
                H[0] = np.eye(4)
            return values.atVector(this.keys()[0]) - x0

        return error

    # --- graph construction ----------------------------------------------
    def _build_graph(
        self, x0: np.ndarray, xref: np.ndarray
    ) -> gtsam.NonlinearFactorGraph:
        N = self.horizon
        cost_x = gtsam.noiseModel.Gaussian.Information(self.Q)
        cost_u = gtsam.noiseModel.Gaussian.Information(self.R)
        cost_xf = gtsam.noiseModel.Gaussian.Information(self.Qf)
        constraint = gtsam.noiseModel.Constrained.All(4)
        u_barrier = gtsam.noiseModel.Diagonal.Precisions(
            np.full(2, self.barrier_weight)
        )

        graph = gtsam.NonlinearFactorGraph()
        graph.add(gtsam.CustomFactor(constraint, [X(0)], self._prior_error(x0)))
        for k in range(N):
            graph.add(
                gtsam.CustomFactor(
                    constraint, [X(k), U(k), X(k + 1)], self._dynamics_error()
                )
            )
            graph.add(
                gtsam.CustomFactor(cost_x, [X(k)], self._state_cost_error(xref))
            )
            graph.add(gtsam.CustomFactor(cost_u, [U(k)], self._control_cost_error))
            graph.add(
                gtsam.CustomFactor(
                    u_barrier, [U(k)], self._box_error(self.u_lower, self.u_upper)
                )
            )
            if self.v_bounds is not None:
                lo = np.array([-np.inf, -np.inf, -np.inf, self.v_bounds[0]])
                hi = np.array([np.inf, np.inf, np.inf, self.v_bounds[1]])
                v_barrier = gtsam.noiseModel.Diagonal.Precisions(
                    np.array([0.0, 0.0, 0.0, self.barrier_weight])
                )
                graph.add(
                    gtsam.CustomFactor(v_barrier, [X(k)], self._box_error(lo, hi))
                )
        graph.add(
            gtsam.CustomFactor(cost_xf, [X(N)], self._state_cost_error(xref))
        )
        return graph

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
    def solve(
        self,
        x0: np.ndarray,
        xref: np.ndarray,
        warm_start: bool = True,
    ) -> MPCResult:
        """Optimize the trajectory from ``x0`` toward target state ``xref``.

        Consecutive calls reuse the previous solution as the initial guess
        (``warm_start=True``), which keeps re-optimization fast for closed-loop
        / interactive use.
        """
        x0 = np.asarray(x0, dtype=float).reshape(4)
        xref = np.asarray(xref, dtype=float).reshape(4)

        graph = self._build_graph(x0, xref)
        if warm_start and self._last_solution is not None:
            initial = self._last_solution
        else:
            initial = self._rollout_init(x0)

        result = gtsam.LevenbergMarquardtOptimizer(
            graph, initial, self._params
        ).optimize()
        self._last_solution = result

        states = np.array([result.atVector(X(k)) for k in range(self.horizon + 1)])
        controls = np.array([result.atVector(U(k)) for k in range(self.horizon)])
        return MPCResult(states=states, controls=controls)

    def control(self, x0: np.ndarray, xref: np.ndarray) -> np.ndarray:
        """Return the first optimal control for ``x0`` (one receding-horizon step)."""
        return self.solve(x0, xref).u0

    def reset(self) -> None:
        """Forget the cached warm-start solution."""
        self._last_solution = None

    def simulate(self, x0: np.ndarray, xref: np.ndarray, steps: int) -> MPCResult:
        """Closed-loop receding-horizon rollout for ``steps`` steps toward ``xref``."""
        x = np.asarray(x0, dtype=float).reshape(4)
        self.reset()
        states = [x.copy()]
        controls = []
        for _ in range(steps):
            u = self.control(x, xref)
            x = self.model.step(x, u)
            controls.append(u)
            states.append(x.copy())
        return MPCResult(states=np.array(states), controls=np.array(controls))
