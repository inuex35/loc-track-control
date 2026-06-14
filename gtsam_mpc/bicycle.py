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
    """Discrete-time kinematic bicycle model (classic RK4 integration).

    The continuous dynamics ``xdot = f(x, u)`` are integrated over ``dt`` with
    the 4th-order Runge-Kutta rule. RK4 is ``O(dt^5)`` accurate per step versus
    the ``O(dt^2)`` of forward Euler, which removes the systematic discretization
    bias Euler introduces on curved motion (e.g. a steady cross-track offset
    when tracking a circle), letting the MPC use a coarse prediction grid.

    For extra accuracy on a coarse grid the ``dt`` interval can be split into
    ``substeps`` equal RK4 sub-integrations (as acados does with
    ``sim_method_num_steps``); the control ``u`` is held constant across them.

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
        safety_radius: float = 2.0,
        obstacle_weight: float = 150.0,
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
        self.safety_radius = float(safety_radius)
        self.obstacle_weight = float(obstacle_weight)

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

    @staticmethod
    def _avoid_error(p_obs: np.ndarray, radius: float):
        """Soft keep-out: penalize being closer than ``radius`` to ``p_obs``.

        This is the position control-barrier ``h(x) = ||p - p_obs||^2 - r^2 >= 0``
        imposed softly: the residual is the depth of incursion into the keep-out
        disk (zero outside it), so a stiff noise model pushes the trajectory out.
        """
        def error(this, values, H):
            x = values.atVector(this.keys()[0])
            d = x[:2] - p_obs
            dist = float(np.hypot(d[0], d[1]))
            if dist < radius:
                if H is not None:
                    grad = np.zeros((1, 4))
                    if dist > 1e-6:
                        grad[0, 0] = -d[0] / dist
                        grad[0, 1] = -d[1] / dist
                    H[0] = grad
                return np.array([radius - dist])
            if H is not None:
                H[0] = np.zeros((1, 4))
            return np.array([0.0])

        return error

    def _normalize_xref(self, xref) -> np.ndarray:
        """Reference -> a ``(horizon + 1, 4)`` array of per-step target states.

        Accepts a single state ``(4,)`` (broadcast to every step) or a full
        reference trajectory ``(horizon + 1, 4)``.
        """
        arr = np.asarray(xref, dtype=float)
        if arr.shape == (4,):
            return np.tile(arr, (self.horizon + 1, 1))
        if arr.shape == (self.horizon + 1, 4):
            return arr
        raise ValueError(
            f"xref must be shape (4,) or ({self.horizon + 1}, 4), got {arr.shape}"
        )

    def _normalize_obstacles(self, obstacles) -> list[np.ndarray]:
        """Each obstacle -> a (horizon+1, 2) array of predicted centre positions.

        Accepts a single point ``(2,)`` (static), or a per-step prediction
        ``(horizon+1, 2)``.
        """
        out = []
        for obs in obstacles:
            arr = np.asarray(obs, dtype=float)
            if arr.shape == (2,):
                arr = np.tile(arr, (self.horizon + 1, 1))
            elif arr.shape != (self.horizon + 1, 2):
                raise ValueError(
                    f"obstacle must be shape (2,) or ({self.horizon + 1}, 2), "
                    f"got {arr.shape}"
                )
            out.append(arr)
        return out

    # --- graph construction ----------------------------------------------
    def _build_graph(
        self, x0: np.ndarray, xref: np.ndarray, obstacles: list[np.ndarray] | None
    ) -> gtsam.NonlinearFactorGraph:
        N = self.horizon
        cost_x = gtsam.noiseModel.Gaussian.Information(self.Q)
        cost_u = gtsam.noiseModel.Gaussian.Information(self.R)
        cost_xf = gtsam.noiseModel.Gaussian.Information(self.Qf)
        constraint = gtsam.noiseModel.Constrained.All(4)
        u_barrier = gtsam.noiseModel.Diagonal.Precisions(
            np.full(2, self.barrier_weight)
        )
        obs_barrier = gtsam.noiseModel.Diagonal.Precisions(
            np.array([self.obstacle_weight])
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
                gtsam.CustomFactor(cost_x, [X(k)], self._state_cost_error(xref[k]))
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
            gtsam.CustomFactor(cost_xf, [X(N)], self._state_cost_error(xref[N]))
        )

        if obstacles:
            for obs in obstacles:
                for k in range(N + 1):
                    graph.add(
                        gtsam.CustomFactor(
                            obs_barrier, [X(k)],
                            self._avoid_error(obs[k], self.safety_radius),
                        )
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
        obstacles=None,
        warm_start: bool = True,
    ) -> MPCResult:
        """Optimize the trajectory from ``x0`` toward the reference ``xref``.

        Args:
            x0: Current state ``[x, y, theta, v]``.
            xref: Either a single target state ``(4,)`` applied at every horizon
                step (set-point tracking), or a full reference trajectory of
                shape ``(horizon + 1, 4)`` giving the desired state at each step
                (path / trajectory tracking). The heading component is compared
                on the wrapped error.
            obstacles: Optional iterable of obstacles to avoid. Each is either a
                static centre ``(2,)`` or a per-horizon-step prediction
                ``(horizon + 1, 2)`` (e.g. from a tracker). Avoidance is a soft
                keep-out within ``safety_radius``.
            warm_start: Reuse the previous solution as the initial guess, which
                keeps re-optimization fast for closed-loop / interactive use.
        """
        x0 = np.asarray(x0, dtype=float).reshape(4)
        xref = self._normalize_xref(xref)
        obs = self._normalize_obstacles(obstacles) if obstacles else None

        graph = self._build_graph(x0, xref, obs)
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

    def control(self, x0: np.ndarray, xref: np.ndarray, obstacles=None) -> np.ndarray:
        """Return the first optimal control for ``x0`` (one receding-horizon step)."""
        return self.solve(x0, xref, obstacles=obstacles).u0

    def reset(self) -> None:
        """Forget the cached warm-start solution."""
        self._last_solution = None

    def simulate(
        self, x0: np.ndarray, xref: np.ndarray, steps: int, obstacles=None
    ) -> MPCResult:
        """Closed-loop receding-horizon rollout for ``steps`` steps toward ``xref``.

        ``obstacles`` (static centres) are avoided at every step.
        """
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
