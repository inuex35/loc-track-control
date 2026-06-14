"""Unified estimation + control on a single factor graph.

Where a pipeline runs an MHE solve and then a separate MPC solve (coupled only
by passing the estimate), :class:`JointEstimatorMPC` builds the whole thing --
past *and* future -- as one ``NonlinearFactorGraph`` and solves it with a single
``optimize()``. One shared current-state node ``X(k)`` joins

    * the estimation window  X(k-W) ... X(k)   (anchor + motion + GPS + speed)
    * the control horizon     X(k) ... X(k+N), U(k) ... U(k+N-1)
      (hard dynamics + tracking cost + input barrier)

The current state ``X(k)`` carries no tracking cost and no hard prior: it is
pinned by the measurement factors alone, while the controller plans forward from
it. Input bounds use the barrier penalty so the whole thing stays a single
solve (no Augmented-Lagrangian outer loop).
"""

from __future__ import annotations

import gtsam
import numpy as np
from gtsam.symbol_shorthand import U, X

from . import factors
from .constraints import Inequality, barrier_factor
from .estimation import MovingHorizonEstimator
from .models import BicycleModel

# Defaults reproduce the path-following controller tuning.
_DEFAULT_Q = np.diag([4.0, 4.0, 1.5, 0.6])
_DEFAULT_R = np.diag([0.1, 0.1])
_DEFAULT_QF = np.diag([20.0, 20.0, 4.0, 2.0])


class JointEstimatorMPC:
    """Estimation window + control horizon in one graph, one solve per step.

    Args:
        window: Estimation window length ``W``.
        horizon: Control horizon ``N``.
        gps_sigma: GPS position noise for the estimation factors.
        wheelbase: Bicycle wheelbase ``L``.
        dt_plant: Timestep of the estimation-window motion factors.
        dt_mpc: Prediction timestep of the control-horizon dynamics.
        mpc_substeps: RK4 sub-integrations of the prediction model.
        Q, R, Qf: Tracking / control / terminal cost matrices.
        a_bounds, delta_bounds, v_bounds: Input / speed limits (v unused here).
        barrier_weight: Precision of the input-barrier penalty.
    """

    def __init__(self, window: int = 12, horizon: int = 6, gps_sigma: float = 1.0,
                 wheelbase: float = 2.5, dt_plant: float = 0.1, dt_mpc: float = 0.5,
                 mpc_substeps: int = 2, Q=_DEFAULT_Q, R=_DEFAULT_R, Qf=_DEFAULT_QF,
                 a_bounds=(-4.0, 4.0), delta_bounds=(-0.6, 0.6), v_bounds=(-2.0, 14.0),
                 barrier_weight: float = 500.0):
        # Prediction model (control horizon) runs at the coarse MPC timestep;
        # the estimation window uses a plant-rate model for its motion factors.
        self.model = BicycleModel(wheelbase=wheelbase, dt=dt_mpc, substeps=mpc_substeps)
        self.est = MovingHorizonEstimator(
            BicycleModel(wheelbase=wheelbase, dt=dt_plant), window=window,
            gps_sigma=gps_sigma)
        self.N = int(horizon)
        self.W = int(window)
        self.barrier_weight = float(barrier_weight)

        self.u_lower = np.array([a_bounds[0], delta_bounds[0]])
        self.u_upper = np.array([a_bounds[1], delta_bounds[1]])
        self.v_bounds = v_bounds

        self.cost_x = factors.information(np.asarray(Q, float))
        self.cost_xf = factors.information(np.asarray(Qf, float))
        self.cost_u = factors.information(np.asarray(R, float))
        self.constrained = factors.hard(4)

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(60)
        self._params = params

    def set_gps_sigma(self, sigma: float) -> None:
        self.est.set_gps_sigma(sigma)

    def reset(self, x0: np.ndarray) -> None:
        x0 = np.asarray(x0, float)
        self.k = 0
        self.controls = []                  # controls[t]: t -> t+1 (applied)
        self.gps = [x0[:2].copy()]
        self.vmeas = [float(x0[3])]
        self.Xv = {0: x0.copy()}            # warm-start state values
        self.Uv = {}                        # warm-start control values
        self.estimate = x0.copy()

    def _u_inequality(self, key) -> Inequality:
        u_jac = np.vstack([np.eye(2), -np.eye(2)])

        def ev(u):
            return np.concatenate([u - self.u_upper, self.u_lower - u]), u_jac

        return Inequality(key, 4, ev)

    def step(self, u_applied, gps, v_meas, reference):
        """One joint solve. Returns ``(estimate, plan_states, plan_controls)``."""
        e = self.est
        self.controls.append(np.asarray(u_applied, float).copy())
        self.gps.append(np.asarray(gps, float).copy())
        self.vmeas.append(float(v_meas))
        self.k += 1
        k, N, W = self.k, self.N, self.W
        lo = max(0, k - W)

        # Warm-start guesses: predict the new current state, roll the horizon.
        self.Xv[k] = self.model.step(self.Xv[k - 1], u_applied)
        for t in range(k, k + N):
            self.Uv.setdefault(t, np.zeros(2))
        for t in range(k + 1, k + N + 1):
            self.Xv.setdefault(t, self.model.step(self.Xv[t - 1], self.Uv[t - 1]))

        graph = gtsam.NonlinearFactorGraph()
        # --- estimation window (past + current) ---
        graph.add(factors.prior(X(lo), self.Xv[lo], e.anchor_noise))
        for t in range(lo, k):
            graph.add(factors.motion(e.model, X(t), X(t + 1),
                                     self.controls[t], e.proc_noise))
        for t in range(lo, k + 1):
            graph.add(factors.position_measurement(X(t), self.gps[t], e.gps_noise))
            graph.add(factors.scalar_measurement(X(t), self.vmeas[t], e.speed_noise))
        # --- control horizon (current + future) ---
        for t in range(k, k + N):
            graph.add(factors.dynamics(self.model, X(t), U(t), X(t + 1), self.constrained))
            graph.add(factors.zero_cost(U(t), 2, self.cost_u))
            graph.add(barrier_factor(self._u_inequality(U(t)), self.barrier_weight))
        for j in range(1, N + 1):                 # X(k+1..k+N) track the reference
            noise = self.cost_xf if j == N else self.cost_x
            graph.add(factors.state_cost(X(k + j), reference[j], noise))

        values = gtsam.Values()
        for t in range(lo, k + N + 1):
            values.insert(X(t), self.Xv[t])
        for t in range(k, k + N):
            values.insert(U(t), self.Uv[t])

        result = gtsam.LevenbergMarquardtOptimizer(graph, values, self._params).optimize()

        for t in range(lo, k + N + 1):
            self.Xv[t] = result.atVector(X(t))
        for t in range(k, k + N):
            self.Uv[t] = result.atVector(U(t))
        # Prune values that have fallen out of both windows.
        for t in [t for t in self.Xv if t < lo]:
            del self.Xv[t]
        for t in [t for t in self.Uv if t < k]:
            del self.Uv[t]

        self.estimate = self.Xv[k].copy()
        states = np.array([self.Xv[k + j] for j in range(N + 1)])
        controls = np.array([self.Uv[k + t] for t in range(N)])
        return self.estimate, states, controls
