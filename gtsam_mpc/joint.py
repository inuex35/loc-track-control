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

import gtsam as _gtsam

from . import factors
from .constraints import Inequality, barrier_factor
from .estimation import MovingHorizonEstimator
from .models import BicycleModel

# Obstacle state keys: char 'o', index encodes (obstacle, time) without colliding
# with the ego X/U keys. (Assumes fewer than _OBS_STRIDE simulation steps.)
_OBS_STRIDE = 100_000_000


def _obs_key(j: int, t: int) -> int:
    return _gtsam.symbol("o", j * _OBS_STRIDE + t)


def _cv_matrix(dt: float) -> np.ndarray:
    return np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                     [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


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

    # --- graph assembly broken into reusable pieces (shared with subclasses) ---
    def _push(self, u_applied, gps, v_meas) -> tuple[int, int]:
        """Advance the time index, record measurements, roll the warm-start guess."""
        self.controls.append(np.asarray(u_applied, float).copy())
        self.gps.append(np.asarray(gps, float).copy())
        self.vmeas.append(float(v_meas))
        self.k += 1
        k, N = self.k, self.N
        self.Xv[k] = self.model.step(self.Xv[k - 1], u_applied)
        for t in range(k, k + N):
            self.Uv.setdefault(t, np.zeros(2))
        for t in range(k + 1, k + N + 1):
            self.Xv.setdefault(t, self.model.step(self.Xv[t - 1], self.Uv[t - 1]))
        return k, max(0, k - self.W)

    def _add_ego_factors(self, graph, lo, reference) -> None:
        """Estimation window (past + current) + control horizon (current + future)."""
        e, k, N = self.est, self.k, self.N
        graph.add(factors.prior(X(lo), self.Xv[lo], e.anchor_noise))
        for t in range(lo, k):
            graph.add(factors.motion(e.model, X(t), X(t + 1),
                                     self.controls[t], e.proc_noise))
        for t in range(lo, k + 1):
            graph.add(factors.position_measurement(X(t), self.gps[t], e.gps_noise))
            graph.add(factors.scalar_measurement(X(t), self.vmeas[t], e.speed_noise))
        for t in range(k, k + N):
            graph.add(factors.dynamics(self.model, X(t), U(t), X(t + 1), self.constrained))
            graph.add(factors.zero_cost(U(t), 2, self.cost_u))
            graph.add(barrier_factor(self._u_inequality(U(t)), self.barrier_weight))
        for j in range(1, N + 1):
            noise = self.cost_xf if j == N else self.cost_x
            graph.add(factors.state_cost(X(k + j), reference[j], noise))

    def _insert_ego_values(self, values, lo) -> None:
        k, N = self.k, self.N
        for t in range(lo, k + N + 1):
            values.insert(X(t), self.Xv[t])
        for t in range(k, k + N):
            values.insert(U(t), self.Uv[t])

    def _extract_ego(self, result, lo):
        k, N = self.k, self.N
        for t in range(lo, k + N + 1):
            self.Xv[t] = result.atVector(X(t))
        for t in range(k, k + N):
            self.Uv[t] = result.atVector(U(t))
        for t in [t for t in self.Xv if t < lo]:
            del self.Xv[t]
        for t in [t for t in self.Uv if t < k]:
            del self.Uv[t]
        self.estimate = self.Xv[k].copy()
        states = np.array([self.Xv[k + j] for j in range(N + 1)])
        controls = np.array([self.Uv[k + t] for t in range(N)])
        return states, controls

    def step(self, u_applied, gps, v_meas, reference):
        """One joint solve. Returns ``(estimate, plan_states, plan_controls)``."""
        k, lo = self._push(u_applied, gps, v_meas)
        graph = gtsam.NonlinearFactorGraph()
        self._add_ego_factors(graph, lo, reference)
        values = gtsam.Values()
        self._insert_ego_values(values, lo)
        result = gtsam.LevenbergMarquardtOptimizer(graph, values, self._params).optimize()
        states, controls = self._extract_ego(result, lo)
        return self.estimate, states, controls


class JointLocTrackControl(JointEstimatorMPC):
    """Localization + obstacle tracking + control in a single factor graph.

    Extends :class:`JointEstimatorMPC` with, in the *same* graph and the *same*
    ``optimize()`` call:

        * each obstacle ``j`` tracked as constant-velocity states ``O(j, t)`` over
          the estimation window (with detection factors) and predicted forward
          over the control horizon (motion factors only), and
        * keep-out factors between each ego horizon node and the corresponding
          obstacle node -- one-sided, with the Jacobian on the ego only so the
          avoidance shapes the plan without bending the obstacle estimate.

    The keep-out radius is *uncertainty-aware*: after each solve the obstacle
    nodes' marginal position covariance is read with ``gtsam.Marginals`` and used
    to inflate the radius for the next solve (a one-step lag), so the car gives a
    wider berth to obstacles whose track is uncertain.

    Args:
        n_obstacles: Number of tracked obstacles ``M`` (known data association).
        safety_radius: Base keep-out radius.
        obstacle_weight: Barrier precision of the keep-out penalty.
        n_sigma: Covariance inflation: ``radius = safety_radius + n_sigma * std``.
        obs_proc_sigma, obs_anchor_sigma: Obstacle CV process / window-anchor noise.
        obs_sigma_near, obs_sigma_rate, obs_sigma_max: Range-dependent detection
            noise -- a detection's sigma is
            ``min(obs_sigma_max, obs_sigma_near + obs_sigma_rate * range)``, so a
            nearby obstacle is observed more strongly (smaller sigma) than a far
            one, which in turn shrinks its track covariance and keep-out margin.
        **kwargs: Forwarded to :class:`JointEstimatorMPC`.
    """

    def __init__(self, n_obstacles: int, safety_radius: float = 3.0,
                 obstacle_weight: float = 400.0, n_sigma: float = 2.0,
                 obs_proc_sigma=(0.05, 0.05, 0.4, 0.4), obs_anchor_sigma: float = 10.0,
                 obs_sigma_near: float = 0.25, obs_sigma_rate: float = 0.045,
                 obs_sigma_max: float = 1.2, **kwargs):
        super().__init__(**kwargs)
        self.M = int(n_obstacles)
        self.safety_radius = float(safety_radius)
        self.obstacle_weight = float(obstacle_weight)
        self.n_sigma = float(n_sigma)
        self.obs_proc = factors.sigmas(obs_proc_sigma)
        self.obs_sigma_near = float(obs_sigma_near)
        self.obs_sigma_rate = float(obs_sigma_rate)
        self.obs_sigma_max = float(obs_sigma_max)
        self.obs_anchor = factors.isotropic(4, obs_anchor_sigma)
        self.obs_barrier = factors.precisions([obstacle_weight])
        self.Fp = _cv_matrix(self.est.model.dt)   # plant-rate CV (window)
        self.Fm = _cv_matrix(self.model.dt)        # mpc-rate CV (horizon)

    def reset(self, x0: np.ndarray, obstacles) -> None:
        """Reset with the ego state and the initial obstacle states/positions."""
        super().reset(x0)
        self.Ov = {}        # Ov[j][t] : warm-start obstacle state
        self.Odet = {}      # Odet[j][t] : obstacle position detection at step t
        for j, o in enumerate(obstacles):
            o = np.asarray(o, float)
            state = o if o.shape == (4,) else np.array([o[0], o[1], 0.0, 0.0])
            self.Ov[j] = {0: state.copy()}
            self.Odet[j] = [state[:2].copy()]
        # std[(j, i)] : obstacle position std at horizon step i, from last solve.
        self.std = {}
        self.ego_pos_cov = None   # ego current-state position covariance (2x2)

    def _add_obstacle_factors(self, graph, lo) -> None:
        k, N = self.k, self.N
        for j in range(self.M):
            graph.add(factors.prior(_obs_key(j, lo), self.Ov[j][lo], self.obs_anchor))
            for t in range(lo, k):       # window: plant-rate constant velocity
                graph.add(factors.linear_motion(_obs_key(j, t), _obs_key(j, t + 1),
                                                 self.Fp, self.obs_proc))
            for t in range(k, k + N):    # horizon: mpc-rate prediction
                graph.add(factors.linear_motion(_obs_key(j, t), _obs_key(j, t + 1),
                                                 self.Fm, self.obs_proc))
            for t in range(lo, k + 1):
                # Range-dependent sensor: a detection is trusted more (smaller
                # sigma) the closer the obstacle was to the ego when observed.
                dist = float(np.linalg.norm(self.Xv[t][:2] - self.Odet[j][t]))
                graph.add(factors.position_measurement(
                    _obs_key(j, t), self.Odet[j][t], self._det_noise(dist)))

    def _det_noise(self, dist: float):
        sigma = min(self.obs_sigma_max, self.obs_sigma_near + self.obs_sigma_rate * dist)
        return factors.isotropic(2, sigma)

    def _add_keepout_factors(self, graph) -> None:
        k, N = self.k, self.N
        for j in range(self.M):
            for i in range(N + 1):
                radius = self.safety_radius + self.n_sigma * self.std.get((j, i), 0.0)
                graph.add(factors.keepout(X(k + i), _obs_key(j, k + i),
                                          radius, self.obs_barrier))

    def _push_obstacles(self, detections, lo) -> None:
        k, N = self.k, self.N
        for j in range(self.M):
            self.Odet[j].append(np.asarray(detections[j], float))
            self.Ov[j][k] = self.Fp @ self.Ov[j][k - 1]      # new window node
            for t in range(k + 1, k + N + 1):                # predicted horizon
                self.Ov[j][t] = self.Fm @ self.Ov[j][t - 1]
            for t in [t for t in self.Ov[j] if t < lo]:
                del self.Ov[j][t]

    def _update_radii(self, graph, result) -> None:
        """Read obstacle marginal covariance -> per-horizon position std (next solve).

        Also caches the ego current-state position covariance (``ego_pos_cov``)
        for localization-uncertainty visualization.
        """
        k, N = self.k, self.N
        try:
            marg = _gtsam.Marginals(graph, result)
        except Exception:
            return
        try:
            self.ego_pos_cov = np.asarray(marg.marginalCovariance(X(k)))[:2, :2]
        except Exception:
            self.ego_pos_cov = None
        for j in range(self.M):
            for i in range(N + 1):
                try:
                    cov = marg.marginalCovariance(_obs_key(j, k + i))[:2, :2]
                    self.std[(j, i)] = float(np.sqrt(max(np.linalg.eigvalsh(cov)[-1], 0.0)))
                except Exception:
                    self.std[(j, i)] = 0.0

    def step(self, u_applied, gps, v_meas, detections, reference):
        """One joint solve over ego + obstacles + control.

        Returns ``(estimate, plan_states, plan_controls, obstacle_estimates,
        obstacle_predictions)`` where the last two are per-obstacle current
        states and ``(horizon + 1, 2)`` predicted position trajectories.
        """
        k, lo = self._push(u_applied, gps, v_meas)
        self._push_obstacles(detections, lo)

        graph = gtsam.NonlinearFactorGraph()
        self._add_ego_factors(graph, lo, reference)
        self._add_obstacle_factors(graph, lo)
        self._add_keepout_factors(graph)

        values = gtsam.Values()
        self._insert_ego_values(values, lo)
        for j in range(self.M):
            for t in range(lo, k + self.N + 1):
                values.insert(_obs_key(j, t), self.Ov[j][t])

        result = gtsam.LevenbergMarquardtOptimizer(graph, values, self._params).optimize()

        states, controls = self._extract_ego(result, lo)
        for j in range(self.M):
            for t in range(lo, k + self.N + 1):
                self.Ov[j][t] = result.atVector(_obs_key(j, t))
        self._update_radii(graph, result)

        obstacle_estimates = {j: self.Ov[j][k].copy() for j in range(self.M)}
        obstacle_predictions = {
            j: np.array([self.Ov[j][k + i][:2] for i in range(self.N + 1)])
            for j in range(self.M)
        }
        return self.estimate, states, controls, obstacle_estimates, obstacle_predictions
