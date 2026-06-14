"""State estimators built on factor graphs.

* :class:`MovingHorizonEstimator` -- a sliding-window MAP estimator for the
  bicycle model (the receding-horizon dual of MPC), built from the same
  :mod:`gtsam_mpc.factors` vocabulary as the controllers.
* :class:`ConstantVelocityTracker` -- a linear constant-velocity smoother for
  2-D position tracks (the FGO-MOT motion-prior + measurement smoother), plus a
  constant-velocity forward predictor.
"""

from __future__ import annotations

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X

from . import factors

DEFAULT_SPEED_SIGMA = 0.3   # wheel-speed measurement noise [m/s]


class MovingHorizonEstimator:
    """Sliding-window MAP state estimator on a bicycle factor graph.

    Over the last ``window`` steps the graph has, built from
    :mod:`gtsam_mpc.factors`:

        * a bicycle **motion** factor between consecutive states given the
          applied control (soft, with process noise),
        * a **GPS** position factor and a **speed** factor on each state,
        * a loose **anchor** prior on the oldest state in the window (an
          arrival-cost stand-in for the marginalized older states).

    Solving the window yields the smoothed estimate of the newest state. The
    noise models are exposed as attributes so a joint estimator+controller can
    reuse them.
    """

    def __init__(self, model, window: int = 12, gps_sigma: float = 0.6,
                 proc_sigma=(0.05, 0.05, 0.03, 0.10),
                 speed_sigma: float = DEFAULT_SPEED_SIGMA,
                 anchor_sigma=(0.3, 0.3, 0.2, 0.3)):
        self.model = model
        self.W = int(window)
        self.gps_noise = factors.isotropic(2, gps_sigma)
        self.proc_noise = factors.sigmas(proc_sigma)
        self.speed_noise = factors.isotropic(1, speed_sigma)
        self.anchor_noise = factors.sigmas(anchor_sigma)
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(40)
        self._params = params
        self.reset(np.zeros(4))

    def set_gps_sigma(self, sigma: float) -> None:
        self.gps_noise = factors.isotropic(2, sigma)

    def reset(self, x0: np.ndarray) -> None:
        self.est = [np.asarray(x0, float).copy()]   # est[i] = estimate at step i
        self.controls = []                          # controls[i] : i -> i+1
        self.gps = [np.asarray(x0, float)[:2].copy()]
        self.vmeas = [float(x0[3])]
        self.k = 0

    def update(self, u_prev: np.ndarray, gps: np.ndarray, v_meas: float) -> np.ndarray:
        """Advance one step with the applied control and new measurements."""
        self.controls.append(np.asarray(u_prev, float).copy())
        self.gps.append(np.asarray(gps, float).copy())
        self.vmeas.append(float(v_meas))
        self.k += 1
        k = self.k
        self.est.append(self.model.step(self.est[k - 1], u_prev))  # initial guess

        lo = max(0, k - self.W)
        graph = gtsam.NonlinearFactorGraph()
        graph.add(factors.prior(X(lo), self.est[lo], self.anchor_noise))
        for i in range(lo, k):
            graph.add(factors.motion(self.model, X(i), X(i + 1),
                                     self.controls[i], self.proc_noise))
        for i in range(lo, k + 1):
            graph.add(factors.position_measurement(X(i), self.gps[i], self.gps_noise))
            graph.add(factors.scalar_measurement(X(i), self.vmeas[i], self.speed_noise))

        values = gtsam.Values()
        for i in range(lo, k + 1):
            values.insert(X(i), self.est[i])
        result = gtsam.LevenbergMarquardtOptimizer(graph, values, self._params).optimize()
        for i in range(lo, k + 1):
            self.est[i] = result.atVector(X(i))
        return self.est[k].copy()


class ConstantVelocityTracker:
    """Linear constant-velocity smoother for 2-D position tracks.

    State is ``[px, py, vx, vy]``; only position is observed. The graph is a
    motion prior ``x_{t+1} = F x_t`` plus a position measurement per frame --
    a batch (fixed-lag-style) smoother solved as a *linear* Gaussian factor
    graph. Used to smooth detections and to predict obstacle motion.
    """

    def __init__(self, dt: float = 0.1, process_sigma=(0.02, 0.02, 0.3, 0.3),
                 meas_sigma: float = 0.7, anchor_sigma: float = 10.0):
        self.dt = float(dt)
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                           [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self._proc = gtsam.noiseModel.Diagonal.Sigmas(np.asarray(process_sigma, float))
        self._meas = gtsam.noiseModel.Isotropic.Sigma(2, meas_sigma)
        self._anchor = gtsam.noiseModel.Isotropic.Sigma(4, anchor_sigma)

    def smooth(self, measurements) -> np.ndarray:
        """Batch-smooth a list of 2-D position detections into states ``(T+1, 4)``."""
        dets = [np.asarray(z, float) for z in measurements]
        T = len(dets) - 1
        if T < 1:
            return np.array([[dets[-1][0], dets[-1][1], 0.0, 0.0]])
        graph = gtsam.GaussianFactorGraph()
        graph.add(X(0), np.eye(4), np.zeros(4), self._anchor)
        for t in range(T):
            graph.add(X(t + 1), np.eye(4), X(t), -self.F, np.zeros(4), self._proc)
        for t in range(T + 1):
            graph.add(X(t), self.H, dets[t], self._meas)
        sol = graph.optimize()
        return np.array([sol.at(X(t)) for t in range(T + 1)])

    def estimate(self, measurements) -> np.ndarray:
        """Smoothed estimate of the *newest* state ``[px, py, vx, vy]``."""
        return self.smooth(measurements)[-1]

    def predict(self, state: np.ndarray, horizon: int, dt: float | None = None) -> np.ndarray:
        """Constant-velocity rollout of ``state`` -> positions ``(horizon + 1, 2)``."""
        dt = self.dt if dt is None else dt
        p, v = np.asarray(state, float)[:2], np.asarray(state, float)[2:]
        return np.array([p + (k * dt) * v for k in range(horizon + 1)])
