"""State estimators built on factor graphs.

:class:`MovingHorizonEstimator` -- a sliding-window MAP estimator for the
bicycle model (the receding-horizon dual of MPC), built from the same
:mod:`gtsam_mpc.factors` vocabulary as the controllers.
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
