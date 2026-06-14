"""Coupled localization + control: drive a noisy-GPS car along a path.

This couples the two halves of a robotics stack on factor graphs (cf. Fig. 5 of
Abdelkarim et al., "Factor Graphs in Optimization-Based Robotic Control", IEEE
Access 2025): an *estimator* over past states and a *controller* over future
states, both solved with GTSAM.

Each control step:

    1. The true car state is hidden. We get a noisy **GPS** position fix and a
       noisy **wheel-speed** reading.
    2. A moving-horizon estimator (:class:`gtsam_mpc.MovingHorizonEstimator`) --
       a sliding-window factor graph with bicycle motion factors + GPS/speed
       measurement factors -- fuses the last W steps into a smoothed estimate
       (the receding-horizon dual of MPC).
    3. The path-following MPC plans from the *estimate* (not the truth) and the
       first control is applied to the true plant.

So estimation error feeds straight into the controller and vice versa. Without
the estimator the controller would chase raw GPS noise; the MHE smooths it.

Controls:
    * 1..5                  -- select a path (same set as path_following_sim)
    * Up / Down             -- raise / lower target speed
    * [ / ]                 -- decrease / increase GPS noise
    * E                     -- toggle: control on estimate vs. raw GPS
    * Space                 -- pause / resume
    * R                     -- reset
    * Esc / Q               -- quit

Run from the repository root:

    python examples/loc_control_sim.py

Runs headless for testing with ``SDL_VIDEODRIVER=dummy`` and
``GTSAM_MPC_MAX_FRAMES=<n>``.
"""

import os
import sys

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame

from gtsam_mpc import BicycleModel, MovingHorizonEstimator
import examples.path_following_sim as pf
# Shared world/render config, measurement constants and interactive harness.
# Re-exported here so ``joint_loc_control_sim`` can keep importing them from us.
from examples._racecar_app import (  # noqa: F401  (re-exported for joint sim)
    RacecarApp, WIDTH, HEIGHT, DT, MPC_DT, WHEELBASE,
    GPS_SIGMAS, SPEED_SIGMA,
    TRUE_COLOR, EST_COLOR, GPS_COLOR, WINDOW_COLOR, TEXT_COLOR,
)


def run(seed: int = 0, steps: int = 300, gps_sigma: float = 1.0,
        use_estimate: bool = True, path_gen=None) -> dict:
    """Headless closed-loop run returning localization + tracking metrics.

    With ``use_estimate=True`` the controller drives on the MHE-fused estimate;
    with ``False`` it uses the raw GPS position and a heading inferred from the
    noisy GPS displacement (the honest no-fusion baseline, which has no direct
    heading sensor).
    """
    rng = np.random.default_rng(seed)
    mpc = pf.make_controller()
    plant = BicycleModel(wheelbase=WHEELBASE, dt=DT)
    estimator = MovingHorizonEstimator(BicycleModel(wheelbase=WHEELBASE, dt=DT))
    estimator.set_gps_sigma(gps_sigma)

    gen = path_gen if path_gen is not None else pf.PATHS["4"][1]   # racetrack
    path = pf.make_path(gen)
    curv = pf.path_curvature(path)
    true = pf.reset_on_path(path, 8.0)
    estimator.reset(true)
    est = true.copy()
    prev_gps = true[:2].copy()
    heading = float(true[2])

    loc, cte, deltas = [], [], []
    for _ in range(steps):
        near = pf.nearest_index(path, est[:2])
        refs = pf.reference_trajectory(path, curv, near, 8.0, mpc.horizon, MPC_DT)
        u = mpc.solve(est, refs).u0
        deltas.append(float(u[1]))
        true = plant.step(true, u)
        gps = true[:2] + rng.normal(0.0, gps_sigma, size=2)
        v_meas = true[3] + rng.normal(0.0, SPEED_SIGMA)
        if use_estimate:
            est = estimator.update(u, gps, v_meas)
        else:
            d = gps - prev_gps
            if np.hypot(d[0], d[1]) > 0.2:
                heading = float(np.arctan2(d[1], d[0]))
                prev_gps = gps.copy()
            est = np.array([gps[0], gps[1], heading, v_meas])
        loc.append(float(np.linalg.norm(true[:2] - est[:2])))
        cte.append(pf.cross_track_error(path, true[:2]))

    loc = np.array(loc[50:])
    cte = np.array(cte[50:])
    rate = np.diff(np.array(deltas[50:]))
    return {
        "loc_rmse": float(np.sqrt(np.mean(loc ** 2))),
        "cte_mean": float(cte.mean()),
        "steer_rate_std": float(np.std(rate)),
    }


class LocControlApp(RacecarApp):
    """Two-graph pipeline: an MHE solve feeds an independent MPC solve."""

    caption = "gtsam-mpc: coupled localization + control"

    def setup(self) -> None:
        self.mpc = pf.make_controller()
        self.estimator = MovingHorizonEstimator(BicycleModel(wheelbase=WHEELBASE, dt=DT))
        self.use_estimate = True
        self.raw_prev_gps = None
        self.raw_heading = 0.0
        self.set_gps_sigma(GPS_SIGMAS[self.gps_idx])

    def set_gps_sigma(self, sigma: float) -> None:
        self.estimator.set_gps_sigma(sigma)

    def reset_engine(self) -> None:
        self.mpc.reset()
        self.estimator.reset(self.true_state)
        self.estimate = self.true_state.copy()
        self.raw_prev_gps = self.true_state[:2].copy()
        self.raw_heading = float(self.true_state[2])

    def horizon(self) -> int:
        return self.mpc.horizon

    def window_points(self) -> list:
        e = self.estimator
        lo = max(0, e.k - e.W)
        return [pf.world_to_screen(e.est[i][:2]) for i in range(lo, e.k + 1)]

    def extra_key(self, event) -> bool:
        if event.key == pygame.K_e:
            self.use_estimate = not self.use_estimate
            return True
        return False

    def advance(self, refs):
        # The controller only ever sees the estimate (or raw GPS if toggled).
        plan = self.mpc.solve(self.estimate, refs)
        u = plan.u0
        self.true_state = self.plant.step(self.true_state, u)
        gps, v_meas = self.measure()
        if self.use_estimate:
            self.estimate = self.estimator.update(u, gps, v_meas)
        else:
            # Naive baseline: no fusion. GPS gives position only, so heading must
            # come from the (noisy) GPS displacement -- which is junk and makes
            # path tracking diverge. Keep the window populated for display.
            self.estimator.update(u, gps, v_meas)
            d = gps - self.raw_prev_gps
            if np.hypot(d[0], d[1]) > 0.2:
                self.raw_heading = float(np.arctan2(d[1], d[0]))
                self.raw_prev_gps = gps.copy()
            self.estimate = np.array([gps[0], gps[1], self.raw_heading, v_meas])
        self._last_gps = gps
        self._last_delta = float(u[1])
        return plan.states

    def hud_lines(self) -> list:
        source = "ESTIMATE (MHE fused)" if self.use_estimate else "RAW GPS"
        return self.status_lines() + [
            f"control source: {source}   green=true blue=est red=GPS",
            "1-5 path  up/down speed  [ ] gps-noise  e toggle  space pause  "
            "r reset  esc quit" + self.paused_suffix(),
        ]


def main() -> None:
    LocControlApp().run()


if __name__ == "__main__":
    main()
