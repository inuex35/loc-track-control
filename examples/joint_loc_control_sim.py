"""Unified estimation + control on a *single* factor graph (one optimize()).

Where ``loc_control_sim.py`` runs two graphs in a pipeline (an MHE solve, then a
separate MPC solve, coupled only by passing the estimate), this demo uses
:class:`gtsam_mpc.JointEstimatorMPC`, which builds the whole thing -- past *and*
future -- as one ``NonlinearFactorGraph`` and solves it with a single
``optimize()`` call (the unified estimation+control formulation of Fig. 5 in
Abdelkarim et al., IEEE Access 2025).

Controls / run: same as loc_control_sim.py.

    python examples/joint_loc_control_sim.py

Headless test: SDL_VIDEODRIVER=dummy GTSAM_MPC_MAX_FRAMES=<n>.
"""

import os
import sys

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from gtsam_mpc import BicycleModel, JointEstimatorMPC
import examples.path_following_sim as pf
from examples._racecar_app import (
    RacecarApp, DT, MPC_DT, WHEELBASE, GPS_SIGMAS, SPEED_SIGMA,
)


def run(seed=0, steps=300, gps_sigma=1.0, target_speed=8.0, path_gen=None):
    """Headless joint estimation+control run; returns localization/tracking metrics."""
    rng = np.random.default_rng(seed)
    joint = JointEstimatorMPC(gps_sigma=gps_sigma)
    plant = BicycleModel(wheelbase=WHEELBASE, dt=DT)
    gen = path_gen if path_gen is not None else pf.PATHS["4"][1]   # racetrack
    path = pf.make_path(gen)
    curv = pf.path_curvature(path)

    true = pf.reset_on_path(path, target_speed)
    joint.reset(true)
    estimate = true.copy()
    u = np.zeros(2)
    loc, cte, deltas = [], [], []
    for _ in range(steps):
        true = plant.step(true, u)
        gps = true[:2] + rng.normal(0.0, gps_sigma, size=2)
        v_meas = true[3] + rng.normal(0.0, SPEED_SIGMA)
        near = pf.nearest_index(path, estimate[:2])
        refs = pf.reference_trajectory(path, curv, near, target_speed, joint.N, MPC_DT)
        estimate, _states, ctrls = joint.step(u, gps, v_meas, refs)
        u = ctrls[0]
        deltas.append(float(u[1]))
        loc.append(float(np.linalg.norm(true[:2] - estimate[:2])))
        cte.append(pf.cross_track_error(path, true[:2]))

    loc = np.array(loc[50:]); cte = np.array(cte[50:]); rate = np.diff(np.array(deltas[50:]))
    return {
        "loc_rmse": float(np.sqrt(np.mean(loc ** 2))),
        "cte_mean": float(cte.mean()),
        "steer_rate_std": float(np.std(rate)),
    }


class JointApp(RacecarApp):
    """Single-graph estimation+control: one optimize() per step."""

    caption = "gtsam-mpc: joint estimation + control (single graph)"

    def setup(self) -> None:
        self.joint = JointEstimatorMPC(gps_sigma=GPS_SIGMAS[self.gps_idx])
        self.u_cmd = np.zeros(2)

    def set_gps_sigma(self, sigma: float) -> None:
        self.joint.set_gps_sigma(sigma)

    def reset_engine(self) -> None:
        self.joint.reset(self.true_state)
        self.estimate = self.true_state.copy()
        self.u_cmd = np.zeros(2)

    def horizon(self) -> int:
        return self.joint.N

    def window_points(self) -> list:
        j = self.joint
        lo = max(0, j.k - j.W)
        return [pf.world_to_screen(j.Xv[t][:2]) for t in range(lo, j.k + 1)]

    def advance(self, refs):
        self.true_state = self.plant.step(self.true_state, self.u_cmd)
        gps, v_meas = self.measure()
        self.estimate, states, ctrls = self.joint.step(self.u_cmd, gps, v_meas, refs)
        self.u_cmd = ctrls[0]
        self._last_gps = gps
        self._last_delta = float(self.u_cmd[1])
        return states

    def hud_lines(self) -> list:
        return self.status_lines() + [
            "SINGLE GRAPH: one optimize() solves estimation window + control horizon",
            "1-5 path  up/down speed  [ ] gps-noise  space pause  r reset  esc quit"
            + self.paused_suffix(),
        ]


def main():
    JointApp().run()


if __name__ == "__main__":
    main()
