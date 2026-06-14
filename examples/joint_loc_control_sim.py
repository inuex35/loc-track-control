"""Unified estimation + control on a *single* factor graph (one optimize()).

Where ``loc_control_sim.py`` runs two graphs in a pipeline (an MHE solve, then a
separate MPC solve, coupled only by passing the estimate), this demo builds the
whole thing -- past *and* future -- as one ``NonlinearFactorGraph`` and solves
it with a single ``optimize()`` call. This is the unified estimation+control
formulation of Fig. 5 in Abdelkarim et al. (IEEE Access 2025): one shared
current-state node ``X(k)`` joins

    * the estimation window  X(k-W) ... X(k)   (anchor + motion + GPS + speed)
    * the control horizon    X(k) ... X(k+N), U(k) ... U(k+N-1)
      (hard dynamics + tracking cost + input barriers)

The current state X(k) carries no tracking cost and no hard prior: it is fixed
by the measurement factors alone, while the controller plans forward from it.
Solving the joint graph yields the smoothed estimate and the optimal plan at
once.

Controls / run: same as loc_control_sim.py.

    python examples/joint_loc_control_sim.py

Headless test: SDL_VIDEODRIVER=dummy GTSAM_MPC_MAX_FRAMES=<n>.
"""

import os
import sys

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gtsam
import pygame
from gtsam.symbol_shorthand import U, X

from gtsam_mpc import BicycleMPC, BicycleModel
import examples.path_following_sim as pf
from examples.loc_control_sim import (
    MovingHorizonEstimator, GPS_SIGMAS, SPEED_SIGMA,
    TRUE_COLOR, EST_COLOR, GPS_COLOR, WINDOW_COLOR, TEXT_COLOR,
)

WIDTH, HEIGHT = pf.WIDTH, pf.HEIGHT
DT, MPC_DT, WHEELBASE = pf.DT, pf.MPC_DT, pf.WHEELBASE


class JointEstimatorMPC:
    """Estimation window + control horizon in one graph, one solve per step.

    Reuses the factor error builders and noise models of
    :class:`MovingHorizonEstimator` (estimation side) and
    :class:`~gtsam_mpc.BicycleMPC` (control side); only the graph assembly is
    new. Input bounds use the barrier penalty so the whole thing stays a single
    ``optimize()`` (no Augmented-Lagrangian outer loop).
    """

    def __init__(self, window=12, horizon=None, gps_sigma=1.0, target_speed=8.0):
        self.model = BicycleModel(wheelbase=WHEELBASE, dt=MPC_DT)
        # est uses a plant-rate model for its motion factors over the window.
        self.est = MovingHorizonEstimator(
            BicycleModel(wheelbase=WHEELBASE, dt=DT), window=window,
            gps_sigma=gps_sigma)
        self.mpc = pf.make_controller()      # supplies cost/dyn/box builders
        self.mpc.constraint_mode = "barrier"  # keep to a single optimize()
        self.mpc.barrier_weight = 500.0
        self.N = self.mpc.horizon if horizon is None else int(horizon)
        self.W = window

        # GTSAM noise models.
        self.cost_x = gtsam.noiseModel.Gaussian.Information(self.mpc.Q)
        self.cost_u = gtsam.noiseModel.Gaussian.Information(self.mpc.R)
        self.cost_xf = gtsam.noiseModel.Gaussian.Information(self.mpc.Qf)
        self.constrained = gtsam.noiseModel.Constrained.All(4)
        self.u_barrier = gtsam.noiseModel.Diagonal.Precisions(
            np.full(2, self.mpc.barrier_weight))

        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(60)
        self._params = params

    def set_gps_sigma(self, sigma):
        self.est.set_gps_sigma(sigma)

    def reset(self, x0):
        x0 = np.asarray(x0, float)
        self.k = 0
        self.controls = []                      # controls[t]: t -> t+1 (applied)
        self.gps = [x0[:2].copy()]
        self.vmeas = [float(x0[3])]
        self.Xv = {0: x0.copy()}                # warm-start state values
        self.Uv = {}                            # warm-start control values
        self.estimate = x0.copy()

    def step(self, u_applied, gps, v_meas, reference):
        """One joint solve. Returns (estimate, plan_states, plan_controls)."""
        e, m = self.est, self.mpc
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
        graph.add(gtsam.CustomFactor(e.anchor_noise, [X(lo)],
                                     e._anchor_error(self.Xv[lo])))
        for t in range(lo, k):
            graph.add(gtsam.CustomFactor(e.proc_noise, [X(t), X(t + 1)],
                                         e._motion_error(self.controls[t])))
        for t in range(lo, k + 1):
            graph.add(gtsam.CustomFactor(e.gps_noise, [X(t)], e._gps_error(self.gps[t])))
            graph.add(gtsam.CustomFactor(e.speed_noise, [X(t)],
                                         e._speed_error(self.vmeas[t])))
        # --- control horizon (current + future) ---
        for t in range(k, k + N):
            graph.add(gtsam.CustomFactor(self.constrained, [X(t), U(t), X(t + 1)],
                                         m._dynamics_error()))
            graph.add(gtsam.CustomFactor(self.cost_u, [U(t)], m._control_cost_error))
            graph.add(gtsam.CustomFactor(self.u_barrier, [U(t)],
                                         m._box_error(m.u_lower, m.u_upper)))
        for j in range(1, N + 1):                # X(k+1..k+N) track the reference
            noise = self.cost_xf if j == N else self.cost_x
            graph.add(gtsam.CustomFactor(noise, [X(k + j)],
                                         m._state_cost_error(reference[j])))

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


def main():
    max_frames_env = os.environ.get("GTSAM_MPC_MAX_FRAMES")
    max_frames = int(max_frames_env) if max_frames_env else None
    rng = np.random.default_rng(0)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("gtsam-mpc: joint estimation + control (single graph)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    plant = BicycleModel(wheelbase=WHEELBASE, dt=DT)
    gps_idx = 1
    joint = JointEstimatorMPC(gps_sigma=GPS_SIGMAS[gps_idx])

    path_key = "1"
    path_name, generator = pf.PATHS[path_key]
    path = pf.make_path(generator)
    curvature = pf.path_curvature(path)
    target_speed = 8.0
    paused = False

    def reset_all():
        nonlocal true_state, estimate, u_cmd, true_trail, est_trail, gps_pts, rmse_sq, nrmse
        true_state = pf.reset_on_path(path, target_speed)
        joint.reset(true_state)
        estimate = true_state.copy()
        u_cmd = np.zeros(2)
        true_trail, est_trail, gps_pts = [], [], []
        rmse_sq, nrmse = 0.0, 0

    true_state = estimate = u_cmd = None
    true_trail = est_trail = gps_pts = None
    rmse_sq = nrmse = 0
    reset_all()

    frame = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset_all()
                elif event.key == pygame.K_UP:
                    target_speed = min(target_speed + 1.0, pf.V_BOUNDS[1])
                elif event.key == pygame.K_DOWN:
                    target_speed = max(target_speed - 1.0, 0.0)
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                    gps_idx = int(np.clip(
                        gps_idx + (1 if event.key == pygame.K_RIGHTBRACKET else -1),
                        0, len(GPS_SIGMAS) - 1))
                    joint.set_gps_sigma(GPS_SIGMAS[gps_idx])
                else:
                    name = pygame.key.name(event.key)
                    if name in pf.PATHS:
                        path_key = name
                        path_name, generator = pf.PATHS[path_key]
                        path = pf.make_path(generator)
                        curvature = pf.path_curvature(path)
                        reset_all()

        near = pf.nearest_index(path, estimate[:2])
        refs = pf.reference_trajectory(path, curvature, near, target_speed,
                                       joint.N, MPC_DT)
        if not paused:
            true_state = plant.step(true_state, u_cmd)
            gps = true_state[:2] + rng.normal(0.0, GPS_SIGMAS[gps_idx], size=2)
            v_meas = true_state[3] + rng.normal(0.0, SPEED_SIGMA)
            estimate, plan_states, plan_ctrls = joint.step(u_cmd, gps, v_meas, refs)
            u_cmd = plan_ctrls[0]

            true_trail.append(pf.world_to_screen(true_state[:2]))
            est_trail.append(pf.world_to_screen(estimate[:2]))
            gps_pts.append(pf.world_to_screen(gps))
            for buf in (true_trail, est_trail, gps_pts):
                if len(buf) > 400:
                    buf.pop(0)
            rmse_sq += float(np.sum((true_state[:2] - estimate[:2]) ** 2))
            nrmse += 1
        else:
            plan_states = np.array([joint.Xv[joint.k + j] for j in range(joint.N + 1)])

        # --- render ---
        screen.fill(pf.BG)
        pf.draw_path(screen, path)
        for pt in gps_pts[-120:]:
            pygame.draw.circle(screen, GPS_COLOR, pt, 2)
        if len(true_trail) > 1:
            pygame.draw.lines(screen, TRUE_COLOR, False, true_trail, 2)
        if len(est_trail) > 1:
            pygame.draw.lines(screen, EST_COLOR, False, est_trail, 2)

        lo = max(0, joint.k - joint.W)
        for t in range(lo, joint.k + 1):
            pygame.draw.circle(screen, WINDOW_COLOR, pf.world_to_screen(joint.Xv[t][:2]), 3, 1)
        plan_pts = [pf.world_to_screen(s[:2]) for s in plan_states]
        if len(plan_pts) > 1:
            pygame.draw.lines(screen, pf.PLAN_COLOR, False, plan_pts, 2)

        pf.draw_car(screen, true_state, float(u_cmd[1]))
        ex, ey = pf.world_to_screen(estimate[:2])
        pygame.draw.circle(screen, EST_COLOR, (ex, ey), 6, 2)

        loc_err = float(np.linalg.norm(true_state[:2] - estimate[:2]))
        rmse = float(np.sqrt(rmse_sq / nrmse)) if nrmse else 0.0
        cte = pf.cross_track_error(path, true_state[:2])
        lines = [
            f"path: {path_name}   target speed={target_speed:.1f} m/s   v={true_state[3]:+.2f}",
            f"GPS sigma={GPS_SIGMAS[gps_idx]:.1f} m   loc err={loc_err:.2f} m   "
            f"loc RMSE={rmse:.2f} m   true cross-track={cte:.2f} m",
            "SINGLE GRAPH: one optimize() solves estimation window + control horizon",
            "1-5 path  up/down speed  [ ] gps-noise  space pause  r reset  esc quit"
            + ("   [PAUSED]" if paused else ""),
        ]
        for i, text in enumerate(lines):
            screen.blit(font.render(text, True, TEXT_COLOR), (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(pf.FPS)
        frame += 1
        if max_frames is not None and frame >= max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
