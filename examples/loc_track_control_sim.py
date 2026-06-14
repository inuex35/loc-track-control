"""統合 (full stack): localization + obstacle tracking + control, one graph.

Combines all three factor-graph layers via :class:`gtsam_mpc.JointLocTrackControl`
in a single ``optimize()`` per step:

    * the ego car localizes from noisy **GPS** + wheel-speed (estimation window),
    * a moving **obstacle** is tracked from noisy detections and predicted over
      the horizon, and
    * the car follows the path while **avoiding** the obstacle, with the keep-out
      radius inflated by the obstacle track's covariance (uncertainty-aware).

Controls:
    * Space  -- pause / resume
    * R      -- reset
    * Esc/Q  -- quit

Run:  python examples/loc_track_control_sim.py
Headless test:  SDL_VIDEODRIVER=dummy GTSAM_MPC_MAX_FRAMES=<n> python examples/loc_track_control_sim.py
"""

import os
import sys

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame

from gtsam_mpc import BicycleModel, JointLocTrackControl, paths
import examples.path_following_sim as pf
from examples._racecar_app import (
    DT, MPC_DT, WHEELBASE, TRUE_COLOR, EST_COLOR, GPS_COLOR,
)
from examples._viz import max_frames_from_env

SAFETY_RADIUS = 4.0
N_SIGMA = 1.5
GPS_SIGMA = 0.6
SPEED_SIGMA = 0.3
OBS_DET_SIGMA = 0.6
OBS_COLOR = (255, 90, 90)
PRED_COLOR = (255, 150, 150)

_Fp = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def _scenario(target_speed=8.0):
    """Circle path with one obstacle sitting on the path ahead, drifting across."""
    path = paths.make_path(paths.circle)
    curv = paths.path_curvature(path)
    true = paths.reset_on_path(path, target_speed)
    near0 = paths.nearest_index(path, true[:2])
    p = path[(near0 + 60) % len(path)]
    obstacle = np.array([p[0], p[1], 0.3, 0.0])
    return path, curv, true, obstacle


def run(seed: int = 0, steps: int = 160, gps_sigma: float = GPS_SIGMA,
        safety_radius: float = SAFETY_RADIUS, n_sigma: float = N_SIGMA) -> dict:
    """Headless full-stack run; returns localization + clearance metrics."""
    rng = np.random.default_rng(seed)
    jtc = JointLocTrackControl(n_obstacles=1, gps_sigma=gps_sigma,
                               safety_radius=safety_radius, n_sigma=n_sigma)
    plant = BicycleModel(wheelbase=WHEELBASE, dt=DT)
    path, curv, true, obs = _scenario()
    jtc.reset(true, [obs.copy()])
    estimate = true.copy()
    u = np.zeros(2)

    loc, clear = [], []
    for _ in range(steps):
        true = plant.step(true, u)
        obs = _Fp @ obs
        gps = true[:2] + rng.normal(0.0, gps_sigma, 2)
        v_meas = true[3] + rng.normal(0.0, SPEED_SIGMA)
        det = obs[:2] + rng.normal(0.0, OBS_DET_SIGMA, 2)
        near = paths.nearest_index(path, estimate[:2])
        refs = paths.reference_trajectory(path, curv, near, 8.0, jtc.N, MPC_DT)
        estimate, _states, ctrls, _oest, _opred = jtc.step(u, gps, v_meas, [det], refs)
        u = ctrls[0]
        loc.append(np.linalg.norm(true[:2] - estimate[:2]))
        clear.append(np.linalg.norm(true[:2] - obs[:2]))

    loc = np.array(loc[30:])
    return {
        "loc_rmse": float(np.sqrt(np.mean(loc ** 2))),
        "min_clearance": float(min(clear)),
        "cte": float(paths.cross_track_error(path, true[:2])),
    }


def main() -> None:
    max_frames = max_frames_from_env()
    rng = np.random.default_rng(0)

    pygame.init()
    screen = pygame.display.set_mode((pf.WIDTH, pf.HEIGHT))
    pygame.display.set_caption("gtsam-mpc: localization + tracking + control")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    jtc = JointLocTrackControl(n_obstacles=1, gps_sigma=GPS_SIGMA,
                               safety_radius=SAFETY_RADIUS, n_sigma=N_SIGMA)
    plant = BicycleModel(wheelbase=WHEELBASE, dt=DT)
    path, curv, true, obs = _scenario()
    jtc.reset(true, [obs.copy()])
    estimate = true.copy()
    u = np.zeros(2)
    det_hist: list = []
    pred = np.tile(obs[:2], (jtc.N + 1, 1))
    paused = False

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
                    path, curv, true, obs = _scenario()
                    jtc.reset(true, [obs.copy()])
                    estimate = true.copy(); u = np.zeros(2); det_hist = []

        if not paused:
            true = plant.step(true, u)
            obs = _Fp @ obs
            gps = true[:2] + rng.normal(0.0, GPS_SIGMA, 2)
            v_meas = true[3] + rng.normal(0.0, SPEED_SIGMA)
            det = obs[:2] + rng.normal(0.0, OBS_DET_SIGMA, 2)
            near = paths.nearest_index(path, estimate[:2])
            refs = paths.reference_trajectory(path, curv, near, 8.0, jtc.N, MPC_DT)
            estimate, states, ctrls, oest, opred = jtc.step(u, gps, v_meas, [det], refs)
            u = ctrls[0]
            pred = opred[0]
            det_hist.append(det)
            if len(det_hist) > 80:
                det_hist.pop(0)

        # --- render ---
        screen.fill(pf.BG)
        pf.draw_path(screen, path)
        for d in det_hist:
            pygame.draw.circle(screen, GPS_COLOR, pf.world_to_screen(d), 2)
        # obstacle prediction + uncertainty-inflated safety ring
        pygame.draw.lines(screen, PRED_COLOR, False,
                          [pf.world_to_screen(p) for p in pred], 1)
        oc = pf.world_to_screen(obs[:2])
        radius = SAFETY_RADIUS + N_SIGMA * jtc.std.get((0, 0), 0.0)
        pygame.draw.circle(screen, OBS_COLOR, oc, int(0.8 * pf.PIXELS_PER_METER))
        pygame.draw.circle(screen, OBS_COLOR, oc, int(radius * pf.PIXELS_PER_METER), 1)
        pygame.draw.lines(screen, pf.PLAN_COLOR, False,
                          [pf.world_to_screen(s[:2]) for s in states], 2)
        pf.draw_car(screen, true, float(u[1]))
        ex, ey = pf.world_to_screen(estimate[:2])
        pygame.draw.circle(screen, EST_COLOR, (ex, ey), 6, 2)

        loc_err = float(np.linalg.norm(true[:2] - estimate[:2]))
        clr = float(np.linalg.norm(true[:2] - obs[:2]))
        lines = [
            f"loc err={loc_err:.2f} m   true cross-track={pf.cross_track_error(path, true[:2]):.2f} m",
            f"obstacle clearance={clr:.2f} m   safety radius={radius:.2f} m "
            f"(base {SAFETY_RADIUS:.0f} + {N_SIGMA}*std)",
            "ONE GRAPH: localize + track + avoid    green=true blue=est red=obstacle",
            "space pause   r reset   esc quit" + ("   [PAUSED]" if paused else ""),
        ]
        for i, text in enumerate(lines):
            screen.blit(font.render(text, True, pf.TEXT_COLOR), (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(pf.FPS)
        frame += 1
        if max_frames is not None and frame >= max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
