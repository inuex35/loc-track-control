"""統合 (full stack): localization + multi-obstacle tracking + control, one graph.

Combines all three factor-graph layers via :class:`gtsam_mpc.JointLocTrackControl`
in a single ``optimize()`` per step, on a racetrack with slow *moving* traffic:

    * the ego car localizes from noisy **GPS** + wheel-speed (estimation window),
    * each **obstacle** is a slower car driving along the track; it is tracked
      from noisy detections (as a constant-velocity target) and predicted over
      the horizon, and
    * the (faster) ego follows the path while **overtaking / avoiding** every
      obstacle, with each keep-out radius inflated by that obstacle track's
      covariance (uncertainty-aware).

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
from examples._racecar_app import EST_COLOR, GPS_COLOR
from examples._viz import max_frames_from_env

COURSE = "4"            # racetrack
N_OBSTACLES = 5
OBS_OFFSET = 3.0        # lateral offset of obstacles from the path [m]
OBS_SPACING = 55        # path-index spacing between obstacles
OBS_SPEED = 4.0         # obstacle speed along the track (slow traffic) [m/s]
SAFETY_RADIUS = 3.0
N_SIGMA = 1.5
GPS_SIGMA = 0.6
SPEED_SIGMA = 0.3
OBS_DET_SIGMA = 0.6
TARGET_SPEED = 8.0
WHEELBASE = pf.WHEELBASE

OBS_COLOR = (255, 90, 90)
PRED_COLOR = (255, 150, 150)


def _tangent_normal(path, idx):
    n = len(path)
    t = path[(idx + 1) % n] - path[(idx - 1) % n]
    t = t / (np.linalg.norm(t) + 1e-9)
    return t, np.array([-t[1], t[0]])   # (tangent, left normal)


class TrafficObstacle:
    """A slower car driving along the track, offset to one side of the path."""

    def __init__(self, path, idx, offset, speed):
        self.path = path
        self.n = len(path)
        self.idx = float(idx)
        self.offset = offset
        self.speed = speed

    def advance(self, dt, resolution=paths.PATH_RESOLUTION):
        self.idx = (self.idx + self.speed * dt / resolution) % self.n

    def position(self):
        i = int(self.idx) % self.n
        _, normal = _tangent_normal(self.path, i)
        return self.path[i] + self.offset * normal

    def cv_state(self):
        """Constant-velocity state ``[px, py, vx, vy]`` (for the tracker init)."""
        i = int(self.idx) % self.n
        tangent, normal = _tangent_normal(self.path, i)
        p = self.path[i] + self.offset * normal
        v = self.speed * tangent
        return np.array([p[0], p[1], v[0], v[1]])


def _scenario(course=COURSE, n_obstacles=N_OBSTACLES, offset=OBS_OFFSET,
              spacing=OBS_SPACING, obs_speed=OBS_SPEED):
    path = paths.make_path(paths.PATHS[course][1])
    curv = paths.path_curvature(path)
    true = paths.reset_on_path(path, TARGET_SPEED)
    near0 = paths.nearest_index(path, true[:2])
    obstacles = [
        TrafficObstacle(path, (near0 + 60 + j * spacing) % len(path),
                        (1.0 if j % 2 == 0 else -1.0) * offset, obs_speed)
        for j in range(n_obstacles)
    ]
    return path, curv, true, obstacles


def run(seed: int = 0, steps: int = 240, gps_sigma: float = GPS_SIGMA,
        n_obstacles: int = N_OBSTACLES, safety_radius: float = SAFETY_RADIUS,
        n_sigma: float = N_SIGMA) -> dict:
    """Headless full-stack run; returns localization / clearance / progress."""
    rng = np.random.default_rng(seed)
    jtc = JointLocTrackControl(n_obstacles=n_obstacles, gps_sigma=gps_sigma,
                               safety_radius=safety_radius, n_sigma=n_sigma)
    plant = BicycleModel(wheelbase=WHEELBASE, dt=pf.DT)
    path, curv, true, obs = _scenario(n_obstacles=n_obstacles)
    jtc.reset(true, [o.cv_state() for o in obs])
    estimate = true.copy()
    u = np.zeros(2)

    loc, min_clear = [], np.inf
    prev_idx = paths.nearest_index(path, true[:2])
    progress = 0
    for _ in range(steps):
        for o in obs:
            o.advance(pf.DT)
        true = plant.step(true, u)
        gps = true[:2] + rng.normal(0.0, gps_sigma, 2)
        v_meas = true[3] + rng.normal(0.0, SPEED_SIGMA)
        dets = [o.position() + rng.normal(0.0, OBS_DET_SIGMA, 2) for o in obs]
        near = paths.nearest_index(path, estimate[:2])
        refs = paths.reference_trajectory(path, curv, near, TARGET_SPEED, jtc.N, pf.MPC_DT)
        estimate, _s, ctrls, _oe, _op = jtc.step(u, gps, v_meas, dets, refs)
        u = ctrls[0]
        loc.append(np.linalg.norm(true[:2] - estimate[:2]))
        min_clear = min(min_clear,
                        min(np.linalg.norm(true[:2] - o.position()) for o in obs))
        cur = paths.nearest_index(path, true[:2])
        d = (cur - prev_idx) % len(path)
        progress += d if d < len(path) // 2 else 0
        prev_idx = cur

    loc = np.array(loc[30:])
    return {
        "loc_rmse": float(np.sqrt(np.mean(loc ** 2))),
        "min_clearance": float(min_clear),
        "progress": int(progress),
        "path_len": len(path),
    }


def main() -> None:
    max_frames = max_frames_from_env()
    rng = np.random.default_rng(0)

    pygame.init()
    screen = pygame.display.set_mode((pf.WIDTH, pf.HEIGHT))
    pygame.display.set_caption("gtsam-mpc: localization + moving-obstacle tracking + control")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    plant = BicycleModel(wheelbase=WHEELBASE, dt=pf.DT)

    def setup():
        path, curv, true, obs = _scenario()
        jtc = JointLocTrackControl(n_obstacles=len(obs), gps_sigma=GPS_SIGMA,
                                   safety_radius=SAFETY_RADIUS, n_sigma=N_SIGMA)
        jtc.reset(true, [o.cv_state() for o in obs])
        det_hist = [[] for _ in obs]
        preds = [np.tile(o.position(), (jtc.N + 1, 1)) for o in obs]
        return jtc, path, curv, true, obs, det_hist, preds

    jtc, path, curv, true, obs, det_hist, preds = setup()
    estimate = true.copy()
    u = np.zeros(2)
    states = np.tile(true, (jtc.N + 1, 1))
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
                    jtc, path, curv, true, obs, det_hist, preds = setup()
                    estimate = true.copy(); u = np.zeros(2)

        if not paused:
            for o in obs:
                o.advance(pf.DT)
            true = plant.step(true, u)
            gps = true[:2] + rng.normal(0.0, GPS_SIGMA, 2)
            v_meas = true[3] + rng.normal(0.0, SPEED_SIGMA)
            dets = [o.position() + rng.normal(0.0, OBS_DET_SIGMA, 2) for o in obs]
            near = paths.nearest_index(path, estimate[:2])
            refs = paths.reference_trajectory(path, curv, near, TARGET_SPEED, jtc.N, pf.MPC_DT)
            estimate, states, ctrls, oest, opred = jtc.step(u, gps, v_meas, dets, refs)
            u = ctrls[0]
            for j in range(len(obs)):
                preds[j] = opred[j]
                det_hist[j].append(dets[j])
                if len(det_hist[j]) > 60:
                    det_hist[j].pop(0)

        # --- render ---
        screen.fill(pf.BG)
        pf.draw_path(screen, path)
        for j, o in enumerate(obs):
            for d in det_hist[j]:
                pygame.draw.circle(screen, GPS_COLOR, pf.world_to_screen(d), 2)
            pygame.draw.lines(screen, PRED_COLOR, False,
                              [pf.world_to_screen(p) for p in preds[j]], 1)
            oc = pf.world_to_screen(o.position())
            radius = SAFETY_RADIUS + N_SIGMA * jtc.std.get((j, 0), 0.0)
            pygame.draw.circle(screen, OBS_COLOR, oc, int(0.8 * pf.PIXELS_PER_METER))
            pygame.draw.circle(screen, OBS_COLOR, oc, int(radius * pf.PIXELS_PER_METER), 1)
        pygame.draw.lines(screen, pf.PLAN_COLOR, False,
                          [pf.world_to_screen(s[:2]) for s in states], 2)
        pf.draw_car(screen, true, float(u[1]))
        ex, ey = pf.world_to_screen(estimate[:2])
        pygame.draw.circle(screen, EST_COLOR, (ex, ey), 6, 2)

        loc_err = float(np.linalg.norm(true[:2] - estimate[:2]))
        clr = min(float(np.linalg.norm(true[:2] - o.position())) for o in obs)
        lines = [
            f"loc err={loc_err:.2f} m   moving obstacles={len(obs)} @ {OBS_SPEED:.0f} m/s   "
            f"nearest clearance={clr:.2f} m",
            f"base safety radius={SAFETY_RADIUS:.0f} m + {N_SIGMA}*track-std (uncertainty-aware)",
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
