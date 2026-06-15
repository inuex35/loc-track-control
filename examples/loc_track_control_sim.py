"""統合 (full stack): localization + multi-obstacle tracking + control, one graph.

Combines all three factor-graph layers via :class:`gtsam_mpc.JointLocTrackControl`
in a single ``optimize()`` per step, on a racetrack with slow *moving* traffic:

    * the ego car localizes from noisy **GPS** + wheel-speed (estimation window),
    * each **obstacle** is a slower car driving along the track; it is tracked
      from noisy detections (range-dependent: closer = observed more strongly)
      and predicted over the horizon, and
    * the (faster) ego follows the path while **overtaking / avoiding** every
      obstacle, with each keep-out radius inflated by that obstacle track's
      covariance (uncertainty-aware).

The whole scenario is encapsulated in :class:`Sim`, which ``main`` (interactive),
``run`` (headless metrics) and ``iter_frames`` (GIF recording) all drive.

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

COURSE = "1"            # circle (gentle curves -- the car never has to reverse)
N_OBSTACLES = 5
OBS_OFFSET = 3.0        # lateral offset of obstacles from the path [m]
OBS_SPACING = 55        # path-index spacing between obstacles
OBS_SPEED = 4.0         # obstacle speed along the track (slow traffic) [m/s]
SAFETY_RADIUS = 3.0
N_SIGMA = 1.5
GPS_SIGMA = 0.3
SPEED_SIGMA = 0.3
# Range-dependent detection noise (matches JointLocTrackControl's sensor model):
# a closer obstacle is detected more accurately, so it is observed more strongly.
DET_SIGMA_NEAR = 0.25
DET_SIGMA_RATE = 0.045
DET_SIGMA_MAX = 1.2
TARGET_SPEED = 8.0
WHEELBASE = pf.WHEELBASE
WARMUP = 25             # steps to settle the MHE window + obstacle tracks before display

OBS_COLOR = (255, 90, 90)
PRED_COLOR = (255, 150, 150)
EGO_GPS_COLOR = (255, 210, 90)    # raw GPS fixes (the localization noise)
TRUE_TRAIL_COLOR = (90, 200, 140)
EST_TRAIL_COLOR = (90, 150, 255)
UNCERT_COLOR = (130, 170, 255)    # localization uncertainty ellipse


def _cov_ellipse_points(center, cov, n_sigma=2.0, k=24):
    """World-frame points of the ``n_sigma`` covariance ellipse around ``center``."""
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    axes = n_sigma * np.sqrt(vals)
    return [center + vecs @ (axes * np.array([np.cos(a), np.sin(a)]))
            for a in np.linspace(0.0, 2 * np.pi, k, endpoint=False)]


def _det_sigma(dist: float) -> float:
    """Detection 1-sigma grows with range: closer obstacles are seen better."""
    return min(DET_SIGMA_MAX, DET_SIGMA_NEAR + DET_SIGMA_RATE * dist)


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


class Sim:
    """One closed-loop full-stack scenario: ego + traffic + the joint solver.

    Owns the RNG and all visual state, so ``step`` advances one control step and
    ``draw`` renders the current frame. ``main``/``run``/``iter_frames`` differ
    only in how they drive it.
    """

    def __init__(self, seed: int = 0, gps_sigma: float = GPS_SIGMA,
                 n_obstacles: int = N_OBSTACLES, safety_radius: float = SAFETY_RADIUS,
                 n_sigma: float = N_SIGMA):
        self.rng = np.random.default_rng(seed)
        self.gps_sigma = gps_sigma
        self.jtc = JointLocTrackControl(n_obstacles=n_obstacles, gps_sigma=gps_sigma,
                                        safety_radius=safety_radius, n_sigma=n_sigma)
        self.plant = BicycleModel(wheelbase=WHEELBASE, dt=pf.DT)
        self.path, self.curv, self.true, self.obs = _scenario(n_obstacles=n_obstacles)
        self.jtc.reset(self.true, [o.cv_state() for o in self.obs])
        self.estimate = self.true.copy()
        self.u = np.zeros(2)
        self.states = np.tile(self.true, (self.jtc.N + 1, 1))
        self.preds = [np.tile(o.position(), (self.jtc.N + 1, 1)) for o in self.obs]
        self.det_hist = [[] for _ in self.obs]
        self.ego_gps: list = []
        self.true_trail: list = []
        self.est_trail: list = []

    def step(self) -> None:
        for o in self.obs:
            o.advance(pf.DT)
        self.true = self.plant.step(self.true, self.u)
        gps = self.true[:2] + self.rng.normal(0.0, self.gps_sigma, 2)
        v_meas = self.true[3] + self.rng.normal(0.0, SPEED_SIGMA)
        dets = [o.position()
                + self.rng.normal(0.0, _det_sigma(np.linalg.norm(self.true[:2] - o.position())), 2)
                for o in self.obs]
        near = paths.nearest_index(self.path, self.estimate[:2])
        refs = paths.reference_trajectory(self.path, self.curv, near, TARGET_SPEED,
                                          self.jtc.N, pf.MPC_DT)
        self.estimate, self.states, ctrls, _oe, opred = self.jtc.step(
            self.u, gps, v_meas, dets, refs)
        self.u = ctrls[0]
        for j in range(len(self.obs)):
            self.preds[j] = opred[j]
            self.det_hist[j].append(dets[j])
            if len(self.det_hist[j]) > 60:
                self.det_hist[j].pop(0)
        self.ego_gps.append(gps.copy())
        self.true_trail.append(self.true[:2].copy())
        self.est_trail.append(self.estimate[:2].copy())
        for buf in (self.ego_gps, self.true_trail, self.est_trail):
            if len(buf) > 90:
                buf.pop(0)

    def clearance(self) -> float:
        return min(float(np.linalg.norm(self.true[:2] - o.position())) for o in self.obs)

    def clear_trails(self) -> None:
        self.ego_gps.clear()
        self.true_trail.clear()
        self.est_trail.clear()
        for h in self.det_hist:
            h.clear()

    def warmup(self, n: int = WARMUP) -> None:
        """Run ``n`` steps and drop their trails, so display starts settled."""
        for _ in range(n):
            self.step()
        self.clear_trails()

    def draw(self, screen, font, paused: bool = False) -> None:
        screen.fill(pf.BG)
        pf.draw_path(screen, self.path)
        for j, o in enumerate(self.obs):
            for d in self.det_hist[j]:
                pygame.draw.circle(screen, OBS_COLOR, pf.world_to_screen(d), 1)
            pygame.draw.lines(screen, PRED_COLOR, False,
                              [pf.world_to_screen(p) for p in self.preds[j]], 1)
            oc = pf.world_to_screen(o.position())
            radius = SAFETY_RADIUS + N_SIGMA * self.jtc.std.get((j, 0), 0.0)
            pygame.draw.circle(screen, OBS_COLOR, oc, int(0.8 * pf.PIXELS_PER_METER))
            pygame.draw.circle(screen, OBS_COLOR, oc, int(radius * pf.PIXELS_PER_METER), 1)

        # Ego localization noise: raw GPS fixes, true vs estimated trail, 2-sigma ellipse.
        for g in self.ego_gps:
            pygame.draw.circle(screen, EGO_GPS_COLOR, pf.world_to_screen(g), 2)
        if len(self.true_trail) > 1:
            pygame.draw.lines(screen, TRUE_TRAIL_COLOR, False,
                              [pf.world_to_screen(p) for p in self.true_trail], 2)
        if len(self.est_trail) > 1:
            pygame.draw.lines(screen, EST_TRAIL_COLOR, False,
                              [pf.world_to_screen(p) for p in self.est_trail], 2)
        if self.jtc.ego_pos_cov is not None:
            ell = _cov_ellipse_points(self.estimate[:2], self.jtc.ego_pos_cov, 2.0)
            pygame.draw.polygon(screen, UNCERT_COLOR,
                                [pf.world_to_screen(p) for p in ell], 1)

        pygame.draw.lines(screen, pf.PLAN_COLOR, False,
                          [pf.world_to_screen(s[:2]) for s in self.states], 2)
        pf.draw_car(screen, self.true, float(self.u[1]))
        ex, ey = pf.world_to_screen(self.estimate[:2])
        pygame.draw.circle(screen, EST_COLOR, (ex, ey), 6, 2)

        loc_err = float(np.linalg.norm(self.true[:2] - self.estimate[:2]))
        sd = (np.sqrt(np.trace(self.jtc.ego_pos_cov) / 2.0)
              if self.jtc.ego_pos_cov is not None else 0.0)
        lines = [
            f"loc err={loc_err:.2f} m   est 1-sigma={sd:.2f} m   "
            f"obstacles={len(self.obs)} @ {OBS_SPEED:.0f} m/s   "
            f"nearest clearance={self.clearance():.2f} m",
            "amber=raw GPS fixes   green=true path   blue=estimate path   "
            "blue ellipse=2-sigma localization uncertainty",
            "ONE GRAPH (single optimize): localize + track + avoid   "
            "red ring=safety (shrinks when obstacle is close = confident)",
            "space pause   r reset   esc quit" + ("   [PAUSED]" if paused else ""),
        ]
        for i, text in enumerate(lines):
            screen.blit(font.render(text, True, pf.TEXT_COLOR), (10, 10 + i * 20))


def run(seed: int = 0, steps: int = 240, gps_sigma: float = GPS_SIGMA,
        n_obstacles: int = N_OBSTACLES, safety_radius: float = SAFETY_RADIUS,
        n_sigma: float = N_SIGMA) -> dict:
    """Headless full-stack run; returns localization / clearance / progress."""
    sim = Sim(seed=seed, gps_sigma=gps_sigma, n_obstacles=n_obstacles,
              safety_radius=safety_radius, n_sigma=n_sigma)
    loc, min_clear = [], np.inf
    prev_idx = paths.nearest_index(sim.path, sim.true[:2])
    progress = 0
    for _ in range(steps):
        sim.step()
        loc.append(float(np.linalg.norm(sim.true[:2] - sim.estimate[:2])))
        min_clear = min(min_clear, sim.clearance())
        cur = paths.nearest_index(sim.path, sim.true[:2])
        d = (cur - prev_idx) % len(sim.path)
        progress += d if d < len(sim.path) // 2 else 0
        prev_idx = cur

    loc = np.array(loc[30:])
    return {
        "loc_rmse": float(np.sqrt(np.mean(loc ** 2))),
        "min_clearance": float(min_clear),
        "progress": int(progress),
        "path_len": len(sim.path),
    }


def iter_frames(frames: int, seed: int = 0):
    """Yield a rendered surface per frame (headless), for GIF recording."""
    pygame.init()
    pygame.font.init()
    surface = pygame.Surface((pf.WIDTH, pf.HEIGHT))
    font = pygame.font.SysFont("monospace", 15)
    sim = Sim(seed=seed)
    sim.warmup()
    for _ in range(frames):
        sim.step()
        sim.draw(surface, font)
        yield surface
    pygame.quit()


def main() -> None:
    max_frames = max_frames_from_env()

    pygame.init()
    screen = pygame.display.set_mode((pf.WIDTH, pf.HEIGHT))
    pygame.display.set_caption("gtsam-mpc: localization + moving-obstacle tracking + control")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    sim = Sim()
    sim.warmup()
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
                    sim = Sim()
                    sim.warmup()

        if not paused:
            sim.step()
        sim.draw(screen, font, paused)
        pygame.display.flip()
        clock.tick(pf.FPS)

        frame += 1
        if max_frames is not None and frame >= max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
