"""Pygame demo: path following with *hard* constraints (SQP on GTSAM 4.3).

A car follows a racetrack whose centreline is blocked by static obstacles. The
:class:`~gtsam_mpc.BicycleMPC` runs in ``constraint_mode="sqp"``: every frame
it solves a sequence of QPs with GTSAM's constrained QP solver
(``gtsam.QpProblem`` + ``gtsam.LinearConstraint``), so the input box, the
speed limit and each obstacle keep-out ring are enforced as hard constraints
rather than penalties. The HUD tracks the worst values seen so far: the speed
stays pinned at the limit on the straights (the target speed is set above it)
and the clearance never drops below the keep-out radius.

Controls: space pause, r reset, esc quit.

Headless test:  SDL_VIDEODRIVER=dummy GTSAM_MPC_MAX_FRAMES=<n> python examples/sqp_obstacle_sim.py
"""

import os
import sys

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame

from gtsam_mpc import BicycleMPC, BicycleModel, paths
from examples import path_following_sim as pf
from examples._viz import (
    GOAL_COLOR, PLAN_COLOR, TEXT_COLOR, TRAIL_COLOR, View, draw_car, max_frames_from_env,
)

COURSE = "4"                    # racetrack
DT = 0.1                        # plant + MPC step [s] (same grid: no inter-node cutting)
HORIZON = 20                    # 2 s look-ahead
TARGET_SPEED = 10.0             # deliberately above the speed limit
A_BOUNDS = (-4.0, 4.0)
DELTA_BOUNDS = (-0.6, 0.6)
V_BOUNDS = (-2.0, 8.0)
SAFETY_RADIUS = 2.5
SENSE_RANGE = 25.0              # only obstacles this close enter the QP
# Obstacles: (path index, lateral offset [m]) -- slightly off the centreline.
OBSTACLES = [(90, 0.4), (200, -0.5), (320, 0.3), (430, -0.4), (520, 0.5)]
KEEPOUT_COLOR = (255, 90, 90)
PIXELS_PER_METER = 16.0         # zoomed in: the racetrack spans ~52 x 32 m
_VIEW = View(pf.WIDTH, pf.HEIGHT, PIXELS_PER_METER)
world_to_screen = _VIEW.world_to_screen


def make_controller() -> BicycleMPC:
    return BicycleMPC(
        BicycleModel(wheelbase=pf.WHEELBASE, dt=DT),
        Q=np.diag([4.0, 4.0, 1.5, 0.6]), R=np.diag([0.1, 0.1]), horizon=HORIZON,
        Qf=np.diag([20.0, 20.0, 4.0, 2.0]),
        a_bounds=A_BOUNDS, delta_bounds=DELTA_BOUNDS, v_bounds=V_BOUNDS,
        constraint_mode="sqp", safety_radius=SAFETY_RADIUS,
    )


def _obstacle_points(path) -> np.ndarray:
    pts = []
    for idx, offset in OBSTACLES:
        tangent = path[(idx + 1) % len(path)] - path[idx - 1]
        normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
        pts.append(path[idx] + offset * normal)
    return np.array(pts)


class Sim:
    """Closed-loop path following around hard keep-out rings."""

    def __init__(self):
        self.path = paths.make_path(paths.PATHS[COURSE][1])
        self.curvature = paths.path_curvature(self.path)
        self.obstacles = _obstacle_points(self.path)
        self.mpc = make_controller()
        self.plant = BicycleModel(wheelbase=pf.WHEELBASE, dt=DT)
        self.reset()

    def reset(self) -> None:
        self.state = paths.reset_on_path(self.path, V_BOUNDS[1])
        self.mpc.reset()
        self.plan = None
        self.trail: list[tuple[int, int]] = []
        self.progress = 0
        self._last_idx = paths.nearest_index(self.path, self.state[:2])
        self.worst = {"clear": np.inf, "v": -np.inf, "a": 0.0, "delta": 0.0}

    def clearance(self) -> float:
        return float(np.min(np.linalg.norm(self.obstacles - self.state[:2], axis=1)))

    def step(self) -> None:
        near = paths.nearest_index(self.path, self.state[:2])
        refs = paths.reference_trajectory(self.path, self.curvature, near,
                                          TARGET_SPEED, HORIZON, DT)
        dist = np.linalg.norm(self.obstacles - self.state[:2], axis=1)
        nearby = [o for o, d in zip(self.obstacles, dist) if d < SENSE_RANGE]
        self.plan = self.mpc.solve(self.state, refs, obstacles=nearby or None)
        u = self.plan.u0
        self.state = self.plant.step(self.state, u)

        n = len(self.path)
        idx = paths.nearest_index(self.path, self.state[:2])
        self.progress += (idx - self._last_idx + n // 2) % n - n // 2
        self._last_idx = idx
        self.trail.append(world_to_screen(self.state[:2]))
        if len(self.trail) > 600:
            self.trail.pop(0)

        w = self.worst
        w["clear"] = min(w["clear"], self.clearance())
        w["v"] = max(w["v"], float(self.state[3]))
        w["a"] = max(w["a"], abs(float(u[0])))
        w["delta"] = max(w["delta"], abs(float(u[1])))

    def draw(self, screen, font, paused: bool = False) -> None:
        screen.fill(pf.BG)
        pygame.draw.lines(screen, pf.PATH_COLOR, True,
                          [world_to_screen(p) for p in self.path], 2)
        r_px = int(SAFETY_RADIUS * PIXELS_PER_METER)
        for o in self.obstacles:
            c = world_to_screen(o)
            pygame.draw.circle(screen, KEEPOUT_COLOR, c, r_px, 1)
            pygame.draw.circle(screen, KEEPOUT_COLOR, c, 6)
        if len(self.trail) > 1:
            pygame.draw.lines(screen, TRAIL_COLOR, False, self.trail, 2)
        delta = 0.0
        if self.plan is not None:
            pts = [world_to_screen(s[:2]) for s in self.plan.states]
            pygame.draw.lines(screen, PLAN_COLOR, False, pts, 2)
            for p in pts:
                pygame.draw.circle(screen, PLAN_COLOR, p, 2)
            delta = float(self.plan.controls[0, 1])
        draw_car(screen, _VIEW, self.state, delta, pf.WHEELBASE)

        w = self.worst
        lines = [
            "HARD constraints: SQP on GTSAM 4.3 QpProblem / LinearConstraint",
            f"v={self.state[3]:5.2f} m/s (limit {V_BOUNDS[1]:.1f}, target {TARGET_SPEED:.0f})"
            f"   clearance={self.clearance():5.2f} m (keep-out {SAFETY_RADIUS:.1f})",
            f"worst so far:  max v={w['v']:.3f}   min clearance={w['clear']:.3f}"
            f"   max|a|={w['a']:.3f}/{A_BOUNDS[1]:.0f}"
            f"   max|delta|={w['delta']:.3f}/{DELTA_BOUNDS[1]:.1f}",
            "red ring = keep-out   green = MPC plan   space pause   r reset   esc quit"
            + ("   [PAUSED]" if paused else ""),
        ]
        for i, text in enumerate(lines):
            color = GOAL_COLOR if i == 0 else TEXT_COLOR
            screen.blit(font.render(text, True, color), (10, 10 + i * 20))


def run(steps: int = 200) -> dict:
    """Headless closed loop; returns the worst constraint values seen."""
    sim = Sim()
    for _ in range(steps):
        sim.step()
    w = sim.worst
    return {"min_clearance": w["clear"], "max_v": w["v"], "max_a": w["a"],
            "max_delta": w["delta"], "progress": sim.progress}


def iter_frames(frames: int, seed: int = 0):
    """Yield a rendered surface per frame (headless), for GIF/MP4 recording."""
    pygame.init()
    pygame.font.init()
    surface = pygame.Surface((pf.WIDTH, pf.HEIGHT))
    font = pygame.font.SysFont("monospace", 15)
    sim = Sim()
    for _ in range(frames):
        sim.step()
        sim.draw(surface, font)
        yield surface
    pygame.quit()


def main() -> None:
    max_frames = max_frames_from_env()
    pygame.init()
    screen = pygame.display.set_mode((pf.WIDTH, pf.HEIGHT))
    pygame.display.set_caption("gtsam-mpc: hard-constrained path following (SQP)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)
    sim = Sim()
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
                    sim.reset()
        if not paused:
            sim.step()
        sim.draw(screen, font, paused)
        pygame.display.flip()
        clock.tick(int(round(1.0 / DT)))
        frame += 1
        if max_frames is not None and frame >= max_frames:
            running = False
    pygame.quit()


if __name__ == "__main__":
    main()
