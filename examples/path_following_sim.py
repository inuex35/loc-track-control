"""Interactive pygame demo: bicycle-model MPC tracking arbitrary paths.

A car (kinematic bicycle model) follows a chosen reference path using the
receding-horizon :class:`~gtsam_mpc.BicycleMPC`. Each frame we find the point on
the path closest to the car and sample the path *ahead* of it into a full
reference trajectory (see :func:`gtsam_mpc.paths.reference_trajectory`): handing
the MPC this time-varying reference lets it see the upcoming curvature and
anticipate corners. The nonlinear factor graph is re-optimized every frame
(warm-started), the first control is applied, and the predicted plan is drawn.

Paths (switch live with 1..5): circle, figure-eight, sine wave, racetrack,
rounded square.

Controls:
    * 1..5                  -- select a path
    * Up / Down             -- raise / lower target speed
    * Space                 -- pause / resume
    * R                     -- snap the car back onto the path start
    * Esc / Q               -- quit

Run from the repository root:

    python examples/path_following_sim.py

Runs headless for testing with ``SDL_VIDEODRIVER=dummy`` and
``GTSAM_MPC_MAX_FRAMES=<n>``.
"""

import os
import sys

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame

from gtsam_mpc import BicycleMPC, BicycleModel
# Path geometry lives in the package; re-export so the localization demos that
# build on this one can keep using ``pf.make_path`` etc.
from gtsam_mpc.paths import (  # noqa: F401  (re-exported for the loc/joint demos)
    PATHS, PATH_RESOLUTION, LAT_ACCEL_MAX,
    make_path, path_curvature, nearest_index, cross_track_error,
    reference_trajectory, reset_on_path,
)
from examples._viz import (
    BG, GOAL_COLOR as TARGET_COLOR, PLAN_COLOR, TEXT_COLOR, TRAIL_COLOR,
    View, draw_car as _draw_car, max_frames_from_env,
)

# --- World / rendering configuration -------------------------------------
WIDTH, HEIGHT = 1000, 720
PIXELS_PER_METER = 11.0
DT = 0.1                         # control / sim timestep (re-solve every step) [s]
FPS = int(round(1.0 / DT))
WHEELBASE = 2.5

# The MPC predicts on a coarser grid than the control rate: 0.5 s steps out to
# 3 s ahead (6 steps). The plant is stepped at the finer DT and re-optimized
# every DT.
MPC_DT = 0.5
MPC_HORIZON = 6                  # 6 * 0.5 s = 3 s prediction horizon

PATH_COLOR = (90, 90, 120)       # demo-specific colour (rest come from _viz)

A_BOUNDS = (-4.0, 4.0)
DELTA_BOUNDS = (-0.6, 0.6)
V_BOUNDS = (-2.0, 14.0)

_VIEW = View(WIDTH, HEIGHT, PIXELS_PER_METER)
world_to_screen = _VIEW.world_to_screen


def make_controller() -> BicycleMPC:
    # Prediction model uses the coarse MPC timestep (0.5 s steps, 3 s horizon),
    # split into 2 RK4 sub-integrations for accuracy on a coarse grid.
    model = BicycleModel(wheelbase=WHEELBASE, dt=MPC_DT, substeps=2)
    Q = np.diag([4.0, 4.0, 1.5, 0.6])      # track position firmly, follow heading/speed
    R = np.diag([0.1, 0.1])
    Qf = np.diag([20.0, 20.0, 4.0, 2.0])
    # Augmented Lagrangian keeps acceleration / steering / speed within bounds
    # tightly (no penalty-weight tuning); it costs almost nothing until a bound
    # actually binds.
    return BicycleMPC(
        model, Q=Q, R=R, horizon=MPC_HORIZON, Qf=Qf,
        a_bounds=A_BOUNDS, delta_bounds=DELTA_BOUNDS, v_bounds=V_BOUNDS,
        constraint_mode="al",
    )


def draw_car(screen, state: np.ndarray, delta: float) -> None:
    _draw_car(screen, _VIEW, state, delta, WHEELBASE)


def draw_path(screen, path: np.ndarray) -> None:
    pts = [world_to_screen(p) for p in path]
    pygame.draw.lines(screen, PATH_COLOR, True, pts, 2)


def draw_scene(surface, font, path, state, plan, refs,
               path_name, target_speed, trail, paused=False) -> None:
    """Render one frame: path, trail, predicted plan, reference and car + HUD."""
    surface.fill(BG)
    draw_path(surface, path)

    if len(trail) > 1:
        pygame.draw.lines(surface, TRAIL_COLOR, False, trail, 2)

    plan_pts = [world_to_screen(s[:2]) for s in plan.states]
    if len(plan_pts) > 1:
        pygame.draw.lines(surface, PLAN_COLOR, False, plan_pts, 2)
    for pt in plan_pts:
        pygame.draw.circle(surface, PLAN_COLOR, pt, 2)

    # Reference trajectory the MPC is tracking over its horizon (path ahead).
    for r in refs:
        rx, ry = world_to_screen(r[:2])
        pygame.draw.circle(surface, TARGET_COLOR, (rx, ry), 4, 1)

    draw_car(surface, state, float(plan.controls[0, 1]))

    cross_track = cross_track_error(path, state[:2])
    lines = [
        f"path: {path_name}   target speed={target_speed:.1f} m/s   v={state[3]:+.2f} m/s",
        f"cross-track err={cross_track:.2f} m   a={plan.controls[0,0]:+.2f}   "
        f"delta={np.degrees(plan.controls[0,1]):+.1f} deg",
        "1-5: path   up/down: speed   space: pause   r: reset   esc: quit"
        + ("   [PAUSED]" if paused else ""),
    ]
    for i, text in enumerate(lines):
        surface.blit(font.render(text, True, TEXT_COLOR), (10, 10 + i * 20))


def main() -> None:
    max_frames = max_frames_from_env()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("gtsam-mpc: path-following MPC")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    mpc = make_controller()
    # Plant stepped at the fine control timestep (independent of the MPC's
    # coarser prediction grid).
    plant = BicycleModel(wheelbase=WHEELBASE, dt=DT)

    path_key = "1"
    path_name, generator = PATHS[path_key]
    path = make_path(generator)
    curvature = path_curvature(path)

    target_speed = 8.0
    state = reset_on_path(path, target_speed)
    paused = False
    trail: list[tuple[int, int]] = []

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
                    state = reset_on_path(path, target_speed)
                    mpc.reset()
                    trail.clear()
                elif event.key == pygame.K_UP:
                    target_speed = min(target_speed + 1.0, V_BOUNDS[1])
                elif event.key == pygame.K_DOWN:
                    target_speed = max(target_speed - 1.0, 0.0)
                else:
                    name = pygame.key.name(event.key)
                    if name in PATHS:
                        path_key = name
                        path_name, generator = PATHS[path_key]
                        path = make_path(generator)
                        curvature = path_curvature(path)
                        state = reset_on_path(path, target_speed)
                        mpc.reset()
                        trail.clear()

        near = nearest_index(path, state[:2])
        # Reference is sampled on the MPC's prediction grid (MPC_DT per step).
        refs = reference_trajectory(path, curvature, near, target_speed,
                                    mpc.horizon, MPC_DT)

        plan = mpc.solve(state, refs)
        if not paused:
            state = plant.step(state, plan.u0)
            trail.append(world_to_screen(state[:2]))
            if len(trail) > 400:
                trail.pop(0)

        draw_scene(screen, font, path, state, plan, refs,
                   path_name, target_speed, trail, paused)
        pygame.display.flip()
        clock.tick(FPS)

        frame += 1
        if max_frames is not None and frame >= max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
