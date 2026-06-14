"""Interactive pygame simulation of factor-graph MPC.

A 2-D point mass is steered toward a goal by the receding-horizon
:class:`~gtsam_mpc.LinearMPC` controller. At every frame the full horizon is
re-optimized as a GTSAM factor graph; the first control is applied and the
predicted plan is drawn ahead of the point.

Controls:
    * Left click            -- set a new goal position
    * Space                 -- pause / resume
    * R                     -- reset the point to the centre at rest
    * Esc / window close    -- quit

Run from the repository root:

    python examples/pygame_sim.py

The simulation also runs headless for testing: set ``SDL_VIDEODRIVER=dummy``
and ``GTSAM_MPC_MAX_FRAMES=<n>`` to run ``n`` frames without a window.
"""

import os
import sys

# The demo uses no sound; disable the audio backend so it runs cleanly on
# headless machines without an audio device.
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame

from gtsam_mpc import LinearMPC, point_mass_2d
from examples._viz import (
    BG, GOAL_COLOR, TEXT_COLOR, View, max_frames_from_env,
)

# --- World / rendering configuration -------------------------------------
WIDTH, HEIGHT = 900, 700
PIXELS_PER_METER = 30.0          # world (metres) -> screen (pixels)
DT = 0.1                         # control / sim timestep [s]
FPS = int(round(1.0 / DT))

POINT_COLOR = (80, 200, 255)
PLAN_COLOR = (90, 110, 140)      # dimmer than the shared plan colour for points
VEL_COLOR = (250, 220, 120)

_VIEW = View(WIDTH, HEIGHT, PIXELS_PER_METER)
world_to_screen = _VIEW.world_to_screen
screen_to_world = _VIEW.screen_to_world


def make_controller() -> LinearMPC:
    system = point_mass_2d(dt=DT)
    # Penalize position error firmly, velocity lightly, control cheaply, with a
    # strong terminal cost so the point settles at the goal within the horizon.
    Q = np.diag([4.0, 4.0, 0.5, 0.5])
    R = np.diag([0.05, 0.05])
    Qf = np.diag([40.0, 40.0, 4.0, 4.0])
    return LinearMPC(system, Q=Q, R=R, horizon=30, Qf=Qf)


def draw_grid(screen: pygame.Surface) -> None:
    _VIEW.draw_grid(screen, spacing_m=1.0)


def main() -> None:
    max_frames = max_frames_from_env()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("gtsam-mpc: factor-graph MPC")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    mpc = make_controller()
    system = mpc.system

    state = np.zeros(4)                       # [px, py, vx, vy]
    goal = np.array([4.0, 2.0])               # target position
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
                    state = np.zeros(4)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                goal = screen_to_world(*event.pos)

        # Goal as a zero-velocity setpoint (an equilibrium of the point mass).
        xref = np.array([goal[0], goal[1], 0.0, 0.0])

        plan = mpc.solve(state, xref=xref)
        if not paused:
            state = system.step(state, plan.u0)

        # --- render ---
        screen.fill(BG)
        draw_grid(screen)

        # Predicted plan over the horizon.
        plan_pts = [world_to_screen(s[:2]) for s in plan.states]
        if len(plan_pts) > 1:
            pygame.draw.lines(screen, PLAN_COLOR, False, plan_pts, 2)
        for pt in plan_pts:
            pygame.draw.circle(screen, PLAN_COLOR, pt, 2)

        # Goal.
        gx, gy = world_to_screen(goal)
        pygame.draw.circle(screen, GOAL_COLOR, (gx, gy), 9, 2)
        pygame.draw.line(screen, GOAL_COLOR, (gx - 12, gy), (gx + 12, gy), 1)
        pygame.draw.line(screen, GOAL_COLOR, (gx, gy - 12), (gx, gy + 12), 1)

        # Point mass and its velocity vector.
        px, py = world_to_screen(state[:2])
        vel_end = world_to_screen(state[:2] + state[2:] * 0.3)
        pygame.draw.line(screen, VEL_COLOR, (px, py), vel_end, 2)
        pygame.draw.circle(screen, POINT_COLOR, (px, py), 8)

        # HUD.
        speed = float(np.linalg.norm(state[2:]))
        dist = float(np.linalg.norm(state[:2] - goal))
        lines = [
            f"pos=({state[0]:+.2f}, {state[1]:+.2f}) m   speed={speed:.2f} m/s",
            f"goal=({goal[0]:+.2f}, {goal[1]:+.2f}) m    dist={dist:.2f} m",
            "click: set goal   space: pause   r: reset   esc: quit"
            + ("   [PAUSED]" if paused else ""),
        ]
        for i, text in enumerate(lines):
            screen.blit(font.render(text, True, TEXT_COLOR), (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(FPS)

        frame += 1
        if max_frames is not None and frame >= max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
