"""Interactive pygame simulation of nonlinear bicycle-model MPC.

A car (kinematic bicycle model) is steered toward a goal by the receding-horizon
:class:`~gtsam_mpc.BicycleMPC`. Each frame the nonlinear factor graph is
re-optimized (warm-started from the previous solution); the first control is
applied and the predicted path is drawn ahead of the car. Acceleration and
steering limits are enforced as soft barrier factors.

Controls:
    * Left click            -- set a new goal position
    * Space                 -- pause / resume
    * R                     -- reset the car to the centre at rest
    * Esc / Q               -- quit

Run from the repository root:

    python examples/bicycle_sim.py

Runs headless for testing with ``SDL_VIDEODRIVER=dummy`` and
``GTSAM_MPC_MAX_FRAMES=<n>``.
"""

import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from gtsam_mpc import BicycleMPC, BicycleModel

WIDTH, HEIGHT = 1000, 720
PIXELS_PER_METER = 18.0
DT = 0.1
FPS = int(round(1.0 / DT))
WHEELBASE = 2.5

BG = (18, 18, 24)
GRID = (38, 38, 48)
CAR_BODY = (80, 200, 255)
WHEEL = (235, 235, 245)
GOAL_COLOR = (255, 120, 120)
PLAN_COLOR = (120, 200, 140)
TEXT_COLOR = (210, 210, 220)

A_BOUNDS = (-3.0, 3.0)
DELTA_BOUNDS = (-0.6, 0.6)
V_BOUNDS = (-3.0, 8.0)


def world_to_screen(p) -> tuple[int, int]:
    sx = WIDTH / 2 + p[0] * PIXELS_PER_METER
    sy = HEIGHT / 2 - p[1] * PIXELS_PER_METER
    return int(sx), int(sy)


def screen_to_world(sx: float, sy: float) -> np.ndarray:
    return np.array(
        [(sx - WIDTH / 2) / PIXELS_PER_METER, (HEIGHT / 2 - sy) / PIXELS_PER_METER]
    )


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def make_controller() -> BicycleMPC:
    model = BicycleModel(wheelbase=WHEELBASE, dt=DT)
    Q = np.diag([3.0, 3.0, 0.8, 0.4])
    R = np.diag([0.1, 0.1])
    Qf = np.diag([30.0, 30.0, 3.0, 3.0])
    return BicycleMPC(
        model, Q=Q, R=R, horizon=25, Qf=Qf,
        a_bounds=A_BOUNDS, delta_bounds=DELTA_BOUNDS, v_bounds=V_BOUNDS,
        barrier_weight=500.0,
    )


def draw_grid(screen) -> None:
    step = int(PIXELS_PER_METER * 2)
    for x in range(WIDTH // 2 % step, WIDTH, step):
        pygame.draw.line(screen, GRID, (x, 0), (x, HEIGHT))
    for y in range(HEIGHT // 2 % step, HEIGHT, step):
        pygame.draw.line(screen, GRID, (0, y), (WIDTH, y))


def draw_car(screen, state: np.ndarray, delta: float) -> None:
    px, py, theta, _ = state
    pos = np.array([px, py])
    R = _rot(theta)
    length, width = WHEELBASE + 1.0, 1.8

    # Body rectangle centred on the wheelbase midpoint.
    half = np.array([length / 2, width / 2])
    corners = np.array([[-half[0], -half[1]], [half[0], -half[1]],
                        [half[0], half[1]], [-half[0], half[1]]])
    body = [world_to_screen(pos + R @ (c + np.array([length / 2 - 0.5, 0]))) for c in corners]
    pygame.draw.polygon(screen, CAR_BODY, body)

    # Wheels: rear (aligned with body) and front (steered by delta).
    def wheel(center_body, ang):
        wl, ww = 0.7, 0.25
        wr = _rot(theta + ang)
        pts = np.array([[-wl, -ww], [wl, -ww], [wl, ww], [-wl, ww]])
        screen_pts = [world_to_screen(pos + R @ center_body + wr @ p) for p in pts]
        pygame.draw.polygon(screen, WHEEL, screen_pts)

    wheel(np.array([0.0, width / 2]), 0.0)
    wheel(np.array([0.0, -width / 2]), 0.0)
    wheel(np.array([WHEELBASE, width / 2]), delta)
    wheel(np.array([WHEELBASE, -width / 2]), delta)


def main() -> None:
    max_frames_env = os.environ.get("GTSAM_MPC_MAX_FRAMES")
    max_frames = int(max_frames_env) if max_frames_env else None

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("gtsam-mpc: nonlinear bicycle MPC")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    mpc = make_controller()
    model = mpc.model

    state = np.array([-10.0, 0.0, 0.0, 0.0])     # [x, y, theta, v]
    goal_xy = np.array([10.0, 4.0])
    last_delta = 0.0
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
                    state = np.array([0.0, 0.0, 0.0, 0.0])
                    mpc.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                goal_xy = screen_to_world(*event.pos)

        # Heading goal points from the car toward the target; zero terminal speed.
        heading = np.arctan2(goal_xy[1] - state[1], goal_xy[0] - state[0])
        xref = np.array([goal_xy[0], goal_xy[1], heading, 0.0])

        plan = mpc.solve(state, xref)
        last_delta = float(plan.controls[0, 1])
        if not paused:
            state = model.step(state, plan.u0)

        # --- render ---
        screen.fill(BG)
        draw_grid(screen)

        plan_pts = [world_to_screen(s[:2]) for s in plan.states]
        if len(plan_pts) > 1:
            pygame.draw.lines(screen, PLAN_COLOR, False, plan_pts, 2)
        for pt in plan_pts:
            pygame.draw.circle(screen, PLAN_COLOR, pt, 2)

        gx, gy = world_to_screen(goal_xy)
        pygame.draw.circle(screen, GOAL_COLOR, (gx, gy), 9, 2)
        pygame.draw.line(screen, GOAL_COLOR, (gx - 12, gy), (gx + 12, gy), 1)
        pygame.draw.line(screen, GOAL_COLOR, (gx, gy - 12), (gx, gy + 12), 1)

        draw_car(screen, state, last_delta)

        dist = float(np.linalg.norm(state[:2] - goal_xy))
        lines = [
            f"pos=({state[0]:+.1f}, {state[1]:+.1f})  heading={np.degrees(state[2]):+.0f}deg  v={state[3]:+.2f} m/s",
            f"control: a={plan.controls[0,0]:+.2f} m/s^2   delta={np.degrees(last_delta):+.1f} deg   dist={dist:.2f} m",
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
