"""統合デモ: track moving obstacles, then avoid them with MPC.

This wires the three factor-graph layers into one pipeline:

    perception  ->  prediction  ->  control

    * Moving obstacles emit noisy position **detections** each frame.
    * A constant-velocity **factor-graph tracker** (FGO-MOT style) smooths each
      obstacle's recent detections into a state estimate ``[x, y, vx, vy]`` and
      **predicts** its position over the MPC horizon.
    * The ego car (kinematic bicycle) is driven to a goal by ``BicycleMPC``,
      which avoids each predicted obstacle trajectory via soft keep-out
      (control-barrier-style) factors.

Controls:
    * Left click            -- set a new goal position
    * Space                 -- pause / resume
    * R                     -- reset the car
    * Esc / Q               -- quit

Run:  python examples/integrated_sim.py

Headless test:  SDL_VIDEODRIVER=dummy GTSAM_MPC_MAX_FRAMES=<n> python examples/integrated_sim.py
"""

import os
import sys
from collections import deque

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Allow ``python examples/integrated_sim.py`` to import the sibling example.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame

import examples.bicycle_sim as bsim
from gtsam_mpc import BicycleMPC, BicycleModel, ConstantVelocityTracker

DT = bsim.DT
HORIZON = 25
SAFETY_RADIUS = 3.0
OBS_DRAW_RADIUS = 0.8
TRACK_WINDOW = 12

OBS_TRUE = (255, 90, 90)
OBS_RING = (255, 90, 90)
DET_COLOR = (255, 200, 80)
PRED_COLOR = (255, 140, 140)

# Constant-velocity factor-graph tracker for the moving obstacles.
_TRACKER = ConstantVelocityTracker(dt=DT, process_sigma=(0.05, 0.05, 0.4, 0.4),
                                   meas_sigma=0.6)
_F = _TRACKER.F


def make_controller() -> BicycleMPC:
    model = BicycleModel(wheelbase=bsim.WHEELBASE, dt=DT)
    return BicycleMPC(
        model,
        Q=np.diag([3.0, 3.0, 0.6, 0.3]),
        R=np.diag([0.1, 0.1]),
        horizon=HORIZON,
        Qf=np.diag([30.0, 30.0, 3.0, 3.0]),
        a_bounds=(-4.0, 4.0),
        delta_bounds=(-0.6, 0.6),
        v_bounds=(-3.0, 9.0),
        barrier_weight=500.0,
        safety_radius=SAFETY_RADIUS,
        obstacle_weight=400.0,
    )


def estimate_state(detections: deque, dt: float) -> np.ndarray:
    """Smooth recent detections into ``[x, y, vx, vy]`` (CV factor-graph tracker)."""
    return _TRACKER.estimate(list(detections))


def predict_horizon(state: np.ndarray, horizon: int, dt: float) -> np.ndarray:
    """Constant-velocity rollout of an obstacle over the horizon -> (N+1, 2)."""
    return _TRACKER.predict(state, horizon, dt)


class Obstacle:
    """Ground-truth moving obstacle with a detection history."""

    def __init__(self, pos, vel):
        self.state = np.array([pos[0], pos[1], vel[0], vel[1]], dtype=float)
        self.history = deque(maxlen=TRACK_WINDOW)

    def step(self, dt, rng, bound=28.0):
        self.state = _F @ self.state
        # Bounce off the field edges so obstacles stay in view.
        for i in (0, 1):
            if abs(self.state[i]) > bound:
                self.state[i] = np.clip(self.state[i], -bound, bound)
                self.state[i + 2] *= -1
        self.history.append(self.state[:2] + rng.normal(0, 0.6, 2))


def draw_scene(screen, font, ego, goal_xy, obstacles, predictions, plan,
               paused: bool = False) -> None:
    """Render one frame of the track-and-avoid scene (shared by sim and GIF)."""
    screen.fill(bsim.BG)
    bsim.draw_grid(screen)

    for obs, pred in zip(obstacles, predictions):
        for d in obs.history:
            pygame.draw.circle(screen, DET_COLOR, bsim.world_to_screen(d), 2)
        pred_pts = [bsim.world_to_screen(p) for p in pred]
        if len(pred_pts) > 1:
            pygame.draw.lines(screen, PRED_COLOR, False, pred_pts, 1)
        c = bsim.world_to_screen(obs.state[:2])
        pygame.draw.circle(screen, OBS_TRUE, c,
                           int(OBS_DRAW_RADIUS * bsim.PIXELS_PER_METER))
        pygame.draw.circle(screen, OBS_RING, c,
                           int(SAFETY_RADIUS * bsim.PIXELS_PER_METER), 1)

    plan_pts = [bsim.world_to_screen(s[:2]) for s in plan.states]
    if len(plan_pts) > 1:
        pygame.draw.lines(screen, bsim.PLAN_COLOR, False, plan_pts, 2)

    gx, gy = bsim.world_to_screen(goal_xy)
    pygame.draw.circle(screen, bsim.GOAL_COLOR, (gx, gy), 9, 2)

    bsim.draw_car(screen, ego, float(plan.controls[0, 1]))

    dist_goal = float(np.linalg.norm(ego[:2] - goal_xy))
    min_obs = min(
        (float(np.linalg.norm(ego[:2] - o.state[:2])) for o in obstacles),
        default=float("inf"),
    )
    lines = [
        f"ego=({ego[0]:+.1f},{ego[1]:+.1f}) v={ego[3]:+.2f}  goal dist={dist_goal:.1f}",
        f"tracked obstacles={len(predictions)}  nearest={min_obs:.1f} m  "
        f"(safety {SAFETY_RADIUS:.0f} m)",
        "click: set goal   space: pause   r: reset   esc: quit"
        + ("   [PAUSED]" if paused else ""),
    ]
    for i, text in enumerate(lines):
        screen.blit(font.render(text, True, bsim.TEXT_COLOR), (10, 10 + i * 20))


def run(frames: int = 70, seed: int = 0) -> dict:
    """Headless track-and-avoid rollout (no rendering) for testing/metrics."""
    rng = np.random.default_rng(seed)
    mpc = make_controller()
    model = mpc.model
    ego = np.array([-22.0, 0.0, 0.0, 0.0])
    goal = np.array([22.0, 0.0])
    obstacles = [
        Obstacle((-2.0, -12.0), (0.3, 2.2)),
        Obstacle((6.0, 12.0), (-0.4, -2.0)),
    ]
    min_clearance = float("inf")
    for _ in range(frames):
        for obs in obstacles:
            obs.step(DT, rng)
        predictions = [
            predict_horizon(estimate_state(obs.history, DT), HORIZON, DT)
            for obs in obstacles
            if len(obs.history) > 0
        ]
        heading = np.arctan2(goal[1] - ego[1], goal[0] - ego[0])
        xref = np.array([goal[0], goal[1], heading, 0.0])
        u = mpc.control(ego, xref, obstacles=predictions)
        ego = model.step(ego, u)
        min_clearance = min(
            min_clearance,
            min(np.linalg.norm(ego[:2] - o.state[:2]) for o in obstacles),
        )
    return {
        "ego": ego,
        "min_clearance": float(min_clearance),
        "goal_dist": float(np.linalg.norm(ego[:2] - goal)),
    }


def iter_frames(frames: int, seed: int = 0):
    """Yield a rendered surface per frame (headless), for GIF recording."""
    pygame.init()
    pygame.font.init()
    surface = pygame.Surface((bsim.WIDTH, bsim.HEIGHT))
    font = pygame.font.SysFont("monospace", 16)
    rng = np.random.default_rng(seed)
    mpc = make_controller()
    model = mpc.model
    ego = np.array([-22.0, 0.0, 0.0, 0.0])
    goal_xy = np.array([22.0, 0.0])
    obstacles = [
        Obstacle((-2.0, -12.0), (0.3, 2.2)),
        Obstacle((6.0, 12.0), (-0.4, -2.0)),
    ]
    for _ in range(frames):
        for obs in obstacles:
            obs.step(DT, rng)
        predictions = [
            predict_horizon(estimate_state(o.history, DT), HORIZON, DT)
            for o in obstacles if len(o.history) > 0
        ]
        heading = np.arctan2(goal_xy[1] - ego[1], goal_xy[0] - ego[0])
        xref = np.array([goal_xy[0], goal_xy[1], heading, 0.0])
        plan = mpc.solve(ego, xref, obstacles=predictions)
        ego = model.step(ego, plan.u0)
        draw_scene(surface, font, ego, goal_xy, obstacles, predictions, plan)
        yield surface
    pygame.quit()


def main() -> None:
    max_frames_env = os.environ.get("GTSAM_MPC_MAX_FRAMES")
    max_frames = int(max_frames_env) if max_frames_env else None
    rng = np.random.default_rng(0)

    pygame.init()
    screen = pygame.display.set_mode((bsim.WIDTH, bsim.HEIGHT))
    pygame.display.set_caption("gtsam-mpc: track + avoid (integrated)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 16)

    mpc = make_controller()
    model = mpc.model

    ego = np.array([-22.0, 0.0, 0.0, 0.0])     # [x, y, theta, v]
    goal_xy = np.array([22.0, 0.0])
    obstacles = [
        Obstacle((-2.0, -12.0), (0.3, 2.2)),
        Obstacle((6.0, 12.0), (-0.4, -2.0)),
    ]
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
                    ego = np.array([-22.0, 0.0, 0.0, 0.0]); mpc.reset()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                goal_xy = bsim.screen_to_world(*event.pos)

        if not paused:
            for obs in obstacles:
                obs.step(DT, rng)

        # Perception + prediction: track each obstacle and roll it forward.
        predictions = []
        estimates = []
        for obs in obstacles:
            if len(obs.history) == 0:
                continue
            est = estimate_state(obs.history, DT)
            estimates.append(est)
            predictions.append(predict_horizon(est, HORIZON, DT))

        # Control: drive to the goal while avoiding the predicted obstacles.
        heading = np.arctan2(goal_xy[1] - ego[1], goal_xy[0] - ego[0])
        xref = np.array([goal_xy[0], goal_xy[1], heading, 0.0])
        plan = mpc.solve(ego, xref, obstacles=predictions)
        if not paused:
            ego = model.step(ego, plan.u0)

        draw_scene(screen, font, ego, goal_xy, obstacles, predictions, plan, paused)
        pygame.display.flip()
        clock.tick(int(round(1.0 / DT)))

        frame += 1
        if max_frames is not None and frame >= max_frames:
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
