"""Interactive pygame demo: bicycle-model MPC tracking arbitrary paths.

A car (kinematic bicycle model) follows a chosen reference path using the
receding-horizon :class:`~gtsam_mpc.BicycleMPC`. Each frame we find the point
on the path closest to the car and sample the path *ahead* of it into a full
reference trajectory: step ``k`` of the horizon targets the path point reached
by travelling ``v * dt`` of arc length per step (with ``v`` capped by curvature),
tagged with the path-tangent heading and that speed. Handing the MPC this
time-varying reference (instead of a single look-ahead point) lets it see the
upcoming curvature and corner-cut/anticipate properly. The nonlinear factor
graph is re-optimized every frame (warm-started), the first control is applied,
and the predicted plan is drawn ahead of the car.

Several paths are provided and can be switched live:

    * 1 -- circle
    * 2 -- figure-eight (lemniscate)
    * 3 -- sine wave (wraps around)
    * 4 -- racetrack / stadium oval
    * 5 -- rounded square

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

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

from gtsam_mpc import BicycleMPC, BicycleModel

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

BG = (18, 18, 24)
GRID = (38, 38, 48)
CAR_BODY = (80, 200, 255)
WHEEL = (235, 235, 245)
PATH_COLOR = (90, 90, 120)
TARGET_COLOR = (255, 120, 120)
PLAN_COLOR = (120, 200, 140)
TEXT_COLOR = (210, 210, 220)
TRAIL_COLOR = (60, 120, 90)

A_BOUNDS = (-4.0, 4.0)
DELTA_BOUNDS = (-0.6, 0.6)
V_BOUNDS = (-2.0, 14.0)

PATH_RESOLUTION = 0.25            # arc-length spacing of resampled path [m]
LAT_ACCEL_MAX = 3.0              # comfort lateral-accel limit for corner speed [m/s^2]


def world_to_screen(p) -> tuple[int, int]:
    sx = WIDTH / 2 + p[0] * PIXELS_PER_METER
    sy = HEIGHT / 2 - p[1] * PIXELS_PER_METER
    return int(sx), int(sy)


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


# --- Reference paths ------------------------------------------------------
# Each generator returns a closed-loop polyline as raw (un-resampled) points;
# ``make_path`` resamples it to uniform arc-length spacing for stable lookahead.
def _circle(radius=22.0, n=400):
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def _figure_eight(scale=26.0, n=600):
    # Lemniscate of Gerono: a smooth self-crossing figure-eight.
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([scale * np.cos(t), scale * np.sin(t) * np.cos(t)])


def _sine_wave(length=64.0, amp=5.0, cycles=2, base=12.0, n=800):
    # Closed serpentine: two sine legs joined by semicircular U-turns. Using
    # ``cos`` with an integer cycle count makes each leg start and end with a
    # horizontal tangent, so it meets the semicircle ends smoothly.
    radius = base
    xs = np.linspace(-length / 2, length / 2, n // 4)
    wave = amp * np.cos(2 * np.pi * cycles * (xs + length / 2) / length)
    top = np.column_stack([xs, base + wave])                 # left -> right
    bottom = np.column_stack([xs[::-1], -base + wave[::-1]])  # right -> left
    # Right U-turn: top-right (heading +x) down to bottom-right (heading -x),
    # sweeping through the rightmost point.
    a = np.linspace(np.pi / 2, -np.pi / 2, n // 4)
    right = np.column_stack([length / 2 + radius * np.cos(a),
                             amp + radius * np.sin(a)])
    # Left U-turn: bottom-left (heading -x) up to top-left (heading +x),
    # sweeping through the leftmost point.
    a2 = np.linspace(3 * np.pi / 2, np.pi / 2, n // 4)
    left = np.column_stack([-length / 2 + radius * np.cos(a2),
                            amp + radius * np.sin(a2)])
    return np.vstack([top, right, bottom, left])


def _racetrack(half=26.0, radius=16.0, n=600):
    # Stadium / oval: two straights joined by semicircular ends.
    straight = half - radius
    pts = []
    # right semicircle
    for a in np.linspace(-np.pi / 2, np.pi / 2, n // 4):
        pts.append([straight + radius * np.cos(a), radius * np.sin(a)])
    # top straight (right -> left)
    for x in np.linspace(straight, -straight, n // 4):
        pts.append([x, radius])
    # left semicircle
    for a in np.linspace(np.pi / 2, 3 * np.pi / 2, n // 4):
        pts.append([-straight + radius * np.cos(a), radius * np.sin(a)])
    # bottom straight (left -> right)
    for x in np.linspace(-straight, straight, n // 4):
        pts.append([x, -radius])
    return np.array(pts)


def _rounded_square(half=24.0, radius=8.0, n=600):
    s = half - radius
    centers = [(s, s), (-s, s), (-s, -s), (s, -s)]
    starts = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    pts = []
    for (cx, cy), a0 in zip(centers, starts):
        for a in np.linspace(a0, a0 + np.pi / 2, n // 4):
            pts.append([cx + radius * np.cos(a), cy + radius * np.sin(a)])
    return np.array(pts)


PATHS = {
    "1": ("circle", _circle),
    "2": ("figure-eight", _figure_eight),
    "3": ("sine wave", _sine_wave),
    "4": ("racetrack", _racetrack),
    "5": ("rounded square", _rounded_square),
}


def make_path(generator) -> np.ndarray:
    """Resample a raw polyline to uniform arc-length spacing (closed loop)."""
    raw = generator()
    closed = np.vstack([raw, raw[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(int(total / PATH_RESOLUTION), 8)
    su = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(su, s, closed[:, 0])
    y = np.interp(su, s, closed[:, 1])
    return np.column_stack([x, y])


def path_curvature(path: np.ndarray) -> np.ndarray:
    """Discrete curvature (1/radius) at each point of a closed path."""
    prv = np.roll(path, 1, axis=0)
    nxt = np.roll(path, -1, axis=0)
    d1 = path - prv
    d2 = nxt - path
    # Signed area of the triangle (prv, path, nxt) gives curvature via the
    # circumscribed-circle radius: kappa = 4*area / (|a||b||c|).
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    a = np.linalg.norm(d1, axis=1)
    b = np.linalg.norm(d2, axis=1)
    c = np.linalg.norm(nxt - prv, axis=1)
    denom = a * b * c
    kappa = np.where(denom > 1e-9, 2.0 * np.abs(cross) / np.maximum(denom, 1e-9), 0.0)
    return kappa


def make_controller() -> BicycleMPC:
    # Prediction model uses the coarse MPC timestep (0.5 s steps, 3 s horizon).
    # Split each step into 2 RK4 sub-integrations for accuracy (as acados does
    # with sim_method_num_steps), so the coarse grid stays unbiased on curves.
    model = BicycleModel(wheelbase=WHEELBASE, dt=MPC_DT, substeps=2)
    # Track position firmly, follow the path heading, hold target speed.
    Q = np.diag([4.0, 4.0, 1.5, 0.6])
    R = np.diag([0.1, 0.1])
    Qf = np.diag([20.0, 20.0, 4.0, 2.0])
    # Augmented Lagrangian keeps acceleration / steering / speed within their
    # bounds tightly (no penalty-weight tuning); it converges in one outer
    # iteration when, as here, the curvature-limited reference keeps inputs
    # feasible, so it costs almost nothing until a bound actually binds.
    return BicycleMPC(
        model, Q=Q, R=R, horizon=MPC_HORIZON, Qf=Qf,
        a_bounds=A_BOUNDS, delta_bounds=DELTA_BOUNDS, v_bounds=V_BOUNDS,
        constraint_mode="al",
    )


def nearest_index(path: np.ndarray, p: np.ndarray) -> int:
    return int(np.argmin(np.sum((path - p) ** 2, axis=1)))


def cross_track_error(path: np.ndarray, p: np.ndarray) -> float:
    """Perpendicular distance from ``p`` to the path polyline.

    Uses the nearest vertex and projects onto its two adjacent segments, so the
    result is the true lateral offset -- not the (quantized, partly along-track)
    distance to the nearest discrete vertex.
    """
    i = nearest_index(path, p)
    n = len(path)
    best = float(np.linalg.norm(p - path[i]))
    for j in (i - 1, i):
        a, b = path[j % n], path[(j + 1) % n]
        ab = b - a
        denom = float(ab @ ab)
        if denom < 1e-12:
            continue
        t = np.clip((p - a) @ ab / denom, 0.0, 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def _curve_speed(kappa: float, speed: float) -> float:
    """Speed capped so curvature ``kappa`` stays within the lateral-accel limit."""
    if kappa <= 1e-4:
        return speed
    return min(speed, float(np.sqrt(LAT_ACCEL_MAX / kappa)))


def reference_trajectory(path, curvature, near_idx, speed, horizon, dt):
    """Build the per-step reference the MPC should track over its horizon.

    Rather than chasing a single look-ahead point, we sample the path *ahead* of
    the car: starting at the closest path point, each horizon step advances
    along the path by ``v * dt`` of arc length, where ``v`` is the desired speed
    locally capped by curvature. Each reference state is the path point plus its
    tangent heading and that speed, giving a full path-tracking reference of
    shape ``(horizon + 1, 4)``.
    """
    n = len(path)
    refs = np.empty((horizon + 1, 4))
    pos = float(near_idx)              # fractional index along the path
    for k in range(horizon + 1):
        idx = int(pos) % n
        nxt = (idx + 1) % n
        tangent = path[nxt] - path[idx]
        heading = np.arctan2(tangent[1], tangent[0])
        v = _curve_speed(curvature[idx], speed)
        refs[k] = [path[idx, 0], path[idx, 1], heading, v]
        pos += v * dt / PATH_RESOLUTION
    return refs


def draw_car(screen, state: np.ndarray, delta: float) -> None:
    px, py, theta, _ = state
    pos = np.array([px, py])
    R = _rot(theta)
    length, width = WHEELBASE + 1.0, 1.8
    half = np.array([length / 2, width / 2])
    corners = np.array([[-half[0], -half[1]], [half[0], -half[1]],
                        [half[0], half[1]], [-half[0], half[1]]])
    body = [world_to_screen(pos + R @ (c + np.array([length / 2 - 0.5, 0]))) for c in corners]
    pygame.draw.polygon(screen, CAR_BODY, body)

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


def draw_path(screen, path: np.ndarray) -> None:
    pts = [world_to_screen(p) for p in path]
    pygame.draw.lines(screen, PATH_COLOR, True, pts, 2)


def reset_on_path(path: np.ndarray, speed: float) -> np.ndarray:
    """Place the car at the path start, headed along the tangent, at speed."""
    tangent = path[1] - path[0]
    heading = np.arctan2(tangent[1], tangent[0])
    return np.array([path[0, 0], path[0, 1], heading, speed])


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
    max_frames_env = os.environ.get("GTSAM_MPC_MAX_FRAMES")
    max_frames = int(max_frames_env) if max_frames_env else None

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
