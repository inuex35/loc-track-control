"""Shared interactive harness for the localization+control racecar demos.

``loc_control_sim`` and ``joint_loc_control_sim`` differ only in their
estimation/control *engine*: everything around it -- the pygame event loop,
the noisy-measurement model, the trail / RMSE bookkeeping and the world
rendering (path, GPS dots, true/estimate trails, estimator window, plan,
car, HUD) -- is identical. That scaffolding lives here as :class:`RacecarApp`;
each demo subclasses it and fills in a few hooks:

    * :meth:`setup`        -- build the engine (once)
    * :meth:`reset_engine` -- reset it to ``self.true_state``
    * :meth:`set_gps_sigma`-- push a new GPS noise level to the engine
    * :meth:`advance`      -- run one closed-loop step, return the plan states
    * :meth:`horizon`      -- reference-trajectory length to request
    * :meth:`window_points`-- estimator-window points to draw (optional)
    * :meth:`extra_key` / :meth:`hud_lines` -- demo-specific keys / HUD text
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pygame

import examples.path_following_sim as pf
from examples._viz import max_frames_from_env
from gtsam_mpc import BicycleModel

# Reuse the path-following world/render setup.
WIDTH, HEIGHT = pf.WIDTH, pf.HEIGHT
DT, MPC_DT, WHEELBASE = pf.DT, pf.MPC_DT, pf.WHEELBASE

# Default measurement noise (1-sigma).
GPS_SIGMAS = [0.3, 0.6, 1.0, 1.6]   # cycled with [ and ]
SPEED_SIGMA = 0.3                   # wheel-speed noise [m/s]

TRUE_COLOR = (90, 200, 140)
EST_COLOR = (90, 150, 255)
GPS_COLOR = (235, 120, 120)
WINDOW_COLOR = (250, 220, 120)
TEXT_COLOR = (210, 210, 220)


class RacecarApp:
    """Interactive base app: shared event loop, measurement model and render."""

    caption = "gtsam-mpc"

    def __init__(self):
        self.rng = np.random.default_rng(0)
        self.plant = BicycleModel(wheelbase=WHEELBASE, dt=DT)
        self.target_speed = 8.0
        self.gps_idx = 1
        self.paused = False

        self.true_trail: list = []
        self.est_trail: list = []
        self.gps_pts: list = []
        self.rmse_sq = 0.0
        self.nrmse = 0
        self.plan_states = None
        self._last_gps = None
        self._last_delta = 0.0

        self.true_state = None
        self.estimate = None

        self._set_path("1")
        self.setup()
        self.reset()

    # --- hooks for subclasses --------------------------------------------
    def setup(self) -> None:
        """Build the engine once (called before the first reset)."""

    def reset_engine(self) -> None:
        """Reset the engine to ``self.true_state`` and set ``self.estimate``."""
        self.estimate = self.true_state.copy()

    def set_gps_sigma(self, sigma: float) -> None:
        """Push a new GPS noise sigma to the engine."""

    def advance(self, refs: np.ndarray) -> np.ndarray:
        """Run one closed-loop step (not paused). Return the plan states.

        Implementations update ``self.true_state`` and ``self.estimate`` and set
        ``self._last_gps`` and ``self._last_delta``.
        """
        raise NotImplementedError

    def horizon(self) -> int:
        """Number of reference-trajectory steps to request from the path."""
        raise NotImplementedError

    def window_points(self) -> list:
        """Screen points of the estimator window to outline (optional)."""
        return []

    def extra_key(self, event) -> bool:
        """Handle a demo-specific key. Return True if consumed."""
        return False

    def hud_lines(self) -> list[str]:
        """Full list of HUD text lines for this demo."""
        return self.status_lines()

    # --- shared machinery ------------------------------------------------
    def _set_path(self, key: str) -> None:
        self.path_key = key
        self.path_name, gen = pf.PATHS[key]
        self.path = pf.make_path(gen)
        self.curvature = pf.path_curvature(self.path)

    def reset(self) -> None:
        self.true_state = pf.reset_on_path(self.path, self.target_speed)
        self.true_trail.clear()
        self.est_trail.clear()
        self.gps_pts.clear()
        self.rmse_sq, self.nrmse = 0.0, 0
        self.plan_states = None
        self._last_delta = 0.0
        self.reset_engine()

    def measure(self) -> tuple[np.ndarray, float]:
        """Noisy GPS position and wheel-speed reading of the hidden true state."""
        gps = self.true_state[:2] + self.rng.normal(0.0, GPS_SIGMAS[self.gps_idx], 2)
        v = self.true_state[3] + self.rng.normal(0.0, SPEED_SIGMA)
        return gps, float(v)

    def paused_suffix(self) -> str:
        return "   [PAUSED]" if self.paused else ""

    def status_lines(self) -> list[str]:
        """The two HUD lines common to both demos (path/speed and localization)."""
        loc_err = float(np.linalg.norm(self.true_state[:2] - self.estimate[:2]))
        rmse = float(np.sqrt(self.rmse_sq / self.nrmse)) if self.nrmse else 0.0
        cte = pf.cross_track_error(self.path, self.true_state[:2])
        return [
            f"path: {self.path_name}   target speed={self.target_speed:.1f} m/s   "
            f"v={self.true_state[3]:+.2f}",
            f"GPS sigma={GPS_SIGMAS[self.gps_idx]:.1f} m   loc err={loc_err:.2f} m   "
            f"loc RMSE={rmse:.2f} m   true cross-track={cte:.2f} m",
        ]

    def _record(self) -> None:
        self.true_trail.append(pf.world_to_screen(self.true_state[:2]))
        self.est_trail.append(pf.world_to_screen(self.estimate[:2]))
        self.gps_pts.append(pf.world_to_screen(self._last_gps))
        for buf in (self.true_trail, self.est_trail, self.gps_pts):
            if len(buf) > 400:
                buf.pop(0)
        self.rmse_sq += float(np.sum((self.true_state[:2] - self.estimate[:2]) ** 2))
        self.nrmse += 1

    def _handle_event(self, event) -> bool:
        """Process one event; return False to quit."""
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if event.key == pygame.K_SPACE:
                self.paused = not self.paused
            elif event.key == pygame.K_r:
                self.reset()
            elif event.key == pygame.K_UP:
                self.target_speed = min(self.target_speed + 1.0, pf.V_BOUNDS[1])
            elif event.key == pygame.K_DOWN:
                self.target_speed = max(self.target_speed - 1.0, 0.0)
            elif event.key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
                step = 1 if event.key == pygame.K_RIGHTBRACKET else -1
                self.gps_idx = int(np.clip(self.gps_idx + step, 0, len(GPS_SIGMAS) - 1))
                self.set_gps_sigma(GPS_SIGMAS[self.gps_idx])
            elif not self.extra_key(event):
                name = pygame.key.name(event.key)
                if name in pf.PATHS:
                    self._set_path(name)
                    self.reset()
        return True

    def _draw(self, screen, font) -> None:
        screen.fill(pf.BG)
        pf.draw_path(screen, self.path)

        for pt in self.gps_pts[-120:]:
            pygame.draw.circle(screen, GPS_COLOR, pt, 2)
        if len(self.true_trail) > 1:
            pygame.draw.lines(screen, TRUE_COLOR, False, self.true_trail, 2)
        if len(self.est_trail) > 1:
            pygame.draw.lines(screen, EST_COLOR, False, self.est_trail, 2)

        for pt in self.window_points():
            pygame.draw.circle(screen, WINDOW_COLOR, pt, 3, 1)

        if self.plan_states is not None:
            plan_pts = [pf.world_to_screen(s[:2]) for s in self.plan_states]
            if len(plan_pts) > 1:
                pygame.draw.lines(screen, pf.PLAN_COLOR, False, plan_pts, 2)

        pf.draw_car(screen, self.true_state, self._last_delta)
        ex, ey = pf.world_to_screen(self.estimate[:2])
        pygame.draw.circle(screen, EST_COLOR, (ex, ey), 6, 2)

        for i, text in enumerate(self.hud_lines()):
            screen.blit(font.render(text, True, TEXT_COLOR), (10, 10 + i * 20))

    def run(self) -> None:
        max_frames = max_frames_from_env()
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(self.caption)
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("monospace", 16)

        frame = 0
        running = True
        while running:
            for event in pygame.event.get():
                if not self._handle_event(event):
                    running = False

            near = pf.nearest_index(self.path, self.estimate[:2])
            refs = pf.reference_trajectory(self.path, self.curvature, near,
                                           self.target_speed, self.horizon(), MPC_DT)
            if not self.paused:
                self.plan_states = self.advance(refs)
                self._record()

            self._draw(screen, font)
            pygame.display.flip()
            clock.tick(pf.FPS)

            frame += 1
            if max_frames is not None and frame >= max_frames:
                running = False

        pygame.quit()
