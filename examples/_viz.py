"""Shared pygame rendering helpers for the example simulations.

The example sims share a common 2-D world->screen camera, a colour palette, and
car/grid drawing. Those primitives live here so each demo only contains its own
scenario logic. Sims expose thin module-level aliases (e.g. ``world_to_screen``,
``draw_car``) that delegate here, so cross-sim imports keep working unchanged.
"""

from __future__ import annotations

import os

import numpy as np
import pygame

# --- Shared colour palette -----------------------------------------------
BG = (18, 18, 24)
GRID = (38, 38, 48)
CAR_BODY = (80, 200, 255)
WHEEL = (235, 235, 245)
GOAL_COLOR = (255, 120, 120)
PLAN_COLOR = (120, 200, 140)
TEXT_COLOR = (210, 210, 220)
TRAIL_COLOR = (60, 120, 90)


def rot(theta: float) -> np.ndarray:
    """2x2 rotation matrix for ``theta`` radians."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


class View:
    """A fixed 2-D camera mapping world metres to screen pixels.

    World origin is the screen centre with ``y`` pointing up.
    """

    def __init__(self, width: int, height: int, pixels_per_meter: float):
        self.width = int(width)
        self.height = int(height)
        self.ppm = float(pixels_per_meter)

    def world_to_screen(self, p) -> tuple[int, int]:
        return (
            int(self.width / 2 + p[0] * self.ppm),
            int(self.height / 2 - p[1] * self.ppm),
        )

    def screen_to_world(self, sx: float, sy: float) -> np.ndarray:
        return np.array(
            [(sx - self.width / 2) / self.ppm, (self.height / 2 - sy) / self.ppm]
        )

    def draw_grid(self, screen, spacing_m: float = 1.0, color=GRID) -> None:
        step = max(int(self.ppm * spacing_m), 2)
        for x in range(self.width // 2 % step, self.width, step):
            pygame.draw.line(screen, color, (x, 0), (x, self.height))
        for y in range(self.height // 2 % step, self.height, step):
            pygame.draw.line(screen, color, (0, y), (self.width, y))


def draw_car(screen, view: View, state: np.ndarray, delta: float,
             wheelbase: float = 2.5) -> None:
    """Draw a bicycle-model car: oriented body with steered front wheels."""
    px, py, theta, _ = state
    pos = np.array([px, py])
    R = rot(theta)
    length, width = wheelbase + 1.0, 1.8

    # Body rectangle centred on the wheelbase midpoint.
    half = np.array([length / 2, width / 2])
    corners = np.array([[-half[0], -half[1]], [half[0], -half[1]],
                        [half[0], half[1]], [-half[0], half[1]]])
    body = [view.world_to_screen(pos + R @ (c + np.array([length / 2 - 0.5, 0])))
            for c in corners]
    pygame.draw.polygon(screen, CAR_BODY, body)

    def wheel(center_body, ang):
        wl, ww = 0.7, 0.25
        wr = rot(theta + ang)
        pts = np.array([[-wl, -ww], [wl, -ww], [wl, ww], [-wl, ww]])
        screen_pts = [view.world_to_screen(pos + R @ center_body + wr @ p)
                      for p in pts]
        pygame.draw.polygon(screen, WHEEL, screen_pts)

    wheel(np.array([0.0, width / 2]), 0.0)
    wheel(np.array([0.0, -width / 2]), 0.0)
    wheel(np.array([wheelbase, width / 2]), delta)
    wheel(np.array([wheelbase, -width / 2]), delta)


def max_frames_from_env() -> int | None:
    """Frame cap from ``GTSAM_MPC_MAX_FRAMES`` (for headless smoke runs)."""
    value = os.environ.get("GTSAM_MPC_MAX_FRAMES")
    return int(value) if value else None
