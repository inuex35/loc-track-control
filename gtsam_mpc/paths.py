"""Reference paths and path-tracking geometry for the bicycle MPC.

Pure NumPy helpers (no rendering): a set of closed reference paths, uniform
arc-length resampling, discrete curvature, nearest-point / cross-track-error
queries, and a curvature-aware reference-trajectory sampler that turns a path
into the per-step reference ``(horizon + 1, 4)`` the MPC tracks.
"""

from __future__ import annotations

import numpy as np

PATH_RESOLUTION = 0.25   # arc-length spacing of resampled paths [m]
LAT_ACCEL_MAX = 3.0      # comfort lateral-accel limit for corner speed [m/s^2]


# --- reference path generators -------------------------------------------
# Each returns a closed-loop polyline as raw (un-resampled) points; ``make_path``
# resamples it to uniform arc-length spacing for stable lookahead.
def circle(radius: float = 22.0, n: int = 400) -> np.ndarray:
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t)])


def figure_eight(scale: float = 26.0, n: int = 600) -> np.ndarray:
    # Lemniscate of Gerono: a smooth self-crossing figure-eight.
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([scale * np.cos(t), scale * np.sin(t) * np.cos(t)])


def sine_wave(length: float = 64.0, amp: float = 5.0, cycles: int = 2,
              base: float = 12.0, n: int = 800) -> np.ndarray:
    # Closed serpentine: two sine legs joined by semicircular U-turns. Using
    # ``cos`` with an integer cycle count makes each leg start and end with a
    # horizontal tangent, so it meets the semicircle ends smoothly.
    radius = base
    xs = np.linspace(-length / 2, length / 2, n // 4)
    wave = amp * np.cos(2 * np.pi * cycles * (xs + length / 2) / length)
    top = np.column_stack([xs, base + wave])                 # left -> right
    bottom = np.column_stack([xs[::-1], -base + wave[::-1]])  # right -> left
    a = np.linspace(np.pi / 2, -np.pi / 2, n // 4)
    right = np.column_stack([length / 2 + radius * np.cos(a), amp + radius * np.sin(a)])
    a2 = np.linspace(3 * np.pi / 2, np.pi / 2, n // 4)
    left = np.column_stack([-length / 2 + radius * np.cos(a2), amp + radius * np.sin(a2)])
    return np.vstack([top, right, bottom, left])


def racetrack(half: float = 26.0, radius: float = 16.0, n: int = 600) -> np.ndarray:
    # Stadium / oval: two straights joined by semicircular ends.
    straight = half - radius
    pts = []
    for a in np.linspace(-np.pi / 2, np.pi / 2, n // 4):
        pts.append([straight + radius * np.cos(a), radius * np.sin(a)])
    for x in np.linspace(straight, -straight, n // 4):
        pts.append([x, radius])
    for a in np.linspace(np.pi / 2, 3 * np.pi / 2, n // 4):
        pts.append([-straight + radius * np.cos(a), radius * np.sin(a)])
    for x in np.linspace(-straight, straight, n // 4):
        pts.append([x, -radius])
    return np.array(pts)


def rounded_square(half: float = 24.0, radius: float = 8.0, n: int = 600) -> np.ndarray:
    s = half - radius
    centers = [(s, s), (-s, s), (-s, -s), (s, -s)]
    starts = [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]
    pts = []
    for (cx, cy), a0 in zip(centers, starts):
        for a in np.linspace(a0, a0 + np.pi / 2, n // 4):
            pts.append([cx + radius * np.cos(a), cy + radius * np.sin(a)])
    return np.array(pts)


# Keyed for the interactive demos (1..5).
PATHS = {
    "1": ("circle", circle),
    "2": ("figure-eight", figure_eight),
    "3": ("sine wave", sine_wave),
    "4": ("racetrack", racetrack),
    "5": ("rounded square", rounded_square),
}


# --- geometry helpers ----------------------------------------------------
def make_path(generator, resolution: float = PATH_RESOLUTION) -> np.ndarray:
    """Resample a raw polyline to uniform arc-length spacing (closed loop)."""
    raw = generator()
    closed = np.vstack([raw, raw[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    n = max(int(total / resolution), 8)
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
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    a = np.linalg.norm(d1, axis=1)
    b = np.linalg.norm(d2, axis=1)
    c = np.linalg.norm(nxt - prv, axis=1)
    denom = a * b * c
    return np.where(denom > 1e-9, 2.0 * np.abs(cross) / np.maximum(denom, 1e-9), 0.0)


def nearest_index(path: np.ndarray, p: np.ndarray) -> int:
    """Index of the path vertex closest to ``p``."""
    return int(np.argmin(np.sum((path - p) ** 2, axis=1)))


def cross_track_error(path: np.ndarray, p: np.ndarray) -> float:
    """Perpendicular distance from ``p`` to the path polyline.

    Projects onto the two segments adjacent to the nearest vertex, so the result
    is the true lateral offset (not the quantized distance to a discrete vertex).
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


def _curve_speed(kappa: float, speed: float, lat_accel_max: float = LAT_ACCEL_MAX) -> float:
    """Speed capped so curvature ``kappa`` stays within the lateral-accel limit."""
    if kappa <= 1e-4:
        return speed
    return min(speed, float(np.sqrt(lat_accel_max / kappa)))


def reference_trajectory(path, curvature, near_idx, speed, horizon, dt,
                         resolution: float = PATH_RESOLUTION) -> np.ndarray:
    """Build the per-step reference ``(horizon + 1, 4)`` the MPC tracks.

    Starting at the closest path point, each horizon step advances along the
    path by ``v * dt`` of arc length (``v`` locally capped by curvature). Each
    reference state is the path point with its tangent heading and that speed.
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
        pos += v * dt / resolution
    return refs


def reset_on_path(path: np.ndarray, speed: float) -> np.ndarray:
    """Place a car at the path start, headed along the tangent, at ``speed``."""
    tangent = path[1] - path[0]
    heading = np.arctan2(tangent[1], tangent[0])
    return np.array([path[0, 0], path[0, 1], heading, speed])
