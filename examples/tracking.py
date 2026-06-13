"""トラッキング (Tracking): multi-object tracking on a factor graph.

This is the *perception / estimation* layer (cf. FGO-MOT). Several targets move
with constant velocity in 2-D. A sensor returns noisy position detections each
frame. For each track we build a linear Gaussian factor graph with:

    * a constant-velocity **motion model** factor between consecutive states
      x_{t+1} = F x_t                                  (the "motion prior")
    * a **measurement** factor at every frame (observes position only)

Solving the graph is a batch (fixed-lag-style) smoother that produces much
smoother, more accurate trajectories than the raw detections. Data association
(which detection belongs to which track) is assumed known here -- that is the
hard part of real MOT and is out of scope for this didactic sample.

Run:  python examples/tracking.py
"""

from __future__ import annotations

import numpy as np
import gtsam
from gtsam.symbol_shorthand import X


# State is [px, py, vx, vy]; we observe [px, py].
def _F(dt: float) -> np.ndarray:
    return np.array(
        [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float
    )


_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)


def _smooth_track(measurements: list[np.ndarray], dt: float,
                  meas_sigma: float) -> np.ndarray:
    """Batch-smooth one track from its position detections."""
    T = len(measurements) - 1
    F = _F(dt)
    graph = gtsam.GaussianFactorGraph()
    # Loose prior to anchor the (otherwise gauge-free) initial state.
    graph.add(X(0), np.eye(4), np.zeros(4),
              gtsam.noiseModel.Diagonal.Sigmas(np.full(4, 10.0)))
    proc = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.02, 0.02, 0.3, 0.3]))
    meas = gtsam.noiseModel.Diagonal.Sigmas(np.full(2, meas_sigma))
    for t in range(T):
        # x_{t+1} - F x_t = 0  (constant-velocity motion prior)
        graph.add(X(t + 1), np.eye(4), X(t), -F, np.zeros(4), proc)
    for t in range(T + 1):
        graph.add(X(t), _H, measurements[t], meas)
    sol = graph.optimize()
    return np.array([sol.at(X(t)) for t in range(T + 1)])


def run(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    dt, T, meas_sigma = 0.1, 40, 0.7
    F = _F(dt)

    # A few targets with different initial position/velocity.
    inits = [
        np.array([0.0, 0.0, 2.0, 1.0]),
        np.array([5.0, -2.0, -1.0, 1.5]),
        np.array([-3.0, 3.0, 1.2, -0.8]),
    ]

    raw_err, smooth_err = [], []
    tracks = []
    for x0 in inits:
        truth = [x0]
        for _ in range(T):
            truth.append(F @ truth[-1])
        detections = [_H @ truth[t] + rng.normal(0, meas_sigma, 2) for t in range(T + 1)]
        estimate = _smooth_track(detections, dt, meas_sigma)

        truth_pos = np.array([s[:2] for s in truth])
        det = np.array(detections)
        raw_err.append(np.sqrt(np.mean(np.sum((det - truth_pos) ** 2, axis=1))))
        smooth_err.append(
            np.sqrt(np.mean(np.sum((estimate[:, :2] - truth_pos) ** 2, axis=1)))
        )
        tracks.append((truth_pos, det, estimate))

    return {
        "tracks": tracks,
        "rmse_raw": float(np.mean(raw_err)),
        "rmse_smoothed": float(np.mean(smooth_err)),
    }


def main() -> None:
    out = run()
    print("トラッキング (Tracking): factor-graph multi-object tracking")
    print(f"  targets               = {len(out['tracks'])}")
    print(f"  raw-detection   RMSE  = {out['rmse_raw']:.3f} m")
    print(f"  smoothed (graph) RMSE = {out['rmse_smoothed']:.3f} m")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 6))
        colors = ["C0", "C1", "C2", "C3"]
        for i, (truth, det, est) in enumerate(out["tracks"]):
            c = colors[i % len(colors)]
            plt.plot(truth[:, 0], truth[:, 1], c + "-", lw=2,
                     label="truth" if i == 0 else None)
            plt.scatter(det[:, 0], det[:, 1], s=10, c=c, alpha=0.35,
                        label="detections" if i == 0 else None)
            plt.plot(est[:, 0], est[:, 1], c + "--",
                     label="smoothed" if i == 0 else None)
        plt.axis("equal"); plt.grid(alpha=0.3); plt.legend()
        plt.title("Tracking: factor-graph MOT (constant-velocity smoother)")
        plt.savefig("tracking.png", dpi=120)
        print("  saved plot to tracking.png")
    except ImportError:
        print("  (matplotlib not installed -- skipping plot)")


if __name__ == "__main__":
    main()
