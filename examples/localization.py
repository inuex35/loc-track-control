"""位置推定 (Localization): odometry + GPS sensor fusion on a factor graph.

This mirrors the *estimation* layer of a robotics stack (cf. the positioning
half of JPCM / IPN_MPC). A 2-D robot drives along an arc. We have:

    * noisy wheel **odometry** between consecutive poses  -> BetweenFactorPose2
    * occasional noisy **GPS** absolute position fixes     -> PriorFactorPose2

Both are fused by finding the MAP trajectory of a Pose2 factor graph with
GTSAM. Odometry alone drifts without bound; the GPS fixes pin it down, and the
factor graph optimally blends the two.

Run:  python examples/localization.py
"""

from __future__ import annotations

import numpy as np
import gtsam
from gtsam.symbol_shorthand import X


def run(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    N = 60
    odom_step = gtsam.Pose2(0.5, 0.0, 0.05)          # true per-step motion (arc)
    odom_noise = np.array([0.08, 0.08, 0.03])        # odometry sigma (x, y, theta)
    gps_noise = 0.9                                  # GPS position sigma [m]
    gps_every = 8

    # Ground-truth trajectory.
    true = [gtsam.Pose2(0.0, 0.0, 0.0)]
    for _ in range(N):
        true.append(true[-1].compose(odom_step))

    odom_model = gtsam.noiseModel.Diagonal.Sigmas(odom_noise)
    # GPS constrains position only: make the heading sigma huge.
    gps_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([gps_noise, gps_noise, 1e6]))

    graph = gtsam.NonlinearFactorGraph()
    graph.add(
        gtsam.PriorFactorPose2(
            X(0), true[0], gtsam.noiseModel.Diagonal.Sigmas(np.full(3, 1e-3))
        )
    )

    noisy_odom = []
    for t in range(N):
        rel = true[t].between(true[t + 1])
        nd = gtsam.Pose2(
            rel.x() + rng.normal(0, odom_noise[0]),
            rel.y() + rng.normal(0, odom_noise[1]),
            rel.theta() + rng.normal(0, odom_noise[2]),
        )
        noisy_odom.append(nd)
        graph.add(gtsam.BetweenFactorPose2(X(t), X(t + 1), nd, odom_model))
        if t % gps_every == 0:
            gx = true[t + 1].x() + rng.normal(0, gps_noise)
            gy = true[t + 1].y() + rng.normal(0, gps_noise)
            graph.add(gtsam.PriorFactorPose2(X(t + 1), gtsam.Pose2(gx, gy, 0.0), gps_model))

    # Initial guess = dead reckoning (compose noisy odometry from the start).
    init = gtsam.Values()
    p = true[0]
    init.insert(X(0), p)
    for t in range(N):
        p = p.compose(noisy_odom[t])
        init.insert(X(t + 1), p)

    result = gtsam.LevenbergMarquardtOptimizer(
        graph, init, gtsam.LevenbergMarquardtParams()
    ).optimize()

    def pos_rmse(values: gtsam.Values) -> float:
        err = [
            (values.atPose2(X(t)).x() - true[t].x()) ** 2
            + (values.atPose2(X(t)).y() - true[t].y()) ** 2
            for t in range(N + 1)
        ]
        return float(np.sqrt(np.mean(err)))

    return {
        "N": N,
        "true": true,
        "init": init,
        "result": result,
        "rmse_dead_reckoning": pos_rmse(init),
        "rmse_fused": pos_rmse(result),
    }


def main() -> None:
    out = run()
    print("位置推定 (Localization): odometry + GPS fusion")
    print(f"  dead-reckoning position RMSE = {out['rmse_dead_reckoning']:.3f} m")
    print(f"  fused (factor graph)    RMSE = {out['rmse_fused']:.3f} m")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        N, true = out["N"], out["true"]
        tx = [p.x() for p in true]; ty = [p.y() for p in true]
        ix = [out["init"].atPose2(X(t)).x() for t in range(N + 1)]
        iy = [out["init"].atPose2(X(t)).y() for t in range(N + 1)]
        rx = [out["result"].atPose2(X(t)).x() for t in range(N + 1)]
        ry = [out["result"].atPose2(X(t)).y() for t in range(N + 1)]
        plt.figure(figsize=(7, 6))
        plt.plot(tx, ty, "k-", lw=2, label="ground truth")
        plt.plot(ix, iy, "r--", label="dead reckoning")
        plt.plot(rx, ry, "b-", label="fused (factor graph)")
        plt.axis("equal"); plt.grid(alpha=0.3); plt.legend()
        plt.title("Localization: odometry + GPS fusion")
        plt.savefig("localization.png", dpi=120)
        print("  saved plot to localization.png")
    except ImportError:
        print("  (matplotlib not installed -- skipping plot)")


if __name__ == "__main__":
    main()
