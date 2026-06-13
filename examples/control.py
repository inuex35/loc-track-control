"""制御 (Control): receding-horizon MPC as factor-graph optimization.

This is the *control* layer (cf. the MPC half of JPCM / IPN_MPC, and this
package's own solvers). A 2-D point mass is regulated to a goal by
:class:`~gtsam_mpc.LinearMPC`, which solves a Gaussian factor graph at every
step. For the nonlinear car version see ``examples/bicycle_sim.py``.

Run:  python examples/control.py
"""

from __future__ import annotations

import numpy as np

from gtsam_mpc import LinearMPC, point_mass_2d


def run() -> dict:
    dt = 0.1
    system = point_mass_2d(dt=dt)
    mpc = LinearMPC(
        system,
        Q=np.diag([4.0, 4.0, 0.5, 0.5]),       # x, y, vx, vy
        R=np.diag([0.05, 0.05]),               # ax, ay
        horizon=30,
        Qf=np.diag([40.0, 40.0, 4.0, 4.0]),
    )
    x0 = np.array([-6.0, -3.0, 0.0, 0.0])
    goal = np.array([5.0, 4.0, 0.0, 0.0])

    traj = mpc.simulate(x0, steps=120, xref=goal)
    final_err = float(np.linalg.norm(traj.states[-1] - goal))
    max_accel = float(np.max(np.linalg.norm(traj.controls, axis=1)))
    return {"dt": dt, "goal": goal, "traj": traj,
            "final_error": final_err, "max_accel": max_accel}


def main() -> None:
    out = run()
    traj = out["traj"]
    print("制御 (Control): point-mass MPC to a goal")
    print(f"  steps           = {traj.controls.shape[0]}")
    print(f"  final state err = {out['final_error']:.3f}")
    print(f"  peak |accel|    = {out['max_accel']:.2f} m/s^2")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = traj.states
        plt.figure(figsize=(7, 6))
        plt.plot(s[:, 0], s[:, 1], "b-", lw=2, label="closed-loop path")
        plt.plot(s[0, 0], s[0, 1], "go", label="start")
        plt.plot(out["goal"][0], out["goal"][1], "r*", ms=15, label="goal")
        plt.axis("equal"); plt.grid(alpha=0.3); plt.legend()
        plt.title("Control: receding-horizon MPC")
        plt.savefig("control.png", dpi=120)
        print("  saved plot to control.png")
    except ImportError:
        print("  (matplotlib not installed -- skipping plot)")


if __name__ == "__main__":
    main()
