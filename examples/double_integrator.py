"""Drive a double integrator to the origin with factor-graph MPC.

Run from the repository root:

    python examples/double_integrator.py

It prints the open-loop optimal trajectory and the closed-loop (receding
horizon) trajectory. If matplotlib is installed it also saves a plot to
``double_integrator.png``.
"""

import numpy as np

from gtsam_mpc import LinearMPC, double_integrator


def main() -> None:
    dt = 0.1
    system = double_integrator(dt=dt)

    # Penalize position and velocity error, with cheap control and a heavy
    # terminal cost so the state is driven close to the origin by the horizon end.
    Q = np.diag([1.0, 1.0])
    R = np.array([[0.1]])
    Qf = np.diag([100.0, 100.0])

    mpc = LinearMPC(system, Q=Q, R=R, horizon=25, Qf=Qf)

    x0 = np.array([5.0, 0.0])  # start 5 m from the origin, at rest

    open_loop = mpc.solve(x0)
    print("Open-loop optimal plan from x0 =", x0)
    print(f"  first control u0 = {open_loop.u0}")
    print(f"  terminal state   = {open_loop.states[-1]}")

    closed_loop = mpc.simulate(x0, steps=60)
    print("\nClosed-loop (receding horizon) trajectory:")
    print(f"  final state after {closed_loop.controls.shape[0]} steps "
          f"= {closed_loop.states[-1]}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.arange(closed_loop.states.shape[0]) * dt
        fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        axes[0].plot(t, closed_loop.states[:, 0]); axes[0].set_ylabel("position")
        axes[1].plot(t, closed_loop.states[:, 1]); axes[1].set_ylabel("velocity")
        axes[2].step(t[:-1], closed_loop.controls[:, 0], where="post")
        axes[2].set_ylabel("control"); axes[2].set_xlabel("time [s]")
        for ax in axes:
            ax.axhline(0.0, color="0.7", lw=0.8); ax.grid(alpha=0.3)
        fig.suptitle("Double integrator: closed-loop factor-graph MPC")
        fig.tight_layout()
        fig.savefig("double_integrator.png", dpi=120)
        print("\nSaved plot to double_integrator.png")
    except ImportError:
        print("\n(matplotlib not installed -- skipping plot)")


if __name__ == "__main__":
    main()
