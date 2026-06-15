# gtsam-mpc

**Estimation, control and tracking as factor-graph optimization** with [GTSAM](https://gtsam.org/).

One idea throughout: estimation, control and tracking are all MAP inference on factor graphs built from a shared vocabulary of factors. (Linear MPC, for instance, is exactly a Gaussian factor graph — eliminating it is the Riccati recursion.) See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the layout.

| Component | Problem |
| --------- | ------- |
| `LinearMPC` | linear-quadratic MPC (= Riccati) |
| `BicycleMPC` | nonlinear bicycle MPC; barrier / aug-Lagrangian / slack constraints |
| `MovingHorizonEstimator` | sliding-window localization |
| `ConstantVelocityTracker` | multi-object tracking / prediction |
| `JointEstimatorMPC` | localization + control, one graph |
| `JointLocTrackControl` | localization + tracking + control, one graph |

## Install

```bash
pip install -e ".[sim]"   # core + pygame; use [dev] for pytest, [plot] for matplotlib
```

## Quickstart

```python
import numpy as np
from gtsam_mpc import BicycleMPC, BicycleModel

mpc = BicycleMPC(
    BicycleModel(wheelbase=2.5, dt=0.1),   # state [x, y, θ, v], control [a, δ]
    Q=np.diag([3., 3., .8, .4]), R=np.diag([.1, .1]), horizon=25,
    a_bounds=(-3, 3), delta_bounds=(-.6, .6), v_bounds=(-3, 8),
    constraint_mode="al",                  # "barrier" | "al" | "slack"
)
goal = np.array([10., 4., 0., 0.])
u = mpc.control(np.zeros(4), goal)                       # first [a, δ]
traj = mpc.simulate(np.zeros(4), goal, steps=120)        # closed loop
```

Dynamics (RK4) and costs are `gtsam.CustomFactor`s solved with Levenberg-Marquardt (warm-started). `xref` is a single state or a `(horizon+1, 4)` reference trajectory; obstacles to avoid go in `obstacles=`. `LinearMPC(system, Q, R, horizon)` is the linear analogue with `.solve/.control/.simulate`.

## Demos

Interactive (space = pause, r = reset, esc = quit); all run headless with `SDL_VIDEODRIVER=dummy GTSAM_MPC_MAX_FRAMES=<n>`.

| Script | Shows |
| ------ | ----- |
| `pygame_sim.py` / `bicycle_sim.py` | point-mass / bicycle MPC to a clickable goal |
| `path_following_sim.py` | bicycle MPC tracking selectable paths (1–5) |
| `loc_control_sim.py` / `joint_loc_control_sim.py` | localization + control (two-graph pipeline / single graph) |
| `integrated_sim.py` | track moving obstacles, then avoid them |
| `loc_track_control_sim.py` | **full stack**: localization + tracking + control in one graph |
| `localization.py` / `control.py` / `tracking.py` | minimal per-layer samples (print metric vs. baseline) |

Record any demo to a GIF (no display needed):

```bash
python examples/make_gif.py out.gif --demo loc_track_control_sim --frames 300
```

The full-stack demo inflates each obstacle's keep-out radius by its track covariance (wider berth for uncertain tracks) and observes nearer obstacles more strongly.

## Test

```bash
pytest   # 44 tests
```

LinearMPC vs. a Riccati reference, bicycle Jacobians vs. finite differences, the three constraint strategies, the estimators/tracker, path geometry, and each demo's `run()`.

## License

MIT — see [LICENSE](LICENSE).
