# gtsam-mpc

**Estimation, control and tracking** as **factor-graph optimization**, solved with [GTSAM](https://gtsam.org/).

The package is built around one idea: **estimation, control and tracking are all maximum-a-posteriori (MAP) inference on factor graphs built from a shared vocabulary of factors.** A finite-horizon linear-quadratic optimal control problem, for instance, is exactly MAP inference on a Gaussian factor graph — eliminating it *is* the classical Riccati recursion, expressed declaratively as factors. The same factors are reused to localize the robot, track moving obstacles, and fuse everything into a single optimization.

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the package layout (`models`, `factors`, `constraints`, `control`, `estimation`, `joint`, `paths`).

| Solver / estimator | Problem | Graph |
| ------------------ | ------- | ----- |
| `LinearMPC`              | linear-quadratic MPC                          | Gaussian (= Riccati) |
| `BicycleMPC`             | nonlinear kinematic-bicycle MPC               | nonlinear (LM); barrier / aug-Lagrangian / slack constraints |
| `MovingHorizonEstimator` | sliding-window localization (MHE)             | nonlinear (LM) |
| `ConstantVelocityTracker`| 2-D multi-object tracking / prediction        | Gaussian |
| `JointEstimatorMPC`      | localization **+** control in one graph        | nonlinear (LM) |
| `JointLocTrackControl`   | localization **+** tracking **+** control, one graph | nonlinear (LM), uncertainty-aware avoidance |

## Installation

```bash
pip install -e .          # core (gtsam, numpy)
pip install -e ".[dev]"   # + pytest
pip install -e ".[plot]"  # + matplotlib (example plots)
pip install -e ".[sim]"   # + pygame (interactive demos / GIF recording)
```

## Quickstart

### Linear MPC

```python
import numpy as np
from gtsam_mpc import LinearMPC, point_mass_2d

mpc = LinearMPC(
    point_mass_2d(dt=0.1),               # state = [px, py, vx, vy]
    Q=np.diag([4.0, 4.0, 0.5, 0.5]),
    R=np.diag([0.05, 0.05]),
    horizon=30,
    Qf=np.diag([40.0, 40.0, 4.0, 4.0]),
)
goal = np.array([3.0, -2.0, 0.0, 0.0])   # position goal, zero velocity
u = mpc.control(np.zeros(4), xref=goal)  # first control to apply
traj = mpc.simulate(np.zeros(4), steps=60, xref=goal)
```

`xref` defaults to the origin; tracking is exact when it is an equilibrium (e.g. a zero-velocity goal). The whole problem is a *linear* Gaussian factor graph:

| MPC ingredient | Factor graph element |
| -------------- | -------------------- |
| Initial state `x_0 = x_init` | hard equality constraint |
| Dynamics `x_{k+1} = A x_k + B u_k` | hard equality constraint on `x_k, u_k, x_{k+1}` |
| State / control / terminal cost | quadratic factor with information `Q` / `R` / `Qf` |

### Nonlinear bicycle MPC

```python
import numpy as np
from gtsam_mpc import BicycleMPC, BicycleModel

mpc = BicycleMPC(
    BicycleModel(wheelbase=2.5, dt=0.1),   # state [x, y, θ, v], control [a, δ]
    Q=np.diag([3.0, 3.0, 0.8, 0.4]),
    R=np.diag([0.1, 0.1]),
    horizon=25,
    Qf=np.diag([30.0, 30.0, 3.0, 3.0]),
    a_bounds=(-3.0, 3.0), delta_bounds=(-0.6, 0.6), v_bounds=(-3.0, 8.0),
    constraint_mode="al",                  # "barrier" | "al" | "slack"
)
goal = np.array([10.0, 4.0, 0.0, 0.0])
u = mpc.control(np.zeros(4), goal)
traj = mpc.simulate(np.zeros(4), goal, steps=120)
```

The dynamics (RK4, with analytic Jacobians) and costs are `gtsam.CustomFactor`s, optimized with Levenberg-Marquardt and warm-started between steps. `xref` may be a single state (set-point) or a `(horizon+1, 4)` reference trajectory (path tracking). Obstacles to avoid are passed as `obstacles=` (static centres `(2,)` or per-step predictions `(horizon+1, 2)`).

**Inequality constraints** (input / speed / obstacle keep-out) are handled by a pluggable [`ConstraintStrategy`](gtsam_mpc/constraints.py):

- `BarrierStrategy` — fixed one-sided quadratic penalty (soft, fast).
- `AugmentedLagrangianStrategy` — outer multiplier loop; meets bounds tightly without tuning a weight.
- `SlackStrategy` — `g + s² = 0` hard equality with a slack variable; a single solve.

Pass `constraint_mode="barrier"|"al"|"slack"` or a `constraints=` strategy instance.

> The barrier/slack avoidance is **soft** — it strongly avoids but does not *guarantee* a hard minimum distance. Augmented Lagrangian meets the bounds tightly. See the hard-constraints discussion in [ARCHITECTURE.md](ARCHITECTURE.md).

## Demos

All demos run headless for CI/testing with `SDL_VIDEODRIVER=dummy GTSAM_MPC_MAX_FRAMES=<n>`.

**Minimal samples** (print a metric vs. a naive baseline; save a plot if matplotlib is installed) — one per autonomy layer, echoing three reference projects:

| Script | Layer | Echoes |
| ------ | ----- | ------ |
| `examples/localization.py` | estimation: Pose2 odometry + GPS fusion | JPCM / IPN_MPC |
| `examples/control.py` | control: receding-horizon MPC | this package |
| `examples/tracking.py` | perception: multi-object CV smoother | FGO-MOT |
| `examples/double_integrator.py` | control: 1-D LQR vs. Riccati | — |

**Interactive pygame demos** (left-click / keys to drive; **space** pause, **r** reset, **esc** quit):

| Script | What it shows |
| ------ | ------------- |
| `examples/pygame_sim.py` | point-mass MPC chasing a clickable goal |
| `examples/bicycle_sim.py` | bicycle MPC driving to a clickable goal |
| `examples/path_following_sim.py` | bicycle MPC tracking selectable paths (1–5) at the curvature-limited speed |
| `examples/loc_control_sim.py` | **localization + control**: drive on the MHE estimate vs. raw GPS (two-graph pipeline) |
| `examples/joint_loc_control_sim.py` | the same, as a **single graph** (one `optimize()`) |
| `examples/integrated_sim.py` | **track + avoid**: CV tracker predicts moving obstacles, `BicycleMPC` avoids them |
| `examples/loc_track_control_sim.py` | **full stack**: localization + tracking + control in one graph, uncertainty-aware avoidance |

### Full stack: `JointLocTrackControl`

`examples/loc_track_control_sim.py` drives `JointLocTrackControl` — ego localization (GPS + speed), constant-velocity tracking of several moving obstacles, and path-following control with obstacle avoidance, **all in one `NonlinearFactorGraph` solved once per step**. Two touches make the coupling pay off:

- the keep-out radius for each obstacle is **inflated by that obstacle's track covariance** (read from `gtsam.Marginals`) — the car gives a wider berth to uncertain tracks;
- detections are **range-dependent** — a closer obstacle is observed more strongly, so its track tightens and the car can pass it more confidently.

```python
import numpy as np
from gtsam_mpc import JointLocTrackControl

jtc = JointLocTrackControl(n_obstacles=3, safety_radius=3.0, n_sigma=1.5)
jtc.reset(x0, [obs0, obs1, obs2])        # obstacle [px,py,vx,vy] or (px,py)
est, plan_states, plan_u, obs_est, obs_pred = jtc.step(u, gps, v_meas, dets, reference)
```

### Recording a GIF

No display needed — record any demo exposing `iter_frames` to an animated GIF:

```bash
python examples/make_gif.py out.gif --demo loc_track_control_sim --frames 300 --scale 0.6
python examples/make_gif.py out.gif --demo integrated_sim --frames 90
```

## Tests

```bash
pytest          # 44 tests, ~4–5 min (the joint/marginals demos dominate)
```

Coverage spans the whole package: `LinearMPC` against an independent finite-horizon LQR (Riccati) reference; `BicycleModel` Jacobians against finite differences; the three constraint strategies; `MovingHorizonEstimator` / `ConstantVelocityTracker`; the path geometry; and each demo's `run()` entry point, asserting the factor-graph result beats its naive baseline (fused vs. raw, smoothed vs. detections, localizes + tracks + avoids).

## API reference

- **Models** (`gtsam_mpc.models`): `LinearSystem(A, B)`, `double_integrator(dt)`, `point_mass_2d(dt)`, `BicycleModel(wheelbase, dt, substeps)` (`.step`, `.jacobians`).
- **Control** (`gtsam_mpc.control`):
  - `LinearMPC(system, Q, R, horizon, Qf=None)` — `.solve(x0, xref=None)`, `.control`, `.simulate`.
  - `BicycleMPC(model, Q, R, horizon, Qf=None, a_bounds, delta_bounds, v_bounds, constraints=None, constraint_mode="barrier", barrier_weight, safety_radius, obstacle_weight, ...)` — `.solve(x0, xref, obstacles=None, warm_start=True)`, `.control`, `.simulate`, `.reset`.
  - `MPCResult` — `.states`, `.controls`, `.u0`.
- **Constraints** (`gtsam_mpc.constraints`): `BarrierStrategy`, `AugmentedLagrangianStrategy`, `SlackStrategy`, `Inequality`, `make_strategy`.
- **Estimation** (`gtsam_mpc.estimation`):
  - `MovingHorizonEstimator(model, window, gps_sigma, ...)` — `.reset(x0)`, `.update(u, gps, v) -> state`.
  - `ConstantVelocityTracker(dt, process_sigma, meas_sigma)` — `.smooth(detections)`, `.estimate(detections)`, `.predict(state, horizon, dt=None)`.
- **Joint** (`gtsam_mpc.joint`):
  - `JointEstimatorMPC(...)` — `.reset(x0)`, `.step(u, gps, v, reference) -> (estimate, states, controls)`.
  - `JointLocTrackControl(n_obstacles, safety_radius, n_sigma, ...)` — `.reset(x0, obstacles)`, `.step(u, gps, v, detections, reference) -> (estimate, states, controls, obstacle_estimates, obstacle_predictions)`.
- **Factors / paths** (`gtsam_mpc.factors`, `gtsam_mpc.paths`): the reusable `CustomFactor` vocabulary and path geometry (`make_path`, `path_curvature`, `reference_trajectory`, `cross_track_error`, …).

## References

- F. Dellaert and M. Kaess, *Factor Graphs for Robot Perception*, Foundations and Trends in Robotics, 2017.
- Reference projects this package echoes: JPCM / IPN_MPC (joint positioning + control), FGO-MOT (factor-graph multi-object tracking).
- GTSAM: <https://gtsam.org/>

## License

MIT — see [LICENSE](LICENSE).
