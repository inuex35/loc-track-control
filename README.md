# gtsam-mpc

**Model Predictive Control (MPC)** formulated as **factor-graph optimization** and solved with [GTSAM](https://gtsam.org/).

A finite-horizon linear-quadratic optimal control problem is mathematically equivalent to maximum-a-posteriori (MAP) inference on a Gaussian factor graph. This package builds that graph and lets GTSAM's sparse linear solver recover the optimal trajectory — eliminating the graph is the same computation as the classical Riccati recursion, but expressed declaratively as factors.

Two solvers are provided:

| Solver           | System                          | Graph            | Constraints |
| ---------------- | ------------------------------- | ---------------- | ----------- |
| `LinearMPC`      | linear (`A`, `B`)               | Gaussian (LQR)   | equality only (dynamics + initial state) |
| `BicycleMPC`     | nonlinear kinematic bicycle     | nonlinear (LM)   | equality + **soft** input/speed limits (barrier factors) |

## The problem

For a discrete-time linear system `x_{k+1} = A x_k + B u_k`, solve

```
minimize    sum_{k=0}^{N-1} ( x_kᵀ Q x_k + u_kᵀ R u_k ) + x_Nᵀ Qf x_N
subject to  x_0 = x_init
            x_{k+1} = A x_k + B u_k
```

## How it maps to a factor graph

| MPC ingredient                       | Factor graph element                                  |
| ------------------------------------ | ----------------------------------------------------- |
| Initial state `x_0 = x_init`         | Hard equality constraint (`noiseModel.Constrained`)   |
| Dynamics `x_{k+1} = A x_k + B u_k`   | Hard equality constraint linking `x_k`, `u_k`, `x_{k+1}` |
| State cost `x_kᵀ Q x_k`              | Quadratic factor with information matrix `Q`          |
| Control cost `u_kᵀ R u_k`           | Quadratic factor with information matrix `R`           |
| Terminal cost `x_Nᵀ Qf x_N`         | Quadratic factor with information matrix `Qf`          |

The MAP solution of the resulting Gaussian factor graph is exactly the optimal control sequence.

## Installation

```bash
pip install -e .          # core (gtsam, numpy)
pip install -e ".[dev]"   # + pytest
pip install -e ".[plot]"  # + matplotlib for the example plot
pip install -e ".[sim]"   # + pygame for the interactive simulation
```

## Usage

```python
import numpy as np
from gtsam_mpc import LinearMPC, double_integrator

system = double_integrator(dt=0.1)          # state = [position, velocity]
mpc = LinearMPC(
    system,
    Q=np.diag([1.0, 1.0]),                  # running state cost
    R=np.array([[0.1]]),                    # running control cost
    horizon=25,
    Qf=np.diag([100.0, 100.0]),             # terminal cost
)

x0 = np.array([5.0, 0.0])

# Open-loop optimal plan over the whole horizon:
plan = mpc.solve(x0)
print(plan.controls.shape)   # (25, 1)
print(plan.states.shape)     # (26, 2)

# One receding-horizon step (the control you'd actually apply):
u = mpc.control(x0)

# Full closed-loop simulation (re-optimize each step, apply first control):
traj = mpc.simulate(x0, steps=60)
```

### Tracking a goal

Pass `xref` to regulate toward a target state instead of the origin (exact when
`xref` is an equilibrium, e.g. a zero-velocity position goal):

```python
import numpy as np
from gtsam_mpc import LinearMPC, point_mass_2d

mpc = LinearMPC(
    point_mass_2d(dt=0.1),                       # state = [px, py, vx, vy]
    Q=np.diag([4.0, 4.0, 0.5, 0.5]),
    R=np.diag([0.05, 0.05]),
    horizon=30,
    Qf=np.diag([40.0, 40.0, 4.0, 4.0]),
)
goal = np.array([3.0, -2.0, 0.0, 0.0])           # position goal, zero velocity
u = mpc.control(np.zeros(4), xref=goal)
```

## Example

```bash
python examples/double_integrator.py
```

Drives a double integrator from `[5, 0]` to the origin and (if matplotlib is installed) saves `double_integrator.png`.

### Interactive pygame simulation

```bash
python examples/pygame_sim.py
```

A 2-D point mass chases a goal under receding-horizon MPC, with the predicted
horizon plan drawn ahead of it. **Left click** sets a new goal, **space**
pauses, **r** resets, **esc** quits.

```bash
python examples/bicycle_sim.py
```

A car (nonlinear kinematic bicycle model) drives to a clickable goal under
`BicycleMPC`, with steering and acceleration enforced as soft limits. Same key
bindings.

## Three factor-graph layers: estimation, control, tracking

The same idea — *write the problem as factors and find the MAP estimate* —
spans an entire autonomy stack. `examples/` has one self-contained sample per
layer, echoing three reference projects:

| Sample | Layer | Factor graph | Reference project |
| ------ | ----- | ------------ | ----------------- |
| `examples/localization.py` | 位置推定 (estimation) | Pose2 odometry (`BetweenFactor`) + GPS (`PriorFactor`) fusion | JPCM / IPN_MPC (positioning half) |
| `examples/control.py`      | 制御 (control)        | receding-horizon MPC (Gaussian / LQR) | this package, JPCM (control half) |
| `examples/tracking.py`     | トラッキング (perception) | constant-velocity motion prior + position measurements | FGO-MOT |

```bash
python examples/localization.py   # odometry drifts; GPS fusion tracks truth
python examples/control.py        # point-mass MPC drives to a goal
python examples/tracking.py       # smooths noisy multi-object detections
```

Each prints a metric showing the factor-graph estimate beats the naive
baseline (fused vs dead-reckoning, smoothed vs raw detections) and, with
matplotlib, saves a plot.

### Integrated demo: track obstacles, then avoid them with MPC

```bash
python examples/integrated_sim.py
```

Wires all three layers into one pipeline — **perception → prediction → control**:

1. Moving obstacles emit noisy **detections** each frame.
2. A constant-velocity **factor-graph tracker** smooths each obstacle and
   **predicts** its position over the MPC horizon.
3. `BicycleMPC` drives the ego car to a (clickable) goal while **avoiding** each
   predicted obstacle trajectory via soft keep-out (control-barrier-style)
   factors within `safety_radius`.

This mirrors the tightly-coupled estimation+control idea behind JPCM/IPN_MPC,
with FGO-MOT-style tracking feeding the obstacle predictions.

`BicycleMPC.solve` / `control` / `simulate` accept an `obstacles` argument — a
list of static centres `(2,)` or per-step predictions `(horizon+1, 2)`:

```python
preds = [predicted_obstacle_xy]            # shape (horizon+1, 2) from a tracker
u = mpc.control(ego_state, goal, obstacles=preds)
```

> The avoidance is **soft** (a penalty/barrier factor), so it strongly avoids
> but does not *guarantee* a hard minimum distance — see the note on hard
> constraints below.

### Recording an animation (GIF)

No display needed — render any of the pygame demos to an animated GIF:

```bash
python examples/make_gif.py integrated.gif --frames 90 --scale 0.6 --fps 20
```

This drives the integrated track-and-avoid demo headless and writes a GIF
(handy for docs/CI where an interactive window isn't available).

## Nonlinear bicycle MPC

For the kinematic bicycle model

```
ẋ = v·cos θ,   ẏ = v·sin θ,   θ̇ = (v/L)·tan δ,   v̇ = a
state = [x, y, θ, v],   control = [a, δ]
```

`BicycleMPC` builds a **nonlinear** factor graph: the discretized dynamics and
costs are `gtsam.CustomFactor` factors with analytic Jacobians, optimized with
Levenberg-Marquardt (warm-started between calls). Input limits `|a| ≤ a_max`,
`|δ| ≤ δ_max` and optional speed limits are added as **soft barrier factors** —
a one-sided quadratic penalty, zero inside the feasible box and growing with the
violation, whose stiffness is set by `barrier_weight`.

```python
import numpy as np
from gtsam_mpc import BicycleMPC, BicycleModel

mpc = BicycleMPC(
    BicycleModel(wheelbase=2.5, dt=0.1),
    Q=np.diag([3.0, 3.0, 0.8, 0.4]),     # x, y, heading, speed
    R=np.diag([0.1, 0.1]),               # accel, steering
    horizon=25,
    Qf=np.diag([30.0, 30.0, 3.0, 3.0]),
    a_bounds=(-3.0, 3.0),
    delta_bounds=(-0.6, 0.6),
    v_bounds=(-3.0, 8.0),
    barrier_weight=500.0,
)

x0 = np.array([0.0, 0.0, 0.0, 0.0])      # [x, y, theta, v]
goal = np.array([10.0, 4.0, 0.0, 0.0])
u = mpc.control(x0, goal)                # first [a, delta] to apply
traj = mpc.simulate(x0, goal, steps=120) # closed-loop rollout
```

> The soft barrier means inputs may slightly exceed the limits; raise
> `barrier_weight` to tighten. The dynamics are nonlinear, so the predicted
> trajectory satisfies them to a small numerical residual (the *applied*
> closed-loop control uses the true model).

## Tests

```bash
pytest
```

The suite validates the factor-graph solution against an independent finite-horizon LQR (Riccati) reference, checks dynamics/initial-condition consistency, and verifies closed-loop regulation to the origin.

## API

- **`LinearSystem(A, B)`** — discrete-time LTI system; `double_integrator(dt)` and `point_mass_2d(dt)` are ready-made examples.
- **`LinearMPC(system, Q, R, horizon, Qf=None)`** — the solver.
  - `.solve(x0, xref=None) -> MPCResult` — open-loop optimal `states` / `controls`.
  - `.control(x0, xref=None) -> np.ndarray` — first optimal control (one receding-horizon step).
  - `.simulate(x0, steps, xref=None) -> MPCResult` — closed-loop receding-horizon rollout.
- **`MPCResult`** — `.states`, `.controls`, `.u0`.
- **`BicycleModel(wheelbase, dt)`** — nonlinear kinematic bicycle; `.step`, `.jacobians`.
- **`BicycleMPC(model, Q, R, horizon, Qf=None, a_bounds, delta_bounds, v_bounds, barrier_weight, safety_radius, obstacle_weight, max_iterations)`**
  - `.solve(x0, xref, obstacles=None, warm_start=True) -> MPCResult`
  - `.control(x0, xref, obstacles=None) -> np.ndarray`
  - `.simulate(x0, xref, steps, obstacles=None) -> MPCResult`
  - `.reset()` — clear the cached warm-start solution.

## References

- F. Dellaert and M. Kaess, *Factor Graphs for Robot Perception*, Foundations and Trends in Robotics, 2017.
- GTSAM: https://gtsam.org/

## License

MIT — see [LICENSE](LICENSE).
