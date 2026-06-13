# gtsam-mpc

Linear **Model Predictive Control (MPC)** formulated as **factor-graph optimization** and solved with [GTSAM](https://gtsam.org/).

A finite-horizon linear-quadratic optimal control problem is mathematically equivalent to maximum-a-posteriori (MAP) inference on a Gaussian factor graph. This package builds that graph and lets GTSAM's sparse linear solver recover the optimal trajectory — eliminating the graph is the same computation as the classical Riccati recursion, but expressed declaratively as factors.

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

## Example

```bash
python examples/double_integrator.py
```

Drives a double integrator from `[5, 0]` to the origin and (if matplotlib is installed) saves `double_integrator.png`.

## Tests

```bash
pytest
```

The suite validates the factor-graph solution against an independent finite-horizon LQR (Riccati) reference, checks dynamics/initial-condition consistency, and verifies closed-loop regulation to the origin.

## API

- **`LinearSystem(A, B)`** — discrete-time LTI system; `double_integrator(dt)` is a ready-made example.
- **`LinearMPC(system, Q, R, horizon, Qf=None)`** — the solver.
  - `.solve(x0) -> MPCResult` — open-loop optimal `states` / `controls`.
  - `.control(x0) -> np.ndarray` — first optimal control (one receding-horizon step).
  - `.simulate(x0, steps) -> MPCResult` — closed-loop receding-horizon rollout.
- **`MPCResult`** — `.states`, `.controls`, `.u0`.

## References

- F. Dellaert and M. Kaess, *Factor Graphs for Robot Perception*, Foundations and Trends in Robotics, 2017.
- GTSAM: https://gtsam.org/

## License

MIT — see [LICENSE](LICENSE).
