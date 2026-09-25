# loc-track-control

**Localization + multi-object tracking + control on a single factor graph**, with [GTSAM](https://gtsam.org/).

`JointLocTrackControl` fuses an entire autonomy stack into **one `NonlinearFactorGraph`, solved once per step**: the ego car localizes from noisy GPS, every moving obstacle is tracked and predicted, and a path-following MPC overtakes/avoids them — all jointly optimized.

![full stack demo](media/loc_track_control.gif)

Full-resolution video: [media/loc_track_control.mp4](media/loc_track_control.mp4)

*Amber = raw GPS · green = true path · blue = estimate (+2σ ellipse) · red = tracked obstacles · green line = MPC plan. The red safety ring shrinks as an obstacle is observed more strongly up close, and grows with track uncertainty.*

```python
import numpy as np
from gtsam_mpc import JointLocTrackControl

jtc = JointLocTrackControl(n_obstacles=3, safety_radius=3.0, n_sigma=1.5)
jtc.reset(x0, [obs0, obs1, obs2])          # each obstacle [px,py,vx,vy] or (px,py)

# one optimize() does estimation window + obstacle tracks + control horizon:
est, plan_states, plan_u, obs_est, obs_pred = jtc.step(u, gps, v_meas, dets, reference)
```

Two couplings make the single graph pay off:

- **uncertainty-aware avoidance** — each obstacle's keep-out radius is inflated by its track covariance (read from `gtsam.Marginals`), so the car gives uncertain tracks a wider berth;
- **range-aware sensing** — a closer obstacle is observed more strongly, tightening its track so the car can pass confidently.

Run it: `python examples/loc_track_control_sim.py` (space/r/esc). See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the design.

## Why a factor graph

Estimation, control and tracking are all MAP inference on factor graphs built from one shared vocabulary of factors — so `JointLocTrackControl` is assembled from reusable pieces, each usable on its own:

| Component | Problem |
| --------- | ------- |
| `BicycleMPC` | nonlinear bicycle MPC; barrier / aug-Lagrangian / slack / **SQP (hard)** constraints |
| `MovingHorizonEstimator` | sliding-window localization |
| `JointEstimatorMPC` | localization + control, one graph |

```python
from gtsam_mpc import BicycleMPC, BicycleModel
mpc = BicycleMPC(BicycleModel(2.5, 0.1), Q=..., R=..., horizon=25, constraint_mode="al")
u = mpc.control(x0, goal)            # or mpc.simulate / a (horizon+1,4) reference + obstacles=
```

`constraint_mode="sqp"` uses GTSAM 4.3's constrained QP solver (`gtsam.QpProblem` + `gtsam.LinearConstraint`): each SQP step turns the hard dynamics into linear equalities and input / speed / keep-out bounds into linear inequalities, so the bounds hold exactly instead of up to a penalty slack. Degenerate or infeasible QPs fall back to `gtsam.AugmentedLagrangianOptimizer`. It is slower than the penalty modes (~0.2 s per step at horizon 20).

## Install

```bash
pip install -e ".[sim]"   # core (GTSAM >= 4.3) + pygame; [dev] for pytest, [plot] for matplotlib, [video] for MP4
```

## Demos

Interactive (space pause, r reset, esc quit); all run headless with `SDL_VIDEODRIVER=dummy GTSAM_MPC_MAX_FRAMES=<n>`.

| Script | Shows |
| ------ | ----- |
| `loc_track_control_sim.py` | **full stack** (the headline above) |
| `loc_control_sim.py` / `joint_loc_control_sim.py` | localization + control (pipeline / single graph) |
| `path_following_sim.py` | bicycle MPC tracking selectable paths (1–5) |
| `bicycle_sim.py` | bicycle MPC to a clickable goal |
| `localization.py` | minimal odometry + GPS fusion sample |

Record the demo to a GIF or MP4 (by extension): `python examples/make_gif.py out.gif --frames 300` / `python examples/make_gif.py out.mp4 --frames 300` (MP4 needs `pip install -e ".[video]"`).

## Test

```bash
pytest   # 30 tests
```

## License

MIT — see [LICENSE](LICENSE).
