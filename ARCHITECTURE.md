# Architecture

gtsam-mpc is organized around a single idea: **estimation, control and tracking
are all maximum-a-posteriori (MAP) inference on factor graphs built from one
shared vocabulary of factors.** The package layers that vocabulary into
reusable solvers; the `examples/` are thin demos and visualizations on top.

## Layers

```
gtsam_mpc/
  models.py        BicycleModel  (RK4 step + exact analytic Jacobians)

  factors.py       The factor vocabulary: GTSAM CustomFactor builders, each
                   with an analytic Jacobian + noise-model shorthands
                     prior, dynamics (ternary), motion (binary, fixed u),
                     linear_motion (CV prior), keepout, state_cost
                     (heading-wrapped), zero_cost, position/scalar_measurement

  constraints.py   Inequality g(x) <= 0 handling as pluggable strategies
                     Inequality                      (key + g, dg/dx evaluator)
                     BarrierStrategy                 (soft one-sided penalty)
                     AugmentedLagrangianStrategy     (outer multiplier loop)
                     SlackStrategy                   (g + s^2 = 0, single solve)
                     barrier_factor, make_strategy

  control.py       BicycleMPC  (nonlinear graph from factors + a constraint
                   strategy), MPCResult, require_pd

  estimation.py    MovingHorizonEstimator   (sliding-window bicycle MHE)

  joint.py         JointEstimatorMPC      -- estimation window + control horizon
                   JointLocTrackControl   -- + obstacle tracking & uncertainty-
                                             aware avoidance, all in one graph

  paths.py         Reference paths + path-tracking geometry (pure NumPy)
                     circle/figure_eight/sine_wave/racetrack/rounded_square,
                     make_path, path_curvature, nearest_index,
                     cross_track_error, reference_trajectory, reset_on_path

examples/          Demos + visualization only (no reusable algorithms)
  _viz.py            shared pygame camera (View), palette, draw_car, draw_grid
  _racecar_app.py    shared interactive harness (event loop + render); RacecarApp
  *_sim.py           interactive pygame demos
  localization.py    minimal odometry + GPS fusion sample
  make_gif.py        headless GIF recorder
```

## Dependency direction

`models` <- `factors` <- {`constraints`, `control`, `estimation`} <- `joint`.
`paths` is standalone (pure NumPy). Examples depend on the package and on the
two example-only helpers (`_viz`, `_racecar_app`); nothing in the package
depends on `examples/`.

## Why this shape

* **One vocabulary, many graphs.** `BicycleMPC`, `MovingHorizonEstimator` and
  `JointEstimatorMPC` all assemble their graphs from `factors.py` instead of
  re-deriving the same dynamics/measurement error functions. This mirrors the
  reference projects (JPCM/IPN_MPC, FGO-MOT) where perception and control share
  a factor-graph backbone. (Obstacle tracking in `JointLocTrackControl` reuses
  the same constant-velocity motion + position-measurement factors directly in
  the joint graph.)
* **Constraints are a strategy, not a branch.** Inequalities are described once
  as `Inequality` records; `barrier` / `al` / `slack` are interchangeable
  `ConstraintStrategy` objects. `BicycleMPC` accepts either a strategy instance
  (`constraints=`) or a convenience mode name (`constraint_mode=`).
* **Library vs. demo.** Anything reusable (estimators, trackers, path geometry)
  lives in the package and is unit-tested directly; `examples/` only wires
  things together and draws them.

## Tests

`tests/` mirrors the package: `test_bicycle.py` (BicycleMPC + constraint modes),
`test_constraints.py` (strategy API), `test_estimation.py` (MHE), `test_paths.py`,
and `test_examples.py` (the demo `run()` entry points, asserting each
factor-graph result beats its baseline).
