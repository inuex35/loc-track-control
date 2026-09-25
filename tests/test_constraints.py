"""Tests for the constraint-strategy API (objects + the convenience factory)."""

import numpy as np
import pytest

from gtsam_mpc import (
    AugmentedLagrangianStrategy,
    BarrierStrategy,
    BicycleMPC,
    BicycleModel,
    SlackStrategy,
    SQPStrategy,
    make_strategy,
)


def _mpc(strategy):
    return BicycleMPC(
        BicycleModel(wheelbase=2.5, dt=0.1),
        Q=np.diag([2.0, 2.0, 0.5, 0.5]), R=np.diag([0.1, 0.1]), horizon=15,
        Qf=np.diag([20.0, 20.0, 2.0, 2.0]),
        a_bounds=(-2.0, 2.0), delta_bounds=(-0.4, 0.4),
        constraints=strategy,
    )


def test_make_strategy_types():
    assert isinstance(make_strategy("barrier"), BarrierStrategy)
    assert isinstance(make_strategy("al"), AugmentedLagrangianStrategy)
    assert isinstance(make_strategy("slack"), SlackStrategy)
    assert isinstance(make_strategy("sqp"), SQPStrategy)
    with pytest.raises(ValueError):
        make_strategy("penalty")


def test_strategy_object_overrides_mode():
    # Passing a strategy instance is the clean alternative to the mode string.
    mpc = _mpc(BarrierStrategy(weight=400.0))
    assert isinstance(mpc.strategy, BarrierStrategy)
    out = mpc.solve(np.zeros(4), np.array([6.0, 2.0, 0.0, 0.0]), warm_start=False)
    assert out.controls.shape == (15, 2)
    assert np.all(np.isfinite(out.u0))


def test_al_strategy_object_respects_bounds():
    mpc = _mpc(AugmentedLagrangianStrategy())
    res = mpc.simulate(np.zeros(4), np.array([15.0, 8.0, 0.0, 0.0]), steps=15)
    a, delta = res.controls[:, 0], res.controls[:, 1]
    assert a.max() <= 2.0 + 1e-2 and a.min() >= -2.0 - 1e-2
    assert delta.max() <= 0.4 + 1e-2 and delta.min() >= -0.4 - 1e-2


def test_sqp_enforces_bounds_as_hard_constraints():
    # GTSAM's constrained QP solver: input and speed bounds hold exactly, not
    # just up to a penalty-dependent slack.
    mpc = _mpc(SQPStrategy())
    mpc.v_bounds = (-2.0, 3.0)
    res = mpc.simulate(np.zeros(4), np.array([15.0, 8.0, 0.0, 0.0]), steps=15)
    a, delta = res.controls[:, 0], res.controls[:, 1]
    assert a.max() <= 2.0 + 1e-6 and a.min() >= -2.0 - 1e-6
    assert delta.max() <= 0.4 + 1e-6 and delta.min() >= -0.4 - 1e-6
    assert res.states[:, 3].max() <= 3.0 + 1e-6
    assert a.max() > 2.0 - 1e-6  # the bound is actually active


def test_sqp_keeps_out_of_obstacle():
    mpc = _mpc(SQPStrategy())
    mpc.safety_radius = 2.0
    obstacle = np.array([6.0, 0.8])
    res = mpc.simulate(np.zeros(4), np.array([12.0, 0.0, 0.0, 0.0]), steps=60,
                       obstacles=[obstacle])
    min_dist = np.min(np.linalg.norm(res.states[:, :2] - obstacle, axis=1))
    assert min_dist > 2.0 - 1e-2
    assert res.states[-1, 0] > 9.0
