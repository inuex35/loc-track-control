"""Tests for the constraint-strategy API (objects + the convenience factory)."""

import numpy as np
import pytest

from gtsam_mpc import (
    AugmentedLagrangianStrategy,
    BarrierStrategy,
    BicycleMPC,
    BicycleModel,
    SlackStrategy,
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
