"""Tests for the factor-graph estimators."""

import numpy as np

from gtsam_mpc import BicycleModel, MovingHorizonEstimator


def test_mhe_beats_raw_gps():
    rng = np.random.default_rng(0)
    model = BicycleModel(wheelbase=2.5, dt=0.1)
    mhe = MovingHorizonEstimator(model, window=8, gps_sigma=0.6)
    true = np.array([0.0, 0.0, 0.0, 5.0])
    mhe.reset(true)
    u = np.array([0.3, 0.1])

    fused_err, raw_err = [], []
    for _ in range(40):
        true = model.step(true, u)
        gps = true[:2] + rng.normal(0, 0.6, 2)
        v_meas = true[3] + rng.normal(0, 0.3)
        est = mhe.update(u, gps, v_meas)
        fused_err.append(np.linalg.norm(true[:2] - est[:2]))
        raw_err.append(np.linalg.norm(true[:2] - gps))

    # Skip the warm-up; fusion should beat raw GPS position.
    assert np.mean(fused_err[10:]) < np.mean(raw_err[10:])
