"""Tests for the factor-graph estimators."""

import numpy as np

from gtsam_mpc import BicycleModel, ConstantVelocityTracker, MovingHorizonEstimator


def test_cv_tracker_smooths_better_than_raw():
    rng = np.random.default_rng(0)
    dt, T, sigma = 0.1, 40, 0.7
    tracker = ConstantVelocityTracker(dt=dt, meas_sigma=sigma)
    F, H = tracker.F, tracker.H

    x = np.array([0.0, 0.0, 2.0, 1.0])
    truth = [x]
    for _ in range(T):
        truth.append(F @ truth[-1])
    truth_pos = np.array([s[:2] for s in truth])
    det = np.array([H @ s + rng.normal(0, sigma, 2) for s in truth])

    est = tracker.smooth(det)
    raw_rmse = np.sqrt(np.mean(np.sum((det - truth_pos) ** 2, axis=1)))
    smooth_rmse = np.sqrt(np.mean(np.sum((est[:, :2] - truth_pos) ** 2, axis=1)))
    assert smooth_rmse < 0.6 * raw_rmse


def test_cv_tracker_predict_shape_and_motion():
    tracker = ConstantVelocityTracker(dt=0.1)
    pred = tracker.predict(np.array([1.0, 2.0, 3.0, 0.0]), horizon=5)
    assert pred.shape == (6, 2)
    # Constant velocity: x advances by v*dt each step, y stays.
    np.testing.assert_allclose(pred[1], [1.0 + 3.0 * 0.1, 2.0], atol=1e-9)


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
