"""Tests for the nonlinear kinematic-bicycle MPC."""

import numpy as np
import pytest

from gtsam_mpc import BicycleMPC, BicycleModel


def test_model_jacobians_match_finite_difference():
    model = BicycleModel(wheelbase=2.5, dt=0.1)
    rng = np.random.default_rng(0)
    for _ in range(10):
        x = rng.normal(size=4)
        u = np.array([rng.normal(), rng.uniform(-0.4, 0.4)])  # keep |delta| < pi/2
        dfdx, dfdu = model.jacobians(x, u)

        eps = 1e-6
        num_dfdx = np.zeros((4, 4))
        for i in range(4):
            dx = np.zeros(4); dx[i] = eps
            num_dfdx[:, i] = (model.step(x + dx, u) - model.step(x - dx, u)) / (2 * eps)
        num_dfdu = np.zeros((4, 2))
        for i in range(2):
            du = np.zeros(2); du[i] = eps
            num_dfdu[:, i] = (model.step(x, u + du) - model.step(x, u - du)) / (2 * eps)

        np.testing.assert_allclose(dfdx, num_dfdx, atol=1e-5)
        np.testing.assert_allclose(dfdu, num_dfdu, atol=1e-5)


def _default_mpc(**kwargs):
    model = BicycleModel(wheelbase=2.5, dt=0.1)
    return BicycleMPC(
        model,
        Q=np.diag([2.0, 2.0, 0.5, 0.5]),
        R=np.diag([0.1, 0.1]),
        horizon=20,
        Qf=np.diag([20.0, 20.0, 2.0, 2.0]),
        **kwargs,
    )


def test_dynamics_residual_small():
    mpc = _default_mpc()
    x0 = np.zeros(4)
    xref = np.array([8.0, 3.0, 0.0, 0.0])
    result = mpc.solve(x0, xref, warm_start=False)
    # Each predicted state should obey the (nonlinear) dynamics closely.
    worst = max(
        np.linalg.norm(result.states[k + 1] - mpc.model.step(result.states[k], result.controls[k]))
        for k in range(mpc.horizon)
    )
    assert worst < 1e-2


def test_initial_condition_enforced():
    mpc = _default_mpc()
    x0 = np.array([1.0, -2.0, 0.3, 1.5])
    result = mpc.solve(x0, np.array([5.0, 0.0, 0.0, 0.0]), warm_start=False)
    np.testing.assert_allclose(result.states[0], x0, atol=1e-6)


def test_input_bounds_respected_softly():
    # Tight barrier should keep inputs close to the box despite an aggressive goal.
    mpc = _default_mpc(a_bounds=(-2.0, 2.0), delta_bounds=(-0.4, 0.4),
                       barrier_weight=2000.0)
    result = mpc.simulate(np.zeros(4), np.array([15.0, 8.0, 0.0, 0.0]), steps=10)
    a, delta = result.controls[:, 0], result.controls[:, 1]
    # Soft constraint: allow a small tolerance over the hard limit.
    assert a.max() <= 2.0 + 0.1 and a.min() >= -2.0 - 0.1
    assert delta.max() <= 0.4 + 0.05 and delta.min() >= -0.4 - 0.05


def test_closed_loop_reaches_goal():
    mpc = _default_mpc(v_bounds=(-2.0, 4.0))
    goal = np.array([10.0, 4.0, 0.0, 0.0])
    result = mpc.simulate(np.zeros(4), goal, steps=120)
    assert result.states.shape == (121, 4)
    assert result.controls.shape == (120, 2)
    # Position should be reached (heading/speed are softer to satisfy).
    assert np.linalg.norm(result.states[-1, :2] - goal[:2]) < 0.5


def test_at_goal_needs_little_control():
    mpc = _default_mpc()
    goal = np.array([3.0, 1.0, 0.0, 0.0])
    result = mpc.solve(goal, goal, warm_start=False)
    assert np.linalg.norm(result.u0) < 1e-2


def test_avoids_static_obstacle():
    mpc = _default_mpc(v_bounds=(-2.0, 4.0), safety_radius=2.0, obstacle_weight=400.0)
    goal = np.array([12.0, 0.0, 0.0, 0.0])
    # Near the straight-line path but slightly off-axis (a symmetric obstacle
    # dead-ahead is a potential-field local minimum; a tracker rarely reports
    # one perfectly centred anyway).
    obstacle = np.array([6.0, 0.8])
    result = mpc.simulate(np.zeros(4), goal, steps=120, obstacles=[obstacle])

    min_dist = np.min(np.linalg.norm(result.states[:, :2] - obstacle, axis=1))
    # Soft keep-out: should stay close to the safety radius (small slack allowed).
    assert min_dist > 2.0 - 0.4
    # Still makes meaningful progress toward the goal past the obstacle.
    assert result.states[-1, 0] > 9.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"Q": np.eye(2), "R": np.eye(2), "horizon": 5},          # Q wrong size
        {"Q": np.eye(4), "R": np.eye(3), "horizon": 5},          # R wrong size
        {"Q": np.eye(4), "R": np.eye(2), "horizon": 0},          # bad horizon
    ],
)
def test_invalid_arguments_raise(kwargs):
    model = BicycleModel()
    with pytest.raises(ValueError):
        BicycleMPC(model, **kwargs)


def test_invalid_constraint_mode_raises():
    with pytest.raises(ValueError):
        _default_mpc(constraint_mode="penalty")


def test_al_satisfies_input_bounds_tightly():
    # Augmented Lagrangian should respect the box much more tightly than the
    # fixed-weight barrier, even when the goal makes the bounds active.
    a_lim, d_lim = 2.0, 0.4
    mpc = _default_mpc(a_bounds=(-a_lim, a_lim), delta_bounds=(-d_lim, d_lim),
                       constraint_mode="al")
    result = mpc.simulate(np.zeros(4), np.array([15.0, 8.0, 0.0, 0.0]), steps=20)
    a, delta = result.controls[:, 0], result.controls[:, 1]
    assert a.max() <= a_lim + 1e-2 and a.min() >= -a_lim - 1e-2
    assert delta.max() <= d_lim + 1e-2 and delta.min() >= -d_lim - 1e-2


def test_al_respects_speed_bounds():
    mpc = _default_mpc(v_bounds=(-1.0, 3.0), constraint_mode="al")
    result = mpc.simulate(np.zeros(4), np.array([20.0, 0.0, 0.0, 0.0]), steps=40)
    v = result.states[:, 3]
    assert v.max() <= 3.0 + 2e-2 and v.min() >= -1.0 - 2e-2


def test_slack_single_solve_respects_bounds():
    # The slack reformulation handles every constraint inside one GTSAM solve
    # (no Python outer loop). It is tight on steering; the s^2 degeneracy at an
    # active bound lets acceleration overshoot a little, hence the looser tol.
    a_lim, d_lim = 2.0, 0.4
    mpc = _default_mpc(a_bounds=(-a_lim, a_lim), delta_bounds=(-d_lim, d_lim),
                       constraint_mode="slack")
    result = mpc.simulate(np.zeros(4), np.array([15.0, 8.0, 0.0, 0.0]), steps=20)
    a, delta = result.controls[:, 0], result.controls[:, 1]
    assert a.max() <= a_lim + 0.15 and a.min() >= -a_lim - 0.15
    assert delta.max() <= d_lim + 1e-2 and delta.min() >= -d_lim - 1e-2


def test_al_avoids_static_obstacle():
    mpc = _default_mpc(v_bounds=(-2.0, 4.0), safety_radius=2.0,
                       constraint_mode="al")
    goal = np.array([12.0, 0.0, 0.0, 0.0])
    obstacle = np.array([6.0, 0.8])
    result = mpc.simulate(np.zeros(4), goal, steps=120, obstacles=[obstacle])
    min_dist = np.min(np.linalg.norm(result.states[:, :2] - obstacle, axis=1))
    # AL drives the keep-out violation toward zero -> tighter than the barrier.
    assert min_dist > 2.0 - 0.1
    assert result.states[-1, 0] > 9.0
