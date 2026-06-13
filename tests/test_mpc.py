"""Tests for the factor-graph linear MPC solver.

The key correctness check compares the factor-graph solution against an
independent finite-horizon LQR computed with the Riccati backward recursion;
the two must agree to numerical precision.
"""

import numpy as np
import pytest

from gtsam_mpc import LinearMPC, LinearSystem, double_integrator, point_mass_2d


def finite_horizon_lqr(system, Q, R, Qf, x0, N):
    """Reference solution: discrete finite-horizon LQR via Riccati recursion."""
    A, B = system.A, system.B
    P = np.asarray(Qf, dtype=float).copy()
    gains = []
    for _ in range(N):
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        gains.append(K)
        P = Q + A.T @ P @ A - A.T @ P @ B @ K
    gains.reverse()

    x = np.asarray(x0, dtype=float).copy()
    controls = []
    for k in range(N):
        u = -gains[k] @ x
        controls.append(u)
        x = system.step(x, u)
    return np.array(controls)


def test_matches_riccati_lqr():
    system = double_integrator(dt=0.1)
    Q = np.diag([1.0, 1.0])
    R = np.array([[0.1]])
    Qf = np.diag([10.0, 10.0])
    N = 20
    x0 = np.array([5.0, 0.0])

    mpc = LinearMPC(system, Q=Q, R=R, horizon=N, Qf=Qf)
    result = mpc.solve(x0)
    reference = finite_horizon_lqr(system, Q, R, Qf, x0, N)

    np.testing.assert_allclose(result.controls, reference, atol=1e-9)


def test_dynamics_and_initial_condition_consistent():
    system = double_integrator(dt=0.05)
    mpc = LinearMPC(system, Q=np.eye(2), R=np.array([[1.0]]), horizon=15)
    x0 = np.array([-2.0, 1.0])
    result = mpc.solve(x0)

    # Initial condition is enforced exactly.
    np.testing.assert_allclose(result.states[0], x0, atol=1e-9)
    # Every predicted state obeys the dynamics.
    for k in range(mpc.horizon):
        expected = system.step(result.states[k], result.controls[k])
        np.testing.assert_allclose(result.states[k + 1], expected, atol=1e-9)


def test_result_shapes_and_u0():
    system = double_integrator()
    N = 12
    mpc = LinearMPC(system, Q=np.eye(2), R=np.array([[0.5]]), horizon=N)
    result = mpc.solve([1.0, 0.0])

    assert result.states.shape == (N + 1, 2)
    assert result.controls.shape == (N, 1)
    np.testing.assert_array_equal(result.u0, result.controls[0])


def test_closed_loop_converges_to_origin():
    system = double_integrator(dt=0.1)
    mpc = LinearMPC(
        system, Q=np.diag([1.0, 1.0]), R=np.array([[0.05]]),
        horizon=25, Qf=np.diag([100.0, 100.0]),
    )
    result = mpc.simulate(np.array([5.0, 0.0]), steps=80)

    assert result.states.shape == (81, 2)
    assert result.controls.shape == (80, 1)
    # The receding-horizon controller should regulate the state to the origin.
    assert np.linalg.norm(result.states[-1]) < 0.05


def test_tracks_nonzero_goal():
    system = point_mass_2d(dt=0.1)
    mpc = LinearMPC(
        system,
        Q=np.diag([4.0, 4.0, 0.5, 0.5]),
        R=np.diag([0.05, 0.05]),
        horizon=30,
        Qf=np.diag([40.0, 40.0, 4.0, 4.0]),
    )
    goal = np.array([3.0, -2.0, 0.0, 0.0])
    result = mpc.simulate(np.zeros(4), steps=120, xref=goal)
    # Closed loop should converge to the (equilibrium) goal.
    np.testing.assert_allclose(result.states[-1], goal, atol=0.05)


def test_reference_at_goal_needs_no_control():
    system = point_mass_2d(dt=0.1)
    mpc = LinearMPC(system, Q=np.eye(4), R=np.eye(2), horizon=10)
    goal = np.array([2.0, 5.0, 0.0, 0.0])
    # Already at the goal (an equilibrium): the plan should command ~zero input.
    result = mpc.solve(goal, xref=goal)
    np.testing.assert_allclose(result.controls, 0.0, atol=1e-9)


def test_zero_initial_state_gives_zero_control():
    system = double_integrator()
    mpc = LinearMPC(system, Q=np.eye(2), R=np.array([[1.0]]), horizon=10)
    result = mpc.solve(np.zeros(2))
    np.testing.assert_allclose(result.controls, 0.0, atol=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"Q": np.zeros((2, 2)), "R": np.array([[1.0]]), "horizon": 5},  # Q not PD
        {"Q": np.eye(2), "R": np.array([[-1.0]]), "horizon": 5},        # R not PD
        {"Q": np.eye(2), "R": np.array([[1.0]]), "horizon": 0},         # bad horizon
    ],
)
def test_invalid_arguments_raise(kwargs):
    system = double_integrator()
    with pytest.raises(ValueError):
        LinearMPC(system, **kwargs)


def test_linear_system_shape_validation():
    with pytest.raises(ValueError):
        LinearSystem(A=np.ones((2, 3)), B=np.ones((2, 1)))  # A not square
    with pytest.raises(ValueError):
        LinearSystem(A=np.eye(2), B=np.ones((3, 1)))  # B rows mismatch
