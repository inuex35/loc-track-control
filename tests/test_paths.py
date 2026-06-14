"""Tests for the path geometry / reference-trajectory helpers."""

import numpy as np

from gtsam_mpc import paths


def test_make_path_uniform_arclength():
    path = paths.make_path(paths.circle, resolution=0.25)
    seg = np.linalg.norm(np.diff(np.vstack([path, path[:1]]), axis=0), axis=1)
    # Uniform arc-length spacing close to the requested resolution.
    assert np.allclose(seg, seg[0], atol=0.05)
    assert abs(seg.mean() - 0.25) < 0.05


def test_curvature_of_circle():
    radius = 22.0
    path = paths.make_path(lambda: paths.circle(radius=radius))
    kappa = paths.path_curvature(path)
    # Curvature of a circle is 1/radius everywhere.
    assert np.allclose(np.median(kappa), 1.0 / radius, rtol=0.1)


def test_cross_track_error_zero_on_path():
    path = paths.make_path(paths.racetrack)
    assert paths.cross_track_error(path, path[10]) < 1e-6


def test_reference_trajectory_shape_and_heading():
    path = paths.make_path(paths.circle)
    curv = paths.path_curvature(path)
    refs = paths.reference_trajectory(path, curv, near_idx=0, speed=8.0,
                                      horizon=6, dt=0.5)
    assert refs.shape == (7, 4)
    # First reference sits on the path with a finite heading and the set speed.
    assert paths.cross_track_error(path, refs[0, :2]) < 0.3
    assert refs[0, 3] <= 8.0 + 1e-9


def test_reset_on_path_heads_along_tangent():
    path = paths.make_path(paths.circle)
    state = paths.reset_on_path(path, speed=5.0)
    np.testing.assert_allclose(state[:2], path[0])
    assert state[3] == 5.0
