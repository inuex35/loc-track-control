"""Smoke + sanity tests for the three factor-graph sample layers.

Each example exposes a ``run()`` returning metrics, so we can assert that the
factor-graph estimate/control actually beats the naive baseline.
"""

import importlib
import sys
from pathlib import Path

# Make the ``examples`` package importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_localization_fusion_beats_dead_reckoning():
    mod = importlib.import_module("examples.localization")
    out = mod.run(seed=0)
    assert out["rmse_fused"] < out["rmse_dead_reckoning"]
    assert out["rmse_fused"] < 1.0


def test_control_reaches_goal():
    mod = importlib.import_module("examples.control")
    out = mod.run()
    assert out["final_error"] < 0.1


def test_tracking_smoothing_beats_raw():
    mod = importlib.import_module("examples.tracking")
    out = mod.run(seed=0)
    assert out["rmse_smoothed"] < out["rmse_raw"]
    assert out["rmse_smoothed"] < 0.5


def test_integrated_tracks_and_avoids():
    mod = importlib.import_module("examples.integrated_sim")
    out = mod.run(frames=70, seed=0)
    # Stayed clear of the moving obstacles (soft keep-out, small slack allowed)...
    assert out["min_clearance"] > mod.SAFETY_RADIUS - 0.5
    # ...and still reached the goal.
    assert out["goal_dist"] < 2.0
