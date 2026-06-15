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


def test_localization_control_coupling_mhe_beats_raw():
    mod = importlib.import_module("examples.loc_control_sim")
    mhe = mod.run(seed=0, steps=250, gps_sigma=1.0, use_estimate=True)
    raw = mod.run(seed=0, steps=250, gps_sigma=1.0, use_estimate=False)
    # Fusing the motion model with a window of GPS fixes cuts localization error...
    assert mhe["loc_rmse"] < 0.6 * raw["loc_rmse"]
    # ...and keeps the car on the path, where the raw-GPS controller diverges
    # (no heading sensor -> heading from noisy GPS deltas is unusable).
    assert mhe["cte_mean"] < 1.0
    assert mhe["cte_mean"] < raw["cte_mean"]


def test_joint_single_graph_matches_pipeline():
    jt = importlib.import_module("examples.joint_loc_control_sim")
    lc = importlib.import_module("examples.loc_control_sim")
    joint = jt.run(seed=0, steps=200, gps_sigma=1.0)
    pipe = lc.run(seed=0, steps=200, gps_sigma=1.0, use_estimate=True)
    # The unified single-optimize() graph stays on the path...
    assert joint["cte_mean"] < 1.0
    # ...and localizes about as well as the two-graph pipeline (the past/future
    # coupling is weak for a deterministic model, so they should be close).
    assert joint["loc_rmse"] < 1.5 * pipe["loc_rmse"]


def test_loc_track_control_single_graph():
    mod = importlib.import_module("examples.loc_track_control_sim")
    out = mod.run(seed=0, steps=240)
    # Localizes from noisy GPS (looser than the static demos: aggressive
    # overtaking of moving traffic exercises the GPS-window estimate harder)...
    assert out["loc_rmse"] < 1.5
    # ...overtakes the moving traffic while keeping clear. The keep-out is soft
    # and uncertainty-aware: a close obstacle is observed strongly, so its track
    # is confident and the margin shrinks -- the car may pass fairly tight.
    assert out["min_clearance"] > mod.SAFETY_RADIUS - 1.5
    # ...and makes real progress around the track (~ at least most of a lap).
    assert out["progress"] > out["path_len"] // 2
