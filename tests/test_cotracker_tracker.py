"""
Unit tests for CoTracker3 dense point tracker and retargeter.
"""

import pytest
import numpy as np
import torch
from src.video_processing.cotracker_tracker import CoTrackerTrajectoryExtractor, GLASS_GRID_SHAPE, HAND_GRID_SHAPE
from src.retargeting.cotracker_to_panda import CoTrackerToPandaRetargeter


def test_cotracker_model_initialization():
    """Verify CoTracker extractor initializes with valid target size and device."""
    extractor = CoTrackerTrajectoryExtractor()
    assert extractor.target_size == (384, 384)
    assert extractor.device in ["mps", "cpu"]


def test_cotracker_query_generation():
    """Verify query tensor has correct (1, N_total, 3) shape."""
    extractor = CoTrackerTrajectoryExtractor()
    glass_roi = (100, 200, 180, 280)
    hand_pts_init = np.zeros((21, 2), dtype=np.float32)
    queries = extractor._build_queries(t_hand=15, glass_roi=glass_roi, hand_pts_init=hand_pts_init)
    expected_pts = (GLASS_GRID_SHAPE[0] * GLASS_GRID_SHAPE[1]) + 21
    assert queries.shape == (1, expected_pts, 3)
    # Check start times
    assert queries[0, 0, 0].item() == 0.0
    assert queries[0, -1, 0].item() == 15.0


def test_cotracker_retargeter_action_bounds(tmp_path):
    """Verify retargeter produces valid 7D actions bounded within limits."""
    # Create synthetic trajectory npz
    T = 20
    synth_deltas = np.random.uniform(-0.05, 0.05, (T, 3)).astype(np.float32)
    synth_grippers = np.full(T, -1.0, dtype=np.float32)
    synth_grippers[10:] = 1.0  # closed from step 10
    synth_traj_file = tmp_path / "test_ct_trajectory.npz"
    np.savez(
        synth_traj_file,
        wrist_deltas=synth_deltas,
        gripper_states=synth_grippers,
        timestamps=np.linspace(0, 1, T, dtype=np.float32),
    )

    retargeter = CoTrackerToPandaRetargeter(max_step_displacement=0.8)
    out_file = tmp_path / "test_ct_actions.npz"
    res = retargeter.retarget_trajectory(str(synth_traj_file), str(out_file))

    actions = res["actions"]
    assert actions.shape == (T, 7)
    # Positions clamped to [-0.8, 0.8]
    assert np.all(actions[:, :3] >= -0.8)
    assert np.all(actions[:, :3] <= 0.8)
    # Rotations zeroed
    assert np.all(actions[:, 3:6] == 0.0)
    # Gripper correctly mapped
    assert actions[0, 6] == -1.0
    assert actions[15, 6] == 1.0
