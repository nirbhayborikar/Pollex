"""
export_lerobot_dataset.py
" This is the data-generation/export stage before SmolVLA training."

- It takes action trajectory such as "in iteration 7"/ iteration 19 we generate from cotracker to panda.
- replay the old retargeted actions to create the expert dataset
"""

import os
import json
import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


DEFAULT_TASK_PROMPTS = {
    "detailed": "approach the glass on the table and grasp the cylinder and lift the cylinder and then bring down the cylinder to the table and then withdraw your hands",
    "partial": "approach the glass, grasp it and lift the cylinder",
    "vague": "lift up and down the glass",
}

# Load expert episode:

def load_expert_episode(
    replay_npz_path: str,
    episode_id: str,
    task_description: str = DEFAULT_TASK_PROMPTS["detailed"],
) -> Dict[str, Any]:

    path = Path(replay_npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Replay file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        # SmolVLA visual observation
        images = np.asarray(data["sim_frames"], dtype=np.uint8)

        # Final closed-loop expert action from Iteration 10
        actions = np.asarray(data["executed_actions"], dtype=np.float32)

        # Actual Panda proprioception from RoboSuite / MuJoCo
        eef_pos = np.asarray(data["robot_eef_pos"], dtype=np.float32)
        eef_quat = np.asarray(data["robot_eef_quat"], dtype=np.float32)
        gripper_qpos = np.asarray(data["robot_gripper_qpos"], dtype=np.float32)

        fps = (
            float(np.asarray(data["replay_fps"]).item())
            if "replay_fps" in data
            else 30.0
        )

        lift_success = (
            bool(np.asarray(data["lift_success"]).item())
            if "lift_success" in data
            else False
        )

        max_lift = (
            float(np.asarray(data["max_tumbler_lift"]).item())
            if "max_tumbler_lift" in data
            else np.nan
        )

    # All training signals must describe the same timestep count.
    T = len(actions)

    if len(images) != T:
        raise ValueError(f"Image/action mismatch: {len(images)} != {T}")

    if len(eef_pos) != T:
        raise ValueError(f"EEF/action mismatch: {len(eef_pos)} != {T}")

    if len(eef_quat) != T:
        raise ValueError(f"Quaternion/action mismatch: {len(eef_quat)} != {T}")

    if len(gripper_qpos) != T:
        raise ValueError(f"Gripper/action mismatch: {len(gripper_qpos)} != {T}")

    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"Expected actions (T, 7), got {actions.shape}")

    if eef_pos.shape != (T, 3):
        raise ValueError(f"Expected EEF position (T, 3), got {eef_pos.shape}")

    if eef_quat.shape != (T, 4):
        raise ValueError(f"Expected EEF quaternion (T, 4), got {eef_quat.shape}")

    if gripper_qpos.ndim != 2 or gripper_qpos.shape[1] < 1:
        raise ValueError(f"Expected gripper state (T, >=1), got {gripper_qpos.shape}")

    if not np.all(np.isfinite(actions)):
        raise ValueError("Expert actions contain NaN/Inf.")

    # Use one physical gripper position value.
    grip = gripper_qpos[:, :1]

    # Real 8D Panda proprioception:
    # [x, y, z, qx, qy, qz, qw, gripper]
    state_8d = np.hstack([
        eef_pos,
        eef_quat,
        grip,
    ]).astype(np.float32)

    if not np.all(np.isfinite(state_8d)):
        raise ValueError("Robot state contains NaN/Inf.")

    # Synchronised timestamp for every observation/action pair.
    timestamps = np.arange(T, dtype=np.float32) / fps

    print(
        f"[+] {episode_id}: frames={T}, "
        f"state={state_8d.shape}, actions={actions.shape}, "
        f"lift={max_lift * 100:.2f} cm, success={lift_success}"
    )

    return {
        "episode_id": episode_id,
        "num_frames": T,

        # SmolVLA inputs
        "images": images,
        "state_8d": state_8d,

        # SmolVLA target
        "actions": actions,

        # Raw proprioception for analysis/debugging
        "eef_pos": eef_pos,
        "eef_quat": eef_quat,
        "gripper_qpos": gripper_qpos,

        "timestamps": timestamps,
        "task": task_description,

        "metrics": {
            "max_lift_cm": float(max_lift * 100.0),
            "success": lift_success,
        },
    }

# storing:
def save_hdf5_dataset(episodes: List[Dict[str, Any]], output_path: str):
    """Save synchronized expert episodes as HDF5: observation_t -> action_t."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as root:
        data_group = root.create_group("data")
        total_frames = 0

        for i, ep in enumerate(episodes):
            group = data_group.create_group(f"demo_{i}")

            # Episode metadata
            group.attrs["episode_id"] = ep["episode_id"]
            group.attrs["num_samples"] = ep["num_frames"]
            group.attrs["task"] = ep["task"]
            group.attrs["success"] = ep["metrics"]["success"]
            group.attrs["max_lift_cm"] = ep["metrics"]["max_lift_cm"]

            # Expert 7D action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
            group.create_dataset("action", data=ep["actions"], dtype="float32")

            # Panda 7D state: [eef_x, eef_y, eef_z, 0, 0, 0, gripper]
            group.create_dataset("observation.state", data=ep["state_8d"], dtype="float32")

            # Actual Panda end-effector XYZ, useful for analysis/debugging
            group.create_dataset("observation.eef_pos", data=ep["eef_pos"], dtype="float32")

            # Time corresponding to every observation/action pair
            group.create_dataset("timestamp", data=ep["timestamps"], dtype="float32")

            # SmolVLA visual observation from RoboSuite agentview
            group.create_dataset(
                "observation.images.agentview",
                data=ep["images"],
                dtype="uint8",
                chunks=(1, 256, 256, 3),
                compression="gzip",
                compression_opts=4,
            )

            total_frames += ep["num_frames"]

        # Dataset-level metadata
        data_group.attrs["num_episodes"] = len(episodes)
        data_group.attrs["total_frames"] = total_frames

    print(f"[+] HDF5 -> {path} | episodes={len(episodes)} | frames={total_frames}")



# dataset building:
def build_full_dataset(
    output_dir: str = "datasets/lerobot_dataset/glass_pick_place",
    replay_file: str = "datasets/cotracker_iterations/iteration_10_smolvla_expert/cup_lifting_01_replay.npz",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode = load_expert_episode(replay_npz_path=replay_file,episode_id="cup_lifting_01")

    if not episode["metrics"]["success"]:
        raise RuntimeError("Refusing to export unsuccessful expert episode.")

    npz_dest = out_dir / "cup_lifting_01_episode.npz"

    np.savez_compressed(
        npz_dest,
        images=episode["images"],
        actions=episode["actions"],
        state_8d=episode["state_8d"],
        timestamps=episode["timestamps"],
        task=np.asarray(episode["task"]),
        max_lift_cm=np.float32(episode["metrics"]["max_lift_cm"]),
        success=np.int8(episode["metrics"]["success"]),
    )

    save_hdf5_dataset(
        [episode],
        str(out_dir / "train_episodes.hdf5"),
    )

    metadata = {
        "dataset_name": "glass_pick_place_iteration10",
        "robot": "Franka Panda",
        "expert_source": "Iteration 10 object-relative closed-loop replay",

        "action_dim": 7,
        "action_format": "[dx, dy, dz, droll, dpitch, dyaw, gripper]",

        "state_dim": 8,
        "state_format": "[eef_x, eef_y, eef_z, qx, qy, qz, qw, gripper_qpos]",

        "image_resolution": [256, 256, 3],
        "camera": "agentview",

        "episodes": ["cup_lifting_01"],
        "frames": episode["num_frames"],
        "task": episode["task"],
        "max_lift_cm": episode["metrics"]["max_lift_cm"],
    }
    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("[+] Dataset export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Iteration 10 SmolVLA expert replay dataset"
    )

    # Final successful Iteration 10 expert replay
    parser.add_argument(
        "--replay",
        type=str,
        default="datasets/cotracker_iterations/iteration_10_smolvla_expert/cup_lifting_01_replay.npz",
        help="Iteration 10 replay NPZ with RGB, executed actions, and real Panda proprioception.",
    )

    # SmolVLA dataset output directory
    parser.add_argument(
        "--out_dir",
        "--output_dir",
        type=str,
        default="datasets/lerobot_dataset/glass_pick_place",
        help="Output directory for SmolVLA NPZ, HDF5, and metadata.",
    )

    args = parser.parse_args()

    build_full_dataset(
        output_dir=args.out_dir,
        replay_file=args.replay,
    )

"""
PYTHONPATH=. python3 src/learning/export_lerobot_dataset.py \
  --replay datasets/cotracker_iterations/iteration_10_smolvla_expert/cup_lifting_01_replay.npz \
  --out_dir datasets/lerobot_dataset/glass_pick_place
"""