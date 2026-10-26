"""
Export Verified Retargeted Trajectories to LeRobot & SmolVLA Dataset Format.

Simulates verified demonstration actions (Vid_0, Vid_2 for train, Vid_5 for test)
in GlassLiftEnv, records synced offscreen RGB frames, robot proprioception,
and 7D OSC actions, and packages them into LeRobot HDF5 and NPZ formats.
"""

import os
import json
import h5py
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from src.simulation.libero_runner import PandaSimEnvironment


DEFAULT_TASK_PROMPTS = {
    "detailed": "approach the glass on the table and grasp the cylinder and lift the cylinder and then bring down the cylinder to the table and then withdraw your hands",
    "partial": "approach the glass, grasp it and lift the cylinder",
    "vague": "lift up and down the glass",
}


def record_simulation_episode(
    actions_npz_path: str,
    episode_id: str,
    task_description: str = DEFAULT_TASK_PROMPTS["detailed"],
) -> Dict[str, Any]:
    """
    Executes retargeted actions in GlassLiftEnv and collects synced observations.
    """
    actions_file = Path(actions_npz_path)
    if not actions_file.exists():
        raise FileNotFoundError(f"Actions file not found: {actions_file}")

    data = np.load(actions_file)
    actions = data["actions"]  # Shape: (T, 7)
    timestamps = data["timestamps"] if "timestamps" in data else np.arange(len(actions)) / 20.0
    T = len(actions)

    print(f"[*] Simulating episode '{episode_id}' ({T} steps) in GlassLiftEnv...")

    sim = PandaSimEnvironment(
        env_name="GlassLift",
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_name="agentview",
        camera_height=256,
        camera_width=256,
    )

    obs = sim.reset()
    sim_frames = []
    eef_positions = []
    eef_quats = []
    gripper_qpos = []
    glass_z_positions = []
    rewards = []

    init_obj_z = None
    max_obj_z = -100.0

    for t in range(T):
        act = actions[t]
        
        # Record state prior to action (standard Markov Decision Process)
        sim_frames.append(obs["rgb"])
        eef_positions.append(obs["robot_eef_pos"])
        eef_quats.append(obs["robot_eef_quat"])
        grip_val = obs["robot_gripper_qpos"][0] if obs["robot_gripper_qpos"] is not None else 0.0
        gripper_qpos.append(grip_val)

        # Track glass object Z
        glass_z = sim.env.get_object_z()
        if init_obj_z is None:
            init_obj_z = glass_z
        glass_z_positions.append(glass_z)
        if glass_z > max_obj_z:
            max_obj_z = glass_z

        # Step simulation
        obs, reward, done, info = sim.step(act)
        rewards.append(reward)

    sim.close()

    init_z = init_obj_z if init_obj_z is not None else 0.8575
    final_z = glass_z_positions[-1]
    net_lift_height = max(0.0, max_obj_z - init_z)
    returned_to_table = abs(final_z - init_z) < 0.02 and final_z > 0.84
    success = (net_lift_height >= 0.05) and returned_to_table

    print(f"  [+] Episode '{episode_id}': Max Lift = {net_lift_height*100:.1f} cm | Table Return = {returned_to_table} | Success = {success}")

    # Build 7D state: [eef_x, eef_y, eef_z, eef_qx, eef_qy, eef_qz, gripper_qpos]
    eef_pos_arr = np.array(eef_positions, dtype=np.float32)      # (T, 3)
    eef_quat_arr = np.array(eef_quats, dtype=np.float32)        # (T, 4)
    grip_arr = np.array(gripper_qpos, dtype=np.float32)[:, None] # (T, 1)

    # 7D state vector: [x, y, z, roll/quat_x, quat_y, quat_z, gripper]
    # For compatibility with 7D state architectures
    state_7d = np.hstack([eef_pos_arr, eef_quat_arr[:, :3], grip_arr])
    # Full 8D state: [pos(3), quat(4), grip(1)]
    state_8d = np.hstack([eef_pos_arr, eef_quat_arr, grip_arr])

    return {
        "episode_id": episode_id,
        "num_frames": T,
        "images": np.array(sim_frames, dtype=np.uint8),  # (T, 256, 256, 3)
        "actions": np.array(actions, dtype=np.float32),   # (T, 7)
        "state_7d": np.array(state_7d, dtype=np.float32), # (T, 7)
        "state_8d": np.array(state_8d, dtype=np.float32), # (T, 8)
        "eef_pos": eef_pos_arr,
        "eef_quat": eef_quat_arr,
        "gripper_qpos": grip_arr,
        "glass_z": np.array(glass_z_positions, dtype=np.float32),
        "timestamps": np.array(timestamps, dtype=np.float32),
        "task": task_description,
        "metrics": {
            "max_lift_cm": float(net_lift_height * 100.0),
            "final_z": float(final_z),
            "init_z": float(init_z),
            "returned_to_table": bool(returned_to_table),
            "success": bool(success),
        }
    }


def save_hdf5_dataset(episodes: List[Dict[str, Any]], output_path: str):
    """
    Saves list of episode dicts into LeRobot-compatible HDF5 file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(out_file), "w") as root:
        data_grp = root.create_group("data")
        total_frames = 0

        for ep_idx, ep in enumerate(episodes):
            ep_grp = data_grp.create_group(f"demo_{ep_idx}")
            ep_grp.attrs["episode_id"] = ep["episode_id"]
            ep_grp.attrs["num_samples"] = ep["num_frames"]
            ep_grp.attrs["task"] = ep["task"]
            ep_grp.attrs["success"] = ep["metrics"]["success"]
            ep_grp.attrs["max_lift_cm"] = ep["metrics"]["max_lift_cm"]

            ep_grp.create_dataset("action", data=ep["actions"], dtype="float32")
            ep_grp.create_dataset("observation.state", data=ep["state_7d"], dtype="float32")
            ep_grp.create_dataset("observation.state_8d", data=ep["state_8d"], dtype="float32")
            ep_grp.create_dataset("observation.eef_pos", data=ep["eef_pos"], dtype="float32")
            ep_grp.create_dataset("observation.eef_quat", data=ep["eef_quat"], dtype="float32")
            ep_grp.create_dataset("observation.glass_z", data=ep["glass_z"], dtype="float32")
            ep_grp.create_dataset("timestamps", data=ep["timestamps"], dtype="float32")

            # Store images with gzip compression
            ep_grp.create_dataset(
                "observation.images.agentview",
                data=ep["images"],
                dtype="uint8",
                chunks=(1, 256, 256, 3),
                compression="gzip",
                compression_opts=4,
            )
            # Alias for camera1
            ep_grp.create_dataset(
                "observation.images.camera1",
                data=ep["images"],
                dtype="uint8",
                chunks=(1, 256, 256, 3),
                compression="gzip",
                compression_opts=4,
            )

            total_frames += ep["num_frames"]

        data_grp.attrs["total_frames"] = total_frames
        data_grp.attrs["num_episodes"] = len(episodes)

    print(f"[+] Saved HDF5 dataset -> {out_file} ({len(episodes)} episodes, {total_frames} frames)")


def build_full_dataset(
    output_dir: str = "data/lerobot_dataset/glass_pick_place",
    trajectories_dir: str = "data/cotracker_trajectories",
):
    """
    Compiles training dataset (Vid_0, Vid_2) and hold-out test dataset (Vid_5).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = Path(trajectories_dir)

    print("[*] ===================================================")
    print("[*] Exporting LeRobot Dataset for Glass Pick-and-Place")
    print("[*] ===================================================")

    # 1. Training Set: Vid_0 and Vid_2 (100% verified pick, lift, table return)
    def resolve_action_path(base_dir: Path, ep_id: str) -> Path:
        cand = base_dir / f"{ep_id}_ct_actions.npz"
        if cand.exists():
            return cand
        cand = base_dir / f"{ep_id}_actions.npz"
        if cand.exists():
            return cand
        cand = Path("data/cotracker_trajectories") / f"{ep_id}_ct_actions.npz"
        if cand.exists():
            return cand
        cand = Path("data/processed_trajectories") / f"{ep_id}_actions.npz"
        if cand.exists():
            return cand
        return base_dir / f"{ep_id}_ct_actions.npz"

    train_sources = [
        ("Vid_0", resolve_action_path(traj_dir, "Vid_0")),
        ("Vid_2", resolve_action_path(traj_dir, "Vid_2")),
    ]
    train_episodes = []
    for ep_id, act_path in train_sources:
        ep = record_simulation_episode(str(act_path), ep_id)
        train_episodes.append(ep)
        # Also save individual episode NPZ
        npz_dest = out_dir / f"{ep_id}_episode.npz"
        np.savez_compressed(
            str(npz_dest),
            images=ep["images"],
            actions=ep["actions"],
            state_7d=ep["state_7d"],
            state_8d=ep["state_8d"],
            glass_z=ep["glass_z"],
            task=ep["task"],
            metrics=ep["metrics"],
        )
        print(f"    [+] Saved standalone NPZ -> {npz_dest}")

    save_hdf5_dataset(train_episodes, str(out_dir / "train_episodes.hdf5"))

    # 2. Test Set: Vid_5 (slanted cylinder orientation testbed)
    test_sources = [
        ("Vid_5", resolve_action_path(traj_dir, "Vid_5")),
    ]
    test_episodes = []
    for ep_id, act_path in test_sources:
        ep = record_simulation_episode(str(act_path), ep_id)
        test_episodes.append(ep)
        npz_dest = out_dir / f"{ep_id}_episode.npz"
        np.savez_compressed(
            str(npz_dest),
            images=ep["images"],
            actions=ep["actions"],
            state_7d=ep["state_7d"],
            state_8d=ep["state_8d"],
            glass_z=ep["glass_z"],
            task=ep["task"],
            metrics=ep["metrics"],
        )
        print(f"    [+] Saved standalone NPZ -> {npz_dest}")

    save_hdf5_dataset(test_episodes, str(out_dir / "test_episodes.hdf5"))

    # 3. Export Metadata Summary
    metadata = {
        "dataset_name": "glass_pick_place",
        "robot": "Franka Panda",
        "action_dim": 7,
        "action_format": "OSC_POSE [dx, dy, dz, droll, dpitch, dyaw, gripper]",
        "state_dim": 7,
        "image_resolution": [256, 256, 3],
        "camera": "agentview",
        "train_episodes": [ep["episode_id"] for ep in train_episodes],
        "test_episodes": [ep["episode_id"] for ep in test_episodes],
        "train_frames": sum(ep["num_frames"] for ep in train_episodes),
        "test_frames": sum(ep["num_frames"] for ep in test_episodes),
        "task_prompts": DEFAULT_TASK_PROMPTS,
        "episode_stats": {
            ep["episode_id"]: {
                "num_frames": ep["num_frames"],
                "max_lift_cm": ep["metrics"]["max_lift_cm"],
                "success": ep["metrics"]["success"],
            }
            for ep in train_episodes + test_episodes
        }
    }

    meta_file = out_dir / "dataset_info.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved metadata summary -> {meta_file}")
    print("[*] Dataset export complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LeRobot dataset for GlassLift")
    parser.add_argument("--out_dir", "--output_dir", type=str, default="data/lerobot_dataset/glass_pick_place", help="Output dataset directory")
    parser.add_argument("--traj_dir", "--actions_dir", type=str, default="data/cotracker_trajectories", help="Directory of input trajectory action NPZs")
    args = parser.parse_args()

    build_full_dataset(output_dir=args.out_dir, trajectories_dir=args.traj_dir)
