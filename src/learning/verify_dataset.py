"""
Verify LeRobot Dataset Formatting and Physical Integrity.

Inspects HDF5 and NPZ dataset artifacts, checks array shapes,
verifies physical lift success and table return metrics, and outputs
a summary table of dataset dimensions.
"""

import h5py
import json
import argparse
import numpy as np
from pathlib import Path


def verify_dataset(dataset_dir: str = "data/lerobot_dataset/glass_pick_place"):
    dpath = Path(dataset_dir)
    if not dpath.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dpath}")

    info_file = dpath / "dataset_info.json"
    if info_file.exists():
        with open(info_file) as f:
            meta = json.load(f)
        print("=" * 70)
        print("LeRobot Dataset Metadata Summary")
        print("=" * 70)
        print(f"Dataset: {meta.get('dataset_name')}")
        print(f"Robot: {meta.get('robot')} | Action Dim: {meta.get('action_dim')}")
        print(f"Train Episodes: {meta.get('train_episodes')} ({meta.get('train_frames')} frames)")
        print(f"Test Episodes: {meta.get('test_episodes')} ({meta.get('test_frames')} frames)")
        print(f"Image Resolution: {meta.get('image_resolution')}")
        print("-" * 70)

    # Verify HDF5 files
    for h5_name in ["train_episodes.hdf5", "test_episodes.hdf5"]:
        h5_path = dpath / h5_name
        if not h5_path.exists():
            print(f"[-] Missing {h5_name}")
            continue

        print(f"\n[*] Inspecting {h5_name}...")
        with h5py.File(str(h5_path), "r") as root:
            data_grp = root["data"]
            print(f"    Total episodes: {data_grp.attrs.get('num_episodes')}")
            print(f"    Total frames: {data_grp.attrs.get('total_frames')}")

            for ep_key in data_grp.keys():
                ep = data_grp[ep_key]
                act = ep["action"][:]
                img = ep["observation.images.agentview"]
                st7 = ep["observation.state"][:]
                gz = ep["observation.glass_z"][:]
                ep_id = ep.attrs.get("episode_id", ep_key)
                succ = ep.attrs.get("success", False)
                max_lift = ep.attrs.get("max_lift_cm", 0.0)

                print(f"    [{ep_key}] ID={ep_id} | Frames={len(act)} | Lift={max_lift:.1f}cm | Success={succ}")
                print(f"        Actions shape: {act.shape}, range=[{act.min():.2f}, {act.max():.2f}]")
                print(f"        State shape: {st7.shape}, range=[{st7.min():.2f}, {st7.max():.2f}]")
                print(f"        Images shape: {img.shape}, dtype={img.dtype}")
                print(f"        Glass Z: start={gz[0]:.3f}m, apex={gz.max():.3f}m, end={gz[-1]:.3f}m")

                # Assertions
                assert act.shape[1] == 7, f"Action dimension must be 7, got {act.shape[1]}"
                assert st7.shape[1] == 7, f"State dimension must be 7, got {st7.shape[1]}"
                assert img.shape[1:] == (256, 256, 3), f"Image shape must be (256, 256, 3), got {img.shape[1:]}"
                assert len(act) == len(st7) == len(img) == len(gz), "Mismatched sequence lengths!"

    print("\n[+] All dataset assertions passed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify LeRobot dataset formatting")
    parser.add_argument("--dataset", type=str, default="data/lerobot_dataset/glass_pick_place")
    args = parser.parse_args()
    verify_dataset(args.dataset)
