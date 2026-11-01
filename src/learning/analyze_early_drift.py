"""
Early closed-loop drift analysis.

Compares Iteration 10 expert vs K=5 SmolVLA rollout for frames 0-100:
    - EEF X / Z
    - dx / dz commands
    - accumulation of state error

Purpose:
    Identify when the SmolVLA rollout first leaves the expert trajectory.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--expert",
        default=(
            "datasets/cotracker_iterations/"
            "iteration_10_smolvla_expert/"
            "cup_lifting_01_replay.npz"
        ),
    )

    parser.add_argument(
        "--rollout",
        default=(
            "datasets/smolvla_comparisons/k_5/"
            "cup_lifting_01_baseline_rollout.npz"
        ),
    )

    parser.add_argument(
        "--end_frame",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--out_dir",
        default="datasets/smolvla_comparisons/early_drift",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.expert, allow_pickle=False) as data:
        expert_pos = np.asarray(
            data["robot_eef_pos"],
            dtype=np.float32,
        )
        expert_action = np.asarray(
            data["executed_actions"],
            dtype=np.float32,
        )

    with np.load(args.rollout, allow_pickle=False) as data:
        rollout_pos = np.asarray(
            data["eef_positions"],
            dtype=np.float32,
        )
        rollout_action = np.asarray(
            data["actions"],
            dtype=np.float32,
        )

    T = min(
        args.end_frame + 1,
        len(expert_pos),
        len(rollout_pos),
        len(expert_action),
        len(rollout_action),
    )

    expert_pos = expert_pos[:T]
    rollout_pos = rollout_pos[:T]
    expert_action = expert_action[:T]
    rollout_action = rollout_action[:T]

    # State errors: rollout - expert
    x_error = rollout_pos[:, 0] - expert_pos[:, 0]
    z_error = rollout_pos[:, 2] - expert_pos[:, 2]

    # Action differences: rollout - expert
    dx_error = rollout_action[:, 0] - expert_action[:, 0]
    dz_error = rollout_action[:, 2] - expert_action[:, 2]

    # Useful drift thresholds
    drift_5mm = np.where(
        np.sqrt(x_error**2 + z_error**2) >= 0.005
    )[0]

    drift_10mm = np.where(
        np.sqrt(x_error**2 + z_error**2) >= 0.010
    )[0]

    z_drift_5mm = np.where(
        np.abs(z_error) >= 0.005
    )[0]

    dz_sign_mismatch = np.where(
        (np.abs(expert_action[:, 2]) > 1e-4)
        & (
            np.sign(expert_action[:, 2])
            != np.sign(rollout_action[:, 2])
        )
    )[0]

    first_5mm = int(drift_5mm[0]) if len(drift_5mm) else -1
    first_10mm = int(drift_10mm[0]) if len(drift_10mm) else -1
    first_z_5mm = int(z_drift_5mm[0]) if len(z_drift_5mm) else -1

    first_dz_sign_mismatch = (
        int(dz_sign_mismatch[0])
        if len(dz_sign_mismatch)
        else -1
    )

    print("=" * 100)
    print("EARLY CLOSED-LOOP DRIFT: ITERATION 10 EXPERT vs SmolVLA K=5")
    print("=" * 100)

    print(
        f"{'frame':>5} | "
        f"{'Exp X':>8} {'VLA X':>8} {'dX cm':>8} | "
        f"{'Exp Z':>8} {'VLA Z':>8} {'dZ cm':>8} | "
        f"{'Exp dx':>8} {'VLA dx':>8} | "
        f"{'Exp dz':>8} {'VLA dz':>8}"
    )

    rows = []

    for frame in range(T):
        row = {
            "frame": frame,
            "expert_x": float(expert_pos[frame, 0]),
            "rollout_x": float(rollout_pos[frame, 0]),
            "x_error_m": float(x_error[frame]),
            "expert_z": float(expert_pos[frame, 2]),
            "rollout_z": float(rollout_pos[frame, 2]),
            "z_error_m": float(z_error[frame]),
            "expert_dx": float(expert_action[frame, 0]),
            "rollout_dx": float(rollout_action[frame, 0]),
            "expert_dz": float(expert_action[frame, 2]),
            "rollout_dz": float(rollout_action[frame, 2]),
        }

        rows.append(row)

        if frame % 10 == 0 or frame in [
            first_5mm,
            first_10mm,
            first_z_5mm,
            first_dz_sign_mismatch,
        ]:
            print(
                f"{frame:5d} | "
                f"{expert_pos[frame, 0]:+8.4f} "
                f"{rollout_pos[frame, 0]:+8.4f} "
                f"{x_error[frame] * 100:+8.2f} | "
                f"{expert_pos[frame, 2]:+8.4f} "
                f"{rollout_pos[frame, 2]:+8.4f} "
                f"{z_error[frame] * 100:+8.2f} | "
                f"{expert_action[frame, 0]:+8.4f} "
                f"{rollout_action[frame, 0]:+8.4f} | "
                f"{expert_action[frame, 2]:+8.4f} "
                f"{rollout_action[frame, 2]:+8.4f}"
            )

    print("\nSUMMARY")
    print("-" * 60)
    print(f"First X/Z state drift >= 5 mm : frame {first_5mm}")
    print(f"First X/Z state drift >= 1 cm : frame {first_10mm}")
    print(f"First |Z error| >= 5 mm       : frame {first_z_5mm}")
    print(f"First DZ sign mismatch         : frame {first_dz_sign_mismatch}")

    max_x_error = float(np.max(np.abs(x_error)))
    max_z_error = float(np.max(np.abs(z_error)))

    print(f"Maximum |X drift| 0-{T-1}     : {max_x_error * 100:.2f} cm")
    print(f"Maximum |Z drift| 0-{T-1}     : {max_z_error * 100:.2f} cm")

    csv_path = out_dir / "early_drift_frames_0_100.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "frames_analyzed": T,
        "first_state_drift_5mm_frame": first_5mm,
        "first_state_drift_1cm_frame": first_10mm,
        "first_z_drift_5mm_frame": first_z_5mm,
        "first_dz_sign_mismatch_frame": first_dz_sign_mismatch,
        "max_abs_x_drift_cm": max_x_error * 100.0,
        "max_abs_z_drift_cm": max_z_error * 100.0,
    }

    json_path = out_dir / "early_drift_summary.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\n[+] CSV:     {csv_path}")
    print(f"[+] Summary: {json_path}")


if __name__ == "__main__":
    main()

"""
cd ~/Documents/Pollex && \
PYTHONPATH=. python3 -m src.learning.analyze_early_drift \
  --expert datasets/cotracker_iterations/iteration_10_smolvla_expert/cup_lifting_01_replay.npz \
  --rollout datasets/smolvla_comparisons/k_5/cup_lifting_01_baseline_rollout.npz \
  --end_frame 100 \
  --out_dir datasets/smolvla_comparisons/early_drift
"""