"""
Post-Grasp Action Analysis: Expert vs Phase-Balanced SmolVLA.

Compares:
    Iteration 11 expert actions
    Phase-balanced SmolVLA K=5 rollout actions

Focus:
    - dz positive / negative / near-zero balance
    - gripper closed/open behavior
    - closed-gripper + positive-dz lift behavior
    - transition from lift -> lower -> release
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DZ_EPS = 0.01
GRIP_CLOSED_THRESHOLD = 0.0


def summarize(name, actions, start, end):
    a = actions[start:end]
    dz = a[:, 2]
    grip = a[:, 6]

    positive = dz > DZ_EPS
    negative = dz < -DZ_EPS
    zero = ~(positive | negative)
    closed = grip > GRIP_CLOSED_THRESHOLD
    open_grip = ~closed

    lift = closed & positive
    lower = closed & negative

    result = {
        "name": name,
        "start": int(start),
        "end": int(end),
        "frames": int(len(a)),
        "dz_mean": float(np.mean(dz)),
        "dz_std": float(np.std(dz)),
        "dz_min": float(np.min(dz)),
        "dz_max": float(np.max(dz)),
        "positive_dz_frames": int(np.sum(positive)),
        "negative_dz_frames": int(np.sum(negative)),
        "zero_dz_frames": int(np.sum(zero)),
        "positive_dz_percent": float(np.mean(positive) * 100.0),
        "negative_dz_percent": float(np.mean(negative) * 100.0),
        "grip_mean": float(np.mean(grip)),
        "closed_frames": int(np.sum(closed)),
        "open_frames": int(np.sum(open_grip)),
        "closed_percent": float(np.mean(closed) * 100.0),
        "closed_positive_dz_frames": int(np.sum(lift)),
        "closed_negative_dz_frames": int(np.sum(lower)),
        "closed_positive_dz_percent": float(np.mean(lift) * 100.0),
    }

    positive_values = dz[positive]
    negative_values = dz[negative]

    result["mean_positive_dz"] = (
        float(np.mean(positive_values))
        if len(positive_values)
        else 0.0
    )

    result["mean_negative_dz"] = (
        float(np.mean(negative_values))
        if len(negative_values)
        else 0.0
    )

    return result


def print_summary(result):
    print("\n" + "=" * 68)
    print(result["name"])
    print("=" * 68)

    print(f"Frames:                       {result['frames']}")
    print(f"dz mean/std:                  {result['dz_mean']:+.4f} / {result['dz_std']:.4f}")
    print(f"dz min/max:                   {result['dz_min']:+.4f} / {result['dz_max']:+.4f}")
    print(f"Positive dz:                  {result['positive_dz_frames']} ({result['positive_dz_percent']:.1f}%)")
    print(f"Negative dz:                  {result['negative_dz_frames']} ({result['negative_dz_percent']:.1f}%)")
    print(f"Mean positive dz:             {result['mean_positive_dz']:+.4f}")
    print(f"Mean negative dz:             {result['mean_negative_dz']:+.4f}")
    print(f"Gripper closed:               {result['closed_frames']} ({result['closed_percent']:.1f}%)")
    print(f"Closed + positive dz:         {result['closed_positive_dz_frames']} ({result['closed_positive_dz_percent']:.1f}%)")
    print(f"Closed + negative dz:         {result['closed_negative_dz_frames']}")


def longest_run(mask):
    best = current = 0

    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def main():
    parser = argparse.ArgumentParser(
        description="Analyze phase-balanced SmolVLA post-grasp behavior"
    )

    parser.add_argument(
        "--expert",
        default=(
            "datasets/cotracker_iterations/"
            "iteration_11_full_mujoco/"
            "cup_lifting_01_replay.npz"
        ),
    )

    parser.add_argument(
        "--rollout",
        default=(
            "datasets/smolvla_comparisons/"
            "phase_balanced/baseline_k5/"
            "cup_lifting_01_baseline_rollout.npz"
        ),
    )

    parser.add_argument(
        "--expert_grasp_frame",
        type=int,
        default=146,
    )

    parser.add_argument(
        "--rollout_grasp_frame",
        type=int,
        default=86,
    )

    parser.add_argument(
        "--out_dir",
        default=(
            "datasets/smolvla_comparisons/"
            "phase_balanced/postgrasp_analysis"
        ),
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.expert, allow_pickle=False) as data:
        expert_actions = np.asarray(
            data["executed_actions"],
            dtype=np.float32,
        )

        expert_grasp_flags = np.asarray(
            data["grasp_flags"],
            dtype=np.float32,
        )

    with np.load(args.rollout, allow_pickle=False) as data:
        rollout_actions = np.asarray(
            data["actions"],
            dtype=np.float32,
        )

        rollout_grasp_flags = np.asarray(
            data["grasp_flags"],
            dtype=np.float32,
        )

    if expert_actions.ndim != 2 or expert_actions.shape[1] != 7:
        raise ValueError(
            f"Expected expert actions (T,7), got {expert_actions.shape}"
        )

    if rollout_actions.ndim != 2 or rollout_actions.shape[1] != 7:
        raise ValueError(
            f"Expected rollout actions (T,7), got {rollout_actions.shape}"
        )

    expert_start = args.expert_grasp_frame
    rollout_start = args.rollout_grasp_frame

    expert_result = summarize(
        "ITERATION 11 EXPERT - POST GRASP",
        expert_actions,
        expert_start,
        len(expert_actions),
    )

    rollout_result = summarize(
        "PHASE-BALANCED SMOLVLA - POST GRASP",
        rollout_actions,
        rollout_start,
        len(rollout_actions),
    )

    print_summary(expert_result)
    print_summary(rollout_result)

    expert_post = expert_actions[expert_start:]
    rollout_post = rollout_actions[rollout_start:]

    expert_closed = expert_post[:, 6] > GRIP_CLOSED_THRESHOLD
    rollout_closed = rollout_post[:, 6] > GRIP_CLOSED_THRESHOLD

    expert_lift = expert_closed & (expert_post[:, 2] > DZ_EPS)
    rollout_lift = rollout_closed & (rollout_post[:, 2] > DZ_EPS)

    expert_lower = expert_closed & (expert_post[:, 2] < -DZ_EPS)
    rollout_lower = rollout_closed & (rollout_post[:, 2] < -DZ_EPS)

    print("\n" + "=" * 68)
    print("SEQUENCE COMPARISON")
    print("=" * 68)

    print(
        f"Expert longest closed-gripper run: "
        f"{longest_run(expert_closed)} frames"
    )

    print(
        f"VLA longest closed-gripper run:    "
        f"{longest_run(rollout_closed)} frames"
    )

    print(
        f"Expert longest lift run:           "
        f"{longest_run(expert_lift)} frames"
    )

    print(
        f"VLA longest lift run:              "
        f"{longest_run(rollout_lift)} frames"
    )

    print(
        f"Expert longest lower run:          "
        f"{longest_run(expert_lower)} frames"
    )

    print(
        f"VLA longest lower run:             "
        f"{longest_run(rollout_lower)} frames"
    )

    # Find first meaningful post-grasp downward transition.
    expert_lower_indices = np.where(expert_lower)[0]
    rollout_lower_indices = np.where(rollout_lower)[0]

    first_expert_lower = (
        int(expert_start + expert_lower_indices[0])
        if len(expert_lower_indices)
        else -1
    )

    first_rollout_lower = (
        int(rollout_start + rollout_lower_indices[0])
        if len(rollout_lower_indices)
        else -1
    )

    # Find release transition.
    expert_open_indices = np.where(~expert_closed)[0]
    rollout_open_indices = np.where(~rollout_closed)[0]

    first_expert_open = (
        int(expert_start + expert_open_indices[0])
        if len(expert_open_indices)
        else -1
    )

    first_rollout_open = (
        int(rollout_start + rollout_open_indices[0])
        if len(rollout_open_indices)
        else -1
    )

    print(f"\nFirst expert lower frame:         {first_expert_lower}")
    print(f"First phase-balanced lower frame: {first_rollout_lower}")
    print(f"First expert open/release frame:  {first_expert_open}")
    print(f"First phase-balanced open frame:  {first_rollout_open}")

    # Frame-level CSV
    rows = []
    T = max(len(expert_actions), len(rollout_actions))

    for frame in range(T):
        row = {"frame": frame}

        if frame < len(expert_actions):
            row.update({
                "expert_dz": float(expert_actions[frame, 2]),
                "expert_grip": float(expert_actions[frame, 6]),
                "expert_grasp": (
                    float(expert_grasp_flags[frame])
                    if frame < len(expert_grasp_flags)
                    else np.nan
                ),
            })

        if frame < len(rollout_actions):
            row.update({
                "rollout_dz": float(rollout_actions[frame, 2]),
                "rollout_grip": float(rollout_actions[frame, 6]),
                "rollout_grasp": (
                    float(rollout_grasp_flags[frame])
                    if frame < len(rollout_grasp_flags)
                    else np.nan
                ),
            })

        rows.append(row)

    csv_path = out_dir / "postgrasp_action_comparison.csv"

    fieldnames = [
        "frame",
        "expert_dz",
        "expert_grip",
        "expert_grasp",
        "rollout_dz",
        "rollout_grip",
        "rollout_grasp",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    summary = {
        "dz_epsilon": DZ_EPS,
        "gripper_closed_threshold": GRIP_CLOSED_THRESHOLD,
        "expert_grasp_frame": expert_start,
        "rollout_grasp_frame": rollout_start,
        "expert": expert_result,
        "phase_balanced": rollout_result,
        "expert_longest_closed_run": longest_run(expert_closed),
        "phase_balanced_longest_closed_run": longest_run(
            rollout_closed
        ),
        "expert_longest_lift_run": longest_run(expert_lift),
        "phase_balanced_longest_lift_run": longest_run(
            rollout_lift
        ),
        "expert_longest_lower_run": longest_run(expert_lower),
        "phase_balanced_longest_lower_run": longest_run(
            rollout_lower
        ),
        "first_expert_lower_frame": first_expert_lower,
        "first_phase_balanced_lower_frame": first_rollout_lower,
        "first_expert_open_frame": first_expert_open,
        "first_phase_balanced_open_frame": first_rollout_open,
    }

    summary_path = out_dir / "postgrasp_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"\n[+] Saved CSV:     {csv_path}")
    print(f"[+] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()

"""
cd ~/Documents/Pollex && \
python3 -m py_compile src/learning/analyze_phase_balanced_postgrasp.py
"""