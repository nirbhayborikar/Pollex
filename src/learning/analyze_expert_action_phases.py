"""
Analyze Iteration 11 expert action / phase distribution.

Purpose:
    Quantify how much of the 340-frame expert episode supervises:
      - hover
      - approach
      - descent / pre-grasp
      - grasp closing
      - closed-gripper lift
      - closed-gripper lowering
      - release / retract

Input:
    datasets/cotracker_iterations/iteration_11_full_mujoco/
    cup_lifting_01_replay.npz

Outputs:
    datasets/smolvla_comparisons/action_phase_analysis/
      phase_statistics.json
      frame_phase_labels.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]


def action_stats(actions: np.ndarray) -> dict:
    """Return per-channel statistics for a set of expert actions."""
    if len(actions) == 0:
        return {}

    stats = {}

    for i, name in enumerate(ACTION_NAMES):
        values = actions[:, i]

        stats[name] = {
            "mean": float(np.mean(values)),
            "mean_abs": float(np.mean(np.abs(values))),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return stats


def phase_stats(
    name: str,
    mask: np.ndarray,
    actions: np.ndarray,
) -> dict:
    """Compute frame counts and action statistics for one phase."""
    selected = actions[mask]
    count = len(selected)

    if count == 0:
        return {
            "phase": name,
            "frames": 0,
            "percent": 0.0,
            "action_stats": {},
        }

    grip_closed = selected[:, 6] > 0.0
    dz_positive = selected[:, 2] > 1e-4
    dz_negative = selected[:, 2] < -1e-4

    return {
        "phase": name,
        "frames": int(count),
        "percent": float(100.0 * count / len(actions)),
        "grip_closed_frames": int(np.sum(grip_closed)),
        "dz_positive_frames": int(np.sum(dz_positive)),
        "dz_negative_frames": int(np.sum(dz_negative)),
        "closed_and_lift_frames": int(
            np.sum(grip_closed & dz_positive)
        ),
        "closed_and_lower_frames": int(
            np.sum(grip_closed & dz_negative)
        ),
        "action_stats": action_stats(selected),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze expert action and phase distribution."
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
        "--out_dir",
        default=(
            "datasets/smolvla_comparisons/"
            "action_phase_analysis"
        ),
    )

    args = parser.parse_args()

    expert_path = Path(args.expert)

    if not expert_path.exists():
        raise FileNotFoundError(
            f"Expert replay not found: {expert_path}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(expert_path, allow_pickle=False) as data:
        actions = np.asarray(
            data["executed_actions"],
            dtype=np.float32,
        )

        contact_flags = np.asarray(
            data["contact_flags"],
            dtype=np.float32,
        )

        grasp_flags = np.asarray(
            data["grasp_flags"],
            dtype=np.float32,
        )

        grasp_trigger_frame = (
            int(np.asarray(data["grasp_trigger_frame"]).item())
            if "grasp_trigger_frame" in data
            else 143
        )

        physical_grasp_frame = (
            int(
                np.asarray(
                    data["physical_grasp_latch_frame"]
                ).item()
            )
            if "physical_grasp_latch_frame" in data
            else 146
        )

    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(
            f"Expected actions (T,7), got {actions.shape}"
        )

    T = len(actions)

    if len(contact_flags) != T or len(grasp_flags) != T:
        raise ValueError(
            "Action/contact/grasp arrays are not synchronized."
        )

    # Known Iteration-11 phase boundaries derived from the replay.
    hover_end = min(40, T)
    descent_start = min(86, T)
    grasp_trigger_frame = int(
        np.clip(grasp_trigger_frame, 0, T)
    )
    physical_grasp_frame = int(
        np.clip(physical_grasp_frame, 0, T)
    )

    frame_ids = np.arange(T)

    hover_mask = frame_ids < hover_end

    approach_mask = (
        (frame_ids >= hover_end)
        & (frame_ids < descent_start)
    )

    descent_mask = (
        (frame_ids >= descent_start)
        & (frame_ids < grasp_trigger_frame)
    )

    grasp_close_mask = (
        (frame_ids >= grasp_trigger_frame)
        & (frame_ids < physical_grasp_frame)
    )

    post_grasp_mask = frame_ids >= physical_grasp_frame

    grip_closed = actions[:, 6] > 0.0
    grip_open = ~grip_closed

    dz_positive = actions[:, 2] > 1e-4
    dz_negative = actions[:, 2] < -1e-4
    dz_static = ~(dz_positive | dz_negative)

    # Physically important post-grasp phases.
    lift_mask = (
        post_grasp_mask
        & grip_closed
        & dz_positive
    )

    lower_mask = (
        post_grasp_mask
        & grip_closed
        & dz_negative
    )

    closed_static_mask = (
        post_grasp_mask
        & grip_closed
        & dz_static
    )

    release_retract_mask = (
        post_grasp_mask
        & grip_open
    )

    phases = {
        "hover": hover_mask,
        "approach": approach_mask,
        "descent_pregrasp": descent_mask,
        "grasp_closing": grasp_close_mask,
        "postgrasp_closed_lift": lift_mask,
        "postgrasp_closed_lower": lower_mask,
        "postgrasp_closed_static": closed_static_mask,
        "release_retract": release_retract_mask,
    }

    statistics = {}

    for name, mask in phases.items():
        statistics[name] = phase_stats(
            name,
            mask,
            actions,
        )

    # Overall action statistics.
    overall_stats = action_stats(actions)

    total_closed_lift = int(
        np.sum(grip_closed & dz_positive)
    )

    postgrasp_closed_lift = int(
        np.sum(lift_mask)
    )

    total_closed = int(np.sum(grip_closed))
    total_positive_dz = int(np.sum(dz_positive))
    total_negative_dz = int(np.sum(dz_negative))

    critical_fraction = (
        100.0 * postgrasp_closed_lift / T
    )

    closed_lift_fraction = (
        100.0 * postgrasp_closed_lift / total_closed
        if total_closed
        else 0.0
    )

    # Assign one phase label to every frame.
    labels = np.full(T, "unclassified", dtype=object)

    for name, mask in phases.items():
        labels[mask] = name

    # Sanity check that all frames received a phase.
    unclassified = np.where(labels == "unclassified")[0]

    if len(unclassified):
        raise RuntimeError(
            f"Unclassified frames: {unclassified.tolist()}"
        )

    print("=" * 82)
    print("ITERATION 11 EXPERT ACTION / PHASE DISTRIBUTION")
    print("=" * 82)

    print(
        f"{'PHASE':30s} "
        f"{'FRAMES':>8s} "
        f"{'% DATA':>9s} "
        f"{'MEAN |DX|':>11s} "
        f"{'MEAN |DZ|':>11s} "
        f"{'MEAN GRIP':>11s}"
    )

    print("-" * 82)

    for name, result in statistics.items():
        if result["frames"] == 0:
            mean_dx = mean_dz = mean_grip = 0.0
        else:
            stats = result["action_stats"]
            mean_dx = stats["dx"]["mean_abs"]
            mean_dz = stats["dz"]["mean_abs"]
            mean_grip = stats["grip"]["mean"]

        print(
            f"{name:30s} "
            f"{result['frames']:8d} "
            f"{result['percent']:8.2f}% "
            f"{mean_dx:11.4f} "
            f"{mean_dz:11.4f} "
            f"{mean_grip:+11.4f}"
        )

    print("\nCRITICAL LIFT SUPERVISION")
    print("-" * 60)
    print(f"Total frames:                       {T}")
    print(f"Closed-gripper frames:              {total_closed}")
    print(f"Positive-dz frames:                 {total_positive_dz}")
    print(f"Negative-dz frames:                 {total_negative_dz}")
    print(f"Closed + positive-dz frames:        {total_closed_lift}")
    print(f"Post-grasp closed + positive-dz:    {postgrasp_closed_lift}")
    print(f"Critical lift fraction of dataset:  {critical_fraction:.2f}%")
    print(f"Lift fraction of closed frames:     {closed_lift_fraction:.2f}%")

    print("\nOVERALL ACTION MAGNITUDES")
    print("-" * 60)

    for name in ACTION_NAMES:
        values = overall_stats[name]

        print(
            f"{name:7s} "
            f"mean|a|={values['mean_abs']:.4f} "
            f"std={values['std']:.4f} "
            f"range=[{values['min']:+.4f}, "
            f"{values['max']:+.4f}]"
        )

    # Save frame-level CSV.
    csv_path = out_dir / "frame_phase_labels.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "frame",
            "phase",
            "dx",
            "dy",
            "dz",
            "droll",
            "dpitch",
            "dyaw",
            "grip",
            "contact",
            "grasp",
            "grip_closed",
            "dz_positive",
            "dz_negative",
        ])

        for frame in range(T):
            writer.writerow([
                frame,
                labels[frame],
                *[float(x) for x in actions[frame]],
                float(contact_flags[frame]),
                float(grasp_flags[frame]),
                int(grip_closed[frame]),
                int(dz_positive[frame]),
                int(dz_negative[frame]),
            ])

    summary = {
        "expert_file": str(expert_path),
        "total_frames": T,
        "phase_boundaries": {
            "hover_end": hover_end,
            "descent_start": descent_start,
            "grasp_trigger_frame": grasp_trigger_frame,
            "physical_grasp_frame": physical_grasp_frame,
        },
        "critical_lift_supervision": {
            "closed_gripper_frames": total_closed,
            "positive_dz_frames": total_positive_dz,
            "negative_dz_frames": total_negative_dz,
            "closed_positive_dz_frames": total_closed_lift,
            "postgrasp_closed_positive_dz_frames": (
                postgrasp_closed_lift
            ),
            "critical_lift_fraction_percent": (
                critical_fraction
            ),
            "lift_fraction_of_closed_frames_percent": (
                closed_lift_fraction
            ),
        },
        "overall_action_stats": overall_stats,
        "phases": statistics,
    }

    json_path = out_dir / "phase_statistics.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"\n[+] Saved CSV:  {csv_path}")
    print(f"[+] Saved JSON: {json_path}")


if __name__ == "__main__":
    main()

"""
cd ~/Documents/Pollex && \
PYTHONPATH=. python3 -m src.learning.analyze_expert_action_phases \
  --expert datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
  --out_dir datasets/smolvla_comparisons/action_phase_analysis

"""