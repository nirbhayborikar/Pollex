import argparse
from pathlib import Path

import numpy as np


def stats(name, expert, predicted, start, end):
    e = expert[start:end]
    p = predicted[start:end]

    print(f"\n{name} [{start}:{end}]")
    print("-" * 60)

    for idx, channel in [(0, "dx"), (2, "dz"), (6, "grip")]:
        print(
            f"{channel:5s} | "
            f"expert mean={e[:, idx].mean():+.4f} "
            f"min={e[:, idx].min():+.4f} "
            f"max={e[:, idx].max():+.4f} | "
            f"smolvla mean={p[:, idx].mean():+.4f} "
            f"min={p[:, idx].min():+.4f} "
            f"max={p[:, idx].max():+.4f}"
        )


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
    args = parser.parse_args()

    with np.load(args.expert, allow_pickle=False) as data:
        expert = np.asarray(
            data["executed_actions"],
            dtype=np.float32,
        )

    with np.load(args.rollout, allow_pickle=False) as data:
        predicted = np.asarray(
            data["actions"],
            dtype=np.float32,
        )

    T = min(len(expert), len(predicted))
    expert = expert[:T]
    predicted = predicted[:T]

    print("=" * 60)
    print("EXPERT vs SmolVLA ACTION DIAGNOSTIC")
    print("=" * 60)
    print("Expert:  ", expert.shape)
    print("SmolVLA: ", predicted.shape)

    phases = [
        ("Approach", 0, 100),
        ("Pre-grasp", 100, 143),
        ("Grasp/contact", 143, 160),
        ("Lift/return", 160, T),
    ]

    for name, start, end in phases:
        stats(name, expert, predicted, start, end)

    print("\nCRITICAL FRAMES")
    print("-" * 76)
    print(
        f"{'frame':>6} | {'expert dx':>10} {'VLA dx':>10} | "
        f"{'expert dz':>10} {'VLA dz':>10} | "
        f"{'exp grip':>9} {'VLA grip':>9}"
    )

    for frame in [0, 50, 100, 120, 140, 143, 146, 150, 160, 180, 200, 250, 300, 339]:
        if frame >= T:
            continue

        e = expert[frame]
        p = predicted[frame]

        print(
            f"{frame:6d} | "
            f"{e[0]:+10.4f} {p[0]:+10.4f} | "
            f"{e[2]:+10.4f} {p[2]:+10.4f} | "
            f"{e[6]:+9.3f} {p[6]:+9.3f}"
        )

    np.savez_compressed(
        "datasets/smolvla_comparisons/"
        "expert_vs_smolvla_actions.npz",
        expert_actions=expert,
        smolvla_actions=predicted,
    )


if __name__ == "__main__":
    main()

"""
cd ~/Documents/Pollex && \
PYTHONPATH=. python3 src/learning/compare_expert_smolvla_actions.py \
  --expert datasets/cotracker_iterations/iteration_10_smolvla_expert/cup_lifting_01_replay.npz \
  --rollout datasets/smolvla_comparisons/k_5/cup_lifting_01_baseline_rollout.npz

"""