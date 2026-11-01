"""
Analyze semantic purity of 50-step SmolVLA expert action chunks.

Purpose:
    Determine whether a training sample labelled with one temporal phase
    contains actions from multiple future phases because H=50 crosses
    grasp/lift/lower/release boundaries.

Uses Iteration 11 executed actions + tumbler trajectory.

Outputs:
    chunk_phase_purity.csv
    chunk_phase_summary.json
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


DZ_EPS = 0.01
CHUNK_SIZE = 50
RELEASE_SETTLE_FRAMES = 12

PHASES = [
    "hover",
    "approach",
    "descent_pregrasp",
    "grasp_closing",
    "postgrasp_lift",
    "postgrasp_lower",
    "release",
    "retract",
]


def classify_frame(
    frame: int,
    hover_end: int,
    descent_start: int,
    grasp_trigger: int,
    physical_grasp: int,
    peak_lift: int,
    release: int,
    retract: int,
) -> str:
    if frame < hover_end:
        return "hover"
    if frame < descent_start:
        return "approach"
    if frame < grasp_trigger:
        return "descent_pregrasp"
    if frame < physical_grasp:
        return "grasp_closing"
    if frame < peak_lift:
        return "postgrasp_lift"
    if frame < release:
        return "postgrasp_lower"
    if frame < retract:
        return "release"
    return "retract"


def longest_run(values):
    best = current = 0
    previous = None

    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        best = max(best, current)

    return best


def main():
    parser = argparse.ArgumentParser(description="Analyze SmolVLA 50-step action-chunk phase purity")

    parser.add_argument(
        "--expert",
        default="datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz",
    )
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument(
        "--out_dir",
        default="datasets/smolvla_comparisons/chunk_phase_purity",
    )

    args = parser.parse_args()

    if args.chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.expert, allow_pickle=False) as data:
        actions = np.asarray(data["executed_actions"], dtype=np.float32)
        tumbler = np.asarray(data["tumbler_positions"], dtype=np.float32)
        grasp_trigger = int(np.asarray(data["grasp_trigger_frame"]).item())
        physical_grasp = int(np.asarray(data["physical_grasp_latch_frame"]).item())

    T = len(actions)

    if actions.shape != (T, 7):
        raise ValueError(f"Expected actions (T,7), got {actions.shape}")
    if tumbler.shape != (T, 3):
        raise ValueError(f"Expected tumbler positions (T,3), got {tumbler.shape}")

    frame_indices = np.arange(T)
    release_candidates = np.where(
        (frame_indices > physical_grasp) & (actions[:, 6] <= 0.0)
    )[0]

    if not len(release_candidates):
        raise RuntimeError("No post-grasp release frame found.")

    release = int(release_candidates[0])
    retract = min(T, release + RELEASE_SETTLE_FRAMES)

    lift_window = tumbler[physical_grasp:release, 2]
    peak_lift = physical_grasp + int(np.argmax(lift_window))

    hover_end = min(40, T)
    descent_start = min(86, T)

    print("=" * 80)
    print("SMOLVLA ACTION-CHUNK PHASE PURITY")
    print("=" * 80)
    print(f"Episode length:        {T}")
    print(f"Chunk horizon H:       {args.chunk_size}")
    print(f"Grasp trigger:         {grasp_trigger}")
    print(f"Physical grasp:        {physical_grasp}")
    print(f"Peak lift:             {peak_lift}")
    print(f"Release:               {release}")
    print(f"Retract:               {retract}")

    frame_phases = [
        classify_frame(
            frame=i,
            hover_end=hover_end,
            descent_start=descent_start,
            grasp_trigger=grasp_trigger,
            physical_grasp=physical_grasp,
            peak_lift=peak_lift,
            release=release,
            retract=retract,
        )
        for i in range(T)
    ]

    rows = []
    phase_stats = {
        phase: {
            "anchors": 0,
            "purities": [],
            "boundary_crossings": [],
            "grip_closed": [],
            "dz_positive": [],
            "dz_negative": [],
        }
        for phase in PHASES
    }

    for anchor in range(T):
        end = min(T, anchor + args.chunk_size)
        indices = np.arange(anchor, end)

        # Match training behavior: final action is repeated if chunk hits episode end.
        real_actions = actions[anchor:end]
        real_phases = frame_phases[anchor:end]

        if len(real_actions) < args.chunk_size:
            pad_count = args.chunk_size - len(real_actions)
            chunk_actions = np.concatenate(
                [
                    real_actions,
                    np.repeat(real_actions[-1][None, :], pad_count, axis=0),
                ],
                axis=0,
            )
            chunk_phases = real_phases + [real_phases[-1]] * pad_count
        else:
            chunk_actions = real_actions
            chunk_phases = list(real_phases)

        anchor_phase = frame_phases[anchor]

        counts = Counter(chunk_phases)
        same_phase = counts[anchor_phase]
        purity = 100.0 * same_phase / args.chunk_size

        boundary_crossings = sum(
            chunk_phases[i] != chunk_phases[i - 1]
            for i in range(1, len(chunk_phases))
        )

        grip_closed = chunk_actions[:, 6] > 0.0
        dz_positive = chunk_actions[:, 2] > DZ_EPS
        dz_negative = chunk_actions[:, 2] < -DZ_EPS

        closed_positive = grip_closed & dz_positive
        closed_negative = grip_closed & dz_negative

        stats = phase_stats[anchor_phase]
        stats["anchors"] += 1
        stats["purities"].append(purity)
        stats["boundary_crossings"].append(boundary_crossings)
        stats["grip_closed"].append(float(np.mean(grip_closed) * 100.0))
        stats["dz_positive"].append(float(np.mean(dz_positive) * 100.0))
        stats["dz_negative"].append(float(np.mean(dz_negative) * 100.0))

        rows.append({
            "anchor": anchor,
            "anchor_phase": anchor_phase,
            "chunk_end": end - 1,
            "purity_percent": purity,
            "boundary_crossings": boundary_crossings,
            "unique_phases": len(counts),
            "phases_in_chunk": ",".join(counts.keys()),
            "grip_closed_percent": float(np.mean(grip_closed) * 100.0),
            "grip_open_percent": float(np.mean(~grip_closed) * 100.0),
            "positive_dz_percent": float(np.mean(dz_positive) * 100.0),
            "negative_dz_percent": float(np.mean(dz_negative) * 100.0),
            "closed_positive_dz_percent": float(np.mean(closed_positive) * 100.0),
            "closed_negative_dz_percent": float(np.mean(closed_negative) * 100.0),
            "mean_dz": float(np.mean(chunk_actions[:, 2])),
            "mean_grip": float(np.mean(chunk_actions[:, 6])),
        })

    print("\n" + "=" * 80)
    print("PURITY BY ANCHOR PHASE")
    print("=" * 80)

    summary = {}

    for phase in PHASES:
        stats = phase_stats[phase]

        if not stats["anchors"]:
            continue

        purities = np.asarray(stats["purities"])
        crossings = np.asarray(stats["boundary_crossings"])

        result = {
            "anchors": stats["anchors"],
            "mean_purity_percent": float(np.mean(purities)),
            "min_purity_percent": float(np.min(purities)),
            "chunks_crossing_boundary_percent": float(np.mean(crossings > 0) * 100.0),
            "mean_boundary_crossings": float(np.mean(crossings)),
            "mean_grip_closed_percent": float(np.mean(stats["grip_closed"])),
            "mean_positive_dz_percent": float(np.mean(stats["dz_positive"])),
            "mean_negative_dz_percent": float(np.mean(stats["dz_negative"])),
        }

        summary[phase] = result

        print(
            f"{phase:24s} "
            f"anchors={result['anchors']:3d} | "
            f"purity={result['mean_purity_percent']:5.1f}% | "
            f"min={result['min_purity_percent']:5.1f}% | "
            f"cross={result['chunks_crossing_boundary_percent']:5.1f}%"
        )

    print("\n" + "=" * 80)
    print("CRITICAL 50-STEP CHUNKS")
    print("=" * 80)

    critical_anchors = [
        120,
        130,
        140,
        143,
        146,
        150,
        160,
        170,
        180,
        186,
        190,
        200,
        210,
        213,
        220,
        225,
    ]

    row_lookup = {row["anchor"]: row for row in rows}

    for anchor in critical_anchors:
        if anchor >= T:
            continue

        row = row_lookup[anchor]

        print(
            f"{anchor:3d} "
            f"{row['anchor_phase']:22s} "
            f"purity={row['purity_percent']:5.1f}% "
            f"cross={row['boundary_crossings']} "
            f"grip+={row['grip_closed_percent']:5.1f}% "
            f"dz+={row['positive_dz_percent']:5.1f}% "
            f"dz-={row['negative_dz_percent']:5.1f}% "
            f"phases=[{row['phases_in_chunk']}]"
        )

    low_purity = [
        row
        for row in rows
        if row["purity_percent"] < 70.0
    ]

    multi_phase = [
        row
        for row in rows
        if row["boundary_crossings"] > 0
    ]

    overall = {
        "episode_frames": T,
        "chunk_size": args.chunk_size,
        "grasp_trigger_frame": grasp_trigger,
        "physical_grasp_frame": physical_grasp,
        "peak_lift_frame": peak_lift,
        "release_frame": release,
        "retract_frame": retract,
        "chunks_crossing_any_boundary": len(multi_phase),
        "chunks_crossing_any_boundary_percent": 100.0 * len(multi_phase) / T,
        "chunks_below_70_percent_purity": len(low_purity),
        "chunks_below_70_percent_purity_percent": 100.0 * len(low_purity) / T,
    }

    print("\n" + "=" * 80)
    print("OVERALL")
    print("=" * 80)

    print(
        f"Chunks crossing >=1 phase boundary: "
        f"{overall['chunks_crossing_any_boundary']}/{T} "
        f"({overall['chunks_crossing_any_boundary_percent']:.1f}%)"
    )

    print(
        f"Chunks with <70% anchor-phase purity: "
        f"{overall['chunks_below_70_percent_purity']}/{T} "
        f"({overall['chunks_below_70_percent_purity_percent']:.1f}%)"
    )

    csv_path = out_dir / "chunk_phase_purity.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "chunk_phase_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "overall": overall,
                "phase_summary": summary,
            },
            file,
            indent=2,
        )

    print(f"\n[+] Saved CSV:     {csv_path}")
    print(f"[+] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()


"""

cd ~/Documents/Pollex && \
python3 -m py_compile src/learning/analyze_chunk_phase_purity.py && \
PYTHONPATH=. python3 -m src.learning.analyze_chunk_phase_purity \
  --expert datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
  --chunk_size 50 \
  --out_dir datasets/smolvla_comparisons/chunk_phase_purity
"""
