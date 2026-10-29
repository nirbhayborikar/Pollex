"""
Compare MediaPipe vs CoTracker3 tracking pipelines.

Loads trajectory .npz files from each tracker and generates:
  1. Gripper state agreement (frame-by-frame OPEN/CLOSED match)
  2. Wrist trajectory smoothness (jerk = 3rd derivative)
  3. Gripper transition timing (grasp/release event timeline)

Usage:
  python3 src/evaluation/compare_trackers.py \
      --mediapipe datasets/processed_trajectories/cup_lifting_01_trajectory.npz \
      --cotracker data/cotracker_trajectories/cup_lifting_01_ct_trajectory.npz \
      --output eval_results

Either --mediapipe or --cotracker can be omitted; the script plots whatever is available.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# Categorical palette (slot 1 = blue, slot 2 = orange) from validated reference
C_MEDIAPIPE = "#2a78d6"
C_COTRACKER = "#eb6834"
C_SURFACE = "#fcfcfb"
C_TEXT = "#0b0b0b"
C_TEXT_SEC = "#52514e"
C_GRID = "#e0dfdb"
C_CLOSED = "#e34948"
C_OPEN = "#1baf7a"


def load_trajectory(path):
    p = Path(path)
    if not p.exists():
        return None
    data = dict(np.load(str(p), allow_pickle=True))
    fps = float(data.get("fps", 30.0))
    T = len(data["gripper_states"])
    if "timestamps" not in data or len(data["timestamps"]) != T:
        data["timestamps"] = np.arange(T, dtype=np.float32) / fps
    return data


def find_transitions(gripper_states):
    transitions = []
    for i in range(1, len(gripper_states)):
        if gripper_states[i] != gripper_states[i - 1]:
            kind = "CLOSE" if gripper_states[i] > 0 else "OPEN"
            transitions.append((i, kind))
    return transitions


def compute_jerk(positions, fps):
    if len(positions) < 4:
        return np.zeros(len(positions))
    dt = 1.0 / fps
    vel = np.gradient(positions, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    jerk = np.gradient(acc, dt, axis=0)
    return np.linalg.norm(jerk, axis=1)


def style_ax(ax):
    ax.set_facecolor(C_SURFACE)
    ax.tick_params(colors=C_TEXT_SEC, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.grid(axis="y", color=C_GRID, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def plot_comparison(mp_data, ct_data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    has_mp = mp_data is not None
    has_ct = ct_data is not None

    if not has_mp and not has_ct:
        print("[-] No trajectory data found. Nothing to plot.")
        return

    # Determine shared time axis
    if has_mp:
        mp_t = mp_data["timestamps"]
        mp_grip = mp_data["gripper_states"]
        mp_wrist = mp_data["wrist_positions"]
        mp_fps = float(mp_data.get("fps", 30.0))
    if has_ct:
        ct_t = ct_data["timestamps"]
        ct_grip = ct_data["gripper_states"]
        ct_wrist = ct_data["wrist_positions"]
        ct_fps = float(ct_data.get("fps", 30.0))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), facecolor=C_SURFACE)
    fig.suptitle("Tracker Comparison: MediaPipe vs CoTracker3",
                 fontsize=14, color=C_TEXT, fontweight="bold", y=0.97)

    # ── Panel 1: Gripper State Agreement ──
    ax1 = axes[0]
    style_ax(ax1)
    ax1.set_title("Gripper State (OPEN = -1, CLOSED = +1)", fontsize=11,
                  color=C_TEXT, loc="left", pad=8)
    ax1.set_ylabel("Gripper Cmd", fontsize=9, color=C_TEXT_SEC)
    ax1.set_xlabel("Time (s)", fontsize=9, color=C_TEXT_SEC)
    ax1.set_yticks([-1, 0, 1])
    ax1.set_yticklabels(["OPEN", "", "CLOSED"], fontsize=8)

    legend_handles = []
    if has_mp:
        ax1.step(mp_t, mp_grip, where="post", color=C_MEDIAPIPE,
                 linewidth=2, alpha=0.9)
        legend_handles.append(Patch(facecolor=C_MEDIAPIPE, label="MediaPipe"))
    if has_ct:
        ax1.step(ct_t, ct_grip, where="post", color=C_COTRACKER,
                 linewidth=2, alpha=0.9, linestyle="--")
        legend_handles.append(Patch(facecolor=C_COTRACKER, label="CoTracker3"))

    if has_mp and has_ct:
        min_len = min(len(mp_grip), len(ct_grip))
        agree = np.sum(mp_grip[:min_len] == ct_grip[:min_len])
        pct = 100.0 * agree / min_len
        ax1.text(0.98, 0.92, f"Agreement: {pct:.1f}%",
                 transform=ax1.transAxes, ha="right", va="top",
                 fontsize=10, color=C_TEXT, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc=C_SURFACE, ec=C_GRID))

    ax1.legend(handles=legend_handles, loc="upper left", frameon=True,
               framealpha=0.9, edgecolor=C_GRID, fontsize=9)

    # ── Panel 2: Wrist Trajectory Smoothness (Jerk) ──
    ax2 = axes[1]
    style_ax(ax2)
    ax2.set_title("Wrist Jerk Magnitude (lower = smoother)", fontsize=11,
                  color=C_TEXT, loc="left", pad=8)
    ax2.set_ylabel("Jerk (norm)", fontsize=9, color=C_TEXT_SEC)
    ax2.set_xlabel("Time (s)", fontsize=9, color=C_TEXT_SEC)

    stats_lines = []
    if has_mp:
        mp_jerk = compute_jerk(mp_wrist, mp_fps)
        ax2.plot(mp_t, mp_jerk, color=C_MEDIAPIPE, linewidth=1.5, alpha=0.8)
        stats_lines.append(f"MediaPipe mean jerk: {np.mean(mp_jerk):.2f}")
    if has_ct:
        ct_jerk = compute_jerk(ct_wrist, ct_fps)
        ax2.plot(ct_t, ct_jerk, color=C_COTRACKER, linewidth=1.5,
                 alpha=0.8, linestyle="--")
        stats_lines.append(f"CoTracker3 mean jerk: {np.mean(ct_jerk):.2f}")

    if stats_lines:
        ax2.text(0.98, 0.92, "\n".join(stats_lines),
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=9, color=C_TEXT_SEC, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", fc=C_SURFACE, ec=C_GRID))

    # ── Panel 3: Gripper Transition Timeline ──
    ax3 = axes[2]
    style_ax(ax3)
    ax3.set_title("Gripper Transition Events", fontsize=11,
                  color=C_TEXT, loc="left", pad=8)
    ax3.set_xlabel("Time (s)", fontsize=9, color=C_TEXT_SEC)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["CoTracker3", "MediaPipe"], fontsize=9)
    ax3.set_ylim(-0.5, 1.5)

    if has_mp:
        mp_trans = find_transitions(mp_grip)
        for frame_idx, kind in mp_trans:
            t = mp_t[frame_idx] if frame_idx < len(mp_t) else frame_idx / mp_fps
            color = C_CLOSED if kind == "CLOSE" else C_OPEN
            ax3.plot(t, 1, marker="v" if kind == "CLOSE" else "^",
                     color=color, markersize=10, zorder=5)
            ax3.annotate(f"{kind}\nf{frame_idx}", (t, 1), textcoords="offset points",
                         xytext=(0, 12), ha="center", fontsize=7, color=C_TEXT_SEC)

    if has_ct:
        ct_trans = find_transitions(ct_grip)
        for frame_idx, kind in ct_trans:
            t = ct_t[frame_idx] if frame_idx < len(ct_t) else frame_idx / ct_fps
            color = C_CLOSED if kind == "CLOSE" else C_OPEN
            ax3.plot(t, 0, marker="v" if kind == "CLOSE" else "^",
                     color=color, markersize=10, zorder=5)
            ax3.annotate(f"{kind}\nf{frame_idx}", (t, 0), textcoords="offset points",
                         xytext=(0, -18), ha="center", fontsize=7, color=C_TEXT_SEC)

    if not has_ct:
        ax3.set_yticks([1])
        ax3.set_yticklabels(["MediaPipe"], fontsize=9)
        ax3.set_ylim(0.3, 1.7)
    if not has_mp:
        ax3.set_yticks([0])
        ax3.set_yticklabels(["CoTracker3"], fontsize=9)
        ax3.set_ylim(-0.7, 0.7)

    event_handles = [
        Patch(facecolor=C_CLOSED, label="CLOSE"),
        Patch(facecolor=C_OPEN, label="OPEN"),
    ]
    ax3.legend(handles=event_handles, loc="upper right", frameon=True,
               framealpha=0.9, edgecolor=C_GRID, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / "tracker_comparison.png"
    fig.savefig(str(out_path), dpi=150, facecolor=C_SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved comparison plot -> {out_path}")

    # Print summary
    print("\n=== Summary ===")
    if has_mp:
        mp_trans = find_transitions(mp_grip)
        print(f"MediaPipe:  {len(mp_grip)} frames, "
              f"{len(mp_trans)} transitions, "
              f"mean jerk = {np.mean(compute_jerk(mp_wrist, mp_fps)):.2f}")
    if has_ct:
        ct_trans = find_transitions(ct_grip)
        print(f"CoTracker3: {len(ct_grip)} frames, "
              f"{len(ct_trans)} transitions, "
              f"mean jerk = {np.mean(compute_jerk(ct_wrist, ct_fps)):.2f}")
    if has_mp and has_ct:
        min_len = min(len(mp_grip), len(ct_grip))
        agree = np.sum(mp_grip[:min_len] == ct_grip[:min_len])
        print(f"Agreement:  {agree}/{min_len} frames ({100.0 * agree / min_len:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare MediaPipe vs CoTracker3 trajectories")
    parser.add_argument("--mediapipe", type=str, default=None, help="MediaPipe trajectory .npz")
    parser.add_argument("--cotracker", type=str, default=None, help="CoTracker3 trajectory .npz")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory for plots")
    args = parser.parse_args()

    mp_data = load_trajectory(args.mediapipe) if args.mediapipe else None
    ct_data = load_trajectory(args.cotracker) if args.cotracker else None
    plot_comparison(mp_data, ct_data, args.output)
