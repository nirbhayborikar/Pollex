"""
End-to-End CoTracker3 Pipeline Runner & Visualizer
Runs dense point tracking on real phone videos, retargets to Franka Panda OSC actions,
executes the actions in the GlassLift (steel cylinder) RoboSuite simulation,
and compiles synchronized side-by-side verification videos.
"""

import os
import cv2
import argparse
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from src.video_processing.cotracker_tracker import CoTrackerTrajectoryExtractor
from src.retargeting.cotracker_to_panda import CoTrackerToPandaRetargeter
from src.simulation.libero_runner import PandaSimEnvironment


def run_single_video_pipeline(
    video_path: str,
    output_dir: str = "data/cotracker_comparisons",
    traj_dir: str = "data/cotracker_trajectories",
) -> Dict[str, Any]:
    """Runs tracking -> retargeting -> simulation rollout for a single video."""
    vpath = Path(video_path)
    stem = vpath.stem
    print(f"\n{'='*70}\n[>>>] Running CoTracker3 Pipeline on: {vpath.name}\n{'='*70}")

    t_start = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    traj_path = Path(traj_dir)
    traj_path.mkdir(parents=True, exist_ok=True)

    # 1. CoTracker3 Dense Tracking
    extractor = CoTrackerTrajectoryExtractor()
    traj_data = extractor.process_video(str(vpath), str(traj_path), save_annotated_video=True)
    traj_file = traj_path / f"{stem}_ct_trajectory.npz"

    # 2. Retargeting to 7D OSC Panda Actions
    retargeter = CoTrackerToPandaRetargeter()
    actions_file = traj_path / f"{stem}_ct_actions.npz"
    retarget_data = retargeter.retarget_trajectory(str(traj_file), str(actions_file))
    actions = retarget_data["actions"]
    human_frames = traj_data["rgb_frames"]
    T = len(actions)

    # 3. Simulate Rollout in GlassLift Environment
    print(f"[*] Replaying {T} action steps in 'GlassLift' simulation...")
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
    robot_eef_positions = []
    object_z_positions = []

    # Track glass initial and maximum Z height
    init_obj_z = None
    max_obj_z = -100.0

    for t in range(T):
        action = actions[t]
        obs, reward, done, info = sim.step(action)

        if obs["rgb"] is not None:
            sim_frames.append(obs["rgb"])
        if obs["robot_eef_pos"] is not None:
            robot_eef_positions.append(obs["robot_eef_pos"])

        # Track glass object Z coordinate
        try:
            glass_z = sim.env.get_object_z()
            if init_obj_z is None:
                init_obj_z = glass_z
            object_z_positions.append(glass_z)
            if glass_z > max_obj_z:
                max_obj_z = glass_z
        except Exception:
            pass

    sim.close()

    init_z = init_obj_z if init_obj_z is not None else 0.85
    final_z = object_z_positions[-1] if object_z_positions else init_z
    net_lift_height = max(0.0, max_obj_z - init_z)
    returned_to_table = abs(final_z - init_z) < 0.02 and final_z > 0.84
    success = (net_lift_height >= 0.05) and returned_to_table

    # 4. Generate Synchronized Side-by-Side Video
    sbs_video_path = out_path / f"{stem}_ct_comparison.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # 256 + 256 = 512 width, 256 height
    writer = cv2.VideoWriter(str(sbs_video_path), fourcc, 20.0, (512, 256))

    num_frames = min(len(sim_frames), len(human_frames))
    annotated_human_vid = traj_path / f"{stem}_ct_annotated.mp4"

    # Use annotated video if available, else raw
    annot_cap = None
    if annotated_human_vid.exists():
        annot_cap = cv2.VideoCapture(str(annotated_human_vid))

    for i in range(num_frames):
        sim_rgb = sim_frames[i]
        sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

        human_bgr = None
        if annot_cap is not None:
            ret, frame = annot_cap.read()
            if ret:
                human_bgr = cv2.resize(frame, (256, 256))

        if human_bgr is None:
            human_bgr = cv2.cvtColor(cv2.resize(human_frames[i], (256, 256)), cv2.COLOR_RGB2BGR)

        # Overlays
        cv2.putText(human_bgr, "Real Video (CoTracker3)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
        if success:
            sim_status = f"PICK & PLACE: {net_lift_height*100:.1f}cm [SUCCESS]"
            status_color = (0, 255, 0)
        elif net_lift_height >= 0.05:
            sim_status = f"LIFTED: {net_lift_height*100:.1f}cm"
            status_color = (0, 200, 255)
        else:
            sim_status = "REPLAY"
            status_color = (255, 255, 255)
        cv2.putText(sim_bgr, sim_status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, status_color, 1)

        sbs = np.hstack([human_bgr, sim_bgr])
        writer.write(sbs)

    if annot_cap is not None:
        annot_cap.release()
    writer.release()

    total_time = time.time() - t_start
    print(f"[+] Saved comparison video -> {sbs_video_path}")
    print(f"[+] Net Glass Lift: {net_lift_height*100:.1f} cm | Table Return: {returned_to_table} | Success: {success} | Pipeline runtime: {total_time:.1f}s")

    metrics = {
        "video": stem,
        "frames": T,
        "max_lift_cm": net_lift_height * 100.0,
        "final_z": final_z,
        "returned_to_table": returned_to_table,
        "success": success,
        "runtime_s": total_time,
        "comparison_video": str(sbs_video_path),
    }
    return metrics


def run_batch_pipeline(
    video_dir: str = "media",
    skip_videos: List[str] = ["Vid_9"],
    output_dir: str = "data/cotracker_comparisons",
    traj_dir: str = "data/cotracker_trajectories",
):
    """Executes pipeline across all valid videos, skipping specified ones."""
    vdir = Path(video_dir)
    all_videos = sorted(list(vdir.glob("Vid_*.mp4")))
    valid_videos = []
    for v in all_videos:
        if not any(skip in v.name for skip in skip_videos) and not v.name.endswith("_side_by_side.mp4"):
            valid_videos.append(v)

    print(f"[*] Found {len(valid_videos)} demonstration videos to process (excluding {skip_videos})")
    results = []

    for v in valid_videos:
        m = run_single_video_pipeline(str(v), output_dir=output_dir, traj_dir=traj_dir)
        results.append(m)

    # Print summary table
    print("\n" + "=" * 80)
    print("CoTracker3 Pipeline Benchmark Summary")
    print("=" * 80)
    print(f"{'Video':<10} | {'Frames':<8} | {'Max Lift (cm)':<14} | {'Table Return':<14} | {'Success':<8} | {'Time (s)':<8}")
    print("-" * 80)
    summary_lines = [
        "# CoTracker3 Retargeting Benchmark Summary\n",
        "| Video | Frames | Max Lift (cm) | Table Return | Success | Runtime (s) | Comparison Video |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ret_str = "YES" if r.get("returned_to_table", False) else "NO"
        succ_str = "PASS" if r["success"] else "FAIL"
        print(f"{r['video']:<10} | {r['frames']:<8} | {r['max_lift_cm']:<14.1f} | {ret_str:<14} | {succ_str:<8} | {r['runtime_s']:<8.1f}")
        summary_lines.append(f"| {r['video']} | {r['frames']} | {r['max_lift_cm']:.1f} | {ret_str} | **{succ_str}** | {r['runtime_s']:.1f}s | `{r['comparison_video']}` |")
    print("=" * 80)

    summary_file = Path(output_dir) / "benchmark_summary.md"
    summary_file.write_text("\n".join(summary_lines) + "\n")
    print(f"[+] Saved benchmark summary report -> {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end CoTracker3 pipeline")
    parser.add_argument("--video", type=str, default=None, help="Path to single video")
    parser.add_argument("--dir", "--input_dir", type=str, default="media", help="Directory of videos")
    parser.add_argument("--skip", nargs="+", default=["Vid_9"], help="Videos to skip")
    parser.add_argument("--out_dir", "--output_dir", type=str, default="data/cotracker_comparisons", help="Output directory for comparison videos")
    parser.add_argument("--traj_dir", type=str, default="data/cotracker_trajectories", help="Output directory for extracted trajectories")
    args = parser.parse_args()

    if args.video:
        run_single_video_pipeline(args.video, output_dir=args.out_dir, traj_dir=args.traj_dir)
    else:
        run_batch_pipeline(video_dir=args.dir, skip_videos=args.skip, output_dir=args.out_dir, traj_dir=args.traj_dir)
