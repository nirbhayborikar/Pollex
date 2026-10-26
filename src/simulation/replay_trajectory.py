"""
Trajectory Replayer & Side-by-Side Visualizer
Executes retargeted 7D actions in the simulation environment,
records robot rollouts, and compiles a synchronized side-by-side video:
[Left: Real Hand Video | Right: Simulated Panda Robot].
"""

import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

from src.simulation.libero_runner import PandaSimEnvironment


def replay_actions_and_record(
    actions_npz_path: str,
    output_video_path: str,
    output_dataset_path: Optional[str] = None,
    task_name: str = "Lift",
    render_gui: bool = False,
):
    """
    Executes actions inside the simulator and outputs synchronized comparison video.
    """
    actions_file = Path(actions_npz_path)
    if not actions_file.exists():
        raise FileNotFoundError(f"Actions file not found: {actions_file}")

    data = np.load(actions_file)
    actions = data["actions"]  # Shape: (T, 7)
    human_frames = data["rgb_frames"] if "rgb_frames" in data else None
    T = len(actions)

    print(f"[*] Replaying {T} action steps in '{task_name}' environment...")

    env = PandaSimEnvironment(
        env_name=task_name,
        has_renderer=render_gui,
        has_offscreen_renderer=True,
        camera_name="agentview",
        camera_height=256,
        camera_width=256,
    )

    obs = env.reset()
    sim_frames = []
    recorded_robot_states = []

    for t in range(T):
        action = actions[t]
        obs, reward, done, info = env.step(action)

        if obs["rgb"] is not None:
            sim_frames.append(obs["rgb"])
        if obs["robot_eef_pos"] is not None:
            recorded_robot_states.append(obs["robot_eef_pos"])

        if render_gui:
            env.render()

    env.close()
    print(f"[+] Replay finished. Captured {len(sim_frames)} simulation frames.")

    # Save paired episode for LeRobot training
    if output_dataset_path:
        out_data_path = Path(output_dataset_path)
        out_data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_data_path,
            sim_frames=np.array(sim_frames, dtype=np.uint8),
            actions=actions[:len(sim_frames)],
            human_frames=human_frames[:len(sim_frames)] if human_frames is not None else None,
            robot_eef_pos=np.array(recorded_robot_states, dtype=np.float32),
        )
        print(f"[+] Saved paired trajectory dataset -> {out_data_path}")

    # Generate Side-by-Side Comparison Video
    out_vid_path = Path(output_video_path)
    out_vid_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Side-by-side width = 256 + 256 = 512, height = 256
    video_writer = cv2.VideoWriter(str(out_vid_path), fourcc, 20.0, (512, 256))

    num_frames = min(len(sim_frames), len(human_frames) if human_frames is not None else len(sim_frames))

    for i in range(num_frames):
        sim_rgb = sim_frames[i]
        sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

        if human_frames is not None:
            human_rgb = human_frames[i]
            human_bgr = cv2.cvtColor(human_rgb, cv2.COLOR_RGB2BGR)
        else:
            # Placeholder black frame if raw human frames were not embedded
            human_bgr = np.zeros_like(sim_bgr)
            cv2.putText(human_bgr, "Human Video", (30, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Annotate labels
        cv2.putText(human_bgr, "Real Egocentric Hand", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(sim_bgr, "Simulated Panda Robot", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Concatenate horizontally: [Human Video | Panda Robot]
        side_by_side = np.hstack([human_bgr, sim_bgr])
        video_writer.write(side_by_side)

    video_writer.release()
    print(f"[+] Saved synchronized side-by-side video -> {out_vid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay retargeted actions in simulation")
    parser.add_argument("--actions", type=str, required=True, help="Path to retargeted actions.npz")
    parser.add_argument("--out_video", type=str, default="media/side_by_side_demo.mp4", help="Output comparison video")
    parser.add_argument("--out_dataset", type=str, default=None, help="Output paired dataset npz")
    parser.add_argument("--task", type=str, default="Lift", help="Simulation task name")
    parser.add_argument("--gui", action="store_true", help="Enable interactive macOS window viewer")
    args = parser.parse_args()

    replay_actions_and_record(
        actions_npz_path=args.actions,
        output_video_path=args.out_video,
        output_dataset_path=args.out_dataset,
        task_name=args.task,
        render_gui=args.gui,
    )
