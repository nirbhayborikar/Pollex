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

LIFT_SUCCESS_THRESHOLD_M = 0.03


def replay_actions_and_record(
    actions_npz_path: str,
    output_video_path: str,
    output_dataset_path: Optional[str] = None,
    task_name: str = "Lift",
    render_gui: bool = False,
    human_video_path: Optional[str] = None,
):
    """
    Executes actions inside the simulator and outputs synchronized comparison video.

        The function optionally saves:

        1. A synchronized side-by-side video:
               [Real Human Video | Simulated Panda]

        2. A paired NPZ dataset containing:
               simulation frames
               human frames
               robot actions
               actual robot end-effector positions

        For the Iteration-1 experiment, the function additionally records:

               actual end-effector positions
               actual tumbler positions, if exposed by PandaSimEnvironment
               contact state, if exposed by PandaSimEnvironment
               grasp state, if exposed by PandaSimEnvironment
               rewards
               executed actions

        It also creates metrics.txt next to the comparison video.
    """

    # 1. INPUT ACTION FILE
    actions_file = Path(actions_npz_path)
    if not actions_file.exists():
        raise FileNotFoundError(f"Actions file not found: {actions_file}")

    # 2. LOAD RETARGETED TRAJECTORY
    data = np.load(actions_file)
    actions = data["actions"]

    # Load human video frames: prefer external MP4 over embedded rgb_frames
    human_frames = None
    if human_video_path is not None:
        vid_path = Path(human_video_path)
        if not vid_path.exists():
            raise FileNotFoundError(f"Human video not found: {vid_path}")
        cap = cv2.VideoCapture(str(vid_path))
        h_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        human_frames = np.array(h_frames, dtype=np.uint8)
        print(f"[+] Loaded {len(human_frames)} frames from human video: {vid_path}")
    elif "rgb_frames" in data:
        human_frames = data["rgb_frames"]

    if "fps" in data:
        replay_fps = float(data["fps"])
    elif "timestamps" in data and len(data["timestamps"]) > 1:
        dt = np.median(np.diff(np.asarray(data["timestamps"], dtype=np.float64)))
        replay_fps = 1.0 / dt if np.isfinite(dt) and dt > 0 else 30.0
    else:
        replay_fps = 30.0

    if not np.isfinite(replay_fps) or replay_fps <= 0:
        replay_fps = 30.0

    T = len(actions)
    print(f"[*] Replaying {T} action steps in '{task_name}' environment...")
    print(f"[*] Replay / comparison FPS: {replay_fps:.2f}")

    # 3. CREATE PANDA / MUJOCO SIMULATION
    env = PandaSimEnvironment(
        env_name=task_name, has_renderer=render_gui, has_offscreen_renderer=True,
        camera_name="agentview", camera_height=256, camera_width=256,
    )

    # 4. RESET SIMULATION
    obs = env.reset()
    if isinstance(obs, dict):
        print("[*] Observation keys:", sorted(obs.keys()))

    sim_frames = []
    recorded_robot_states = []
    recorded_tumbler_positions = []
    recorded_tumbler_frames = []
    recorded_contacts = []
    recorded_grasps = []
    recorded_rewards = []
    recorded_actions = []
    recorded_robot_state_frames = []

    initial_eef_pos = None
    initial_tumbler_pos = None

    if isinstance(obs, dict):
        initial_eef_value = obs.get("robot_eef_pos", None)
        if initial_eef_value is not None:
            initial_eef_pos = np.asarray(initial_eef_value, dtype=np.float32).copy()

    # Tumbler observation key discovery
    tumbler_pos_key = None
    possible_tumbler_pos_keys = ("tumbler_pos", "glass_pos", "object_pos", "Glass_pos", "cube_pos")

    if isinstance(obs, dict):
        for candidate_key in possible_tumbler_pos_keys:
            if candidate_key in obs and obs[candidate_key] is not None:
                if np.asarray(obs[candidate_key]).size >= 3:
                    tumbler_pos_key = candidate_key
                    break

    if tumbler_pos_key is not None:
        initial_tumbler_pos = np.asarray(obs[tumbler_pos_key], dtype=np.float32).reshape(-1)[:3].copy()
        print(f"[+] Using tumbler position observation key: '{tumbler_pos_key}'")
        print(f"[*] Initial tumbler XYZ: {initial_tumbler_pos}")
    else:
        print("[!] PandaSimEnvironment does not currently expose a recognized tumbler XYZ observation.")
        print("[!] Tumbler height / lift metrics will remain UNAVAILABLE until the actual MuJoCo object position is exposed by libero_runner.py.")

    contact_signal_seen = False
    grasp_signal_seen = False
    contact_source_name = None
    grasp_source_name = None
    possible_contact_keys = ("tumbler_contact", "glass_contact", "object_contact", "contact", "has_contact")
    possible_grasp_keys = ("tumbler_grasped", "glass_grasped", "object_grasped", "grasped", "is_grasped", "grasp")

    # 5. REPLAY EVERY ROBOT ACTION
    for t in range(T):
        action = actions[t]
        obs, reward, done, info = env.step(action)

        recorded_actions.append(np.asarray(action, dtype=np.float32).copy())
        recorded_rewards.append(float(reward))

        # 6. RECORD SIMULATION RGB FRAME
        if isinstance(obs, dict) and obs.get("rgb", None) is not None:
            sim_frames.append(np.asarray(obs["rgb"], dtype=np.uint8).copy())

        # 7. RECORD ACTUAL PANDA END-EFFECTOR POSITION
        eef_pos = None
        if isinstance(obs, dict):
            eef_value = obs.get("robot_eef_pos", None)
            if eef_value is not None:
                eef_pos = np.asarray(eef_value, dtype=np.float32).reshape(-1)[:3].copy()
                recorded_robot_states.append(eef_pos)
                recorded_robot_state_frames.append(t)

        # 8. RECORD ACTUAL TUMBLER POSITION
        tumbler_pos = None
        if tumbler_pos_key is not None and isinstance(obs, dict) and obs.get(tumbler_pos_key, None) is not None:
            tumbler_pos = np.asarray(obs[tumbler_pos_key], dtype=np.float32).reshape(-1)[:3].copy()
            recorded_tumbler_positions.append(tumbler_pos)
            recorded_tumbler_frames.append(t)

        # 9. RECORD PHYSICAL CONTACT (must come from SIMULATOR, not gripper command)
        contact_value = None
        if isinstance(info, dict):
            for candidate_key in possible_contact_keys:
                if candidate_key in info:
                    contact_value = bool(info[candidate_key])
                    contact_source_name = f"info['{candidate_key}']"
                    contact_signal_seen = True
                    break

        if contact_value is None and isinstance(obs, dict):
            for candidate_key in possible_contact_keys:
                if candidate_key in obs:
                    contact_value = bool(obs[candidate_key])
                    contact_source_name = f"obs['{candidate_key}']"
                    contact_signal_seen = True
                    break

        # NaN = signal unavailable, False = simulator says no contact
        recorded_contacts.append(np.nan if contact_value is None else float(contact_value))

        # 10. RECORD PHYSICAL GRASP (must be simulation observation, not action vector)
        grasp_value = None
        if isinstance(info, dict):
            for candidate_key in possible_grasp_keys:
                if candidate_key in info:
                    grasp_value = bool(info[candidate_key])
                    grasp_source_name = f"info['{candidate_key}']"
                    grasp_signal_seen = True
                    break

        if grasp_value is None and isinstance(obs, dict):
            for candidate_key in possible_grasp_keys:
                if candidate_key in obs:
                    grasp_value = bool(obs[candidate_key])
                    grasp_source_name = f"obs['{candidate_key}']"
                    grasp_signal_seen = True
                    break

        recorded_grasps.append(np.nan if grasp_value is None else float(grasp_value))

        # 11. ITERATION-1 PER-FRAME REPLAY LOG
        eef_text = np.array2string(eef_pos, precision=4) if eef_pos is not None else "N/A"
        tumbler_text = np.array2string(tumbler_pos, precision=4) if tumbler_pos is not None else "N/A"
        contact_text = str(contact_value) if contact_value is not None else "N/A"
        grasp_text = str(grasp_value) if grasp_value is not None else "N/A"

        print(
            f"[REPLAY] frame={t:04d} action_xyz=[{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}] "
            f"grip={action[6]:+.1f} eef={eef_text} tumbler={tumbler_text} "
            f"contact={contact_text} grasp={grasp_text} reward={float(reward):.4f}"
        )

        # 12. OPTIONAL LIVE RENDERING
        if render_gui:
            env.render()

    env.close()
    print(f"[+] Replay finished. Captured {len(sim_frames)} simulation frames.")

    # 13. CONVERT RECORDED PHYSICAL DATA TO NUMPY ARRAYS
    executed_actions = np.asarray(recorded_actions, dtype=np.float32)
    rewards = np.asarray(recorded_rewards, dtype=np.float32)
    eef_positions = np.asarray(recorded_robot_states, dtype=np.float32)
    eef_position_frames = np.asarray(recorded_robot_state_frames, dtype=np.int32)
    tumbler_positions = np.asarray(recorded_tumbler_positions, dtype=np.float32)
    tumbler_position_frames = np.asarray(recorded_tumbler_frames, dtype=np.int32)
    contact_flags = np.asarray(recorded_contacts, dtype=np.float32)
    grasp_flags = np.asarray(recorded_grasps, dtype=np.float32)

    # 14. CALCULATE ACTUAL PANDA EEF MOTION
    final_eef_pos = None
    max_eef_dx = max_eef_dy = max_eef_dz = np.nan
    min_eef_dx = max_signed_eef_dx = np.nan
    min_eef_dy = max_signed_eef_dy = np.nan
    min_eef_dz = max_signed_eef_dz = np.nan

    if len(eef_positions) > 0:
        if initial_eef_pos is None:
            initial_eef_pos = eef_positions[0].copy()
        final_eef_pos = eef_positions[-1].copy()
        eef_displacements = eef_positions - initial_eef_pos[None, :]
        max_eef_dx = float(np.max(np.abs(eef_displacements[:, 0])))
        max_eef_dy = float(np.max(np.abs(eef_displacements[:, 1])))
        max_eef_dz = float(np.max(np.abs(eef_displacements[:, 2])))
        min_eef_dx = float(np.min(eef_displacements[:, 0]))
        max_signed_eef_dx = float(np.max(eef_displacements[:, 0]))
        min_eef_dy = float(np.min(eef_displacements[:, 1]))
        max_signed_eef_dy = float(np.max(eef_displacements[:, 1]))
        min_eef_dz = float(np.min(eef_displacements[:, 2]))
        max_signed_eef_dz = float(np.max(eef_displacements[:, 2]))

    # 15. CALCULATE ACTUAL TUMBLER LIFT
    initial_tumbler_z = maximum_tumbler_z = maximum_tumbler_lift = np.nan
    lift_success = None

    if len(tumbler_positions) > 0:
        if initial_tumbler_pos is None:
            initial_tumbler_pos = tumbler_positions[0].copy()
        initial_tumbler_z = float(initial_tumbler_pos[2])
        maximum_tumbler_z = float(np.max(tumbler_positions[:, 2]))
        maximum_tumbler_lift = float(maximum_tumbler_z - initial_tumbler_z)
        lift_success = bool(maximum_tumbler_lift >= LIFT_SUCCESS_THRESHOLD_M)

    # 16. DETERMINE CONTACT / GRASP OUTCOMES
    if contact_signal_seen:
        valid_contacts = contact_flags[np.isfinite(contact_flags)]
        contact_occurred = bool(np.any(valid_contacts > 0.5))
    else:
        contact_occurred = None

    if grasp_signal_seen:
        valid_grasps = grasp_flags[np.isfinite(grasp_flags)]
        grasp_occurred = bool(np.any(valid_grasps > 0.5))
    else:
        grasp_occurred = None

    # 17. SAVE ITERATION-1 METRICS.TXT
    out_vid_path = Path(output_video_path)
    out_vid_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = out_vid_path.parent / "metrics.txt"

    def yes_no_unavailable(value):
        if value is None:
            return "UNAVAILABLE"
        return "YES" if value else "NO"

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("ITERATION 1: RAW COTRACKER3 BASELINE\n")
        f.write("=" * 68 + "\n\n")
        f.write("INPUT / REPLAY\n")
        f.write("-" * 68 + "\n")
        f.write(f"Actions file:              {actions_file}\n")
        f.write(f"Task:                      {task_name}\n")
        f.write(f"Action steps:              {T}\n")
        f.write(f"Comparison FPS:            {replay_fps:.2f}\n\n")

        f.write("ROBOT END-EFFECTOR\n")
        f.write("-" * 68 + "\n")
        if initial_eef_pos is not None:
            f.write(f"Initial EEF XYZ [m]:       {np.array2string(initial_eef_pos, precision=6)}\n")
            f.write(f"Final EEF XYZ [m]:         {np.array2string(final_eef_pos, precision=6)}\n")
            f.write(f"Maximum |dX|:             {max_eef_dx * 100.0:.2f} cm\n")
            f.write(f"Maximum |dY|:             {max_eef_dy * 100.0:.2f} cm\n")
            f.write(f"Maximum |dZ|:             {max_eef_dz * 100.0:.2f} cm\n")
            f.write(f"Signed dX range:           {min_eef_dx * 100.0:+.2f} to {max_signed_eef_dx * 100.0:+.2f} cm\n")
            f.write(f"Signed dY range:           {min_eef_dy * 100.0:+.2f} to {max_signed_eef_dy * 100.0:+.2f} cm\n")
            f.write(f"Signed dZ range:           {min_eef_dz * 100.0:+.2f} to {max_signed_eef_dz * 100.0:+.2f} cm\n")
        else:
            f.write("EEF state:                 UNAVAILABLE\n")
        f.write("\n")

        f.write("TUMBLER / OBJECT\n")
        f.write("-" * 68 + "\n")
        if initial_tumbler_pos is not None:
            f.write(f"Tumbler observation key:   {tumbler_pos_key}\n")
            f.write(f"Initial tumbler XYZ [m]:   {np.array2string(initial_tumbler_pos, precision=6)}\n")
            f.write(f"Initial tumbler Z:         {initial_tumbler_z:.6f} m\n")
            f.write(f"Maximum tumbler Z:         {maximum_tumbler_z:.6f} m\n")
            f.write(f"Maximum tumbler lift:      {maximum_tumbler_lift * 100.0:.2f} cm\n")
        else:
            f.write("Tumbler position:           UNAVAILABLE\n")
            f.write("Maximum tumbler lift:       UNAVAILABLE\n")
        f.write("\n")

        f.write("PHYSICAL OUTCOME\n")
        f.write("-" * 68 + "\n")
        f.write(f"Contact signal source:      {contact_source_name or 'UNAVAILABLE'}\n")
        f.write(f"Contact occurred:           {yes_no_unavailable(contact_occurred)}\n")
        f.write(f"Grasp signal source:        {grasp_source_name or 'UNAVAILABLE'}\n")
        f.write(f"Grasp occurred:             {yes_no_unavailable(grasp_occurred)}\n")
        f.write(f"Lift threshold:             {LIFT_SUCCESS_THRESHOLD_M * 100.0:.1f} cm\n")
        f.write(f"Lift success:               {yes_no_unavailable(lift_success)}\n\n")

        f.write("RESULT\n")
        f.write("-" * 68 + "\n")
        if lift_success is True:
            f.write("Measured result: physical lift threshold reached.\n")
        elif lift_success is False:
            f.write("Measured result: physical lift threshold was not reached.\n")
        else:
            f.write("Measured result: lift could not be evaluated because the simulator did not expose the tumbler XYZ position.\n")
        f.write("\nNo qualitative conclusion such as 'arm stationary' is hard-coded. Interpret Iteration 1 from the measured EEF, object, contact, grasp, and lift values above.\n")

    print(f"[+] Saved Iteration-1 metrics -> {metrics_path}")

    # 18. SAVE PAIRED EPISODE FOR LEROBOT TRAINING
    if output_dataset_path:
        out_data_path = Path(output_dataset_path)
        out_data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_data_path,
            sim_frames=np.array(sim_frames, dtype=np.uint8),
            actions=actions[:len(sim_frames)],
            human_frames=human_frames[:len(sim_frames)] if human_frames is not None else None,
            robot_eef_pos=eef_positions,
            robot_eef_frames=eef_position_frames,
            tumbler_positions=tumbler_positions,
            tumbler_position_frames=tumbler_position_frames,
            contact_flags=contact_flags,
            grasp_flags=grasp_flags,
            rewards=rewards,
            executed_actions=executed_actions,
            initial_eef_pos=initial_eef_pos if initial_eef_pos is not None else np.full(3, np.nan, dtype=np.float32),
            initial_tumbler_pos=initial_tumbler_pos if initial_tumbler_pos is not None else np.full(3, np.nan, dtype=np.float32),
            max_tumbler_lift=np.float32(maximum_tumbler_lift),
            lift_success=np.int8(-1 if lift_success is None else int(lift_success)),
            replay_fps=np.float32(replay_fps),
        )
        print(f"[+] Saved paired trajectory dataset -> {out_data_path}")

    # 19. GENERATE SIDE-BY-SIDE COMPARISON VIDEO
    out_vid_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(out_vid_path), fourcc, replay_fps, (512, 256))

    if not video_writer.isOpened():
        raise RuntimeError(f"Could not create comparison video: {out_vid_path}")

    num_frames = min(len(sim_frames), len(human_frames) if human_frames is not None else len(sim_frames))

    for i in range(num_frames):
        sim_rgb = sim_frames[i]
        sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

        if human_frames is not None:
            human_rgb = cv2.resize(human_frames[i], (256, 256), interpolation=cv2.INTER_AREA)
            human_bgr = cv2.cvtColor(human_rgb, cv2.COLOR_RGB2BGR)
        else:
            human_bgr = np.zeros_like(sim_bgr)
            cv2.putText(human_bgr, "Human Video", (30, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        video_writer.write(np.hstack([human_bgr, sim_bgr]))

    video_writer.release()
    print(f"[+] Saved synchronized side-by-side video -> {out_vid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay retargeted actions in simulation")
    parser.add_argument("--actions", type=str, required=True, help="Path to retargeted actions.npz")
    parser.add_argument("--out_video", type=str, default="media/side_by_side_demo.mp4", help="Output comparison video")
    parser.add_argument("--out_dataset", type=str, default=None, help="Output paired dataset npz")
    parser.add_argument("--task", type=str, default="Lift", help="Simulation task name")
    parser.add_argument("--gui", action="store_true", help="Enable interactive window viewer")
    parser.add_argument("--human_video", type=str, default=None, help="MP4 video for left side of comparison")
    args = parser.parse_args()

    replay_actions_and_record(
        actions_npz_path=args.actions, output_video_path=args.out_video,
        output_dataset_path=args.out_dataset, task_name=args.task, render_gui=args.gui,
        human_video_path=args.human_video,
    )

"""
python3 -m src.simulation.replay_trajectory \
    --actions datasets/cotracker_iterations/iteration_01_raw/cup_lifting_01_ct_actions.npz \
    --out_video datasets/cotracker_iterations/iteration_01_raw/cup_lifting_01_iteration_1.mp4 \
    --out_dataset datasets/cotracker_iterations/iteration_01_raw/cup_lifting_01_replay.npz \
    --task GlassLift

"""
# to see the simulation as well
"""

python3 -m src.simulation.replay_trajectory \
    --actions datasets/cotracker_iterations/iteration_01_raw/cup_lifting_01_ct_actions.npz \
    --out_video datasets/cotracker_iterations/iteration_01_raw/cup_lifting_01_iteration_1.mp4 \
    --out_dataset datasets/cotracker_iterations/iteration_01_raw/cup_lifting_01_replay.npz \
    --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
    --task GlassLift \
    --gui

"""