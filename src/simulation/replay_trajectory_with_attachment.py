"""
Trajectory Replayer with Object-Relative Geometry and Co-Motion Gating
======================================================================

Action:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]

Controller:
    - Object-relative finger-pad X/Z feedback
    - Error-dependent feedforward blending
    - Geometry-triggered grasp with demonstrated-frame timeout
    - Co-motion latch gates the post-grasp transport/lift phase
    - Physical grasp confirmation disables pre-grasp feedback
"""
"""
Trajectory replay with object-relative grasp control and attachment-gated lift.

Action:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.simulation.libero_runner import PandaSimEnvironment


LIFT_SUCCESS_THRESHOLD_M = 0.03
DEFAULT_REPLAY_FPS = 30.0


def valid_xyz(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    xyz = np.asarray(value, dtype=np.float32).reshape(-1)
    if xyz.size < 3:
        return None
    xyz = xyz[:3].copy()
    return xyz if np.all(np.isfinite(xyz)) else None


def yes_no(value) -> str:
    return "YES" if bool(value) else "NO"


def frame_or_na(frame: int) -> str:
    return str(frame) if frame >= 0 else "N/A"


def finite_or_na(value, scale=1.0, precision=3, signed=False) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return format(value * scale, f"+.{precision}f" if signed else f".{precision}f")


def control_stats(errors, corrections):
    finite = errors[np.isfinite(errors)]
    return (
        float(finite[0]) if len(finite) else np.nan,
        float(finite[-1]) if len(finite) else np.nan,
        float(np.min(np.abs(finite))) if len(finite) else np.nan,
        float(np.max(np.abs(corrections))) if len(corrections) else np.nan,
    )


def calculate_replay_fps(data):
    if "fps" in data:
        fps = float(np.asarray(data["fps"]).item())
        if np.isfinite(fps) and fps > 0:
            return fps
    if "timestamps" in data:
        differences = np.diff(np.asarray(data["timestamps"], dtype=np.float64))
        differences = differences[np.isfinite(differences) & (differences > 0)]
        if len(differences):
            return 1.0 / float(np.median(differences))
    return DEFAULT_REPLAY_FPS


def load_human_video(video_path):
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Human video not found: {path}")
    capture, frames = cv2.VideoCapture(str(path)), []
    if not capture.isOpened():
        raise RuntimeError(f"Could not open human video: {path}")
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {path}")
    return np.asarray(frames, dtype=np.uint8)


def create_comparison_video(sim_frames, human_frames, output_video_path, replay_fps):
    output = Path(output_video_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), replay_fps, (512, 256))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output}")
    count = min(len(sim_frames), len(human_frames)) if human_frames is not None else len(sim_frames)
    for index in range(count):
        sim = cv2.cvtColor(cv2.resize(sim_frames[index], (256, 256)), cv2.COLOR_RGB2BGR)
        human = cv2.cvtColor(cv2.resize(human_frames[index], (256, 256)), cv2.COLOR_RGB2BGR) if human_frames is not None else np.zeros_like(sim)
        writer.write(np.hstack([human, sim]))
    writer.release()
    print(f"[+] Saved video: {output}")


def replay_actions_and_record(
    actions_npz_path,
    output_video_path,
    output_dataset_path=None,
    task_name="Lift",
    render_gui=False,
    human_video_path=None,
    closed_loop_pregrasp=False,
    target_pad_x_offset=-0.017,
    pregrasp_kp_x=4.0,
    pregrasp_max_correction=0.07,
    pregrasp_deadband_m=0.002,
    target_pad_z_offset=0.015,
    pregrasp_kp_z=3.0,
    pregrasp_max_z_correction=0.04,
    pregrasp_z_deadband_m=0.002,
    pregrasp_control_start_frac=0.45,
    grasp_ready_x_tolerance_m=0.003,
    grasp_ready_z_tolerance_m=0.002,
    grasp_ready_consecutive_frames=3,
    grasp_control_hold_frames=6,
):
    actions_file = Path(actions_npz_path)
    if not actions_file.exists():
        raise FileNotFoundError(actions_file)

    with np.load(actions_file, allow_pickle=False) as data:
        source_actions = np.asarray(data["actions"], dtype=np.float32).copy()
        if source_actions.ndim != 2 or source_actions.shape[1] != 7:
            raise ValueError(f"Expected (T,7), got {source_actions.shape}")

        replay_fps = calculate_replay_fps(data)
        entry_frame = int(np.asarray(data["entry_frame"]).item()) if "entry_frame" in data else 0
        attachment_score = np.asarray(data["attachment_score"], dtype=np.float32).copy() if "attachment_score" in data else np.empty(0, dtype=np.float32)
        attachment_latch_frame = int(np.asarray(data["attachment_latch_frame"]).item()) if "attachment_latch_frame" in data else -1
        attachment_threshold = float(np.asarray(data["attachment_threshold"]).item()) if "attachment_threshold" in data else np.nan
        stored_frames = np.asarray(data["rgb_frames"], dtype=np.uint8).copy() if human_video_path is None and "rgb_frames" in data else None

    number_of_steps = len(source_actions)
    if len(attachment_score) not in (0, number_of_steps):
        raise ValueError("attachment_score length does not match actions.")

    human_frames = load_human_video(human_video_path) if human_video_path else stored_frames
    grasp_indices = np.where(source_actions[:, 6] > 0)[0]
    demonstrated_grasp_frame = int(grasp_indices[0]) if len(grasp_indices) else number_of_steps
    entry_frame = int(np.clip(entry_frame, 0, max(0, demonstrated_grasp_frame - 1)))
    control_start_frame = entry_frame + int(pregrasp_control_start_frac * max(1, demonstrated_grasp_frame - entry_frame))
    control_start_frame = int(np.clip(control_start_frame, entry_frame, max(entry_frame, demonstrated_grasp_frame - 1)))
    control_end_frame = min(number_of_steps, demonstrated_grasp_frame + grasp_control_hold_frames)

    env = PandaSimEnvironment(
        env_name=task_name,
        has_renderer=render_gui,
        has_offscreen_renderer=True,
        camera_name="agentview",
        camera_height=256,
        camera_width=256,
    )

    obs = env.reset()
    initial_tumbler_pos = valid_xyz(obs.get("tumbler_pos"))
    current_tumbler_pos = initial_tumbler_pos.copy() if initial_tumbler_pos is not None else None
    current_left_pad_pos = valid_xyz(obs.get("left_finger_pad_pos"))
    current_right_pad_pos = valid_xyz(obs.get("right_finger_pad_pos"))

    keys = [
        "sim_frames", "eef_positions", "eef_frames", "tumbler_positions", "tumbler_frames",
        "left_pads", "right_pads", "contacts", "grasps", "rewards", "executed_actions",
        "x_errors", "x_corrections", "z_errors", "z_corrections", "pad_x_offsets",
        "pad_z_offsets", "x_blends", "z_blends", "ready_counts", "attachment_values",
        "human_transport_ready", "lift_allowed",
    ]
    logs = {key: [] for key in keys}

    ready_count = 0
    grasp_triggered = False
    grasp_trigger_frame = -1
    grasp_trigger_reason = "none"
    physical_grasp_latched = False
    physical_grasp_latch_frame = -1
    lift_release_frame = -1

    try:
        for frame_index in range(number_of_steps):
            source_action = source_actions[frame_index].copy()
            executed_action = source_action.copy()

            x_error = z_error = np.nan
            x_correction = z_correction = 0.0
            pad_x_offset = pad_z_offset = np.nan
            x_blend = z_blend = 1.0
            x_ready = z_ready = False
            control_active = False

            attachment = float(attachment_score[frame_index]) if frame_index < len(attachment_score) else np.nan
            human_transport_ready = attachment_latch_frame < 0 or frame_index >= attachment_latch_frame
            lift_allowed = physical_grasp_latched and human_transport_ready

            in_control = closed_loop_pregrasp and not physical_grasp_latched and control_start_frame <= frame_index < control_end_frame
            geometry_available = current_tumbler_pos is not None and current_left_pad_pos is not None and current_right_pad_pos is not None

            if in_control and geometry_available:
                control_active = True
                pad_center = 0.5 * (current_left_pad_pos + current_right_pad_pos)
                pad_x_offset = float(pad_center[0] - current_tumbler_pos[0])
                pad_z_offset = float(pad_center[2] - current_tumbler_pos[2])
                x_error = float(target_pad_x_offset - pad_x_offset)
                z_error = float(target_pad_z_offset - pad_z_offset)

                if abs(x_error) > pregrasp_deadband_m:
                    x_correction = float(np.clip(pregrasp_kp_x * x_error, -pregrasp_max_correction, pregrasp_max_correction))
                if abs(z_error) > pregrasp_z_deadband_m:
                    z_correction = float(np.clip(pregrasp_kp_z * z_error, -pregrasp_max_z_correction, pregrasp_max_z_correction))

                x_ready = abs(x_error) <= grasp_ready_x_tolerance_m
                z_ready = abs(z_error) <= grasp_ready_z_tolerance_m
                ready_count = ready_count + 1 if x_ready and z_ready else 0

                if not grasp_triggered and ready_count >= grasp_ready_consecutive_frames:
                    grasp_triggered, grasp_trigger_frame, grasp_trigger_reason = True, frame_index, "geometry"
                    control_end_frame = min(number_of_steps, frame_index + grasp_control_hold_frames + 1)
                    print(f"[GRASP] Geometry close at frame {frame_index}.")

                if not grasp_triggered and frame_index >= demonstrated_grasp_frame:
                    grasp_triggered, grasp_trigger_frame, grasp_trigger_reason = True, frame_index, "timeout"
                    control_end_frame = min(number_of_steps, frame_index + grasp_control_hold_frames + 1)
                    print(f"[GRASP] Timeout close at frame {frame_index}.")

                if grasp_triggered:
                    executed_action[0], executed_action[2], executed_action[6] = np.clip(x_correction, -1, 1), np.clip(z_correction, -1, 1), 1.0
                else:
                    x_blend = float(np.clip(abs(x_error) / (3.0 * grasp_ready_x_tolerance_m), 0.0, 1.0))
                    z_blend = float(np.clip(abs(z_error) / (3.0 * grasp_ready_z_tolerance_m), 0.0, 1.0))
                    executed_action[0] = float(np.clip(x_blend * source_action[0] + x_correction, -1.0, 1.0))
                    executed_action[2] = float(np.clip(z_blend * source_action[2] + z_correction, -1.0, 1.0))

            elif grasp_triggered and not physical_grasp_latched:
                executed_action[:6], executed_action[6] = 0.0, 1.0

            if physical_grasp_latched and not human_transport_ready:
                executed_action[:6], executed_action[6] = 0.0, 1.0
            elif lift_allowed:
                if lift_release_frame < 0:
                    lift_release_frame = frame_index
                    print(f"[ATTACHMENT] Lift released at frame {frame_index}.")
                executed_action = source_action.copy()
                if source_action[6] > 0:
                    executed_action[6] = 1.0

            obs, reward, done, info = env.step(executed_action)
            eef_pos = valid_xyz(obs.get("robot_eef_pos"))
            tumbler_pos = valid_xyz(obs.get("tumbler_pos"))
            left_pad_pos = valid_xyz(obs.get("left_finger_pad_pos"))
            right_pad_pos = valid_xyz(obs.get("right_finger_pad_pos"))

            if eef_pos is not None:
                logs["eef_positions"].append(eef_pos.copy())
                logs["eef_frames"].append(frame_index)
            if tumbler_pos is not None:
                current_tumbler_pos = tumbler_pos.copy()
                logs["tumbler_positions"].append(tumbler_pos.copy())
                logs["tumbler_frames"].append(frame_index)
            if left_pad_pos is not None:
                current_left_pad_pos = left_pad_pos.copy()
            if right_pad_pos is not None:
                current_right_pad_pos = right_pad_pos.copy()

            logs["left_pads"].append(left_pad_pos.copy() if left_pad_pos is not None else np.full(3, np.nan))
            logs["right_pads"].append(right_pad_pos.copy() if right_pad_pos is not None else np.full(3, np.nan))

            contact = float(obs.get("tumbler_contact", np.nan))
            grasp = float(obs.get("tumbler_grasped", np.nan))

            if not physical_grasp_latched and np.isfinite(grasp) and grasp > 0.5:
                physical_grasp_latched = True
                physical_grasp_latch_frame = frame_index
                print(f"[CONTROL] Physical grasp at frame {frame_index}.")

            lift_allowed = physical_grasp_latched and human_transport_ready

            logs["contacts"].append(contact)
            logs["grasps"].append(grasp)
            logs["rewards"].append(float(reward))
            logs["executed_actions"].append(executed_action.copy())
            logs["x_errors"].append(x_error)
            logs["x_corrections"].append(x_correction)
            logs["z_errors"].append(z_error)
            logs["z_corrections"].append(z_correction)
            logs["pad_x_offsets"].append(pad_x_offset)
            logs["pad_z_offsets"].append(pad_z_offset)
            logs["x_blends"].append(x_blend)
            logs["z_blends"].append(z_blend)
            logs["ready_counts"].append(ready_count)
            logs["attachment_values"].append(attachment)
            logs["human_transport_ready"].append(human_transport_ready)
            logs["lift_allowed"].append(lift_allowed)

            rgb = obs.get("rgb")
            if rgb is not None:
                logs["sim_frames"].append(np.asarray(rgb, dtype=np.uint8).copy())

            if physical_grasp_latched and not human_transport_ready:
                phase = "WAIT-ATTACHMENT"
            elif lift_allowed:
                phase = "TRANSPORT"
            elif control_active and grasp_triggered:
                phase = "GRASP-HOLD"
            elif control_active:
                phase = "CL-APPROACH"
            else:
                phase = "SOURCE"

            attachment_text = f"{attachment:.2f}" if np.isfinite(attachment) else "N/A"
            print(
                f"[REPLAY] frame={frame_index:04d} phase={phase} "
                f"attachment={attachment_text} "
                f"human_transport={int(human_transport_ready)} "
                f"physical_grasp={int(physical_grasp_latched)} "
                f"lift_allowed={int(lift_allowed)} "
                f"x_ready={int(x_ready)} z_ready={int(z_ready)} "
                f"cmd_x={executed_action[0]:+.3f} "
                f"cmd_z={executed_action[2]:+.3f} "
                f"grip={executed_action[6]:+.1f} "
                f"contact={contact} grasp={grasp}"
            )

            if render_gui:
                env.render()
            if done:
                print(f"[*] done=True at frame {frame_index}; continuing.")

    finally:
        env.close()

    arrays = {
        key: np.asarray(value, dtype=np.uint8 if key == "sim_frames" else np.float32)
        for key, value in logs.items()
    }

    contacts, grasps = arrays["contacts"], arrays["grasps"]
    contact_indices, grasp_indices_actual = np.where(contacts > 0.5)[0], np.where(grasps > 0.5)[0]
    first_contact = int(contact_indices[0]) if len(contact_indices) else -1
    first_grasp = int(grasp_indices_actual[0]) if len(grasp_indices_actual) else -1

    maximum_lift, lift_success = np.nan, None
    tumbler_positions = arrays["tumbler_positions"]
    if len(tumbler_positions) and initial_tumbler_pos is not None:
        maximum_lift = float(np.max(tumbler_positions[:, 2]) - initial_tumbler_pos[2])
        lift_success = maximum_lift >= LIFT_SUCCESS_THRESHOLD_M

    _, _, minimum_x_error, _ = control_stats(arrays["x_errors"], arrays["x_corrections"])
    _, _, minimum_z_error, _ = control_stats(arrays["z_errors"], arrays["z_corrections"])

    output_video = Path(output_video_path)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_video.parent / "metrics.txt"

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("REPLAY METRICS\n" + "=" * 68 + "\n\n")
        file.write("OBJECT-RELATIVE GRASP + ATTACHMENT-GATED LIFT\n" + "-" * 68 + "\n")
        file.write(f"Demonstrated grasp frame:  {demonstrated_grasp_frame}\n")
        file.write(f"Grasp trigger frame:       {frame_or_na(grasp_trigger_frame)}\n")
        file.write(f"Grasp trigger reason:      {grasp_trigger_reason}\n")
        file.write(f"Physical grasp frame:      {frame_or_na(physical_grasp_latch_frame)}\n")
        file.write(f"Attachment latch frame:    {frame_or_na(attachment_latch_frame)}\n")
        file.write(f"Attachment threshold:      {finite_or_na(attachment_threshold, 1, 2)}\n")
        file.write(f"Lift release frame:        {frame_or_na(lift_release_frame)}\n")
        file.write(f"Minimum |X error|:         {finite_or_na(minimum_x_error, 100, 2)} cm\n")
        file.write(f"Minimum |Z error|:         {finite_or_na(minimum_z_error, 100, 2)} cm\n\n")
        file.write("PHYSICAL OUTCOME\n" + "-" * 68 + "\n")
        file.write(f"Contact occurred:          {yes_no(len(contact_indices))}\n")
        file.write(f"First contact frame:       {frame_or_na(first_contact)}\n")
        file.write(f"Grasp occurred:            {yes_no(len(grasp_indices_actual))}\n")
        file.write(f"First grasp frame:         {frame_or_na(first_grasp)}\n")
        file.write(f"Maximum tumbler lift:      {finite_or_na(maximum_lift, 100, 2)} cm\n")
        file.write(f"Lift success:              {yes_no(lift_success) if lift_success is not None else 'UNAVAILABLE'}\n")

    print(f"[+] Saved metrics: {metrics_path}")

    if output_dataset_path:
        output_dataset = Path(output_dataset_path)
        output_dataset.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dataset,
            **arrays,
            source_actions=source_actions,
            attachment_score=attachment_score,
            attachment_latch_frame=np.int32(attachment_latch_frame),
            attachment_threshold=np.float32(attachment_threshold),
            demonstrated_grasp_frame=np.int32(demonstrated_grasp_frame),
            grasp_trigger_frame=np.int32(grasp_trigger_frame),
            grasp_trigger_reason=np.asarray(grasp_trigger_reason),
            physical_grasp_latch_frame=np.int32(physical_grasp_latch_frame),
            lift_release_frame=np.int32(lift_release_frame),
            max_tumbler_lift=np.float32(maximum_lift),
            lift_success=np.int8(-1 if lift_success is None else int(lift_success)),
            replay_fps=np.float32(replay_fps),
        )
        print(f"[+] Saved replay dataset: {output_dataset}")

    create_comparison_video(arrays["sim_frames"], human_frames, output_video_path, replay_fps)


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Object-relative grasp with attachment-gated lift.")
    parser.add_argument("--actions", required=True)
    parser.add_argument("--out_video", default="media/side_by_side_demo.mp4")
    parser.add_argument("--out_dataset", default=None)
    parser.add_argument("--task", default="Lift")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--human_video", default=None)
    parser.add_argument("--closed_loop_pregrasp", action="store_true")
    parser.add_argument("--target_pad_x_offset", type=float, default=-0.017)
    parser.add_argument("--pregrasp_kp_x", type=float, default=4.0)
    parser.add_argument("--pregrasp_max_correction", type=float, default=0.07)
    parser.add_argument("--pregrasp_deadband_m", type=float, default=0.002)
    parser.add_argument("--target_pad_z_offset", type=float, default=0.015)
    parser.add_argument("--pregrasp_kp_z", type=float, default=3.0)
    parser.add_argument("--pregrasp_max_z_correction", type=float, default=0.04)
    parser.add_argument("--pregrasp_z_deadband_m", type=float, default=0.002)
    parser.add_argument("--pregrasp_control_start_frac", type=float, default=0.45)
    parser.add_argument("--grasp_ready_x_tolerance_m", type=float, default=0.003)
    parser.add_argument("--grasp_ready_z_tolerance_m", type=float, default=0.002)
    parser.add_argument("--grasp_ready_consecutive_frames", type=int, default=3)
    parser.add_argument("--grasp_control_hold_frames", type=int, default=6)
    return parser


if __name__ == "__main__":
    args = build_argument_parser().parse_args()
    replay_actions_and_record(
        actions_npz_path=args.actions,
        output_video_path=args.out_video,
        output_dataset_path=args.out_dataset,
        task_name=args.task,
        render_gui=args.gui,
        human_video_path=args.human_video,
        closed_loop_pregrasp=args.closed_loop_pregrasp,
        target_pad_x_offset=args.target_pad_x_offset,
        pregrasp_kp_x=args.pregrasp_kp_x,
        pregrasp_max_correction=args.pregrasp_max_correction,
        pregrasp_deadband_m=args.pregrasp_deadband_m,
        target_pad_z_offset=args.target_pad_z_offset,
        pregrasp_kp_z=args.pregrasp_kp_z,
        pregrasp_max_z_correction=args.pregrasp_max_z_correction,
        pregrasp_z_deadband_m=args.pregrasp_z_deadband_m,
        pregrasp_control_start_frac=args.pregrasp_control_start_frac,
        grasp_ready_x_tolerance_m=args.grasp_ready_x_tolerance_m,
        grasp_ready_z_tolerance_m=args.grasp_ready_z_tolerance_m,
        grasp_ready_consecutive_frames=args.grasp_ready_consecutive_frames,
        grasp_control_hold_frames=args.grasp_control_hold_frames,
    )
"""
source /home/nvidia/Documents/Pollex/.venv/bin/activate && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$PYTHONPATH" && \
export DISPLAY=:10 && \
export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL && \
unset QT_DEBUG_PLUGINS && \
unset QT_PLUGIN_PATH && \
unset QT_QPA_PLATFORM_PLUGIN_PATH && \
cd /home/nvidia/Documents/Pollex && \

python3 -m src.simulation.replay_trajectory_with_attachment \
  --actions datasets/cotracker_iterations/iteration_21_attachment/cup_lifting_01_ct_actions.npz \
  --out_video datasets/cotracker_iterations/iteration_21_attachment/cup_lifting_01_iteration_21.mp4 \
  --out_dataset datasets/cotracker_iterations/iteration_21_attachment/cup_lifting_01_replay.npz \
  --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
  --task GlassLift \
  --closed_loop_pregrasp \
  --target_pad_x_offset -0.017 \
  --pregrasp_kp_x 4.0 \
  --pregrasp_max_correction 0.07 \
  --pregrasp_deadband_m 0.002 \
  --target_pad_z_offset 0.015 \
  --pregrasp_kp_z 3.0 \
  --pregrasp_max_z_correction 0.04 \
  --pregrasp_z_deadband_m 0.002 \
  --pregrasp_control_start_frac 0.45 \
  --grasp_ready_x_tolerance_m 0.003 \
  --grasp_ready_z_tolerance_m 0.002 \
  --grasp_ready_consecutive_frames 3 \
  --grasp_control_hold_frames 6 \
  --gui
  """