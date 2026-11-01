"""
Iteration 11 Expert Replay with Full MuJoCo State Logging.

Records the successful object-relative closed-loop expert trajectory plus the
complete flattened MuJoCo state required for exact-state SmolVLA handoffs.

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
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.size < 3:
        return None
    value = value[:3].copy()
    return value if np.all(np.isfinite(value)) else None


def yes_no(value) -> str:
    return "YES" if bool(value) else "NO"


def frame_or_na(frame: int) -> str:
    return str(frame) if frame >= 0 else "N/A"


def finite_or_na(value, scale=1.0, precision=3, signed=False) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return format(value * scale, f"+.{precision}f" if signed else f".{precision}f")


def control_stats(errors: np.ndarray, corrections: np.ndarray):
    finite = errors[np.isfinite(errors)]
    return (
        float(finite[0]) if len(finite) else np.nan,
        float(finite[-1]) if len(finite) else np.nan,
        float(np.min(np.abs(finite))) if len(finite) else np.nan,
        float(np.max(np.abs(corrections))) if len(corrections) else np.nan,
    )


def load_human_video(video_path: str) -> np.ndarray:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Human video not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open human video: {path}")

    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    capture.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from: {path}")

    frames = np.asarray(frames, dtype=np.uint8)
    print(f"[+] Loaded {len(frames)} human frames from: {path}")
    return frames


def calculate_replay_fps(data) -> float:
    if "fps" in data:
        fps = float(np.asarray(data["fps"]).item())
        if np.isfinite(fps) and fps > 0:
            return fps

    if "timestamps" in data:
        timestamps = np.asarray(data["timestamps"], dtype=np.float64)
        differences = np.diff(timestamps)
        differences = differences[np.isfinite(differences) & (differences > 0)]
        if len(differences):
            return 1.0 / float(np.median(differences))

    return DEFAULT_REPLAY_FPS


def create_comparison_video(sim_frames, human_frames, output_video_path, replay_fps):
    output_path = Path(output_video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        replay_fps,
        (512, 256),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output_path}")

    count = min(len(sim_frames), len(human_frames)) if human_frames is not None else len(sim_frames)

    for index in range(count):
        sim = cv2.cvtColor(cv2.resize(sim_frames[index], (256, 256)), cv2.COLOR_RGB2BGR)
        human = cv2.cvtColor(cv2.resize(human_frames[index], (256, 256)), cv2.COLOR_RGB2BGR) if human_frames is not None else np.zeros_like(sim)
        writer.write(np.hstack([human, sim]))

    writer.release()
    print(f"[+] Saved synchronized video: {output_path}")


def validate_parameters(
    target_pad_x_offset, pregrasp_kp_x, pregrasp_max_correction,
    pregrasp_deadband_m, target_pad_z_offset, pregrasp_kp_z,
    pregrasp_max_z_correction, pregrasp_z_deadband_m,
    pregrasp_control_start_frac, grasp_ready_x_tolerance_m,
    grasp_ready_z_tolerance_m, grasp_ready_consecutive_frames,
    grasp_control_hold_frames,
):
    values = {
        "target_pad_x_offset": target_pad_x_offset,
        "pregrasp_kp_x": pregrasp_kp_x,
        "pregrasp_max_correction": pregrasp_max_correction,
        "pregrasp_deadband_m": pregrasp_deadband_m,
        "target_pad_z_offset": target_pad_z_offset,
        "pregrasp_kp_z": pregrasp_kp_z,
        "pregrasp_max_z_correction": pregrasp_max_z_correction,
        "pregrasp_z_deadband_m": pregrasp_z_deadband_m,
        "pregrasp_control_start_frac": pregrasp_control_start_frac,
        "grasp_ready_x_tolerance_m": grasp_ready_x_tolerance_m,
        "grasp_ready_z_tolerance_m": grasp_ready_z_tolerance_m,
    }

    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")

    if pregrasp_kp_x < 0 or pregrasp_kp_z < 0:
        raise ValueError("X/Z gains must be non-negative.")
    if not 0 < pregrasp_max_correction <= 1 or not 0 < pregrasp_max_z_correction <= 1:
        raise ValueError("Maximum corrections must be in (0, 1].")
    if pregrasp_deadband_m < 0 or pregrasp_z_deadband_m < 0:
        raise ValueError("Deadbands must be non-negative.")
    if not 0 <= pregrasp_control_start_frac < 1:
        raise ValueError("Control start fraction must be in [0, 1).")
    if grasp_ready_x_tolerance_m <= 0 or grasp_ready_z_tolerance_m <= 0:
        raise ValueError("Ready tolerances must be positive.")
    if grasp_ready_consecutive_frames < 1 or grasp_control_hold_frames < 0:
        raise ValueError("Frame counts are invalid.")


def replay_actions_and_record(
    actions_npz_path: str,
    output_video_path: str,
    output_dataset_path: Optional[str] = None,
    task_name: str = "Lift",
    render_gui: bool = False,
    human_video_path: Optional[str] = None,
    closed_loop_pregrasp: bool = False,
    target_pad_x_offset: float = -0.017,
    pregrasp_kp_x: float = 4.0,
    pregrasp_max_correction: float = 0.07,
    pregrasp_deadband_m: float = 0.002,
    target_pad_z_offset: float = 0.015,
    pregrasp_kp_z: float = 3.0,
    pregrasp_max_z_correction: float = 0.04,
    pregrasp_z_deadband_m: float = 0.002,
    pregrasp_control_start_frac: float = 0.65,
    grasp_ready_x_tolerance_m: float = 0.003,
    grasp_ready_z_tolerance_m: float = 0.002,
    grasp_ready_consecutive_frames: int = 3,
    grasp_control_hold_frames: int = 6,
) -> None:
    validate_parameters(
        target_pad_x_offset, pregrasp_kp_x, pregrasp_max_correction,
        pregrasp_deadband_m, target_pad_z_offset, pregrasp_kp_z,
        pregrasp_max_z_correction, pregrasp_z_deadband_m,
        pregrasp_control_start_frac, grasp_ready_x_tolerance_m,
        grasp_ready_z_tolerance_m, grasp_ready_consecutive_frames,
        grasp_control_hold_frames,
    )

    actions_file = Path(actions_npz_path)
    if not actions_file.exists():
        raise FileNotFoundError(f"Actions file not found: {actions_file}")

    with np.load(actions_file, allow_pickle=False) as data:
        if "actions" not in data:
            raise KeyError(f"No 'actions' array in: {actions_file}")

        source_actions = np.asarray(data["actions"], dtype=np.float32).copy()
        if source_actions.ndim != 2 or source_actions.shape[1] != 7:
            raise ValueError(f"Expected action shape (T, 7), got {source_actions.shape}")
        if not np.all(np.isfinite(source_actions)):
            raise ValueError("Source actions contain NaN or infinity.")

        replay_fps = calculate_replay_fps(data)
        entry_frame = int(np.asarray(data["entry_frame"]).item()) if "entry_frame" in data else 0
        stored_frames = np.asarray(data["rgb_frames"], dtype=np.uint8).copy() if human_video_path is None and "rgb_frames" in data else None

    human_frames = load_human_video(human_video_path) if human_video_path else stored_frames
    number_of_steps = len(source_actions)
    grasp_indices = np.where(source_actions[:, 6] > 0)[0]
    demonstrated_grasp_frame = int(grasp_indices[0]) if len(grasp_indices) else number_of_steps
    entry_frame = int(np.clip(entry_frame, 0, max(0, demonstrated_grasp_frame - 1)))
    control_start_frame = entry_frame + int(pregrasp_control_start_frac * max(1, demonstrated_grasp_frame - entry_frame))
    control_start_frame = int(np.clip(control_start_frame, entry_frame, max(entry_frame, demonstrated_grasp_frame - 1)))
    control_end_frame = min(number_of_steps, demonstrated_grasp_frame + grasp_control_hold_frames)

    print(f"[*] Iteration 11: {number_of_steps} steps at {replay_fps:.2f} FPS")
    print(f"[*] Target pad offsets: X={target_pad_x_offset * 100:+.2f} cm, Z={target_pad_z_offset * 100:+.2f} cm")
    print(f"[*] Control starts={control_start_frame}, timeout={demonstrated_grasp_frame}")

    env = PandaSimEnvironment(
        env_name=task_name,
        has_renderer=render_gui,
        has_offscreen_renderer=True,
        camera_name="agentview",
        camera_height=256,
        camera_width=256,
    )

    obs = env.reset()
    initial_eef_pos = valid_xyz(obs.get("robot_eef_pos"))
    initial_tumbler_pos = valid_xyz(obs.get("tumbler_pos"))
    current_eef_pos = initial_eef_pos.copy() if initial_eef_pos is not None else None
    current_tumbler_pos = initial_tumbler_pos.copy() if initial_tumbler_pos is not None else None
    current_left_pad_pos = valid_xyz(obs.get("left_finger_pad_pos"))
    current_right_pad_pos = valid_xyz(obs.get("right_finger_pad_pos"))

    logs = {key: [] for key in [
        "sim_frames", "mujoco_states", "eef_positions", "eef_quats",
        "gripper_qpos", "eef_frames", "tumbler_positions", "tumbler_frames",
        "left_pads", "right_pads", "contacts", "grasps", "rewards",
        "executed_actions", "x_errors", "x_corrections", "z_errors",
        "z_corrections", "pad_x_offsets", "pad_z_offsets",
        "control_active", "ready_counts",
    ]}

    ready_count = 0
    grasp_triggered = False
    grasp_trigger_frame = -1
    grasp_trigger_reason = "none"
    physical_grasp_latched = False
    physical_grasp_latch_frame = -1

    try:
        for frame_index in range(number_of_steps):
            source_action = source_actions[frame_index].copy()
            executed_action = source_action.copy()
            x_error = z_error = np.nan
            x_correction = z_correction = 0.0
            pad_x_offset = pad_z_offset = np.nan
            x_blend = z_blend = 1.0
            x_ready = z_ready = control_active = False

            in_control = (
                closed_loop_pregrasp
                and not physical_grasp_latched
                and control_start_frame <= frame_index < control_end_frame
            )

            geometry_available = (
                current_tumbler_pos is not None
                and current_left_pad_pos is not None
                and current_right_pad_pos is not None
            )

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
                    grasp_triggered = True
                    grasp_trigger_frame = frame_index
                    grasp_trigger_reason = "geometry"
                    control_end_frame = min(number_of_steps, frame_index + grasp_control_hold_frames + 1)
                    print(f"[GRASP] Geometry-triggered close at frame {frame_index}.")

                if not grasp_triggered and frame_index >= demonstrated_grasp_frame:
                    grasp_triggered = True
                    grasp_trigger_frame = frame_index
                    grasp_trigger_reason = "timeout"
                    control_end_frame = min(number_of_steps, frame_index + grasp_control_hold_frames + 1)
                    print(f"[GRASP] Timeout close at frame {frame_index}.")

                if grasp_triggered:
                    executed_action[0] = float(np.clip(x_correction, -1.0, 1.0))
                    executed_action[2] = float(np.clip(z_correction, -1.0, 1.0))
                    executed_action[6] = 1.0
                else:
                    x_blend = float(np.clip(abs(x_error) / (3.0 * grasp_ready_x_tolerance_m), 0.0, 1.0))
                    z_blend = float(np.clip(abs(z_error) / (3.0 * grasp_ready_z_tolerance_m), 0.0, 1.0))
                    executed_action[0] = float(np.clip(x_blend * source_action[0] + x_correction, -1.0, 1.0))
                    executed_action[2] = float(np.clip(z_blend * source_action[2] + z_correction, -1.0, 1.0))

            elif grasp_triggered and not physical_grasp_latched:
                executed_action[0] = 0.0
                executed_action[2] = 0.0
                executed_action[6] = 1.0

            obs, reward, done, _ = env.step(executed_action)

            # Full MuJoCo state after action frame_index.
            mujoco_state = env.env.sim.get_state().flatten().copy()
            if not np.all(np.isfinite(mujoco_state)):
                raise RuntimeError(f"Non-finite MuJoCo state at frame {frame_index}")
            logs["mujoco_states"].append(mujoco_state)

            eef_pos = valid_xyz(obs.get("robot_eef_pos"))
            tumbler_pos = valid_xyz(obs.get("tumbler_pos"))
            left_pad_pos = valid_xyz(obs.get("left_finger_pad_pos"))
            right_pad_pos = valid_xyz(obs.get("right_finger_pad_pos"))
            eef_quat = np.asarray(obs.get("robot_eef_quat"), dtype=np.float32).reshape(-1)
            gripper_qpos = np.asarray(obs.get("robot_gripper_qpos"), dtype=np.float32).reshape(-1)

            if eef_pos is None or tumbler_pos is None:
                raise RuntimeError(f"Missing synchronized state at frame {frame_index}")
            if eef_quat.size < 4 or gripper_qpos.size < 1:
                raise RuntimeError(f"Invalid proprioception at frame {frame_index}")

            current_eef_pos = eef_pos.copy()
            current_tumbler_pos = tumbler_pos.copy()
            current_left_pad_pos = left_pad_pos.copy() if left_pad_pos is not None else current_left_pad_pos
            current_right_pad_pos = right_pad_pos.copy() if right_pad_pos is not None else current_right_pad_pos

            logs["eef_positions"].append(eef_pos.copy())
            logs["eef_quats"].append(eef_quat[:4].copy())
            logs["gripper_qpos"].append(gripper_qpos.copy())
            logs["eef_frames"].append(frame_index)
            logs["tumbler_positions"].append(tumbler_pos.copy())
            logs["tumbler_frames"].append(frame_index)
            logs["left_pads"].append(left_pad_pos.copy() if left_pad_pos is not None else np.full(3, np.nan, dtype=np.float32))
            logs["right_pads"].append(right_pad_pos.copy() if right_pad_pos is not None else np.full(3, np.nan, dtype=np.float32))

            contact = float(obs.get("tumbler_contact", False))
            grasp = float(obs.get("tumbler_grasped", False))

            if not physical_grasp_latched and grasp > 0.5:
                physical_grasp_latched = True
                physical_grasp_latch_frame = frame_index
                print(f"[CONTROL] Physical grasp latched at frame {frame_index}.")

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
            logs["control_active"].append(control_active)
            logs["ready_counts"].append(ready_count)

            if obs.get("rgb") is None:
                raise RuntimeError(f"Missing RGB observation at frame {frame_index}")
            logs["sim_frames"].append(np.asarray(obs["rgb"], dtype=np.uint8).copy())

            phase = "CL-APPROACH" if control_active and not grasp_triggered else "GRASP-HOLD" if control_active else "GRASP-LATCHED" if physical_grasp_latched else "SOURCE"
            print(
                f"[REPLAY] frame={frame_index:04d} phase={phase} ready={ready_count} "
                f"x_ready={int(x_ready)} z_ready={int(z_ready)} "
                f"cmd_x={executed_action[0]:+.3f} cmd_z={executed_action[2]:+.3f} "
                f"grip={executed_action[6]:+.1f} contact={contact:.0f} grasp={grasp:.0f}"
            )

            if render_gui:
                env.render()

            if done and frame_index < number_of_steps - 1:
                print(f"[!] Episode terminated at frame {frame_index}; stopping replay.")
                break

    finally:
        env.close()

    sim_frames = np.asarray(logs["sim_frames"], dtype=np.uint8)
    mujoco_states = np.asarray(logs["mujoco_states"], dtype=np.float64)
    executed_actions = np.asarray(logs["executed_actions"], dtype=np.float32)
    rewards = np.asarray(logs["rewards"], dtype=np.float32)
    eef_positions = np.asarray(logs["eef_positions"], dtype=np.float32)
    eef_quats = np.asarray(logs["eef_quats"], dtype=np.float32)
    gripper_qpos = np.asarray(logs["gripper_qpos"], dtype=np.float32)
    eef_frames = np.asarray(logs["eef_frames"], dtype=np.int32)
    tumbler_positions = np.asarray(logs["tumbler_positions"], dtype=np.float32)
    tumbler_frames = np.asarray(logs["tumbler_frames"], dtype=np.int32)
    left_pads = np.asarray(logs["left_pads"], dtype=np.float32)
    right_pads = np.asarray(logs["right_pads"], dtype=np.float32)
    contacts = np.asarray(logs["contacts"], dtype=np.float32)
    grasps = np.asarray(logs["grasps"], dtype=np.float32)
    x_errors = np.asarray(logs["x_errors"], dtype=np.float32)
    x_corrections = np.asarray(logs["x_corrections"], dtype=np.float32)
    z_errors = np.asarray(logs["z_errors"], dtype=np.float32)
    z_corrections = np.asarray(logs["z_corrections"], dtype=np.float32)
    pad_x_offsets = np.asarray(logs["pad_x_offsets"], dtype=np.float32)
    pad_z_offsets = np.asarray(logs["pad_z_offsets"], dtype=np.float32)
    ready_counts = np.asarray(logs["ready_counts"], dtype=np.int32)

    lengths = {
        "frames": len(sim_frames),
        "mujoco": len(mujoco_states),
        "actions": len(executed_actions),
        "eef": len(eef_positions),
        "quat": len(eef_quats),
        "gripper": len(gripper_qpos),
        "tumbler": len(tumbler_positions),
    }

    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Iteration 11 synchronization failure: {lengths}")

    print(f"[+] MuJoCo states: {mujoco_states.shape}")
    print(f"[+] Synchronized samples: {lengths['actions']}")

    contact_indices = np.where(contacts > 0.5)[0]
    grasp_indices = np.where(grasps > 0.5)[0]
    first_contact = int(contact_indices[0]) if len(contact_indices) else -1
    first_grasp = int(grasp_indices[0]) if len(grasp_indices) else -1

    maximum_lift = np.nan
    lift_success = None
    if len(tumbler_positions) and initial_tumbler_pos is not None:
        maximum_lift = float(np.max(tumbler_positions[:, 2]) - initial_tumbler_pos[2])
        lift_success = bool(maximum_lift >= LIFT_SUCCESS_THRESHOLD_M)

    finite_x = pad_x_offsets[np.isfinite(pad_x_offsets)]
    finite_z = pad_z_offsets[np.isfinite(pad_z_offsets)]
    _, _, minimum_x_error, _ = control_stats(x_errors, x_corrections)
    _, _, minimum_z_error, _ = control_stats(z_errors, z_corrections)

    output_video = Path(output_video_path)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_video.parent / "metrics.txt"

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("REPLAY METRICS\n" + "=" * 68 + "\n\n")
        file.write("ITERATION 11: FULL MUJOCO STATE EXPERT\n" + "-" * 68 + "\n")
        file.write(f"Recorded samples:          {len(executed_actions)}\n")
        file.write(f"MuJoCo state dimension:   {mujoco_states.shape[1]}\n")
        file.write(f"Demonstrated timeout:     {demonstrated_grasp_frame}\n")
        file.write(f"Grasp trigger frame:      {frame_or_na(grasp_trigger_frame)}\n")
        file.write(f"Grasp trigger reason:     {grasp_trigger_reason}\n")
        file.write(f"Physical grasp frame:     {frame_or_na(physical_grasp_latch_frame)}\n")
        file.write(f"Minimum |X error|:        {finite_or_na(minimum_x_error, 100, 2)} cm\n")
        file.write(f"Minimum |Z error|:        {finite_or_na(minimum_z_error, 100, 2)} cm\n\n")
        file.write("PHYSICAL OUTCOME\n" + "-" * 68 + "\n")
        file.write(f"Contact occurred:         {yes_no(len(contact_indices))}\n")
        file.write(f"First contact frame:      {frame_or_na(first_contact)}\n")
        file.write(f"Grasp occurred:           {yes_no(len(grasp_indices))}\n")
        file.write(f"First grasp frame:        {frame_or_na(first_grasp)}\n")
        file.write(f"Maximum tumbler lift:     {finite_or_na(maximum_lift, 100, 2)} cm\n")
        file.write(f"Lift success:             {yes_no(lift_success) if lift_success is not None else 'UNAVAILABLE'}\n")

    print(f"[+] Saved replay metrics: {metrics_path}")

    if output_dataset_path:
        output_dataset = Path(output_dataset_path)
        output_dataset.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_dataset,
            sim_frames=sim_frames,
            human_frames=human_frames if human_frames is not None else np.empty((0,), dtype=np.uint8),
            mujoco_states=mujoco_states,
            mujoco_state_timing=np.asarray("post_action"),
            source_actions=source_actions[:len(executed_actions)],
            executed_actions=executed_actions,
            rewards=rewards,
            robot_eef_pos=eef_positions,
            robot_eef_quat=eef_quats,
            robot_gripper_qpos=gripper_qpos,
            robot_eef_frames=eef_frames,
            tumbler_positions=tumbler_positions,
            tumbler_position_frames=tumbler_frames,
            left_pad_positions=left_pads,
            right_pad_positions=right_pads,
            contact_flags=contacts,
            grasp_flags=grasps,
            pad_x_offsets=pad_x_offsets,
            pad_z_offsets=pad_z_offsets,
            x_errors=x_errors,
            z_errors=z_errors,
            x_corrections=x_corrections,
            z_corrections=z_corrections,
            geometry_ready_counts=ready_counts,
            target_pad_x_offset=np.float32(target_pad_x_offset),
            target_pad_z_offset=np.float32(target_pad_z_offset),
            demonstrated_grasp_frame=np.int32(demonstrated_grasp_frame),
            grasp_trigger_frame=np.int32(grasp_trigger_frame),
            grasp_trigger_reason=np.asarray(grasp_trigger_reason),
            physical_grasp_latch_frame=np.int32(physical_grasp_latch_frame),
            max_tumbler_lift=np.float32(maximum_lift),
            lift_success=np.int8(-1 if lift_success is None else int(lift_success)),
            replay_fps=np.float32(replay_fps),
        )

        print(f"[+] Saved Iteration 11 replay dataset: {output_dataset}")

    create_comparison_video(sim_frames, human_frames, output_video_path, replay_fps)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iteration 11 expert replay with full MuJoCo state logging.")
    parser.add_argument("--actions", required=True)
    parser.add_argument("--out_video", default="datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_iteration_11.mp4")
    parser.add_argument("--out_dataset", default="datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz")
    parser.add_argument("--task", default="GlassLift")
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
Run Iteration 11:

source /home/nvidia/Documents/Pollex/.venv/bin/activate && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \
export DISPLAY=:10 && \
export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL QT_DEBUG_PLUGINS QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH && \
cd /home/nvidia/Documents/Pollex && \
python3 -m src.simulation.replay_trajectory_full_mujoco \
  --actions datasets/cotracker_iterations/iteration_8_comotion/cup_lifting_01_ct_actions.npz \
  --out_video datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_iteration_11.mp4 \
  --out_dataset datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
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