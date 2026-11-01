"""
Exact-State SmolVLA Handoff Rollout.

Restores a saved post-action MuJoCo state from Iteration 11, verifies the
restored EEF/tumbler positions, and immediately gives control to SmolVLA.

No expert actions are replayed. No X/Z gains or geometry corrections are used.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
from src.simulation.libero_runner import PandaSimEnvironment


DEFAULT_TASK_PROMPT = (
    "approach the glass on the table and grasp the cylinder and lift the cylinder "
    "and then bring down the cylinder to the table and then withdraw your hands"
)

LIFT_SUCCESS_THRESHOLD_M = 0.03


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_smolvla_model(device: torch.device, checkpoint_path: str) -> SmolVLAPolicy:
    checkpoint = Path(checkpoint_path) / "model.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config_path = hf_hub_download(repo_id="lerobot/smolvla_base", filename="config.json")
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)

    config.pop("type", None)
    config["load_vlm_weights"] = False
    config["device"] = str(device)
    config["input_features"] = {
        key: PolicyFeature(type=FeatureType[value["type"]], shape=tuple(value["shape"]))
        for key, value in config["input_features"].items()
    }
    config["output_features"] = {
        key: PolicyFeature(type=FeatureType[value["type"]], shape=tuple(value["shape"]))
        for key, value in config["output_features"].items()
    }
    config["normalization_mapping"] = {
        key: NormalizationMode[value] for key, value in config["normalization_mapping"].items()
    }

    policy = SmolVLAPolicy(SmolVLAConfig(**config))
    policy.load_state_dict(load_file(str(checkpoint)), strict=False)
    policy.to(device)
    policy.eval()

    print(f"[+] Loaded checkpoint: {checkpoint}")
    print(f"[+] Rollout device: {device}")
    if device.type == "cuda":
        print(f"[+] GPU: {torch.cuda.get_device_name(0)}")

    return policy


def build_language_tokens(policy: SmolVLAPolicy, prompt: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    tokens = tokenizer(
        [prompt + "\n"],
        padding="max_length",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
        return_tensors="pt",
    )
    return (
        tokens["input_ids"].to(device),
        tokens["attention_mask"].to(device, dtype=torch.bool),
    )


def build_state_8d(obs: Dict[str, Any]) -> np.ndarray:
    eef_pos = np.asarray(obs.get("robot_eef_pos"), dtype=np.float32).reshape(-1)
    eef_quat = np.asarray(obs.get("robot_eef_quat"), dtype=np.float32).reshape(-1)
    gripper = np.asarray(obs.get("robot_gripper_qpos"), dtype=np.float32).reshape(-1)

    if eef_pos.size < 3 or eef_quat.size < 4 or gripper.size < 1:
        raise RuntimeError(
            f"Invalid state sizes: pos={eef_pos.size}, quat={eef_quat.size}, "
            f"gripper={gripper.size}"
        )

    state = np.concatenate([eef_pos[:3], eef_quat[:4], gripper[:1]]).astype(np.float32)
    if state.shape != (8,) or not np.all(np.isfinite(state)):
        raise RuntimeError(f"Invalid Panda 8D state: {state}")

    return state


def get_restored_observation(env: PandaSimEnvironment) -> Dict[str, Any]:
    if hasattr(env.env, "_get_observations"):
        try:
            raw_obs = env.env._get_observations(force_update=True)
        except TypeError:
            raw_obs = env.env._get_observations()
    else:
        raise RuntimeError("RoboSuite environment has no _get_observations method.")

    return env._format_obs(raw_obs)


def restore_exact_state(
    env: PandaSimEnvironment,
    mujoco_state: np.ndarray,
    expected_eef_pos: np.ndarray,
    expected_tumbler_pos: np.ndarray,
    handoff_frame: int,
) -> Dict[str, Any]:
    state = np.asarray(mujoco_state, dtype=np.float64).reshape(-1)

    if state.size == 0 or not np.all(np.isfinite(state)):
        raise ValueError(f"Invalid MuJoCo state at frame {handoff_frame}")

    print(f"[*] Restoring Iteration 11 post-action state at frame {handoff_frame}")

    if hasattr(env.env.sim, "set_state_from_flattened"):
        env.env.sim.set_state_from_flattened(state)
    else:
        raise RuntimeError("MuJoCo simulator does not expose set_state_from_flattened().")

    env.env.sim.forward()
    obs = get_restored_observation(env)

    restored_eef = np.asarray(obs["robot_eef_pos"], dtype=np.float32)
    restored_tumbler = np.asarray(obs["tumbler_pos"], dtype=np.float32)
    expected_eef_pos = np.asarray(expected_eef_pos, dtype=np.float32)
    expected_tumbler_pos = np.asarray(expected_tumbler_pos, dtype=np.float32)

    eef_error = float(np.linalg.norm(restored_eef - expected_eef_pos))
    tumbler_error = float(np.linalg.norm(restored_tumbler - expected_tumbler_pos))

    print(f"[VERIFY] expected EEF:   {expected_eef_pos}")
    print(f"[VERIFY] restored EEF:   {restored_eef}")
    print(f"[VERIFY] EEF error:      {eef_error * 1000:.3f} mm")
    print(f"[VERIFY] expected glass: {expected_tumbler_pos}")
    print(f"[VERIFY] restored glass: {restored_tumbler}")
    print(f"[VERIFY] glass error:    {tumbler_error * 1000:.3f} mm")

    if eef_error > 0.001 or tumbler_error > 0.001:
        raise RuntimeError(
            f"Exact-state restore failed: EEF={eef_error * 1000:.2f} mm, "
            f"glass={tumbler_error * 1000:.2f} mm"
        )

    return obs


def run_exact_state_handoff(
    policy: SmolVLAPolicy,
    mujoco_state: np.ndarray,
    expected_eef_pos: np.ndarray,
    expected_tumbler_pos: np.ndarray,
    handoff_frame: int,
    original_initial_tumbler_z: float,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    max_steps: int = 200,
    chunk_exec_steps: int = 5,
    device: Optional[torch.device] = None,
    render_gui: bool = False,
) -> Dict[str, Any]:
    if device is None:
        device = select_device()
    if chunk_exec_steps < 1:
        raise ValueError("chunk_exec_steps must be at least 1.")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1.")

    print(
        f"[*] Exact-state SmolVLA handoff: frame={handoff_frame}, "
        f"K={chunk_exec_steps}, rollout_steps={max_steps}"
    )

    lang_tokens, lang_masks = build_language_tokens(policy, task_prompt, device)

    env = PandaSimEnvironment(
        env_name="GlassLift",
        has_renderer=render_gui,
        has_offscreen_renderer=True,
        camera_name="agentview",
        camera_height=256,
        camera_width=256,
    )

    env.reset()
    obs = restore_exact_state(
        env=env,
        mujoco_state=mujoco_state,
        expected_eef_pos=expected_eef_pos,
        expected_tumbler_pos=expected_tumbler_pos,
        handoff_frame=handoff_frame,
    )

    restored_initial_z = float(obs["tumbler_pos"][2])
    maximum_glass_z = restored_initial_z

    sim_frames = []
    actions_executed = []
    eef_positions = []
    tumbler_positions = []
    glass_z_positions = []
    contact_flags = []
    grasp_flags = []
    rewards = []
    d_xy_distances = []
    inference_times = []
    action_queue = []

    try:
        for step in range(max_steps):
            rgb = np.asarray(obs["rgb"], dtype=np.uint8).copy()
            eef_pos = np.asarray(obs["robot_eef_pos"], dtype=np.float32).copy()
            tumbler_pos = np.asarray(obs["tumbler_pos"], dtype=np.float32).copy()

            sim_frames.append(rgb)
            eef_positions.append(eef_pos)
            tumbler_positions.append(tumbler_pos)
            glass_z_positions.append(float(tumbler_pos[2]))
            d_xy_distances.append(float(np.linalg.norm(eef_pos[:2] - tumbler_pos[:2])))
            maximum_glass_z = max(maximum_glass_z, float(tumbler_pos[2]))

            if not action_queue:
                image = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float()
                image = ((image / 255.0) * 2.0 - 1.0).to(device).unsqueeze(0)
                image_mask = torch.ones(1, dtype=torch.bool, device=device)

                state_tensor = torch.from_numpy(build_state_8d(obs)).to(device).unsqueeze(0)
                padded_state = pad_vector(state_tensor, policy.config.max_state_dim)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                else:
                    start_event = end_event = None

                with torch.no_grad():
                    predicted_chunks = policy.model.sample_actions(
                        [image], [image_mask], lang_tokens, lang_masks, padded_state
                    )

                if end_event is not None:
                    end_event.record()
                    torch.cuda.synchronize()
                    inference_times.append(start_event.elapsed_time(end_event) / 1000.0)

                predicted_actions = predicted_chunks[0, :, :7].detach().cpu().numpy()
                for index in range(min(chunk_exec_steps, len(predicted_actions))):
                    action_queue.append(predicted_actions[index].copy())

            action = np.clip(
                np.asarray(action_queue.pop(0), dtype=np.float32),
                -1.0,
                1.0,
            )

            if action.shape != (7,) or not np.all(np.isfinite(action)):
                raise RuntimeError(f"Invalid SmolVLA action at rollout step {step}: {action}")

            actions_executed.append(action.copy())
            obs, reward, done, _ = env.step(action)

            if render_gui:
                env.render()

            contact = float(obs.get("tumbler_contact", False))
            grasp = float(obs.get("tumbler_grasped", False))
            contact_flags.append(contact)
            grasp_flags.append(grasp)
            rewards.append(float(reward))

            if step % 10 == 0:
                print(
                    f"[ROLLOUT] local={step:03d}/{max_steps} "
                    f"expert_frame≈{handoff_frame + step + 1:03d} "
                    f"x={eef_pos[0]:+.4f} z={eef_pos[2]:+.4f} "
                    f"d_xy={d_xy_distances[-1] * 100:.2f}cm "
                    f"dx={action[0]:+.3f} dz={action[2]:+.3f} "
                    f"grip={action[6]:+.3f} contact={int(contact)} grasp={int(grasp)}"
                )

            if done:
                print(f"[!] Episode terminated at local step {step}; stopping rollout.")
                break

    finally:
        env.close()

    actual_steps = len(actions_executed)
    sim_frames = np.asarray(sim_frames, dtype=np.uint8)
    actions_executed = np.asarray(actions_executed, dtype=np.float32)
    eef_positions = np.asarray(eef_positions, dtype=np.float32)
    tumbler_positions = np.asarray(tumbler_positions, dtype=np.float32)
    glass_z_positions = np.asarray(glass_z_positions, dtype=np.float32)
    contact_flags = np.asarray(contact_flags, dtype=np.float32)
    grasp_flags = np.asarray(grasp_flags, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)

    contact_indices = np.where(contact_flags > 0.5)[0]
    grasp_indices = np.where(grasp_flags > 0.5)[0]
    first_contact = int(contact_indices[0]) if len(contact_indices) else -1
    first_grasp = int(grasp_indices[0]) if len(grasp_indices) else -1
    contact_occurred = bool(len(contact_indices))
    grasp_occurred = bool(len(grasp_indices))

    continuation_lift = max(0.0, maximum_glass_z - restored_initial_z)
    full_episode_lift = max(0.0, maximum_glass_z - original_initial_tumbler_z)
    final_continuation_lift = max(
        0.0,
        float(glass_z_positions[-1]) - restored_initial_z,
    )

    success = grasp_occurred and full_episode_lift >= LIFT_SUCCESS_THRESHOLD_M
    min_d_xy = float(np.min(d_xy_distances))
    max_reach_x = float(np.max(eef_positions[:, 0]))
    lowest_eef_z = float(np.min(eef_positions[:, 2]))
    mean_inference_time = float(np.mean(inference_times)) if inference_times else np.nan
    inference_fps = 1.0 / mean_inference_time if mean_inference_time > 0 else np.nan

    print("\n[+] Exact-State Handoff Rollout Completed")
    print(f"    Handoff frame:             {handoff_frame}")
    print(f"    Actual rollout steps:      {actual_steps}")
    print(f"    Contact:                   {contact_occurred}")
    print(f"    First local contact:       {first_contact}")
    print(f"    Grasp:                     {grasp_occurred}")
    print(f"    First local grasp:         {first_grasp}")
    print(f"    Continuation max lift:     {continuation_lift * 100:.2f} cm")
    print(f"    Full-episode max lift:     {full_episode_lift * 100:.2f} cm")
    print(f"    Final continuation lift:   {final_continuation_lift * 100:.2f} cm")
    print(f"    Minimum XY distance:       {min_d_xy * 100:.2f} cm")
    print(f"    Maximum EEF X:             {max_reach_x:.4f} m")
    print(f"    Lowest EEF Z:              {lowest_eef_z:.4f} m")
    print(f"    Mean inference time:       {mean_inference_time:.4f} s")
    print(f"    Inference FPS:             {inference_fps:.2f}")
    print(f"    Lift success:              {success}")

    return {
        "sim_frames": sim_frames,
        "actions": actions_executed,
        "eef_positions": eef_positions,
        "tumbler_positions": tumbler_positions,
        "glass_z": glass_z_positions,
        "contact_flags": contact_flags,
        "grasp_flags": grasp_flags,
        "rewards": rewards,
        "handoff_frame": int(handoff_frame),
        "actual_steps": actual_steps,
        "terminated_early": actual_steps < max_steps,
        "contact_occurred": contact_occurred,
        "first_contact_frame": first_contact,
        "grasp_occurred": grasp_occurred,
        "first_grasp_frame": first_grasp,
        "continuation_max_lift_cm": float(continuation_lift * 100.0),
        "full_episode_max_lift_cm": float(full_episode_lift * 100.0),
        "final_continuation_lift_cm": float(final_continuation_lift * 100.0),
        "success": bool(success),
        "min_d_xy_cm": float(min_d_xy * 100.0),
        "max_reach_x_m": max_reach_x,
        "lowest_eef_z_m": lowest_eef_z,
        "mean_inference_time_s": mean_inference_time,
        "inference_fps": inference_fps,
        "k_steps": int(chunk_exec_steps),
    }


def read_video(path: str, start_frame: int = 0, extract_right_half: bool = False):
    capture = cv2.VideoCapture(str(path))
    frames = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index >= start_frame:
            if extract_right_half and frame.shape[1] >= 512:
                frame = frame[:, frame.shape[1] // 2:, :]
            frames.append(cv2.resize(frame, (256, 256)))

        frame_index += 1

    capture.release()
    return frames


def create_writer(path: Path, fps: float, size):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {path}")
    return writer


def add_header(frame, text, color):
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (256, 30), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color,
        1,
        cv2.LINE_AA,
    )
    return output


def generate_comparison_videos(
    rollout_data: Dict[str, Any],
    demo_video_path: str,
    expert_sim_path: Optional[str],
    output_dir: str,
    prefix: str = "cup_lifting_01",
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frame = rollout_data["handoff_frame"]
    k_steps = rollout_data["k_steps"]
    policy_frames = rollout_data["sim_frames"]
    human_frames = read_video(demo_video_path, start_frame=frame)
    expert_frames = (
        read_video(expert_sim_path, start_frame=frame, extract_right_half=True)
        if expert_sim_path and Path(expert_sim_path).exists()
        else []
    )

    count = min(len(policy_frames), len(human_frames))
    if expert_frames:
        count = min(count, len(expert_frames))
    if count == 0:
        raise RuntimeError("No synchronized frames available for comparison.")

    color = (0, 255, 0) if rollout_data["success"] else (0, 165, 255)
    label = f"SmolVLA exact state {frame}, K={k_steps}"

    side_path = output / f"{prefix}_exact_handoff_{frame}_k{k_steps}_side_by_side.mp4"
    writer = create_writer(side_path, 20.0, (512, 256))

    for index in range(count):
        human = add_header(human_frames[index], f"Human frame {frame + index}", (0, 255, 255))
        policy = cv2.cvtColor(policy_frames[index], cv2.COLOR_RGB2BGR)
        policy = add_header(policy, label, color)
        writer.write(np.hstack([human, policy]))

    writer.release()
    print(f"[+] Saved side-by-side video: {side_path}")

    gif_path = output / f"{prefix}_exact_handoff_{frame}_k{k_steps}_side_by_side.gif"

    try:
        import imageio.v2 as imageio

        gif_frames = []
        for index in range(0, count, 2):
            human = add_header(human_frames[index], f"Human frame {frame + index}", (0, 255, 255))
            policy = cv2.cvtColor(policy_frames[index], cv2.COLOR_RGB2BGR)
            policy = add_header(policy, label, color)
            combined = np.hstack([human, policy])
            gif_frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

        imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
        print(f"[+] Saved looping GIF: {gif_path}")
    except Exception as error:
        print(f"[-] GIF creation failed: {error}")

    if expert_frames:
        tri_path = output / f"{prefix}_exact_handoff_{frame}_k{k_steps}_tri_panel.mp4"
        writer = create_writer(tri_path, 20.0, (768, 256))

        for index in range(count):
            human = add_header(human_frames[index], "1. Human Demo", (0, 255, 255))
            expert = add_header(expert_frames[index], "2. Iteration 11 Expert", (0, 255, 0))
            policy = cv2.cvtColor(policy_frames[index], cv2.COLOR_RGB2BGR)
            policy = add_header(policy, "3. Exact-State SmolVLA", color)
            writer.write(np.hstack([human, expert, policy]))

        writer.release()
        print(f"[+] Saved tri-panel video: {tri_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Exact Iteration 11 MuJoCo-state handoff to SmolVLA"
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/smolvla_glass_expert_iteration10",
    )
    parser.add_argument(
        "--expert_replay",
        default=(
            "datasets/cotracker_iterations/iteration_11_full_mujoco/"
            "cup_lifting_01_replay.npz"
        ),
    )
    parser.add_argument(
        "--human_video",
        default=(
            "datasets/cotracker_trajectories/"
            "cup_lifting_01_ct_annotated.mp4"
        ),
    )
    parser.add_argument(
        "--expert_video",
        default=(
            "datasets/cotracker_iterations/iteration_11_full_mujoco/"
            "cup_lifting_01_iteration_11.mp4"
        ),
    )
    parser.add_argument("--handoff_frame", type=int, default=140)
    parser.add_argument("--chunk_exec_steps", "-k", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument(
        "--out_dir",
        default="datasets/smolvla_comparisons/exact_handoff_140",
    )
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with np.load(args.expert_replay, allow_pickle=False) as data:
        mujoco_states = np.asarray(data["mujoco_states"], dtype=np.float64)
        expert_eef = np.asarray(data["robot_eef_pos"], dtype=np.float32)
        expert_tumbler = np.asarray(data["tumbler_positions"], dtype=np.float32)
        timing = str(np.asarray(data["mujoco_state_timing"]).item())
        original_initial_tumbler_z = float(expert_tumbler[0, 2])

    if timing != "post_action":
        raise ValueError(f"Expected post_action MuJoCo states, got: {timing}")
    if not 0 <= args.handoff_frame < len(mujoco_states):
        raise ValueError(
            f"handoff_frame must be in [0, {len(mujoco_states) - 1}]"
        )

    device = select_device()
    policy = load_smolvla_model(device, args.checkpoint)
    frame = args.handoff_frame

    results = run_exact_state_handoff(
        policy=policy,
        mujoco_state=mujoco_states[frame],
        expected_eef_pos=expert_eef[frame],
        expected_tumbler_pos=expert_tumbler[frame],
        handoff_frame=frame,
        original_initial_tumbler_z=original_initial_tumbler_z,
        task_prompt=DEFAULT_TASK_PROMPT,
        max_steps=args.steps,
        chunk_exec_steps=args.chunk_exec_steps,
        device=device,
        render_gui=args.gui,
    )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    rollout_path = output / f"cup_lifting_01_exact_handoff_{frame}_rollout.npz"
    np.savez_compressed(
        rollout_path,
        sim_frames=results["sim_frames"],
        actions=results["actions"],
        eef_positions=results["eef_positions"],
        tumbler_positions=results["tumbler_positions"],
        glass_z=results["glass_z"],
        contact_flags=results["contact_flags"],
        grasp_flags=results["grasp_flags"],
        rewards=results["rewards"],
        handoff_frame=np.int32(frame),
    )

    metrics = {
        "phase": "exact_state_handoff",
        "checkpoint": args.checkpoint,
        "expert_replay": args.expert_replay,
        "mujoco_state_timing": timing,
        "handoff_frame": frame,
        "requested_steps": args.steps,
        "actual_steps": results["actual_steps"],
        "terminated_early": results["terminated_early"],
        "k_steps": args.chunk_exec_steps,
        "contact_occurred": results["contact_occurred"],
        "first_local_contact_frame": results["first_contact_frame"],
        "grasp_occurred": results["grasp_occurred"],
        "first_local_grasp_frame": results["first_grasp_frame"],
        "continuation_max_lift_cm": results["continuation_max_lift_cm"],
        "full_episode_max_lift_cm": results["full_episode_max_lift_cm"],
        "final_continuation_lift_cm": results["final_continuation_lift_cm"],
        "min_d_xy_cm": results["min_d_xy_cm"],
        "max_reach_x_m": results["max_reach_x_m"],
        "lowest_eef_z_m": results["lowest_eef_z_m"],
        "mean_inference_time_s": results["mean_inference_time_s"],
        "inference_fps": results["inference_fps"],
        "success": results["success"],
    }

    metrics_path = output / f"cup_lifting_01_exact_handoff_{frame}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"[+] Saved rollout data: {rollout_path}")
    print(f"[+] Saved rollout metrics: {metrics_path}")

    generate_comparison_videos(
        rollout_data=results,
        demo_video_path=args.human_video,
        expert_sim_path=args.expert_video,
        output_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()


"""
Run exact-state handoff at frame 140:

source /home/nvidia/Documents/Pollex/.venv/bin/activate && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \
export DISPLAY=:10 && \
export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL QT_DEBUG_PLUGINS QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH && \
cd ~/Documents/Pollex && \
python3 -m src.simulation.rollout_smolvla_exact_state_handoff \
  --checkpoint outputs/smolvla_glass_expert_iteration10 \
  --expert_replay datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_replay.npz \
  --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
  --expert_video datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_iteration_11.mp4 \
  --handoff_frame 140 \
  --chunk_exec_steps 5 \
  --steps 200 \
  --seed 6 \
  --out_dir datasets/smolvla_comparisons/exact_handoff_140 \
  --gui
"""