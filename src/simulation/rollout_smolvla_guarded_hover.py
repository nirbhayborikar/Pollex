"""
Pure Fine-Tuned SmolVLA Baseline Rollout and Video Generator.

Pipeline:
    RGB + 8D Panda state + task prompt
        -> SmolVLA action chunk (H=50)
        -> execute first K actions
        -> observe and replan

No X/Z gain, geometry feedback, or expert-controller correction is applied.
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


def load_smolvla_model(
    device: torch.device,
    checkpoint_path: str,
) -> SmolVLAPolicy:
    checkpoint = Path(checkpoint_path) / "model.safetensors"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg_path = hf_hub_download(
        repo_id="lerobot/smolvla_base",
        filename="config.json",
    )

    with open(cfg_path, encoding="utf-8") as file:
        cfg_dict = json.load(file)

    cfg_dict.pop("type", None)
    cfg_dict["load_vlm_weights"] = False
    cfg_dict["device"] = str(device)

    cfg_dict["input_features"] = {
        key: PolicyFeature(
            type=FeatureType[value["type"]],
            shape=tuple(value["shape"]),
        )
        for key, value in cfg_dict["input_features"].items()
    }

    cfg_dict["output_features"] = {
        key: PolicyFeature(
            type=FeatureType[value["type"]],
            shape=tuple(value["shape"]),
        )
        for key, value in cfg_dict["output_features"].items()
    }

    cfg_dict["normalization_mapping"] = {
        key: NormalizationMode[value]
        for key, value in cfg_dict["normalization_mapping"].items()
    }

    policy = SmolVLAPolicy(SmolVLAConfig(**cfg_dict))
    policy.load_state_dict(load_file(str(checkpoint)), strict=False)
    policy.to(device)
    policy.eval()

    print(f"[+] Loaded checkpoint: {checkpoint}")
    print(f"[+] Rollout device: {device}")

    if device.type == "cuda":
        print(f"[+] GPU: {torch.cuda.get_device_name(0)}")

    return policy


def build_language_tokens(
    policy: SmolVLAPolicy,
    prompt: str,
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    )

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
            f"Invalid state sizes: pos={eef_pos.size}, "
            f"quat={eef_quat.size}, gripper={gripper.size}"
        )

    state = np.concatenate([
        eef_pos[:3],
        eef_quat[:4],
        gripper[:1],
    ]).astype(np.float32)

    if state.shape != (8,) or not np.isfinite(state).all():
        raise RuntimeError(f"Invalid Panda 8D state: {state}")

    return state


def run_baseline_rollout(
    policy: SmolVLAPolicy,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    max_steps: int = 340,
    chunk_exec_steps: int = 5,
    device: Optional[torch.device] = None,
    render_gui: bool = False,
    hover_hold_steps: int = 40,
) -> Dict[str, Any]:
    if device is None:
        device = select_device()

    if chunk_exec_steps < 1:
        raise ValueError("chunk_exec_steps must be at least 1.")
    if not 0 <= hover_hold_steps < max_steps:
        raise ValueError("hover_hold_steps must be in [0, max_steps).")

    print(f"[*] Starting guarded-hover rollout: K={chunk_exec_steps}, hold={hover_hold_steps}, steps={max_steps}")



    lang_tokens, lang_masks = build_language_tokens(
        policy,
        task_prompt,
        device,
    )

    env = PandaSimEnvironment(
        env_name="GlassLift",
        has_renderer=render_gui,
        has_offscreen_renderer=True,
        camera_name="agentview",
        camera_height=256,
        camera_width=256,
    )

    obs = env.reset()
    initial_tumbler = np.asarray(obs["tumbler_pos"], dtype=np.float32)
    initial_glass_z = float(initial_tumbler[2])
    maximum_glass_z = initial_glass_z

    sim_frames = []
    eef_positions = []
    tumbler_positions = []
    glass_z_positions = []
    actions_executed = []

    # clear queue
    actions_raw = []
    guard_active_flags = []

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
                image = rgb.transpose(2, 0, 1)
                image = torch.from_numpy(image).float().to(device).unsqueeze(0)
                image = (image / 255.0) * 2.0 - 1.0
                image_mask = torch.ones(1, dtype=torch.bool, device=device)

                state = build_state_8d(obs)
                state_tensor = torch.from_numpy(state).to(device).unsqueeze(0)
                padded_state = pad_vector(state_tensor, policy.config.max_state_dim)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

                if start_event is not None:
                    start_event.record()

                with torch.no_grad():
                    predicted_chunks = policy.model.sample_actions(
                        [image],
                        [image_mask],
                        lang_tokens,
                        lang_masks,
                        padded_state,
                    )

                if end_event is not None:
                    end_event.record()
                    torch.cuda.synchronize()
                    inference_times.append(start_event.elapsed_time(end_event) / 1000.0)

                predicted_actions = predicted_chunks[0, :, :7].detach().cpu().numpy()

                for index in range(min(chunk_exec_steps, len(predicted_actions))):
                    action_queue.append(predicted_actions[index].copy())

            raw_action = np.asarray(action_queue.pop(0), dtype=np.float32)
            raw_action = np.clip(raw_action, -1.0, 1.0)

            if raw_action.shape != (7,) or not np.isfinite(raw_action).all():
                raise RuntimeError(
                    f"Invalid predicted action at step {step}: {raw_action}"
                )


            action = raw_action.copy()
            guard_active = step < hover_hold_steps

            if guard_active:
                action[:6] = 0.0
                action[6] = -1.0

            if step == hover_hold_steps:
                action_queue.clear()
                print(f"[GUARD] Released at step {step}; queued hover predictions discarded.")

            actions_raw.append(raw_action.copy())
            guard_active_flags.append(guard_active)
            actions_executed.append(action.copy())

            obs, reward, done, _ = env.step(action)

            if render_gui:
                env.render()

            contact_flags.append(float(obs.get("tumbler_contact", False)))
            grasp_flags.append(float(obs.get("tumbler_grasped", False)))
            rewards.append(float(reward))

            if step % 10 == 0:
                print(
                    f"[ROLLOUT] step={step:03d}/{max_steps} "
                    f"x={eef_pos[0]:+.4f} z={eef_pos[2]:+.4f} "
                    f"d_xy={d_xy_distances[-1] * 100:.2f}cm "
                    f"grip={action[6]:+.3f} "
                    f"contact={int(contact_flags[-1])} "
                    f"grasp={int(grasp_flags[-1])}"
                )

            if done:
                print(f"[*] done=True at step {step}; continuing.")

    finally:
        env.close()

    sim_frames = np.asarray(sim_frames, dtype=np.uint8)
    actions_executed = np.asarray(actions_executed, dtype=np.float32)
    actions_raw = np.asarray(actions_raw, dtype=np.float32)
    guard_active_flags = np.asarray(guard_active_flags, dtype=np.int8)
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

    maximum_lift = max(0.0, maximum_glass_z - initial_glass_z)
    final_lift = max(0.0, float(glass_z_positions[-1]) - initial_glass_z)
    success = grasp_occurred and maximum_lift >= LIFT_SUCCESS_THRESHOLD_M

    min_d_xy = float(np.min(d_xy_distances))
    max_reach_x = float(np.max(eef_positions[:, 0]))
    lowest_eef_z = float(np.min(eef_positions[:, 2]))
    mean_inference_time = float(np.mean(inference_times)) if inference_times else np.nan
    inference_fps = 1.0 / mean_inference_time if mean_inference_time > 0 else np.nan

    print("\n[+] Guarded-Hover Rollout Completed")
    print(f"    Contact:             {contact_occurred}")
    print(f"    First contact:       {first_contact}")
    print(f"    Grasp:               {grasp_occurred}")
    print(f"    First grasp:         {first_grasp}")
    print(f"    Max glass lift:      {maximum_lift * 100:.2f} cm")
    print(f"    Final glass lift:    {final_lift * 100:.2f} cm")
    print(f"    Min XY distance:     {min_d_xy * 100:.2f} cm")
    print(f"    Max reach X:         {max_reach_x:.4f} m")
    print(f"    Lowest EEF Z:        {lowest_eef_z:.4f} m")
    print(f"    Mean inference time: {mean_inference_time:.4f} s")
    print(f"    Inference FPS:       {inference_fps:.2f}")
    print(f"    Lift success:        {success}")

    return {
        "sim_frames": sim_frames,
        "actions": actions_executed,
        "raw_actions": actions_raw,
        "guard_active_flags": guard_active_flags,
        "hover_hold_steps": int(hover_hold_steps),
        "eef_positions": eef_positions,
        "tumbler_positions": tumbler_positions,
        "glass_z": glass_z_positions,
        "contact_flags": contact_flags,
        "grasp_flags": grasp_flags,
        "rewards": rewards,
        "contact_occurred": contact_occurred,
        "first_contact_frame": first_contact,
        "grasp_occurred": grasp_occurred,
        "first_grasp_frame": first_grasp,
        "max_lift_cm": float(maximum_lift * 100.0),
        "final_lift_cm": float(final_lift * 100.0),
        "success": bool(success),
        "min_d_xy_cm": float(min_d_xy * 100.0),
        "max_reach_x_m": max_reach_x,
        "lowest_eef_z_m": lowest_eef_z,
        "mean_inference_time_s": mean_inference_time,
        "inference_fps": inference_fps,
        "gamma_x": 1.0,
        "gamma_z": 1.0,
        "k_steps": int(chunk_exec_steps),
    }


def read_video(path: str, extract_right_half: bool = False):
    capture = cv2.VideoCapture(str(path))
    frames = []

    while True:
        success, frame = capture.read()
        if not success:
            break

        if extract_right_half and frame.shape[1] >= 512:
            frame = frame[:, frame.shape[1] // 2:, :]

        frames.append(cv2.resize(frame, (256, 256)))

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
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (256, 30), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        color,
        1,
        cv2.LINE_AA,
    )
    return result


def generate_comparison_videos(
    rollout_data: Dict[str, Any],
    demo_video_path: str,
    expert_sim_path: Optional[str],
    output_dir: str = "datasets/smolvla_comparisons",
    prefix: str = "cup_lifting_01",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    smolvla_frames = rollout_data["sim_frames"]
    human_frames = read_video(demo_video_path)
    expert_frames = read_video(expert_sim_path, True) if expert_sim_path else []

    frame_count = min(len(smolvla_frames), len(human_frames))
    if expert_frames:
        frame_count = min(frame_count, len(expert_frames))

    if frame_count == 0:
        raise RuntimeError("No frames available for comparison video.")

    k_steps = rollout_data["k_steps"]
    success = rollout_data["success"]
    policy_color = (0, 255, 0) if success else (0, 165, 255)

    hold = rollout_data["hover_hold_steps"]
    policy_label = f"Guarded SmolVLA H={hold} K={k_steps}"
    side_path = out_dir / f"{prefix}_guarded_hover_{hold}_k{k_steps}_side_by_side.mp4"

    side_writer = create_writer(side_path, 20.0, (512, 256))

    for index in range(frame_count):
        human = add_header(human_frames[index], "Human Demo", (0, 255, 255))
        policy = cv2.cvtColor(smolvla_frames[index], cv2.COLOR_RGB2BGR)
        policy = add_header(policy, policy_label, policy_color)
        side_writer.write(np.hstack([human, policy]))

    side_writer.release()
    print(f"[+] Saved side-by-side video: {side_path}")

    gif_path = out_dir / f"{prefix}_guarded_hover_{hold}_k{k_steps}_side_by_side.gif"

    try:
        import imageio.v2 as imageio

        gif_frames = []

        for index in range(0, frame_count, 2):
            human = add_header(human_frames[index], "Human Demo", (0, 255, 255))
            policy = cv2.cvtColor(smolvla_frames[index], cv2.COLOR_RGB2BGR)
            policy = add_header(policy, policy_label, policy_color)
            combined = np.hstack([human, policy])
            gif_frames.append(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

        imageio.mimsave(gif_path, gif_frames, fps=10, loop=0)
        print(f"[+] Saved looping GIF: {gif_path}")

    except Exception as error:
        print(f"[-] GIF creation failed: {error}")

    if expert_frames:
        tri_path = out_dir / f"{prefix}_guarded_hover_{hold}_k{k_steps}_tri_panel.mp4"
        tri_writer = create_writer(tri_path, 20.0, (768, 256))

        for index in range(frame_count):
            human = add_header(human_frames[index], "1. Human Demo", (0, 255, 255))
            expert = add_header(expert_frames[index], "2. Iteration 10 Expert", (0, 255, 0))
            policy = cv2.cvtColor(smolvla_frames[index], cv2.COLOR_RGB2BGR)
            policy = add_header(policy, f"3. {policy_label}", policy_color)
            tri_writer.write(np.hstack([human, expert, policy]))

        tri_writer.release()
        print(f"[+] Saved tri-panel video: {tri_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Pure fine-tuned SmolVLA baseline rollout"
    )

    parser.add_argument(
        "--checkpoint",
        default="outputs/smolvla_glass_expert_iteration10",
    )

    parser.add_argument(
        "--human_video",
        default="datasets/cotracker_trajectories/"
                "cup_lifting_01_ct_annotated.mp4",
    )

    parser.add_argument(
        "--expert_video",
        default="datasets/cotracker_iterations/"
                "iteration_10_smolvla_expert/"
                "cup_lifting_01_iteration_10.mp4",
    )

    parser.add_argument(
        "--chunk_exec_steps",
        "-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=340,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--out_dir",
        default="datasets/smolvla_comparisons",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show the live RoboSuite simulation window.",
    )

    parser.add_argument(
        "--hover_hold_steps",
        type=int,
        default=40,
        help="Hold the initial Panda pose before releasing SmolVLA.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device()
    policy = load_smolvla_model(device, args.checkpoint)

    results = run_baseline_rollout(
        policy=policy,
        task_prompt=DEFAULT_TASK_PROMPT,
        max_steps=args.steps,
        chunk_exec_steps=args.chunk_exec_steps,
        device=device,
        render_gui=args.gui,
        hover_hold_steps=args.hover_hold_steps,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rollout_path = (
        out_dir
        / f"cup_lifting_01_guarded_hover_{args.hover_hold_steps}_rollout.npz"
    )

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
        raw_actions=results["raw_actions"],
        guard_active_flags=results["guard_active_flags"],
        hover_hold_steps=np.int32(results["hover_hold_steps"]),
    )

    metrics = {
        "phase": "guarded_hover_ablation",
        "hover_hold_steps": args.hover_hold_steps,
        "checkpoint": args.checkpoint,
        "steps": args.steps,
        "k_steps": args.chunk_exec_steps,
        "gamma_x": 1.0,
        "gamma_z": 1.0,
        "contact_occurred": results["contact_occurred"],
        "first_contact_frame": results["first_contact_frame"],
        "grasp_occurred": results["grasp_occurred"],
        "first_grasp_frame": results["first_grasp_frame"],
        "min_d_xy_cm": results["min_d_xy_cm"],
        "max_reach_x_m": results["max_reach_x_m"],
        "lowest_eef_z_m": results["lowest_eef_z_m"],
        "max_lift_cm": results["max_lift_cm"],
        "final_lift_cm": results["final_lift_cm"],
        "mean_inference_time_s": results["mean_inference_time_s"],
        "inference_fps": results["inference_fps"],
        "success": results["success"],
    }

    metrics_path = (
        out_dir
        / f"cup_lifting_01_guarded_hover_{args.hover_hold_steps}_metrics.json"
    )


    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"[+] Saved rollout data: {rollout_path}")
    print(f"[+] Saved rollout metrics: {metrics_path}")

    generate_comparison_videos(
        rollout_data=results,
        demo_video_path=args.human_video,
        expert_sim_path=args.expert_video,
        output_dir=args.out_dir,
        prefix="cup_lifting_01",
    )


if __name__ == "__main__":
    main()

""" 
source /home/nvidia/Documents/Pollex/.venv/bin/activate && \
export PYTHONPATH="$HOME/Documents/vla/LIBERO:$HOME/Documents/Pollex:$PYTHONPATH" && \
export DISPLAY=:10 && \
export XAUTHORITY="$HOME/.Xauthority" && \
unset MUJOCO_GL QT_DEBUG_PLUGINS QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH && \
cd ~/Documents/Pollex && \
python3 -m src.simulation.rollout_smolvla_guarded_hover \
  --checkpoint outputs/smolvla_glass_expert_iteration10 \
  --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
  --expert_video datasets/cotracker_iterations/iteration_10_smolvla_expert/cup_lifting_01_iteration_10.mp4 \
  --chunk_exec_steps 5 \
  --hover_hold_steps 40 \
  --steps 340 \
  --seed 6 \
  --out_dir datasets/smolvla_comparisons/guarded_hover_40 \
  --gui
"""