"""
SmolVLA Simulation Rollout & Side-by-Side Synchronized Comparison Video Generator.

Executes closed-loop Vision-Language-Action rollouts of SmolVLA (450M) in GlassLiftEnv,
records offscreen camera views, tracks glass height and contact physics,
and compiles synchronized multi-view comparison videos saved to a dedicated subfolder:
  1. Synchronized 3-Way: [Real Human Video | Expert Retargeted Sim (PASS) | SmolVLA Sim]
  2. Synchronized 2-Way: [Real Human Video | SmolVLA Sim]

Destination subfolder: data/smolvla_comparisons/
"""

import gc
import cv2
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
from src.simulation.libero_runner import PandaSimEnvironment


DEFAULT_TASK_PROMPT = (
    "approach the glass on the table and grasp the cylinder and lift the cylinder "
    "and then bring down the cylinder to the table and then withdraw your hands"
)


def load_smolvla_model(device: torch.device, checkpoint_path: Optional[str] = None) -> SmolVLAPolicy:
    """Loads SmolVLA policy weights (either pretrained base or fine-tuned checkpoint)."""
    cfg_path = hf_hub_download(repo_id="lerobot/smolvla_base", filename="config.json")
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    cfg_dict.pop("type", None)
    cfg_dict["load_vlm_weights"] = False
    cfg_dict["device"] = str(device)

    input_features = {
        k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))
        for k, v in cfg_dict["input_features"].items()
    }
    output_features = {
        k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"]))
        for k, v in cfg_dict["output_features"].items()
    }
    cfg_dict["input_features"] = input_features
    cfg_dict["output_features"] = output_features
    cfg_dict["normalization_mapping"] = {
        k: NormalizationMode[v] for k, v in cfg_dict["normalization_mapping"].items()
    }

    cfg = SmolVLAConfig(**cfg_dict)
    policy = SmolVLAPolicy(cfg)

    if checkpoint_path and (Path(checkpoint_path) / "model.safetensors").exists():
        ckpt_file = str(Path(checkpoint_path) / "model.safetensors")
        print(f"[*] Loading fine-tuned SmolVLA checkpoint from {ckpt_file}...")
        state_dict = load_file(ckpt_file)
    else:
        print(f"[*] Loading pretrained base SmolVLA from Hugging Face...")
        weights_path = hf_hub_download(repo_id="lerobot/smolvla_base", filename="model.safetensors")
        state_dict = load_file(weights_path)

    policy.load_state_dict(state_dict, strict=False)
    policy.to(device)
    policy.eval()

    print("[+] SmolVLA ready for simulation rollouts.")
    return policy


def run_smolvla_rollout(
    policy: SmolVLAPolicy,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    max_steps: int = 150,
    chunk_exec_steps: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Executes closed-loop rollout in GlassLiftEnv using SmolVLA action chunking.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"[*] Starting GlassLiftEnv closed-loop rollout (max_steps={max_steps}, chunk_exec={chunk_exec_steps})...")

    # Tokenize language instruction
    tokens = policy.language_tokenizer(
        [task_prompt + "\n"],
        padding="max_length",
        max_length=policy.config.tokenizer_max_length,
        return_tensors="pt",
    )
    lang_tokens = tokens["input_ids"].to(device)
    lang_masks = tokens["attention_mask"].to(device, dtype=torch.bool)

    env = PandaSimEnvironment(
        env_name="GlassLift",
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_name="agentview",
        camera_height=256,
        camera_width=256,
    )

    obs = env.reset()
    sim_frames = []
    eef_positions = []
    glass_z_positions = []
    actions_executed = []

    init_glass_z = env.env.get_object_z()
    max_glass_z = init_glass_z

    action_queue = []
    t = 0

    while t < max_steps:
        # Record observation
        rgb_frame = obs["rgb"].copy()
        sim_frames.append(rgb_frame)
        eef_positions.append(obs["robot_eef_pos"])

        glass_z = env.env.get_object_z()
        glass_z_positions.append(glass_z)
        if glass_z > max_glass_z:
            max_glass_z = glass_z

        # If action queue is depleted, query SmolVLA for the next action chunk
        if len(action_queue) == 0:
            img_np = rgb_frame.transpose(2, 0, 1)  # (3, 256, 256)
            img_tensor = (torch.from_numpy(img_np).float().to(device).unsqueeze(0) / 255.0) * 2.0 - 1.0
            img_mask = torch.ones(1, dtype=torch.bool, device=device)

            # State 7D
            st_7d = np.hstack([
                obs["robot_eef_pos"],
                obs["robot_eef_quat"][:3],
                [obs["robot_gripper_qpos"][0] if obs["robot_gripper_qpos"] is not None else 0.0]
            ])
            st_tensor = torch.from_numpy(st_7d).float().to(device).unsqueeze(0)
            padded_st = pad_vector(st_tensor, policy.config.max_state_dim)

            with torch.no_grad():
                pred_chunks = policy.model.sample_actions(
                    [img_tensor], [img_mask], lang_tokens, lang_masks, padded_st
                )
                # Shape: (1, 50, 32)
                pred_actions = pred_chunks[0, :, :7].cpu().numpy()

            # Queue first K steps of the predicted chunk
            for k in range(min(chunk_exec_steps, len(pred_actions))):
                action_queue.append(pred_actions[k])

            # Note: Do NOT call torch.mps.empty_cache() here as it purges the shared
            # Apple Silicon Unified Memory pool and corrupts MuJoCo's OpenGL CGL framebuffer.

        # Pop next action from queue
        act = action_queue.pop(0)
        # Clamp action deltas within standard physical bounds
        act_clamped = np.clip(act, -1.0, 1.0)
        actions_executed.append(act_clamped)

        # Step simulation
        obs, reward, done, info = env.step(act_clamped)
        t += 1

    env.close()

    final_glass_z = glass_z_positions[-1]
    net_lift = max(0.0, max_glass_z - init_glass_z)
    returned = abs(final_glass_z - init_glass_z) < 0.02 and final_glass_z > 0.84
    success = (net_lift >= 0.05) and returned

    print(f"[+] Rollout complete ({len(sim_frames)} frames). Max Lift: {net_lift*100:.1f} cm | Table Return: {returned} | Success: {success}")

    return {
        "sim_frames": np.array(sim_frames, dtype=np.uint8),
        "actions": np.array(actions_executed, dtype=np.float32),
        "eef_positions": np.array(eef_positions, dtype=np.float32),
        "glass_z": np.array(glass_z_positions, dtype=np.float32),
        "max_lift_cm": float(net_lift * 100.0),
        "final_z": float(final_glass_z),
        "init_z": float(init_glass_z),
        "returned_to_table": bool(returned),
        "success": bool(success),
    }


def generate_comparison_videos(
    rollout_data: Dict[str, Any],
    demo_video_path: str,
    expert_sim_path: Optional[str],
    output_dir: str = "data/smolvla_comparisons",
    prefix: str = "Vid_0",
    is_finetuned: bool = False,
    k_steps: int = 5,
):
    """
    Compiles synchronized 2-way and 3-way side-by-side comparison videos.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    smol_frames = rollout_data["sim_frames"]
    glass_zs = rollout_data["glass_z"]
    max_lift_cm = rollout_data["max_lift_cm"]
    T_smol = len(smol_frames)

    # 1. Load Real Demonstration Video (annotated or raw)
    demo_cap = cv2.VideoCapture(str(demo_video_path))
    real_frames = []
    while True:
        ret, frame = demo_cap.read()
        if not ret:
            break
        real_frames.append(cv2.resize(frame, (256, 256)))
    demo_cap.release()

    if not real_frames:
        print(f"[-] Could not read frames from {demo_video_path}, creating blank placeholder")
        real_frames = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(T_smol)]

    # 2. Load Expert Simulation Video if available
    expert_frames = []
    if expert_sim_path and Path(expert_sim_path).exists():
        exp_cap = cv2.VideoCapture(str(expert_sim_path))
        while True:
            ret, frame = exp_cap.read()
            if not ret:
                break
            # If the video is side-by-side (512x256), take the right half (sim)
            if frame.shape[1] == 512:
                frame_sim = frame[:, 256:, :]
            else:
                frame_sim = frame
            expert_frames.append(cv2.resize(frame_sim, (256, 256)))
        exp_cap.release()

    # Determine total playback length
    num_frames = min(T_smol, len(real_frames))
    if expert_frames:
        num_frames = min(num_frames, len(expert_frames))

    print(f"[*] Compiling {num_frames} frames into comparison videos at 20 FPS...")

    # --- Video 1: Synchronized 2-Way Comparison (512x256) [Real Human | SmolVLA Sim] ---
    suffix_tag = f"finetuned_k{k_steps}" if is_finetuned else f"zero_shot_k{k_steps}"
    vid_2way_path = out_dir / f"{prefix}_smolvla_{suffix_tag}_side_by_side.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer_2way = cv2.VideoWriter(str(vid_2way_path), fourcc, 20.0, (512, 256))

    policy_label = f"SmolVLA Fine-Tuned (K={k_steps})" if is_finetuned else f"SmolVLA Zero-Shot (K={k_steps})"
    policy_color = (0, 255, 0) if is_finetuned else (0, 200, 255)

    for i in range(num_frames):
        left_bgr = real_frames[i].copy()
        right_bgr = cv2.cvtColor(smol_frames[i], cv2.COLOR_RGB2BGR)

        # Overlay labels on Left Panel (Real Human)
        cv2.rectangle(left_bgr, (0, 0), (256, 32), (0, 0, 0), -1)
        cv2.putText(left_bgr, "Real Human Demo", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Overlay labels on Right Panel (SmolVLA Sim)
        cv2.rectangle(right_bgr, (0, 0), (256, 42), (0, 0, 0), -1)
        cv2.putText(right_bgr, policy_label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, policy_color, 1)
        z_curr = (glass_zs[i] - 0.8575) * 100.0
        cv2.putText(right_bgr, f"Z: {z_curr:+.1f}cm | Step {i+1}/{num_frames}", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        combined_2way = np.hstack([left_bgr, right_bgr])
        writer_2way.write(combined_2way)

    writer_2way.release()
    print(f"[+] Saved 2-Way Comparison Video -> {vid_2way_path}")

    # --- Video 2: Synchronized 3-Way Tri-Panel Video (768x256) ---
    # [Real Human Demo | Expert Retargeted Sim (PASS) | SmolVLA Policy Sim]
    if expert_frames:
        vid_3way_path = out_dir / f"{prefix}_smolvla_{suffix_tag}_tri_panel_comparison.mp4"
        writer_3way = cv2.VideoWriter(str(vid_3way_path), fourcc, 20.0, (768, 256))

        for i in range(num_frames):
            p1 = real_frames[i].copy()
            p2 = expert_frames[i].copy()
            p3 = cv2.cvtColor(smol_frames[i], cv2.COLOR_RGB2BGR)

            # Panel 1: Real Human
            cv2.rectangle(p1, (0, 0), (256, 30), (0, 0, 0), -1)
            cv2.putText(p1, "1. Real Human Demo", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

            # Panel 2: Ground Truth Expert Sim
            cv2.rectangle(p2, (0, 0), (256, 30), (0, 0, 0), -1)
            cv2.putText(p2, "2. Expert Retargeted [PASS]", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)

            # Panel 3: SmolVLA Policy
            cv2.rectangle(p3, (0, 0), (256, 30), (0, 0, 0), -1)
            cv2.putText(p3, f"3. {policy_label}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, policy_color, 1)

            combined_3way = np.hstack([p1, p2, p3])
            writer_3way.write(combined_3way)

        writer_3way.release()
        print(f"[+] Saved 3-Way Tri-Panel Video -> {vid_3way_path}")


def main():
    parser = argparse.ArgumentParser(description="Simulate and record SmolVLA comparison videos")
    parser.add_argument("--video_id", type=str, default="Vid_0", help="Video prefix (Vid_0, Vid_2)")
    parser.add_argument("--checkpoint", type=str, default="outputs/smolvla_glass_expert", help="Path to checkpoint")
    parser.add_argument("--chunk_exec_steps", "-k", type=int, default=5, help="Number of chunk actions to execute before replanning")
    parser.add_argument("--steps", type=int, default=160, help="Max rollout steps")
    parser.add_argument("--out_dir", type=str, default="data/smolvla_comparisons", help="Output subfolder")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    is_finetuned = args.checkpoint and Path(args.checkpoint).exists() and (Path(args.checkpoint) / "model.safetensors").exists()
    policy = load_smolvla_model(device, checkpoint_path=args.checkpoint if is_finetuned else None)

    # Resolve paths
    vid_stem = args.video_id
    annotated_demo = f"data/cotracker_trajectories/{vid_stem}_ct_annotated.mp4"
    if not Path(annotated_demo).exists():
        annotated_demo = f"media/{vid_stem}.mp4"

    expert_comp = f"data/cotracker_comparisons/{vid_stem}_ct_comparison.mp4"

    # Execute simulation rollout
    rollout_results = run_smolvla_rollout(
        policy=policy,
        task_prompt=DEFAULT_TASK_PROMPT,
        max_steps=args.steps,
        chunk_exec_steps=args.chunk_exec_steps,
        device=device,
    )

    # Save rollout data
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_tag = f"finetuned_k{args.chunk_exec_steps}" if is_finetuned else f"zero_shot_k{args.chunk_exec_steps}"
    npz_out = out_dir / f"{vid_stem}_smolvla_{suffix_tag}_rollout_data.npz"
    np.savez_compressed(
        str(npz_out),
        sim_frames=rollout_results["sim_frames"],
        actions=rollout_results["actions"],
        glass_z=rollout_results["glass_z"],
        eef_positions=rollout_results["eef_positions"],
        metrics={
            "max_lift_cm": rollout_results["max_lift_cm"],
            "success": rollout_results["success"],
            "returned_to_table": rollout_results["returned_to_table"],
        }
    )
    print(f"[+] Saved rollout telemetry -> {npz_out}")

    # Generate videos
    generate_comparison_videos(
        rollout_data=rollout_results,
        demo_video_path=annotated_demo,
        expert_sim_path=expert_comp if Path(expert_comp).exists() else None,
        output_dir=args.out_dir,
        prefix=vid_stem,
        is_finetuned=is_finetuned,
        k_steps=args.chunk_exec_steps,
    )


if __name__ == "__main__":
    main()
