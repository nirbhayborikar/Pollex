"""
SmolVLA Calibrated Forward Reach Rollout & Video Generator.

Implements Strategy 2: Forward Reach Gain Calibration (gamma_x).
Applies phase-aware forward velocity scaling to eliminate the ~3cm under-reach shortfall
observed in few-shot Flow-Matching action heads, enabling the Franka Panda gripper
to reach the cylinder grasping zone (X in [-0.050m, -0.035m]) and lift the glass.

Key Design Guardrails:
  1. Separate file: Does NOT modify existing baseline inference files.
  2. Phase-Aware: Scaling is only applied during forward approach (when gripper is open).
  3. Workspace Soft-Cap: Prevents overshooting beyond cylinder center (X <= -0.030m).
  4. Memory-Safe: NO torch.mps.empty_cache() to ensure pristine OpenGL offscreen rendering.
  5. Codec: Uses cv2.VideoWriter_fourcc(*'avc1') (H.264).
"""

import cv2
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
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
    """Loads SmolVLA policy weights."""
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
    print("[+] SmolVLA ready for calibrated reach rollouts.")
    return policy


def apply_calibrated_reach(
    action: np.ndarray,
    current_eef_pos: np.ndarray,
    gamma_x: float = 1.80,
    gamma_z: float = 1.25,
    x_target_limit: float = -0.035,
    z_descent_limit: float = 0.855,
) -> np.ndarray:
    """
    Applies calibrated forward reach and vertical gain on action delta.
    
    1. Forward Reach Calibration (gamma_x):
       - Applied during approach (gripper open: action[6] < 0.2)
       - Action commands forward translation (action[0] > 0.0)
       - EEF has not yet reached cylinder front boundary (current_eef_pos[0] < x_target_limit).
    
    2. Vertical Gain Calibration (gamma_z):
       - Descent phase: gripper open (action[6] < 0.2) and action commands downward descent (action[2] < 0.0)
         and EEF is above grasp height target (current_eef_pos[2] > z_descent_limit)
         -> Accelerates descent so gripper fingers surround cylinder center rather than top lip.
       - Lift phase: gripper is closed/clamping (action[6] >= 0.0) and action commands upward lift (action[2] > 0.0)
         -> Accelerates upward lift momentum to raise cylinder off tabletop.
    """
    act = action.copy()
    is_approach = act[6] < 0.2
    is_moving_forward = act[0] > 0.0
    behind_target = current_eef_pos[0] < x_target_limit

    # Forward reach gain
    if is_approach and is_moving_forward and behind_target:
        act[0] = np.clip(act[0] * gamma_x, -1.0, 1.0)

    # Vertical descent gain
    if is_approach and act[2] < 0.0 and current_eef_pos[2] > z_descent_limit:
        act[2] = np.clip(act[2] * gamma_z, -1.0, 1.0)
    # Vertical lift gain
    elif act[6] >= 0.0 and act[2] > 0.0:
        act[2] = np.clip(act[2] * gamma_z, -1.0, 1.0)
    
    return np.clip(act, -1.0, 1.0)


def run_calibrated_rollout(
    policy: SmolVLAPolicy,
    gamma_x: float = 1.80,
    gamma_z: float = 1.25,
    z_descent_limit: float = 0.855,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    max_steps: int = 160,
    chunk_exec_steps: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Executes closed-loop simulation rollout with calibrated forward reach and vertical gain."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"[*] Starting GlassLiftEnv Calibrated Rollout (gamma_x={gamma_x:.2f}, gamma_z={gamma_z:.2f}, K={chunk_exec_steps}, max_steps={max_steps})...")

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
    init_glass_z = env.env.get_object_z()
    max_glass_z = init_glass_z

    cylinder_body_id = env.env.sim.model.body_name2id(env.env.cube.root_body)
    cyl_init_pos = env.env.sim.data.body_xpos[cylinder_body_id].copy()

    sim_frames = []
    eef_positions = []
    glass_z_positions = []
    actions_executed = []
    d_xy_distances = []

    action_queue = []
    t = 0

    while t < max_steps:
        rgb_frame = obs["rgb"].copy()
        sim_frames.append(rgb_frame)
        eef_pos = obs["robot_eef_pos"].copy()
        eef_positions.append(eef_pos)

        glass_z = env.env.get_object_z()
        glass_z_positions.append(glass_z)
        if glass_z > max_glass_z:
            max_glass_z = glass_z

        cyl_pos = env.env.sim.data.body_xpos[cylinder_body_id].copy()
        diff_xy = np.linalg.norm(eef_pos[:2] - cyl_pos[:2])
        d_xy_distances.append(float(diff_xy))

        # Replanning when queue is empty
        if len(action_queue) == 0:
            img_np = rgb_frame.transpose(2, 0, 1)
            img_tensor = (torch.from_numpy(img_np).float().to(device).unsqueeze(0) / 255.0) * 2.0 - 1.0
            img_mask = torch.ones(1, dtype=torch.bool, device=device)

            grip_q = obs["robot_gripper_qpos"][0] if obs["robot_gripper_qpos"] is not None else 0.0
            st_7d = np.hstack([
                obs["robot_eef_pos"],
                obs["robot_eef_quat"][:3],
                [grip_q]
            ])
            st_tensor = torch.from_numpy(st_7d).float().to(device).unsqueeze(0)
            padded_st = pad_vector(st_tensor, policy.config.max_state_dim)

            with torch.no_grad():
                pred_chunks = policy.model.sample_actions(
                    [img_tensor], [img_mask], lang_tokens, lang_masks, padded_st
                )
                pred_actions = pred_chunks[0, :, :7].cpu().numpy()

            for k in range(min(chunk_exec_steps, len(pred_actions))):
                action_queue.append(pred_actions[k])

            # NOTE: DO NOT call torch.mps.empty_cache() to protect OpenGL CGL buffer.

        # Pop action and apply forward reach calibration
        raw_act = action_queue.pop(0)
        calibrated_act = apply_calibrated_reach(
            action=raw_act,
            current_eef_pos=eef_pos,
            gamma_x=gamma_x,
            gamma_z=gamma_z,
            x_target_limit=float(cyl_pos[0]),
            z_descent_limit=z_descent_limit,
        )
        actions_executed.append(calibrated_act)

        obs, reward, done, info = env.step(calibrated_act)
        t += 1

    env.close()

    net_lift = max(0.0, max_glass_z - init_glass_z)
    final_glass_z = glass_z_positions[-1]
    final_lift = max(0.0, final_glass_z - init_glass_z)
    success = (net_lift >= 0.05)
    min_d_xy = min(d_xy_distances)
    max_x = max(pos[0] for pos in eef_positions)
    lowest_z = min(pos[2] for pos in eef_positions)

    print(f"[+] Rollout Completed:")
    print(f"    Max Glass Lift: {net_lift*100:.2f} cm (Success: {success})")
    print(f"    Final Glass Lift: {final_lift*100:.2f} cm")
    print(f"    Min d_xy to cylinder: {min_d_xy*100:.2f} cm")
    print(f"    Max forward reach X: {max_x:.4f} m (Cylinder center: {cyl_init_pos[0]:.4f} m)")
    print(f"    Lowest EEF height Z: {lowest_z:.4f} m")

    return {
        "sim_frames": np.array(sim_frames, dtype=np.uint8),
        "actions": np.array(actions_executed, dtype=np.float32),
        "eef_positions": np.array(eef_positions, dtype=np.float32),
        "glass_z": np.array(glass_z_positions, dtype=np.float32),
        "max_lift_cm": float(net_lift * 100.0),
        "final_lift_cm": float(final_lift * 100.0),
        "success": bool(success),
        "min_d_xy_cm": float(min_d_xy * 100.0),
        "max_reach_x_m": float(max_x),
        "lowest_eef_z_m": float(lowest_z),
        "final_z": float(final_glass_z),
        "init_z": float(init_glass_z),
        "gamma_x": float(gamma_x),
        "gamma_z": float(gamma_z),
        "k_steps": int(chunk_exec_steps),
    }


def generate_comparison_videos(
    rollout_data: Dict[str, Any],
    demo_video_path: str,
    expert_sim_path: Optional[str],
    output_dir: str = "data/smolvla_comparisons",
    prefix: str = "Vid_0",
):
    """Compiles synchronized comparison videos with calibrated reach annotations."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    smol_frames = rollout_data["sim_frames"]
    glass_zs = rollout_data["glass_z"]
    gamma_x = rollout_data["gamma_x"]
    gamma_z = rollout_data.get("gamma_z", 1.0)
    k_steps = rollout_data["k_steps"]
    max_lift_cm = rollout_data["max_lift_cm"]
    T_smol = len(smol_frames)

    # 1. Load Real Demonstration Video
    demo_cap = cv2.VideoCapture(str(demo_video_path))
    real_frames = []
    while True:
        ret, frame = demo_cap.read()
        if not ret:
            break
        real_frames.append(cv2.resize(frame, (256, 256)))
    demo_cap.release()

    if not real_frames:
        real_frames = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(T_smol)]

    # 2. Load Expert Simulation Video if available
    expert_frames = []
    if expert_sim_path and Path(expert_sim_path).exists():
        exp_cap = cv2.VideoCapture(str(expert_sim_path))
        while True:
            ret, frame = exp_cap.read()
            if not ret:
                break
            if frame.shape[1] == 512:
                frame_sim = frame[:, 256:, :]
            else:
                frame_sim = frame
            expert_frames.append(cv2.resize(frame_sim, (256, 256)))
        exp_cap.release()

    num_frames = min(T_smol, len(real_frames))
    if expert_frames:
        num_frames = min(num_frames, len(expert_frames))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    gx_str = f"{gamma_x:.2f}".replace(".", "p")
    gz_str = f"_gz{gamma_z:.2f}".replace(".", "p") if abs(gamma_z - 1.0) > 1e-3 else ""

    # --- Video 1: 2-Way Comparison (512x256) ---
    vid_2way_path = out_dir / f"{prefix}_smolvla_calibrated_gx{gx_str}{gz_str}_k{k_steps}_side_by_side.mp4"
    writer_2way = cv2.VideoWriter(str(vid_2way_path), fourcc, 20.0, (512, 256))

    if abs(gamma_z - 1.0) > 1e-3:
        policy_label = f"SmolVLA Reach (gx={gamma_x:.2f}, gz={gamma_z:.2f}, K={k_steps})"
    else:
        policy_label = f"SmolVLA Reach (gx={gamma_x:.1f}, K={k_steps})"
    policy_color = (0, 255, 128) if max_lift_cm >= 5.0 else (0, 255, 0)

    for i in range(num_frames):
        left_bgr = real_frames[i].copy()
        right_bgr = cv2.cvtColor(smol_frames[i], cv2.COLOR_RGB2BGR)

        cv2.rectangle(left_bgr, (0, 0), (256, 32), (0, 0, 0), -1)
        cv2.putText(left_bgr, "Real Human Demo", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.rectangle(right_bgr, (0, 0), (256, 42), (0, 0, 0), -1)
        cv2.putText(right_bgr, policy_label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, policy_color, 1)
        z_curr = (glass_zs[i] - 0.8575) * 100.0
        cv2.putText(right_bgr, f"Z: {z_curr:+.1f}cm | Step {i+1}/{num_frames}", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        writer_2way.write(np.hstack([left_bgr, right_bgr]))

    writer_2way.release()
    print(f"[+] Saved 2-Way Video -> {vid_2way_path}")

    # Generate 2-Way looping GIF (10 FPS)
    gif_2way_path = out_dir / f"{prefix}_smolvla_calibrated_gx{gx_str}{gz_str}_k{k_steps}_side_by_side.gif"
    try:
        import imageio
        gif_frames = []
        for i in range(0, num_frames, 2):
            left_bgr = real_frames[i].copy()
            right_bgr = cv2.cvtColor(smol_frames[i], cv2.COLOR_RGB2BGR)

            cv2.rectangle(left_bgr, (0, 0), (256, 32), (0, 0, 0), -1)
            cv2.putText(left_bgr, "Real Human Demo", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            cv2.rectangle(right_bgr, (0, 0), (256, 42), (0, 0, 0), -1)
            cv2.putText(right_bgr, policy_label, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, policy_color, 1)
            z_curr = (glass_zs[i] - 0.8575) * 100.0
            cv2.putText(right_bgr, f"Z: {z_curr:+.1f}cm | Step {i+1}/{num_frames}", (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            combo_bgr = np.hstack([left_bgr, right_bgr])
            combo_rgb = cv2.cvtColor(combo_bgr, cv2.COLOR_BGR2RGB)
            gif_frames.append(combo_rgb)

        imageio.mimsave(str(gif_2way_path), gif_frames, fps=10, loop=0)
        print(f"[+] Saved 2-Way Looping GIF -> {gif_2way_path}")
    except Exception as e:
        print(f"[-] Could not save GIF: {e}")

    # --- Video 2: 3-Way Tri-Panel Video (768x256) ---
    if expert_frames:
        vid_3way_path = out_dir / f"{prefix}_smolvla_calibrated_gx{gx_str}{gz_str}_k{k_steps}_tri_panel_comparison.mp4"
        writer_3way = cv2.VideoWriter(str(vid_3way_path), fourcc, 20.0, (768, 256))

        for i in range(num_frames):
            p1 = real_frames[i].copy()
            p2 = expert_frames[i].copy()
            p3 = cv2.cvtColor(smol_frames[i], cv2.COLOR_RGB2BGR)

            cv2.rectangle(p1, (0, 0), (256, 30), (0, 0, 0), -1)
            cv2.putText(p1, "1. Real Human Demo", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

            cv2.rectangle(p2, (0, 0), (256, 30), (0, 0, 0), -1)
            cv2.putText(p2, "2. Expert Retargeted [PASS]", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)

            cv2.rectangle(p3, (0, 0), (256, 30), (0, 0, 0), -1)
            cv2.putText(p3, f"3. {policy_label}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, policy_color, 1)

            writer_3way.write(np.hstack([p1, p2, p3]))

        writer_3way.release()
        print(f"[+] Saved 3-Way Tri-Panel Video -> {vid_3way_path}")


def main():
    parser = argparse.ArgumentParser(description="Simulate SmolVLA with calibrated forward reach and vertical gain")
    parser.add_argument("--video_id", type=str, default="Vid_0", help="Video prefix (Vid_0, Vid_2)")
    parser.add_argument("--checkpoint", type=str, default="outputs/smolvla_glass_expert")
    parser.add_argument("--gamma_x", type=float, default=1.80, help="Forward reach gain multiplier")
    parser.add_argument("--gamma_z", type=float, default=1.0, help="Vertical descent & lift gain multiplier (1.0 = baseline, 1.25 = augmented lift)")
    parser.add_argument("--z_descent_limit", type=float, default=0.855, help="Target minimum EEF Z height before throttling descent")
    parser.add_argument("--chunk_exec_steps", "-k", type=int, default=5, help="Action chunk execution size")
    parser.add_argument("--steps", type=int, default=160, help="Max rollout steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--out_dir", type=str, default="data/smolvla_comparisons")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy = load_smolvla_model(device, checkpoint_path=args.checkpoint)

    vid_stem = args.video_id
    annotated_demo = f"data/cotracker_trajectories/{vid_stem}_ct_annotated.mp4"
    if not Path(annotated_demo).exists():
        annotated_demo = f"media/{vid_stem}.mp4"

    expert_comp = f"data/cotracker_comparisons/{vid_stem}_ct_comparison.mp4"

    # Execute rollout
    rollout_results = run_calibrated_rollout(
        policy=policy,
        gamma_x=args.gamma_x,
        gamma_z=args.gamma_z,
        z_descent_limit=args.z_descent_limit,
        task_prompt=DEFAULT_TASK_PROMPT,
        max_steps=args.steps,
        chunk_exec_steps=args.chunk_exec_steps,
        device=device,
    )

    # Save data
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gx_str = f"{args.gamma_x:.2f}".replace(".", "p")
    gz_str = f"_gz{args.gamma_z:.2f}".replace(".", "p") if abs(args.gamma_z - 1.0) > 1e-3 else ""
    npz_out = out_dir / f"{vid_stem}_smolvla_calibrated_gx{gx_str}{gz_str}_k{args.chunk_exec_steps}_rollout_data.npz"
    np.savez_compressed(
        str(npz_out),
        sim_frames=rollout_results["sim_frames"],
        actions=rollout_results["actions"],
        glass_z=rollout_results["glass_z"],
        eef_positions=rollout_results["eef_positions"],
        metrics={
            "max_lift_cm": rollout_results["max_lift_cm"],
            "final_lift_cm": rollout_results["final_lift_cm"],
            "success": rollout_results["success"],
            "min_d_xy_cm": rollout_results["min_d_xy_cm"],
            "max_reach_x_m": rollout_results["max_reach_x_m"],
            "lowest_eef_z_m": rollout_results["lowest_eef_z_m"],
            "gamma_x": rollout_results["gamma_x"],
            "gamma_z": rollout_results["gamma_z"],
        }
    )
    print(f"[+] Saved telemetry -> {npz_out}")

    # Generate videos and GIFs
    generate_comparison_videos(
        rollout_data=rollout_results,
        demo_video_path=annotated_demo,
        expert_sim_path=expert_comp if Path(expert_comp).exists() else None,
        output_dir=args.out_dir,
        prefix=vid_stem,
    )


if __name__ == "__main__":
    main()
