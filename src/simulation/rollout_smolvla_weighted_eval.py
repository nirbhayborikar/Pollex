"""
SmolVLA Spatial-Reach & Height-Weighted Expert Evaluation Script.

Evaluates the newly fine-tuned SmolVLA Action Expert checkpoint (trained with
elevated loss weights on forward reach X and vertical height Z) in closed-loop
RoboSuite simulation WITHOUT any artificial gain (gamma_x = 1.0).

Generates synchronized side-by-side and 3-way comparative evaluation videos:
  Panel 1: Real-World Demonstration Video
  Panel 2: Baseline Unweighted Expert (outputs/smolvla_glass_expert)
  Panel 3: Spatial-Weighted Expert (outputs/smolvla_glass_expert_reach_weighted)
"""

import cv2
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
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
    """Loads SmolVLA policy weights from specified checkpoint directory."""
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
        print(f"[*] Loading base model (lerobot/smolvla_base)...")
        weights_path = hf_hub_download(repo_id="lerobot/smolvla_base", filename="model.safetensors")
        state_dict = load_file(weights_path)

    policy.load_state_dict(state_dict, strict=False)
    policy.to(device)
    policy.eval()
    return policy


def run_closed_loop_rollout(
    policy: SmolVLAPolicy,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    max_steps: int = 160,
    chunk_exec_steps: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Executes closed-loop simulation rollout with pure model actions.
    No artificial gain or manual reach scaling (gamma_x = 1.0).
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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

            # NOTE: DO NOT call torch.mps.empty_cache() to maintain CGL buffer integrity.

        act = action_queue.pop(0)
        actions_executed.append(act)

        obs, reward, done, info = env.step(act)
        t += 1

    env.close()

    net_lift = max(0.0, max_glass_z - init_glass_z)
    final_glass_z = glass_z_positions[-1]
    success = (net_lift >= 0.03)  # >= 3cm lift
    min_d_xy = min(d_xy_distances)
    max_x = max(pos[0] for pos in eef_positions)
    lowest_z = min(pos[2] for pos in eef_positions)

    print(f"[+] Rollout Completed (Steps: {max_steps}):")
    print(f"    Max Glass Lift: {net_lift*100:.2f} cm (Success: {success})")
    print(f"    Min d_xy to cylinder: {min_d_xy*100:.2f} cm")
    print(f"    Max forward reach X: {max_x:.4f} m (Cylinder center: {cyl_init_pos[0]:.4f} m)")
    print(f"    Lowest EEF height Z: {lowest_z:.4f} m")

    return {
        "sim_frames": np.array(sim_frames, dtype=np.uint8),
        "actions": np.array(actions_executed, dtype=np.float32),
        "eef_positions": np.array(eef_positions, dtype=np.float32),
        "glass_z": np.array(glass_z_positions, dtype=np.float32),
        "max_lift_cm": float(net_lift * 100.0),
        "success": bool(success),
        "min_d_xy_cm": float(min_d_xy * 100.0),
        "max_reach_x_m": float(max_x),
        "lowest_eef_z_m": float(lowest_z),
        "final_z": float(final_glass_z),
        "init_z": float(init_glass_z),
        "cyl_init_pos": cyl_init_pos.tolist(),
        "k_steps": int(chunk_exec_steps),
    }


def create_annotated_panel(frame: np.ndarray, title: str, subtitle: str, color: tuple = (0, 255, 0)) -> np.ndarray:
    """Creates a stylized panel with top header."""
    h, w, c = frame.shape
    header = np.zeros((48, w, 3), dtype=np.uint8)
    header[:] = (20, 20, 20)

    cv2.putText(header, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(header, subtitle, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([header, frame])


def build_evaluation_video(
    demo_video_path: str,
    baseline_sim_frames: Optional[np.ndarray],
    weighted_rollout: Dict[str, Any],
    output_path: str,
    fps: int = 10,
):
    """Compiles side-by-side or tri-panel comparison video with H.264 avc1."""
    cap = cv2.VideoCapture(demo_video_path)
    real_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256))
        real_frames.append(frame)
    cap.release()

    sim_frames = weighted_rollout["sim_frames"]
    N = min(len(sim_frames), len(real_frames))
    if baseline_sim_frames is not None:
        N = min(N, len(baseline_sim_frames))

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = None

    for i in range(N):
        # Panel 1: Real Demo (BGR)
        p1 = create_annotated_panel(
            real_frames[i],
            "Human Demonstration",
            "Ground Truth Trajectory",
            color=(0, 220, 255),
        )

        # Panel 2: Baseline Unweighted (if available)
        if baseline_sim_frames is not None:
            base_bgr = cv2.cvtColor(baseline_sim_frames[i], cv2.COLOR_RGB2BGR)
            p2 = create_annotated_panel(
                base_bgr,
                "Baseline (Unweighted)",
                "Under-reach shortfall (-0.072m)",
                color=(0, 140, 255),
            )

        # Panel 3: Weighted Expert
        we_rgb = sim_frames[i]
        we_bgr = cv2.cvtColor(we_rgb, cv2.COLOR_RGB2BGR)
        eef = weighted_rollout["eef_positions"][i]
        cyl_xy = np.array(weighted_rollout["cyl_init_pos"][:2])
        d_xy = np.linalg.norm(eef[:2] - cyl_xy) * 100.0
        grip = weighted_rollout["actions"][i][6]

        status = f"X:{eef[0]:+.3f}m | Z:{eef[2]:+.3f}m | Grip:{grip:+.2f}"
        we_panel = create_annotated_panel(
            we_bgr,
            "SmolVLA (Reach-Weighted)",
            status,
            color=(0, 255, 120),
        )

        if baseline_sim_frames is not None:
            composite = np.hstack([p1, p2, we_panel])
        else:
            composite = np.hstack([p1, we_panel])

        if writer is None:
            ch, cw, _ = composite.shape
            writer = cv2.VideoWriter(str(out_file), fourcc, fps, (cw, ch))

        writer.write(composite)

    if writer is not None:
        writer.release()
        print(f"[+] Evaluation video saved to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate reach-weighted SmolVLA Expert Head")
    parser.add_argument("--checkpoint", type=str, default="outputs/smolvla_glass_expert_reach_weighted")
    parser.add_argument("--baseline_checkpoint", type=str, default="outputs/smolvla_glass_expert")
    parser.add_argument("--video_id", type=str, default="Vid_0", choices=["Vid_0", "Vid_2"])
    parser.add_argument("--k_steps", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default="data/smolvla_comparisons")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("=" * 80)
    print("SmolVLA Spatial-Reach Weighted Expert Evaluation")
    print(f"Checkpoint: {args.checkpoint} | Baseline: {args.baseline_checkpoint}")
    print(f"Video: {args.video_id} | Horizon K: {args.k_steps} | Device: {device}")
    print("=" * 80)

    demo_map = {
        "Vid_0": {"path": "media/Vid_0.mp4", "steps": 160},
        "Vid_2": {"path": "media/Vid_2.mp4", "steps": 200},
    }
    spec = demo_map[args.video_id]

    # 1. Rollout Baseline (if available)
    baseline_frames = None
    if args.baseline_checkpoint and (Path(args.baseline_checkpoint) / "model.safetensors").exists():
        print("\n--- Running Baseline Unweighted Expert Rollout ---")
        base_policy = load_smolvla_model(device, args.baseline_checkpoint)
        base_res = run_closed_loop_rollout(
            base_policy,
            max_steps=spec["steps"],
            chunk_exec_steps=args.k_steps,
            device=device,
        )
        baseline_frames = base_res["sim_frames"]
        del base_policy

    # 2. Rollout Spatial-Weighted Expert
    print("\n--- Running Spatial-Reach Weighted Expert Rollout ---")
    weighted_policy = load_smolvla_model(device, args.checkpoint)
    weighted_res = run_closed_loop_rollout(
        weighted_policy,
        max_steps=spec["steps"],
        chunk_exec_steps=args.k_steps,
        device=device,
    )

    # 3. Generate Comparative Video
    out_video = Path(args.out_dir) / f"{args.video_id}_weighted_vs_baseline_comparison.mp4"
    build_evaluation_video(
        demo_video_path=spec["path"],
        baseline_sim_frames=baseline_frames,
        weighted_rollout=weighted_res,
        output_path=str(out_video),
    )

    # 4. Save metrics report
    report_file = Path(args.out_dir) / f"{args.video_id}_weighted_eval_metrics.json"
    metrics = {
        "video_id": args.video_id,
        "k_steps": args.k_steps,
        "checkpoint": args.checkpoint,
        "max_lift_cm": weighted_res["max_lift_cm"],
        "min_d_xy_cm": weighted_res["min_d_xy_cm"],
        "max_reach_x_m": weighted_res["max_reach_x_m"],
        "lowest_eef_z_m": weighted_res["lowest_eef_z_m"],
        "success": weighted_res["success"],
    }
    with open(report_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[+] Benchmark Metrics exported to: {report_file}")
    print(f"    Lift: {weighted_res['max_lift_cm']:.2f} cm | Success: {weighted_res['success']}")
    print(f"    Reach X: {weighted_res['max_reach_x_m']:.4f} m | Min d_xy: {weighted_res['min_d_xy_cm']:.2f} cm")


if __name__ == "__main__":
    main()
