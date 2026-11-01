#!/usr/bin/env python3
"""
SmolVLA Calibrated Rollout with Dual-Axis Gain
================================================
Applies phase-aware velocity scaling (γ_x, γ_z) to predicted actions at
inference time to compensate for flow-matching regression-to-the-mean on
low-variance action channels (ΔX, ΔZ).

Based on the approach from the Humanoid-challenge reference repo:
  - γ_x scales forward reach during open-gripper approach
  - γ_z scales descent during approach AND lift during post-grasp
"""

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

import src.simulation.rollout_smolvla_baseline as baseline
from src.simulation.libero_runner import PandaSimEnvironment
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, pad_vector


DEFAULT_TASK_PROMPT = (
    "approach the glass on the table and grasp the cylinder and lift the cylinder "
    "and then bring down the cylinder to the table and then withdraw your hands"
)

LIFT_SUCCESS_THRESHOLD_M = 0.03
RELEASE_LIFT_THRESHOLD_M = 0.02
RELEASE_LOWER_THRESHOLD_M = 0.015


class ReleaseGuard:
    """Detects lift→lower cycle and forces gripper open."""

    def __init__(self):
        self.peak_lift_m = 0.0
        self.triggered = False
        self.init_z = None

    def update(self, glass_z: float, is_grasping: bool) -> bool:
        if self.init_z is None:
            self.init_z = glass_z
        if self.triggered:
            return True

        lift = glass_z - self.init_z
        if lift > self.peak_lift_m:
            self.peak_lift_m = lift

        lowered_from_peak = self.peak_lift_m - lift
        if (
            is_grasping
            and self.peak_lift_m >= RELEASE_LIFT_THRESHOLD_M
            and lowered_from_peak >= RELEASE_LOWER_THRESHOLD_M
        ):
            self.triggered = True
            return True

        return False


def apply_calibrated_gain(
    action: np.ndarray,
    eef_pos: np.ndarray,
    gamma_x: float = 1.80,
    gamma_z: float = 1.25,
    gamma_lift: float = 1.25,
    x_target_limit: float = -0.055,
    z_descent_limit: float = 0.855,
    is_grasping: bool = False,
) -> np.ndarray:
    act = action.copy()
    gripper_open = act[6] < 0.2

    if gripper_open and act[0] > 0.0 and eef_pos[0] < x_target_limit:
        act[0] = np.clip(act[0] * gamma_x, -1.0, 1.0)

    if gripper_open and act[2] < 0.0 and eef_pos[2] > z_descent_limit:
        act[2] = np.clip(act[2] * gamma_z, -1.0, 1.0)
    elif is_grasping and act[2] > 0.0:
        act[2] = np.clip(act[2] * gamma_lift, -1.0, 1.0)

    return np.clip(act, -1.0, 1.0)


def run_calibrated_rollout(
    policy: SmolVLAPolicy,
    gamma_x: float = 1.80,
    gamma_z: float = 1.25,
    gamma_lift: float = 1.25,
    z_descent_limit: float = 0.855,
    task_prompt: str = DEFAULT_TASK_PROMPT,
    max_steps: int = 340,
    chunk_exec_steps: int = 5,
    device: Optional[torch.device] = None,
    render_gui: bool = False,
    seed: int = 6,
    enable_release_guard: bool = True,
) -> Dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(
        f"[*] Calibrated rollout: γ_x={gamma_x:.2f} γ_z={gamma_z:.2f} "
        f"γ_lift={gamma_lift:.2f} K={chunk_exec_steps} steps={max_steps} "
        f"release_guard={enable_release_guard} seed={seed}"
    )

    release_guard = ReleaseGuard() if enable_release_guard else None

    lang_tokens, lang_masks = baseline.build_language_tokens(
        policy, task_prompt, device,
    )

    env = PandaSimEnvironment(
        env_name="GlassLift",
        has_renderer=render_gui,
        has_offscreen_renderer=True,
        camera_name="agentview",
        camera_height=256,
        camera_width=256,
    )

    sim_frames = []
    actions_executed = []
    eef_positions = []
    tumbler_positions = []
    glass_z_list = []
    contact_flags = []
    grasp_flags = []
    rewards = []
    d_xy_distances = []

    action_queue: List[np.ndarray] = []

    try:
        obs = env.reset()
        init_glass_z = env.env.get_object_z()
        cylinder_body_id = env.env.sim.model.body_name2id(env.env.cube.root_body)

        for step in range(max_steps):
            rgb = obs["rgb"].copy()
            sim_frames.append(rgb)

            eef_pos = obs["robot_eef_pos"].copy()
            eef_positions.append(eef_pos)

            cyl_pos = env.env.sim.data.body_xpos[cylinder_body_id].copy()
            tumbler_positions.append(cyl_pos)

            glass_z = env.env.get_object_z()
            glass_z_list.append(glass_z)

            d_xy = float(np.linalg.norm(eef_pos[:2] - cyl_pos[:2]))
            d_xy_distances.append(d_xy)

            if not action_queue:
                image = torch.from_numpy(
                    rgb.transpose(2, 0, 1)
                ).float().to(device).unsqueeze(0)
                image = (image / 255.0) * 2.0 - 1.0
                image_mask = torch.ones(1, dtype=torch.bool, device=device)

                state_8d = baseline.build_state_8d(obs)
                state_tensor = torch.from_numpy(state_8d).to(device).unsqueeze(0)
                padded_state = pad_vector(state_tensor, policy.config.max_state_dim)

                with torch.no_grad():
                    predicted_chunks = policy.model.sample_actions(
                        [image], [image_mask],
                        lang_tokens, lang_masks,
                        padded_state,
                    )

                raw_actions = predicted_chunks[0, :, :7].detach().cpu().numpy()
                for i in range(min(chunk_exec_steps, len(raw_actions))):
                    action_queue.append(raw_actions[i].copy())

            raw_action = action_queue.pop(0)
            currently_grasping = (
                len(grasp_flags) > 0 and grasp_flags[-1] > 0.5
            ) if step > 0 else False
            action = apply_calibrated_gain(
                raw_action, eef_pos,
                gamma_x=gamma_x,
                gamma_z=gamma_z,
                gamma_lift=gamma_lift,
                z_descent_limit=z_descent_limit,
                is_grasping=currently_grasping,
            )

            if release_guard is not None:
                should_release = release_guard.update(
                    glass_z, currently_grasping,
                )
                if should_release:
                    action[6] = -1.0

            if action.shape != (7,) or not np.isfinite(action).all():
                raise RuntimeError(f"Invalid action at step {step}: {action}")

            actions_executed.append(action.copy())
            obs, reward, done, _ = env.step(action)

            if render_gui:
                env.render()

            contact_flags.append(float(obs.get("tumbler_contact", False)))
            grasp_flags.append(float(obs.get("tumbler_grasped", False)))
            rewards.append(float(reward))

            if step % 20 == 0:
                lift_cm = (glass_z - init_glass_z) * 100
                print(
                    f"  step={step:03d} x={eef_pos[0]:+.4f} z={eef_pos[2]:.4f} "
                    f"d_xy={d_xy * 100:.2f}cm lift={lift_cm:+.2f}cm "
                    f"grip={action[6]:+.3f} "
                    f"contact={int(contact_flags[-1])} grasp={int(grasp_flags[-1])}"
                )

    finally:
        env.close()

    sim_frames = np.asarray(sim_frames, dtype=np.uint8)
    actions_arr = np.asarray(actions_executed, dtype=np.float32)
    eef_arr = np.asarray(eef_positions, dtype=np.float32)
    tumbler_arr = np.asarray(tumbler_positions, dtype=np.float32)
    glass_z_arr = np.asarray(glass_z_list, dtype=np.float32)
    contact_arr = np.asarray(contact_flags, dtype=np.float32)
    grasp_arr = np.asarray(grasp_flags, dtype=np.float32)

    max_lift = max(0, float(np.max(glass_z_arr) - init_glass_z))
    final_lift = max(0, float(glass_z_arr[-1] - init_glass_z))
    grasp_occurred = bool(np.any(grasp_arr > 0.5))
    contact_occurred = bool(np.any(contact_arr > 0.5))
    first_contact = int(np.where(contact_arr > 0.5)[0][0]) if contact_occurred else -1
    first_grasp = int(np.where(grasp_arr > 0.5)[0][0]) if grasp_occurred else -1
    min_d_xy = float(np.min(d_xy_distances))

    return {
        "sim_frames": sim_frames,
        "actions": actions_arr,
        "eef_positions": eef_arr,
        "tumbler_positions": tumbler_arr,
        "glass_z": glass_z_arr,
        "contact_flags": contact_arr,
        "grasp_flags": grasp_arr,
        "rewards": np.asarray(rewards, dtype=np.float32),
        "contact_occurred": contact_occurred,
        "first_contact_frame": first_contact,
        "grasp_occurred": grasp_occurred,
        "first_grasp_frame": first_grasp,
        "max_lift_cm": float(max_lift * 100),
        "final_lift_cm": float(final_lift * 100),
        "success": grasp_occurred and max_lift >= LIFT_SUCCESS_THRESHOLD_M,
        "min_d_xy_cm": float(min_d_xy * 100),
        "max_reach_x_m": float(np.min(eef_arr[:, 0])),
        "lowest_eef_z_m": float(np.min(eef_arr[:, 2])),
        "gamma_x": gamma_x,
        "gamma_z": gamma_z,
        "gamma_lift": gamma_lift,
        "k_steps": chunk_exec_steps,
        "seed": seed,
    }
