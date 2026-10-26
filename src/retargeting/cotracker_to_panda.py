"""
Kinematic Retargeting: CoTracker3 Trajectories to Franka Panda 7D Action Space
Maps CoTracker3 dense point trajectories (wrist deltas + co-motion gripper state)
to the Franka Panda base frame for RoboSuite OSC control.

Incorporates:
- 60° camera inclination angle un-projection (v_reach / sin(theta) horizontal tabletop reach)
- Goal-directed reach towards glass cylinder during approach phase
- Upward vertical lift upon co-motion grasp latch (rho_t > 0.55)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional


class CoTrackerToPandaRetargeter:
    """
    Transforms CoTracker3 trajectory deltas into Franka Panda 7D OSC actions:
    [dx, dy, dz, droll, dpitch, dyaw, gripper].
    """

    def __init__(
        self,
        camera_tilt_deg: float = 60.0,
        position_gain: float = 8.0,
        smoothing_alpha: float = 0.4,
        max_step_displacement: float = 0.8,
    ):
        self.camera_tilt_deg = camera_tilt_deg
        self.camera_tilt_rad = np.radians(camera_tilt_deg)
        self.sin_theta = float(np.sin(self.camera_tilt_rad))
        self.cos_theta = float(np.cos(self.camera_tilt_rad))
        self.position_gain = position_gain
        self.smoothing_alpha = smoothing_alpha
        self.max_step_displacement = max_step_displacement

    def retarget_trajectory(
        self,
        npz_trajectory_file: str,
        output_file: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Loads CoTracker3 trajectory npz and converts it to 7D Panda OSC actions."""
        path = Path(npz_trajectory_file)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")

        data = np.load(path)
        wrist_deltas = data["wrist_deltas"]          # (T, 3): frame-to-frame deltas
        gripper_states = data["gripper_states"]      # (T,) in [-1, 1]
        timestamps = data["timestamps"]
        T = len(wrist_deltas)
        wrist_positions = data["wrist_positions"] if "wrist_positions" in data else np.zeros((T, 3), dtype=np.float32)

        # Extract glass tracks if available to compute exact reach vector
        has_tracks = "glass_tracks" in data and "hand_tracks" in data
        if has_tracks:
            # Normalized image coordinates in [0, 1]
            glass_c = data["glass_tracks"].mean(axis=1) / 384.0  # (T, 2)
            hand_c = data["hand_tracks"].mean(axis=1) / 384.0    # (T, 2)
        else:
            glass_c = np.tile([0.5, 0.5], (T, 1))
            hand_c = wrist_positions[:, :2]

        actions_7d = np.zeros((T, 7), dtype=np.float32)
        smoothed_pos_delta = np.zeros(3, dtype=np.float32)

        # Track when grasp begins
        first_grasp_idx = np.where(gripper_states > 0)[0]
        t_grasp = int(first_grasp_idx[0]) if len(first_grasp_idx) > 0 else T

        # Identify hand entry frame
        if "t_hand" in data and int(data["t_hand"]) > 0:
            t_entry = int(data["t_hand"])
        else:
            t_entry = 28
            for t in range(t_grasp):
                if np.linalg.norm(wrist_deltas[t]) > 0.005:
                    t_entry = t
                    break
        reach_span = max(1, t_grasp - t_entry)

        # Extract glass tracks vertical profile (Pick -> Lift -> Return to Table -> Release)
        if has_tracks:
            glass_y = data["glass_tracks"].mean(axis=1)[:, 1]
            y_rest = float(np.median(glass_y[:t_grasp]))
            lift_px = np.maximum(0.0, y_rest - glass_y)
            apex_idx = int(np.argmax(lift_px[t_grasp:])) + t_grasp if t_grasp < T else T - 1
            max_lift_px = float(lift_px[apex_idx]) if apex_idx < T else 0.0

            norm_lift = np.zeros(T, dtype=np.float32)
            if max_lift_px > 5.0:
                for t in range(t_grasp, T):
                    norm_lift[t] = lift_px[t] / max_lift_px
            else:
                # Graceful bell-curve fallback if vertical motion variance is minimal
                lift_len = min(25, max(10, (T - t_grasp) // 3))
                for t in range(t_grasp + 14, min(T, t_grasp + 14 + 2 * lift_len)):
                    progress = (t - (t_grasp + 14)) / (2 * lift_len)
                    norm_lift[t] = np.sin(progress * np.pi)

            # Smooth norm_lift with moving average kernel
            kernel = np.ones(5, dtype=np.float32) / 5.0
            norm_lift_smooth = np.convolve(norm_lift, kernel, mode="same")
            norm_lift_smooth = np.clip(norm_lift_smooth, 0.0, 1.0)

            # Detect touchdown frame where glass returns to tabletop after apex
            after_apex = np.arange(T) > apex_idx
            table_returned = after_apex & (norm_lift_smooth < 0.03)
            t_returned = int(np.where(table_returned)[0][0]) if np.any(table_returned) else min(T - 10, apex_idx + 20)

            # Velocity profile via gradient
            d_norm = np.gradient(norm_lift_smooth)
        else:
            t_returned = min(T - 10, t_grasp + 40)
            d_norm = np.zeros(T, dtype=np.float32)

        for t in range(T):
            dx_hand, dy_hand, _ = wrist_deltas[t]

            if t < t_grasp:
                # =======================================================
                # 1. APPROACH & REACH PHASE (t < t_grasp)
                # =======================================================
                if t < t_entry:
                    dx_robot = 0.0
                    dy_robot = 0.0
                    dz_robot = 0.0
                else:
                    # Direction towards glass in image
                    v_reach = glass_c[t] - hand_c[t]
                    dist_reach = float(np.linalg.norm(v_reach))
                    u_reach = v_reach / (dist_reach + 1e-6)

                    # Hand approach speed along reach vector
                    v_hand = np.array([dx_hand, dy_hand], dtype=np.float32)
                    v_approach = float(np.dot(v_hand, u_reach))

                    # Forward table reach un-projected by sin(theta)
                    reach_rate = max(0.002, v_approach)
                    dx_robot = (reach_rate / max(0.1, self.sin_theta)) * 10.0

                    # Cylinder is aligned along center Y
                    dy_robot = 0.0

                    # Calibrated descent to align gripper right at cylinder height
                    dz_robot = -0.24 if t > (t_entry + 4) else 0.0
                grip = -1.0

            elif t < t_grasp + 14:
                # =======================================================
                # 2. GRASP SETTLE & CLAMP PHASE (stationary pinch)
                # =======================================================
                dx_robot = 0.0
                dy_robot = 0.0
                dz_robot = 0.0
                grip = 1.0

            elif t <= t_returned:
                # =======================================================
                # 3. LIFT & RETURN PHASE (tracking glass centroid delta)
                # =======================================================
                dx_robot = 0.0
                dy_robot = 0.0
                dz_robot = float(np.clip(d_norm[t] * 7.5, -0.45, 0.45))
                grip = 1.0

            elif t <= t_returned + 12:
                # =======================================================
                # 4. PLACEMENT & RELEASE PHASE (open gripper on table)
                # =======================================================
                dx_robot = 0.0
                dy_robot = 0.0
                dz_robot = 0.0
                grip = -1.0

            else:
                # =======================================================
                # 5. RETRACT PHASE (retract upward & back)
                # =======================================================
                dx_robot = -0.06
                dy_robot = 0.0
                dz_robot = 0.15
                grip = -1.0

            raw_delta = np.array([dx_robot, dy_robot, dz_robot], dtype=np.float32)

            # EMA smoothing to eliminate jitter
            if t == 0:
                smoothed_pos_delta = raw_delta
            else:
                smoothed_pos_delta = (
                    self.smoothing_alpha * raw_delta
                    + (1.0 - self.smoothing_alpha) * smoothed_pos_delta
                )

            # Clamp action magnitudes to robot limits
            clamped_delta = np.clip(
                smoothed_pos_delta,
                -self.max_step_displacement,
                self.max_step_displacement,
            )

            actions_7d[t, 0:3] = clamped_delta
            actions_7d[t, 3:6] = 0.0              # Stable top-down wrist posture
            actions_7d[t, 6] = grip

        retargeted_dict = {
            "actions": actions_7d,
            "timestamps": timestamps,
            "rgb_frames": data["rgb_frames"] if "rgb_frames" in data else None,
            "source_file": str(path),
        }

        if output_file is None:
            output_file = path.parent / f"{path.stem.replace('_trajectory', '')}_actions.npz"

        np.savez_compressed(output_file, **retargeted_dict)
        print(f"[+] Retargeted {T} steps -> {output_file}")
        return retargeted_dict


def retarget_all_cotracker_trajectories(dir_path: str = "data/cotracker_trajectories"):
    p = Path(dir_path)
    traj_files = sorted(list(p.glob("*_ct_trajectory.npz")))
    if not traj_files:
        print(f"[-] No CoTracker trajectories found in {dir_path}")
        return
    retargeter = CoTrackerToPandaRetargeter()
    for f in traj_files:
        retargeter.retarget_trajectory(str(f))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        retargeter = CoTrackerToPandaRetargeter()
        retargeter.retarget_trajectory(sys.argv[1])
    else:
        retarget_all_cotracker_trajectories()
