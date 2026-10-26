"""
Kinematic Retargeting: Human Hand Trajectory to Franka Panda 7D Action Space
Maps camera-space wrist displacements (dx, dy, dz) and pinch state to the
Panda robot base frame for RoboSuite / LIBERO 7D Operational Space Control (OSC).
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


class HandToPandaRetargeter:
    """
    Transforms MediaPipe normalized 3D hand displacements into
    Franka Panda 7D OSC actions: [dx, dy, dz, droll, dpitch, dyaw, gripper].
    """

    def __init__(
        self,
        camera_tilt_deg: float = 45.0,
        position_gain: float = 8.0,
        smoothing_alpha: float = 0.4,
        max_step_displacement: float = 0.8,
    ):
        """
        Args:
            camera_tilt_deg: Downward pitch angle of phone camera in degrees (~35-45 deg).
            position_gain: Scaling factor mapping normalized screen deltas to robot action units.
            smoothing_alpha: Exponential moving average filter coefficient (0 = full smoothing, 1 = no smoothing).
            max_step_displacement: Action clamp limit in [-1.0, 1.0].
        """
        self.camera_tilt_rad = np.radians(camera_tilt_deg)
        self.position_gain = position_gain
        self.smoothing_alpha = smoothing_alpha
        self.max_step_displacement = max_step_displacement

        # Rotation matrix around X-axis to compensate for downward camera pitch:
        # Camera frame: +X right, +Y down, +Z forward/depth
        # We rotate by -tilt around X to align with horizontal table plane
        c = np.cos(-self.camera_tilt_rad)
        s = np.sin(-self.camera_tilt_rad)
        self.R_cam_pitch = np.array([
            [1.0, 0.0, 0.0],
            [0.0,   c,  -s],
            [0.0,   s,   c],
        ], dtype=np.float32)

    def retarget_trajectory(
        self,
        npz_trajectory_file: str,
        output_file: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Loads extracted trajectory from hand_tracker.py and converts it to Panda actions.
        """
        path = Path(npz_trajectory_file)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")

        data = np.load(path)
        wrist_deltas = data["wrist_deltas"]          # shape: (T, 3) in cam frame [dx_cam, dy_cam, dz_cam]
        gripper_states = data["gripper_states"]      # shape: (T,) in [-1, 1]
        timestamps = data["timestamps"]
        T = len(wrist_deltas)

        # 7-DOF Panda Action array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        actions_7d = np.zeros((T, 7), dtype=np.float32)
        smoothed_pos_delta = np.zeros(3, dtype=np.float32)

        for t in range(T):
            dx_cam, dy_cam, dz_cam = wrist_deltas[t]

            # Step 1: Rotate to correct for camera pitch
            # [dx_level, dy_level, dz_level]
            v_cam = np.array([dx_cam, dy_cam, dz_cam], dtype=np.float32)
            v_level = self.R_cam_pitch @ v_cam

            # Step 2: Map camera directions to Panda base frame:
            # - Robot +X is Forward across table  <-- corresponds to camera +Z (depth/forward)
            # - Robot +Y is Left across table     <-- corresponds to camera -X (since cam +X is right)
            # - Robot +Z is Up (lift)            <-- corresponds to camera -Y (since cam +Y is down)
            dx_robot = float(v_level[2]) * self.position_gain   # Forward
            dy_robot = -float(v_level[0]) * self.position_gain  # Left/Right
            dz_robot = -float(v_level[1]) * self.position_gain  # Up/Down

            raw_delta = np.array([dx_robot, dy_robot, dz_robot], dtype=np.float32)

            # Step 3: Apply Exponential Moving Average (EMA) smoothing to eliminate hand jitter
            if t == 0:
                smoothed_pos_delta = raw_delta
            else:
                smoothed_pos_delta = (
                    self.smoothing_alpha * raw_delta
                    + (1.0 - self.smoothing_alpha) * smoothed_pos_delta
                )

            # Step 4: Clamp to safe robot action bounds
            clamped_delta = np.clip(
                smoothed_pos_delta,
                -self.max_step_displacement,
                self.max_step_displacement,
            )

            # Keep top-down orientation stable (droll=0, dpitch=0, dyaw=0)
            actions_7d[t, 0:3] = clamped_delta
            actions_7d[t, 3:6] = 0.0
            actions_7d[t, 6] = gripper_states[t]

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


def retarget_all_trajectories(dir_path: str = "data/processed_trajectories"):
    p = Path(dir_path)
    traj_files = sorted(list(p.glob("*_trajectory.npz")))
    if not traj_files:
        print(f"[-] No trajectory files found in {dir_path}")
        return
    print(f"[*] Found {len(traj_files)} trajectories to retarget in {dir_path}")
    retargeter = HandToPandaRetargeter()
    for f in traj_files:
        retargeter.retarget_trajectory(str(f))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        arg_path = Path(sys.argv[1])
        retargeter = HandToPandaRetargeter()
        if arg_path.is_dir():
            retarget_all_trajectories(str(arg_path))
        else:
            retargeter.retarget_trajectory(str(arg_path))
    else:
        retarget_all_trajectories()
