"""
Kinematic Retargeting: CoTracker3 Trajectories to Franka Panda 7D Action Space
Maps CoTracker3 dense point trajectories (wrist deltas + co-motion gripper state)
to the Franka Panda base frame for RoboSuite OSC control.

Incorporates:
- 60° camera inclination angle un-projection (v_reach / sin(theta) horizontal tabletop reach)
- Goal-directed pad targeting: cosine-ramp EEF trajectory so finger pads center on glass axis
- Upward vertical lift upon co-motion grasp latch (rho_t > 0.55)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional

# CoTracker input resolution: 512 x 512
target_frame_size = 512
class CoTrackerToPandaRetargeter:
    """
    Converts motion measured by CoTracker3 into commands that can be sent to the Franka Panda robot.
    Panda OSC action format: [dx, dy, dz, droll, dpitch, dyaw, gripper].
    Where, 
    - dx, dy, dz -> Cartesian end-effector translation commands
    - droll, dpitch, dyaw -> End-effector rotation commands
    - gripper-> Open / close gripper command
    
    So each video frame eventually produces one 7-dimensional
    robot action.
    """


    # edit position_gain, and tilt agnle
    # iteration 1 : 10.0, and angle = 0
    # iteration 2: 8.0 and angle = 60 deg
    # iteration 4: eef_to_pad_offset_x = 0.043 (measured from Iteration 3 diagnostic)
    # iteration 5: goal-directed pad targeting — target EEF X so pads center on glass
    def __init__(self, camera_tilt_deg: float = 60.0, position_gain: float = 8.0, smoothing_alpha: float = 0.4, max_step_displacement: float = 0.8, eef_to_pad_offset_x: float = 0.043,
                 glass_x: float = -0.035, eef_init_x: float = -0.103):
    
        """
        Initializes the retargeter: 
            1. camera_tilt_deg:Tilt angle of the phone/camera relative to the robot/world frame. 
                - Ex:  0 degrees  -> camera considered level
                        30 degrees -> camera pitched by 30 degrees
                - This is useful because movement observed in camera coordinates
                may not directly correspond to Panda X/Y/Z coordinates
            
            2. position_gain: Multiplier applied to the human-hand movement.
                - Ex: raw hand delta = 0.01, position_gain = 8
                    scaled delta = 0.01 * 8 = 0.08
                - Increasing this makes the Panda react more strongly to the same amount of human-hand movement.
        
            3. smoothing_alpha: Controls temporal smoothing of the movement. (Alpha depended, how much new value velocity to take)
                Usually used like: smoothed = alpha * new_delta + (1 - alpha) * previous_smoothed_delta
            - Example with alpha = 0.4:
                    - 40% current measurement
                    - 60% previous "smoothed value"
            - Smaller alpha: smoother motion, but slower response.
            - Larger alpha: faster response, but more tracking noise/jitter.
        """
            
        
        self.camera_tilt_deg = camera_tilt_deg # Store the camera tilt angle in degrees.
        self.camera_tilt_rad = np.radians(camera_tilt_deg) # ex: 30 degrees -> 0.5236 radians

        # sin and cos. This will later be used for rotating the hand trajectory
        # from camera coordinates into robot/world coordinates
        self.sin_theta = float(np.sin(self.camera_tilt_rad)) # numpy to float
        self.cos_theta = float(np.cos(self.camera_tilt_rad))
        self.position_gain = position_gain
        self.smoothing_alpha = smoothing_alpha
        self.max_step_displacement = max_step_displacement
        self.eef_to_pad_offset_x = eef_to_pad_offset_x
        # Iteration 5: Goal-directed pad targeting
        # Target EEF X = glass_x - eef_to_pad_offset_x so that finger pads center on glass
        self.glass_x = glass_x
        self.eef_init_x = eef_init_x
        self.target_eef_x = glass_x - eef_to_pad_offset_x  # -0.035 - 0.043 = -0.078

    def retarget_trajectory(self, npz_trajectory_file: str,output_file: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Loads the trajectory generated from CoTracker3 and converts it into Panda 7D OSC actions.
        
        npz_trajectory_file: path to the .npz file containing extracted human trajectory.
        output file : Optional path where the converted Panda actions can later be saved
        
        It returns a dictionary whose values are NumPy arrays.
        """
        path = Path(npz_trajectory_file) # file path
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")

        data = np.load(path) # load the .npz file

        """ Load the frame-to-frame wrist movement. 
        Example: frame 0 -> [ 0.00,  0.00,  0.00]
        frame 1 -> [ 0.01, -0.02,  0.00]
        frame 2 -> [ 0.02, -0.01, -0.01]
        Each row is approximately:[dx, dy, dz]
        """
        wrist_deltas = data["wrist_deltas"]          # (T, 3): frame-to-frame deltas
        gripper_states = data["gripper_states"]      # (T,) in [-1, 1]
        timestamps = data["timestamps"]  # Load the timestamp belonging to every trajectory frame.

        # If wrist_deltas.shape == (200, 3): T = 200
        T = len(wrist_deltas)
        # Check whether the NPZ also contains absolute wrist positions. If yes use them, otherwise create T X 3 array filled with zeros.
        wrist_positions = data["wrist_positions"] if "wrist_positions" in data else np.zeros((T, 3), dtype=np.float32)

        # Check that BOTH arrays exist. has_tracks becomes either True or False.
        has_tracks = "glass_tracks" in data and "hand_tracks" in data
        if has_tracks:
            # Normalized image coordinates in [0, 1]
            glass_c = data["glass_tracks"].mean(axis=1) / target_frame_size  # (T, 2)
            hand_c = data["hand_tracks"].mean(axis=1) / target_frame_size    # (T, 2)
        else:
            "If I DO NOT have glass_tracks, assume that the glass is sitting at the center of the image. [0.5, 0.5]"
            glass_c = np.tile([0.5, 0.5], (T, 1))
            hand_c = wrist_positions[:, :2]

        actions_7d = np.zeros((T, 7), dtype=np.float32)
        smoothed_pos_delta = np.zeros(3, dtype=np.float32)

        # ============================================================
        # ITERATION 1 EXPERIMENT LOGGING
        # ============================================================
        """
        Store what the retargeter sees and what action it generates at every frame.
        Actual robot/object positions are recorded later 
        by replay_trajectory.py after MuJoCo executes the action.
        """
        retarget_log = [] # python list for every frame


        # ============================================================
        # FIND WHEN THE GRASP STARTS (+1)
        # ============================================================
        first_grasp_idx = np.where(gripper_states > 0)[0]

        # t_grasp = first frame where grasp starts
        # Ex: first_grasp_idx = [84, 85, 86, 87, ...], then: t_grasp = 84
        t_grasp = int(first_grasp_idx[0]) if len(first_grasp_idx) > 0 else T

        # Identify hand entry frame
        # First check whether your .npz file already contains "t_hand".
        if "t_hand" in data and int(data["t_hand"]) > 0:
            t_entry = int(data["t_hand"])
        else:
            t_entry = 28
            for t in range(t_grasp):
                if np.linalg.norm(wrist_deltas[t]) > 0.005:
                    t_entry = t
                    break
        reach_span = max(1, t_grasp - t_entry)

        """ 
            This block converts the observed 2D cup/glass motion in the video into a simplified 7-DoF Franka/Panda action sequence. 
            The main idea is:
            wait → reach cup → close gripper → lift/return using cup motion → open gripper → retract
        """
        # Extract glass vertical profile (Pick -> Lift -> Return to Table -> Release)
        if has_tracks:
            glass_y = data["glass_tracks"].mean(axis=1)[:, 1] # average all glass tracking point: (T, 2) then [:, 1]-> selects only the image Y coordinate.
            """ 
            top of image
            y = 0
            ↑ glass lifted
            |
            |
            y increases
            ↓
            bottom of image

            so when we lift the glass upward, normally its image y value decreases.
            """

            # calculate the median vertical position of the glass before t_grasp. 
            # Before grasping, the glass is assumed to be sitting on the table.

            """ Example:
            Frame       glass_y
            0           370
            1           369
            2           371
            3           370
            ...
            grasp       370
            """
            y_rest = float(np.median(glass_y[:t_grasp])) # median is a good choice because it's less sensitive to a few bad CoTracker points.
            # y_rest will be constant now

            # calculate how much the glass has been lifted
            """ glass_y will be change we subtract it from y_rest and see the difference
                Large lift px means the glass appears higher in the image. """
            lift_px = np.maximum(0.0, y_rest - glass_y)

            # Find the highest point of the lift
            # this search for the frame with the largest lift after grasping
            apex_idx = int(np.argmax(lift_px[t_grasp:])) + t_grasp if t_grasp < T else T - 1 
            # store max lift
            max_lift_px = float(lift_px[apex_idx]) if apex_idx < T else 0.0

            # Normalize the lift
            norm_lift = np.zeros(T, dtype=np.float32)
            if max_lift_px > 5.0:
                for t in range(t_grasp, T):
                    norm_lift[t] = lift_px[t] / max_lift_px
                # This converts pixel movement into approximately:
                # 0.0 = glass on table, 1.0 = maximum observed lift
            else:
                # Graceful bell-curve fallback if vertical motion variance is minimal
                lift_len = min(25, max(10, (T - t_grasp) // 3))
                for t in range(t_grasp + 14, min(T, t_grasp + 14 + 2 * lift_len)):
                    progress = (t - (t_grasp + 14)) / (2 * lift_len)
                    norm_lift[t] = np.sin(progress * np.pi)

            # Smooth norm_lift with moving average kernel
            kernel = np.ones(5, dtype=np.float32) / 5.0
            """ It removes little CoTracker fluctuations.
            5 frames / 30 FPS ≈ 0.167 seconds
            So roughly 167 ms of the trajectory contributes to this smoothing. """
            norm_lift_smooth = np.convolve(norm_lift, kernel, mode="same")
            norm_lift_smooth = np.clip(norm_lift_smooth, 0.0, 1.0)

            # Detect when the glass returns to the table
            #Detect touchdown frame where glass returns to tabletop after apex
            after_apex = np.arange(T) > apex_idx
            # now when to consider it is return to table < 3%
            table_returned = after_apex & (norm_lift_smooth < 0.03)
            t_returned = int(np.where(table_returned)[0][0]) if np.any(table_returned) else min(T - 10, apex_idx + 20)
            
            # Velocity profile via gradient
            d_norm = np.gradient(norm_lift_smooth) # change in normalized height per frame

            """Ex: d_norm > 0 → glass moving upward
                d_norm ≈ 0 → glass stationary
                d_norm < 0 → glass moving downward
            And later you map this to Panda Z motion.
             """
        else:
            t_returned = min(T - 10, t_grasp + 40)
            d_norm = np.zeros(T, dtype=np.float32)

        # Generate one Panda action per video frame
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
                    # After the hand enters
                    # -------------------------------------------------------
                    # Iteration 5: GOAL-DIRECTED APPROACH WITH IMAGE DISTANCE
                    # -------------------------------------------------------
                    # Keep the image-based velocity approach (large enough for
                    # OSC to track) but modulate by image-space hand→glass
                    # distance so the robot decelerates as the hand closes in.
                    # Add the full eef_to_pad_offset as a one-shot bias in
                    # the first approach frame (not spread) so the robot
                    # aims pads at glass, not EEF at glass.
                    # -------------------------------------------------------

                    # Direction towards glass in image
                    v_reach = glass_c[t] - hand_c[t]
                    dist_reach = float(np.linalg.norm(v_reach))
                    u_reach = v_reach / (dist_reach + 1e-6)

                    # Human hand velocity projected onto reach direction
                    v_hand = np.array([dx_hand, dy_hand], dtype=np.float32)
                    v_approach = float(np.dot(v_hand, u_reach))

                    # Scale by image distance: full speed when far, decelerate near glass
                    # dist_reach is in [0,1] normalized image coords; typical start ~0.3
                    dist_gain = min(dist_reach / 0.15, 1.0)  # saturates above 0.15

                    reach_rate = max(0.002, v_approach) * dist_gain
                    dx_robot = (reach_rate / max(0.1, self.sin_theta)) * self.position_gain

                    # Iteration 5: Apply full pad offset as a one-shot impulse
                    # in the first 5 approach frames (concentrated, not spread).
                    # This gives OSC a large enough command to actually move.
                    frames_since_entry = t - t_entry
                    if frames_since_entry < 5:
                        dx_robot += self.eef_to_pad_offset_x / 5.0  # 0.043/5 = 0.0086 per frame

                    # Cylinder is aligned along center Y
                    dy_robot = 0.0

                    # Calibrated descent to align gripper right at cylinder height
                    dz_robot = -0.24 if t > (t_entry + 4) else 0.0
                grip = -1.0

            # Phase 2: Close the gripper. stop moving → close gripper → wait
            # immediately lifting on the exact grasp frame can produce unstable contacts.
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
            # Convert into XYZ action, Cartesian translation:    
            raw_delta = np.array([dx_robot, dy_robot, dz_robot], dtype=np.float32)

            # EMA smoothing to eliminate jitter
            if t == 0:
                smoothed_pos_delta = raw_delta
            else:
                smoothed_pos_delta = (self.smoothing_alpha * raw_delta + (1.0 - self.smoothing_alpha) * smoothed_pos_delta)

            # Clamp action magnitudes to robot limits
            clamped_delta = np.clip(smoothed_pos_delta, -self.max_step_displacement, self.max_step_displacement)

            actions_7d[t, 0:3] = clamped_delta
            actions_7d[t, 3:6] = 0.0              # Stable top-down wrist posture
            actions_7d[t, 6] = grip


            # ============================================================
            # ITERATION 1: RECORD THIS RETARGETING STEP
            # ============================================================


            if has_tracks:
                current_hand_centroid = hand_c[t]
                current_glass_centroid = glass_c[t]
            else:
                current_hand_centroid = np.array([np.nan, np.nan], dtype=np.float32)
                current_glass_centroid = np.array([np.nan, np.nan],dtype=np.float32)

            retarget_log.append(
                {
                    "frame": int(t),

                    "timestamp": float(timestamps[t]),

                    "hand_centroid_x": float(current_hand_centroid[0]),
                    "hand_centroid_y": float(current_hand_centroid[1]),

                    "tumbler_centroid_x": float(current_glass_centroid[0]),
                    "tumbler_centroid_y": float(current_glass_centroid[1]),

                    "dx_command": float(clamped_delta[0]),
                    "dy_command": float(clamped_delta[1]),
                    "dz_command": float(clamped_delta[2]),

                    "gripper_command": float(grip),

                    "source_gripper_state": float(
                        gripper_states[t]
                    ),
                }
            )


        # ============================================================
        # CONVERT EXPERIMENT LOG INTO NUMPY ARRAYS
        # ============================================================

        retarget_hand_centroids = np.array(
            [
                [
                    row["hand_centroid_x"],
                    row["hand_centroid_y"],
                ]
                for row in retarget_log
            ],
            dtype=np.float32,)

        retarget_tumbler_centroids = np.array(
            [
                [
                    row["tumbler_centroid_x"],
                    row["tumbler_centroid_y"],
                ]
                for row in retarget_log
            ],
            dtype=np.float32,
        )

        retarget_xyz_commands = np.array(
            [
                [
                    row["dx_command"],
                    row["dy_command"],
                    row["dz_command"],
                ]
                for row in retarget_log
            ], dtype=np.float32)

        retargeted_dict = {"actions": actions_7d,"timestamps": timestamps,"rgb_frames": data["rgb_frames"] if "rgb_frames" in data else None,"source_file": str(path)}

        retargeted_dict = {
            "actions": actions_7d,
            "timestamps": timestamps,

            "rgb_frames": (
                data["rgb_frames"]
                if "rgb_frames" in data
                else None),
            # Input tracking information
            "hand_centroids": retarget_hand_centroids,
            "tumbler_centroids": retarget_tumbler_centroids,
            # Actual commands generated by retargeter
            "xyz_commands": retarget_xyz_commands,
            # Original inferred gripper state
            "source_gripper_states": gripper_states,
            "source_file": str(path)
            }

        if output_file is None:
            output_file = (
                path.parent
                / f"{path.stem.replace('_trajectory', '')}_actions.npz")

        output_file = Path(output_file)
        # Make sure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Save NPZ
        np.savez_compressed(output_file, **retargeted_dict)
       
        # ============================================================
        # SAVE HUMAN-READABLE RETARGETING LOG
        # ============================================================

        output_file = Path(output_file)
        retarget_log_file = (output_file.parent
            / f"{output_file.stem}_log.txt")

        with open(retarget_log_file, "w") as f:
            f.write("ITERATION 1 - RETARGETING LOG\n")
            f.write("=" * 100 + "\n")

            f.write(
                "frame,time_s,"
                "hand_x,hand_y,"
                "tumbler_x,tumbler_y,"
                "dx,dy,dz,"
                "gripper\n"
            )

            for row in retarget_log:

                f.write(
                    f"{row['frame']},"
                    f"{row['timestamp']:.4f},"
                    f"{row['hand_centroid_x']:.6f},"
                    f"{row['hand_centroid_y']:.6f},"
                    f"{row['tumbler_centroid_x']:.6f},"
                    f"{row['tumbler_centroid_y']:.6f},"
                    f"{row['dx_command']:.6f},"
                    f"{row['dy_command']:.6f},"
                    f"{row['dz_command']:.6f},"
                    f"{row['gripper_command']:.1f}\n"
                )
       
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

    if len(sys.argv) < 2:
        print("Usage: python3 cotracker_to_panda.py "
            "<trajectory.npz> [output_folder]")
        sys.exit(1)

    input_file = sys.argv[1]
    # Optional output folder
    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])
    else:
        output_dir = Path("datasets/cotracker_iterations/iteration_05_goal_directed")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_file)
    output_file = (output_dir
    / f"{input_path.stem.replace('_trajectory', '')}_actions.npz")

    retargeter = CoTrackerToPandaRetargeter()
    retargeter.retarget_trajectory(input_file,output_file=str(output_file))

"""
python3 src/retargeting/cotracker_to_panda.py \
    datasets/cotracker_trajectories/cup_lifting_01_ct_trajectory.npz \
    datasets/cotracker_iterations/iteration_02_camera_roi
"""