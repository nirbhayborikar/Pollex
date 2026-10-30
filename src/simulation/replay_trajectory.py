"""
Trajectory Replayer & Side-by-Side Visualizer
Executes retargeted 7D actions in the simulation environment,
records robot rollouts, and compiles a synchronized side-by-side video:
[Left: Real Hand Video | Right: Simulated Panda Robot].

export PYTHONPATH="$HOME/Documents/vla/LIBERO:$PYTHONPATH"

export DISPLAY=:10
export XAUTHORITY="$HOME/.Xauthority"

unset MUJOCO_GL
unset QT_DEBUG_PLUGINS
unset QT_PLUGIN_PATH
unset QT_QPA_PLATFORM_PLUGIN_PATH

python3 -m src.simulation.replay_trajectory \
    --actions datasets/cotracker_iterations/iteration_01_raw/cup_lifting_01_ct_actions.npz \
    --out_video datasets/cotracker_iterations/iteration_01_raw/cup_lifting_01_iteration_1.mp4 \
    --out_dataset datasets/cotracker_iterations/iteration_01_raw/cup_lifting_01_replay.npz \
    --task GlassLift

To see the simulation window and use annotated human video:

python3 -m src.simulation.replay_trajectory \
    --actions datasets/cotracker_iterations/iteration_01_raw/cup_lifting_01_ct_actions.npz \
    --out_video datasets/cotracker_iterations/iteration_01_raw/cup_lifting_01_iteration_1.mp4 \
    --out_dataset datasets/cotracker_iterations/iteration_01_raw/cup_lifting_01_replay.npz \
    --human_video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 \
    --task GlassLift \
    --gui
"""

import cv2
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

from src.simulation.libero_runner import PandaSimEnvironment

LIFT_SUCCESS_THRESHOLD_M = 0.03


def replay_actions_and_record(
    actions_npz_path: str,
    output_video_path: str,
    output_dataset_path: Optional[str] = None,
    task_name: str = "Lift",
    render_gui: bool = False,
    human_video_path: Optional[str] = None,
):
    """
    Executes actions inside the simulator and outputs synchronized comparison video.
    Records physical measurements from MuJoCo: tumbler position, contact, grasp.
    Saves metrics.txt, optional paired NPZ dataset, and side-by-side video.
    """

    # 1. INPUT ACTION FILE
    actions_file = Path(actions_npz_path)
    if not actions_file.exists():
        raise FileNotFoundError(f"Actions file not found: {actions_file}")

    # 2. LOAD RETARGETED TRAJECTORY
    data = np.load(actions_file)
    actions = data["actions"]

    human_frames = None
    if human_video_path is not None:
        vid_path = Path(human_video_path)
        if not vid_path.exists():
            raise FileNotFoundError(f"Human video not found: {vid_path}")
        cap = cv2.VideoCapture(str(vid_path))
        h_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            h_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        human_frames = np.array(h_frames, dtype=np.uint8)
        print(f"[+] Loaded {len(human_frames)} frames from human video: {vid_path}")
    elif "rgb_frames" in data:
        human_frames = data["rgb_frames"]

    if "fps" in data:
        replay_fps = float(data["fps"])
    elif "timestamps" in data and len(data["timestamps"]) > 1:
        dt = np.median(np.diff(np.asarray(data["timestamps"], dtype=np.float64)))
        replay_fps = 1.0 / dt if np.isfinite(dt) and dt > 0 else 30.0
    else:
        replay_fps = 30.0

    if not np.isfinite(replay_fps) or replay_fps <= 0:
        replay_fps = 30.0

    T = len(actions)
    print(f"[*] Replaying {T} action steps in '{task_name}' environment...")
    print(f"[*] Replay / comparison FPS: {replay_fps:.2f}")

    # 3. CREATE PANDA / MUJOCO SIMULATION
    env = PandaSimEnvironment(
        env_name=task_name, has_renderer=render_gui, has_offscreen_renderer=True,
        camera_name="agentview", camera_height=256, camera_width=256,
    )

    # 4. RESET AND READ INITIAL PHYSICAL STATE
    obs = env.reset()
    print("[*] Observation keys:", sorted(obs.keys()))

    initial_eef_pos = None
    eef_value = obs.get("robot_eef_pos")
    if eef_value is not None:
        initial_eef_pos = np.asarray(eef_value, dtype=np.float32).copy()

    # Read initial tumbler position directly from standardized interface
    initial_tumbler_pos = None
    tumbler_pos_source = obs.get("tumbler_pos_source")
    tumbler_init = obs.get("tumbler_pos")
    if tumbler_init is not None:
        tumbler_init = np.asarray(tumbler_init, dtype=np.float32).reshape(-1)[:3].copy()
        if tumbler_init.shape == (3,) and np.all(np.isfinite(tumbler_init)):
            initial_tumbler_pos = tumbler_init
            print(f"[PHYSICS] Initial tumbler XYZ: {initial_tumbler_pos}")
            print(f"[PHYSICS] Source: {tumbler_pos_source}")
        else:
            print(f"[PHYSICS] ERROR: Invalid initial tumbler position: {tumbler_init}")
    else:
        print("[PHYSICS] ERROR: tumbler_pos is None after reset")

    # Grasp diagnostic: finger pad geometry info
    left_pad_geom_name = obs.get("left_finger_pad_geom")
    right_pad_geom_name = obs.get("right_finger_pad_geom")
    glass_radius = float(obs.get("glass_radius", 0.030))

    # Storage arrays
    sim_frames = []
    recorded_robot_states = []
    recorded_robot_state_frames = []
    recorded_tumbler_positions = []
    recorded_tumbler_frames = []
    recorded_left_pad_positions = []
    recorded_right_pad_positions = []
    recorded_contacts = []
    recorded_grasps = []
    recorded_rewards = []
    recorded_actions = []

    # 5. REPLAY EVERY ROBOT ACTION
    for t in range(T):
        action = actions[t]
        obs, reward, done, info = env.step(action)

        recorded_actions.append(np.asarray(action, dtype=np.float32).copy())
        recorded_rewards.append(float(reward))

        # 6. RECORD SIMULATION RGB FRAME
        if obs.get("rgb") is not None:
            sim_frames.append(np.asarray(obs["rgb"], dtype=np.uint8).copy())

        # 7. RECORD ACTUAL PANDA END-EFFECTOR POSITION
        eef_pos = None
        eef_value = obs.get("robot_eef_pos")
        if eef_value is not None:
            eef_pos = np.asarray(eef_value, dtype=np.float32).reshape(-1)[:3].copy()
            recorded_robot_states.append(eef_pos)
            recorded_robot_state_frames.append(t)

        # 8. RECORD ACTUAL TUMBLER POSITION (from MuJoCo body_xpos)
        tumbler_pos = None
        tp = obs.get("tumbler_pos")
        if tp is not None:
            tumbler_pos = np.asarray(tp, dtype=np.float32).reshape(-1)[:3].copy()
            if tumbler_pos.shape == (3,) and np.all(np.isfinite(tumbler_pos)):
                recorded_tumbler_positions.append(tumbler_pos)
                recorded_tumbler_frames.append(t)
            else:
                print(f"[PHYSICS] ERROR: Invalid tumbler position at frame {t}: {tumbler_pos}")
                tumbler_pos = None
        else:
            print(f"[PHYSICS] ERROR: tumbler_pos missing at frame {t}")

        # 8b. RECORD FINGER PAD POSITIONS (from MuJoCo geom_xpos)
        left_pad = obs.get("left_finger_pad_pos")
        right_pad = obs.get("right_finger_pad_pos")
        recorded_left_pad_positions.append(
            np.asarray(left_pad, dtype=np.float32).copy() if left_pad is not None
            else np.full(3, np.nan, dtype=np.float32)
        )
        recorded_right_pad_positions.append(
            np.asarray(right_pad, dtype=np.float32).copy() if right_pad is not None
            else np.full(3, np.nan, dtype=np.float32)
        )

        # 9. RECORD PHYSICAL CONTACT (from MuJoCo contact pairs)
        contact_value = obs.get("tumbler_contact")
        recorded_contacts.append(float(contact_value) if contact_value is not None else np.nan)

        # 10. RECORD PHYSICAL GRASP (from RoboSuite _check_grasp)
        grasp_value = obs.get("tumbler_grasped")
        recorded_grasps.append(float(grasp_value) if grasp_value is not None else np.nan)

        # 11. PER-FRAME LOG
        eef_text = np.array2string(eef_pos, precision=4) if eef_pos is not None else "N/A"
        tumbler_text = np.array2string(tumbler_pos, precision=4) if tumbler_pos is not None else "N/A"
        print(
            f"[REPLAY] frame={t:04d} action_xyz=[{action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}] "
            f"grip={action[6]:+.1f} eef={eef_text} tumbler={tumbler_text} "
            f"contact={contact_value} grasp={grasp_value} reward={float(reward):.4f}"
        )

        if render_gui:
            env.render()

    env.close()
    print(f"[+] Replay finished. Captured {len(sim_frames)} simulation frames.")

    # 13. CONVERT TO NUMPY ARRAYS
    executed_actions = np.asarray(recorded_actions, dtype=np.float32)
    rewards = np.asarray(recorded_rewards, dtype=np.float32)
    eef_positions = np.asarray(recorded_robot_states, dtype=np.float32)
    eef_position_frames = np.asarray(recorded_robot_state_frames, dtype=np.int32)
    tumbler_positions = np.asarray(recorded_tumbler_positions, dtype=np.float32)
    tumbler_position_frames = np.asarray(recorded_tumbler_frames, dtype=np.int32)
    contact_flags = np.asarray(recorded_contacts, dtype=np.float32)
    grasp_flags = np.asarray(recorded_grasps, dtype=np.float32)
    left_pad_positions = np.asarray(recorded_left_pad_positions, dtype=np.float32)
    right_pad_positions = np.asarray(recorded_right_pad_positions, dtype=np.float32)

    # 14. CALCULATE ACTUAL PANDA EEF MOTION
    final_eef_pos = None
    max_eef_dx = max_eef_dy = max_eef_dz = np.nan
    min_eef_dx = max_signed_eef_dx = np.nan
    min_eef_dy = max_signed_eef_dy = np.nan
    min_eef_dz = max_signed_eef_dz = np.nan

    if len(eef_positions) > 0:
        if initial_eef_pos is None:
            initial_eef_pos = eef_positions[0].copy()
        final_eef_pos = eef_positions[-1].copy()
        eef_displacements = eef_positions - initial_eef_pos[None, :]
        max_eef_dx = float(np.max(np.abs(eef_displacements[:, 0])))
        max_eef_dy = float(np.max(np.abs(eef_displacements[:, 1])))
        max_eef_dz = float(np.max(np.abs(eef_displacements[:, 2])))
        min_eef_dx = float(np.min(eef_displacements[:, 0]))
        max_signed_eef_dx = float(np.max(eef_displacements[:, 0]))
        min_eef_dy = float(np.min(eef_displacements[:, 1]))
        max_signed_eef_dy = float(np.max(eef_displacements[:, 1]))
        min_eef_dz = float(np.min(eef_displacements[:, 2]))
        max_signed_eef_dz = float(np.max(eef_displacements[:, 2]))

    # 15. CALCULATE ACTUAL TUMBLER LIFT
    initial_tumbler_z = maximum_tumbler_z = maximum_tumbler_lift = np.nan
    lift_success = None

    if len(tumbler_positions) > 0 and initial_tumbler_pos is not None:
        initial_tumbler_z = float(initial_tumbler_pos[2])
        maximum_tumbler_z = float(np.max(tumbler_positions[:, 2]))
        maximum_tumbler_lift = float(maximum_tumbler_z - initial_tumbler_z)
        lift_success = bool(maximum_tumbler_lift >= LIFT_SUCCESS_THRESHOLD_M)

    # 16. MINIMUM EEF-TO-TUMBLER DISTANCE
    min_eef_tumbler_distance = np.nan
    closest_approach_frame = -1

    n_paired = min(len(eef_positions), len(tumbler_positions))
    if n_paired > 0:
        eef_tumbler_distances = np.linalg.norm(eef_positions[:n_paired] - tumbler_positions[:n_paired], axis=1)
        min_eef_tumbler_distance = float(np.min(eef_tumbler_distances))
        closest_approach_frame = int(np.argmin(eef_tumbler_distances))

    # 17. CONTACT / GRASP OUTCOMES
    contact_occurred = False
    first_contact_frame = -1
    valid_contacts = contact_flags[np.isfinite(contact_flags)]
    if len(valid_contacts) > 0:
        contact_occurred = bool(np.any(valid_contacts > 0.5))
        contact_indices = np.where(contact_flags > 0.5)[0]
        if len(contact_indices) > 0:
            first_contact_frame = int(contact_indices[0])

    grasp_occurred = False
    first_grasp_frame = -1
    valid_grasps = grasp_flags[np.isfinite(grasp_flags)]
    if len(valid_grasps) > 0:
        grasp_occurred = bool(np.any(valid_grasps > 0.5))
        grasp_indices = np.where(grasp_flags > 0.5)[0]
        if len(grasp_indices) > 0:
            first_grasp_frame = int(grasp_indices[0])

    # 17b. GRASP GEOMETRY DIAGNOSTIC — three independent closest frames
    diag = {"available": False}

    if n_paired > 0 and np.all(np.isfinite(left_pad_positions[:n_paired])) and np.all(np.isfinite(right_pad_positions[:n_paired])):
        diag["available"] = True
        diag["left_pad_geom_name"] = left_pad_geom_name
        diag["right_pad_geom_name"] = right_pad_geom_name
        diag["glass_radius"] = glass_radius

        # EEF -> tumbler per-frame offsets
        eef_obj_delta = tumbler_positions[:n_paired] - eef_positions[:n_paired]
        diag["eef_closest_frame"] = closest_approach_frame
        diag["eef_xyz"] = eef_positions[closest_approach_frame]
        diag["tumbler_xyz_at_eef"] = tumbler_positions[closest_approach_frame]
        diag["eef_dx"] = float(eef_obj_delta[closest_approach_frame, 0])
        diag["eef_dy"] = float(eef_obj_delta[closest_approach_frame, 1])
        diag["eef_dz"] = float(eef_obj_delta[closest_approach_frame, 2])
        diag["eef_horizontal_xy"] = float(np.linalg.norm(eef_obj_delta[closest_approach_frame, :2]))
        diag["eef_vertical_dz"] = float(eef_obj_delta[closest_approach_frame, 2])

        # Left pad -> tumbler center distances (3D) and horizontal-to-axis
        left_pad_dists_3d = np.linalg.norm(left_pad_positions[:n_paired] - tumbler_positions[:n_paired], axis=1)
        diag["left_closest_frame"] = int(np.argmin(left_pad_dists_3d))
        diag["min_left_pad_dist"] = float(left_pad_dists_3d[diag["left_closest_frame"]])
        lf = diag["left_closest_frame"]
        diag["left_pad_xyz"] = left_pad_positions[lf]
        diag["tumbler_xyz_at_left"] = tumbler_positions[lf]
        diag["left_horizontal_to_axis"] = float(np.linalg.norm(
            left_pad_positions[lf, :2] - tumbler_positions[lf, :2]))
        diag["left_vertical_offset"] = float(left_pad_positions[lf, 2] - tumbler_positions[lf, 2])

        # Right pad -> tumbler center distances
        right_pad_dists_3d = np.linalg.norm(right_pad_positions[:n_paired] - tumbler_positions[:n_paired], axis=1)
        diag["right_closest_frame"] = int(np.argmin(right_pad_dists_3d))
        diag["min_right_pad_dist"] = float(right_pad_dists_3d[diag["right_closest_frame"]])
        rf = diag["right_closest_frame"]
        diag["right_pad_xyz"] = right_pad_positions[rf]
        diag["tumbler_xyz_at_right"] = tumbler_positions[rf]
        diag["right_horizontal_to_axis"] = float(np.linalg.norm(
            right_pad_positions[rf, :2] - tumbler_positions[rf, :2]))
        diag["right_vertical_offset"] = float(right_pad_positions[rf, 2] - tumbler_positions[rf, 2])

        # Pad-pad separation at EEF closest frame
        diag["pad_separation_at_eef"] = float(np.linalg.norm(
            left_pad_positions[closest_approach_frame] - right_pad_positions[closest_approach_frame]))

        # Pad-pad separation at each pad's own closest frame
        diag["pad_separation_at_left"] = float(np.linalg.norm(
            left_pad_positions[lf] - right_pad_positions[lf]))
        diag["pad_separation_at_right"] = float(np.linalg.norm(
            left_pad_positions[rf] - right_pad_positions[rf]))

        # Minimum pad separation across entire episode
        all_seps = np.linalg.norm(left_pad_positions[:n_paired] - right_pad_positions[:n_paired], axis=1)
        diag["min_pad_separation_episode"] = float(np.min(all_seps))
        diag["min_pad_separation_frame"] = int(np.argmin(all_seps))

        # Surface clearance estimates (pad half-thickness along closing axis ~1mm from XML)
        pad_half_thickness = 0.001
        diag["left_surface_clearance"] = diag["left_horizontal_to_axis"] - glass_radius - pad_half_thickness
        diag["right_surface_clearance"] = diag["right_horizontal_to_axis"] - glass_radius - pad_half_thickness

        # Glass dimensions for context
        diag["glass_diameter"] = glass_radius * 2

        print(f"\n[DIAG] EEF closest frame: {diag['eef_closest_frame']}  dist={diag['min_pad_separation_episode']:.4f}")
        print(f"[DIAG] Left pad closest frame: {diag['left_closest_frame']}  3D dist={diag['min_left_pad_dist']*100:.2f} cm  horiz-to-axis={diag['left_horizontal_to_axis']*100:.2f} cm")
        print(f"[DIAG] Right pad closest frame: {diag['right_closest_frame']}  3D dist={diag['min_right_pad_dist']*100:.2f} cm  horiz-to-axis={diag['right_horizontal_to_axis']*100:.2f} cm")
        print(f"[DIAG] Pad separation @ EEF closest: {diag['pad_separation_at_eef']*100:.2f} cm  (glass diameter: {diag['glass_diameter']*100:.1f} cm)")
        print(f"[DIAG] Pad separation @ left closest: {diag['pad_separation_at_left']*100:.2f} cm")
        print(f"[DIAG] Pad separation @ right closest: {diag['pad_separation_at_right']*100:.2f} cm")
        print(f"[DIAG] Min pad separation in episode: {diag['min_pad_separation_episode']*100:.2f} cm at frame {diag['min_pad_separation_frame']}")
        print(f"[DIAG] Left surface clearance: {diag['left_surface_clearance']*100:.2f} cm")
        print(f"[DIAG] Right surface clearance: {diag['right_surface_clearance']*100:.2f} cm")

    # 18. SAVE METRICS.TXT
    out_vid_path = Path(output_video_path)
    out_vid_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = out_vid_path.parent / "metrics.txt"

    def yes_no(value):
        return "YES" if value else "NO"

    def frame_or_na(frame):
        return str(frame) if frame >= 0 else "N/A"

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("REPLAY METRICS\n")
        f.write("=" * 68 + "\n\n")

        f.write("INPUT / REPLAY\n")
        f.write("-" * 68 + "\n")
        f.write(f"Actions file:              {actions_file}\n")
        f.write(f"Task:                      {task_name}\n")
        f.write(f"Action steps:              {T}\n")
        f.write(f"Comparison FPS:            {replay_fps:.2f}\n\n")

        f.write("ROBOT END-EFFECTOR\n")
        f.write("-" * 68 + "\n")
        if initial_eef_pos is not None:
            f.write(f"Initial EEF XYZ [m]:       {np.array2string(initial_eef_pos, precision=6)}\n")
            f.write(f"Final EEF XYZ [m]:         {np.array2string(final_eef_pos, precision=6)}\n")
            f.write(f"Maximum |dX|:              {max_eef_dx * 100.0:.2f} cm\n")
            f.write(f"Maximum |dY|:              {max_eef_dy * 100.0:.2f} cm\n")
            f.write(f"Maximum |dZ|:              {max_eef_dz * 100.0:.2f} cm\n")
            f.write(f"Signed dX range:           {min_eef_dx * 100.0:+.2f} to {max_signed_eef_dx * 100.0:+.2f} cm\n")
            f.write(f"Signed dY range:           {min_eef_dy * 100.0:+.2f} to {max_signed_eef_dy * 100.0:+.2f} cm\n")
            f.write(f"Signed dZ range:           {min_eef_dz * 100.0:+.2f} to {max_signed_eef_dz * 100.0:+.2f} cm\n")
        else:
            f.write("EEF state:                 UNAVAILABLE\n")
        f.write("\n")

        f.write("TUMBLER / OBJECT\n")
        f.write("-" * 68 + "\n")
        if initial_tumbler_pos is not None:
            f.write(f"Tumbler body:              {tumbler_pos_source}\n")
            f.write(f"Position source:           MuJoCo body_xpos\n")
            f.write(f"Initial tumbler XYZ [m]:   {np.array2string(initial_tumbler_pos, precision=6)}\n")
            f.write(f"Initial tumbler Z:         {initial_tumbler_z:.6f} m\n")
            f.write(f"Maximum tumbler Z:         {maximum_tumbler_z:.6f} m\n")
            f.write(f"Maximum tumbler lift:      {maximum_tumbler_lift * 100.0:.2f} cm\n")
            f.write(f"Minimum EEF-object dist:   {min_eef_tumbler_distance * 100.0:.2f} cm\n")
            f.write(f"Closest approach frame:    {frame_or_na(closest_approach_frame)}\n")
        else:
            f.write("Tumbler position:          UNAVAILABLE\n")
            f.write("Maximum tumbler lift:      UNAVAILABLE\n")
        f.write("\n")

        # GRASP GEOMETRY DIAGNOSTIC
        if diag["available"]:
            f.write("CLOSEST APPROACH GEOMETRY\n")
            f.write("-" * 68 + "\n\n")

            ef = diag["eef_closest_frame"]
            f.write(f"EEF closest frame:           {ef}\n")
            f.write(f"EEF XYZ:                     {np.array2string(diag['eef_xyz'], precision=4)}\n")
            f.write(f"Tumbler XYZ:                 {np.array2string(diag['tumbler_xyz_at_eef'], precision=4)}\n")
            f.write(f"EEF-object dX:               {diag['eef_dx']*100:+.2f} cm\n")
            f.write(f"EEF-object dY:               {diag['eef_dy']*100:+.2f} cm\n")
            f.write(f"EEF-object dZ:               {diag['eef_dz']*100:+.2f} cm\n")
            f.write(f"Horizontal XY offset:        {diag['eef_horizontal_xy']*100:.2f} cm\n")
            f.write(f"Vertical offset (dZ):        {diag['eef_vertical_dz']*100:+.2f} cm\n\n")

            f.write(f"Left finger pad geom:        {diag['left_pad_geom_name']}\n")
            f.write(f"Left finger closest frame:   {diag['left_closest_frame']}\n")
            f.write(f"Left finger XYZ:             {np.array2string(diag['left_pad_xyz'], precision=4)}\n")
            f.write(f"Tumbler XYZ at left frame:   {np.array2string(diag['tumbler_xyz_at_left'], precision=4)}\n")
            f.write(f"Min left pad->center dist:   {diag['min_left_pad_dist']*100:.2f} cm\n")
            f.write(f"Left horiz-to-axis dist:     {diag['left_horizontal_to_axis']*100:.2f} cm\n")
            f.write(f"Left vertical offset:        {diag['left_vertical_offset']*100:+.2f} cm\n")
            f.write(f"Left surface clearance:      {diag['left_surface_clearance']*100:.2f} cm\n\n")

            f.write(f"Right finger pad geom:       {diag['right_pad_geom_name']}\n")
            f.write(f"Right finger closest frame:  {diag['right_closest_frame']}\n")
            f.write(f"Right finger XYZ:            {np.array2string(diag['right_pad_xyz'], precision=4)}\n")
            f.write(f"Tumbler XYZ at right frame:  {np.array2string(diag['tumbler_xyz_at_right'], precision=4)}\n")
            f.write(f"Min right pad->center dist:  {diag['min_right_pad_dist']*100:.2f} cm\n")
            f.write(f"Right horiz-to-axis dist:    {diag['right_horizontal_to_axis']*100:.2f} cm\n")
            f.write(f"Right vertical offset:       {diag['right_vertical_offset']*100:+.2f} cm\n")
            f.write(f"Right surface clearance:     {diag['right_surface_clearance']*100:.2f} cm\n\n")

            f.write(f"Glass diameter:              {diag['glass_diameter']*100:.1f} cm\n")
            f.write(f"Pad separation @ EEF frame:  {diag['pad_separation_at_eef']*100:.2f} cm\n")
            f.write(f"Pad separation @ left frame: {diag['pad_separation_at_left']*100:.2f} cm\n")
            f.write(f"Pad separation @ right frame:{diag['pad_separation_at_right']*100:.2f} cm\n")
            f.write(f"Min pad separation episode:  {diag['min_pad_separation_episode']*100:.2f} cm\n")
            f.write(f"  at frame:                  {diag['min_pad_separation_frame']}\n\n")

        f.write("PHYSICAL OUTCOME\n")
        f.write("-" * 68 + "\n")
        f.write(f"Contact signal source:     MuJoCo contact pairs\n")
        f.write(f"Contact occurred:          {yes_no(contact_occurred)}\n")
        f.write(f"First contact frame:       {frame_or_na(first_contact_frame)}\n\n")
        f.write(f"Grasp signal source:       RoboSuite _check_grasp\n")
        f.write(f"Grasp occurred:            {yes_no(grasp_occurred)}\n")
        f.write(f"First grasp frame:         {frame_or_na(first_grasp_frame)}\n\n")
        f.write(f"Lift threshold:            {LIFT_SUCCESS_THRESHOLD_M * 100.0:.1f} cm\n")
        f.write(f"Lift success:              {yes_no(lift_success) if lift_success is not None else 'UNAVAILABLE'}\n\n")

        f.write("RESULT\n")
        f.write("-" * 68 + "\n")
        if lift_success is True:
            f.write("Measured result: physical lift threshold reached.\n")
        elif lift_success is False:
            f.write("Measured result: physical lift threshold was not reached.\n")
        else:
            f.write("Measured result: lift could not be evaluated.\n")

    print(f"[+] Saved Iteration-1 metrics -> {metrics_path}")

    # 19. SAVE PAIRED EPISODE
    if output_dataset_path:
        out_data_path = Path(output_dataset_path)
        out_data_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_data_path,
            sim_frames=np.array(sim_frames, dtype=np.uint8),
            actions=actions[:len(sim_frames)],
            human_frames=human_frames[:len(sim_frames)] if human_frames is not None else None,
            robot_eef_pos=eef_positions,
            robot_eef_frames=eef_position_frames,
            tumbler_positions=tumbler_positions,
            tumbler_position_frames=tumbler_position_frames,
            contact_flags=contact_flags,
            grasp_flags=grasp_flags,
            rewards=rewards,
            executed_actions=executed_actions,
            initial_eef_pos=initial_eef_pos if initial_eef_pos is not None else np.full(3, np.nan, dtype=np.float32),
            initial_tumbler_pos=initial_tumbler_pos if initial_tumbler_pos is not None else np.full(3, np.nan, dtype=np.float32),
            max_tumbler_lift=np.float32(maximum_tumbler_lift),
            min_eef_tumbler_distance=np.float32(min_eef_tumbler_distance),
            left_pad_positions=left_pad_positions,
            right_pad_positions=right_pad_positions,
            lift_success=np.int8(-1 if lift_success is None else int(lift_success)),
            replay_fps=np.float32(replay_fps),
        )
        print(f"[+] Saved paired trajectory dataset -> {out_data_path}")

    # 20. GENERATE SIDE-BY-SIDE COMPARISON VIDEO
    out_vid_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(out_vid_path), fourcc, replay_fps, (512, 256))

    if not video_writer.isOpened():
        raise RuntimeError(f"Could not create comparison video: {out_vid_path}")

    num_frames = min(len(sim_frames), len(human_frames) if human_frames is not None else len(sim_frames))

    for i in range(num_frames):
        sim_rgb = sim_frames[i]
        sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)

        if human_frames is not None:
            human_rgb = cv2.resize(human_frames[i], (256, 256), interpolation=cv2.INTER_AREA)
            human_bgr = cv2.cvtColor(human_rgb, cv2.COLOR_RGB2BGR)
        else:
            human_bgr = np.zeros_like(sim_bgr)

        video_writer.write(np.hstack([human_bgr, sim_bgr]))

    video_writer.release()
    print(f"[+] Saved synchronized side-by-side video -> {out_vid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay retargeted actions in simulation")
    parser.add_argument("--actions", type=str, required=True, help="Path to retargeted actions.npz")
    parser.add_argument("--out_video", type=str, default="media/side_by_side_demo.mp4", help="Output comparison video")
    parser.add_argument("--out_dataset", type=str, default=None, help="Output paired dataset npz")
    parser.add_argument("--task", type=str, default="Lift", help="Simulation task name")
    parser.add_argument("--gui", action="store_true", help="Enable interactive window viewer")
    parser.add_argument("--human_video", type=str, default=None, help="MP4 video for left side of comparison")
    args = parser.parse_args()

    replay_actions_and_record(
        actions_npz_path=args.actions, output_video_path=args.out_video,
        output_dataset_path=args.out_dataset, task_name=args.task, render_gui=args.gui,
        human_video_path=args.human_video,
    )
