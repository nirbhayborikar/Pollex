"""
CoTracker3 trajectory retargeting to Franka Panda OSC actions.

The generated RoboSuite action has the form:

    [dx, dy, dz, droll, dpitch, dyaw, gripper]

Important:
    dx, dy, and dz are controller actions. They are not guaranteed to
    represent metric displacement in meters unless the configured RoboSuite
    OSC controller explicitly interprets them that way.

This module generates an offline feedforward action sequence. Actual
closed-loop Cartesian control requires the measured EEF pose from MuJoCo
during replay.

python3 src/retargeting/cotracker_to_panda.py \
  datasets/cotracker_trajectories/cup_lifting_01_ct_trajectory.npz \
  datasets/cotracker_iterations/iteration_09_x_bias \
  --approach-x-bias 0.123 \
  --descent-delay-frac 0.45 \
  --descent-rate -0.28
  
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np


TARGET_FRAME_SIZE = 512


class CoTrackerToPandaRetargeter:
    """
    Convert CoTracker hand/object trajectories into Panda OSC actions.

    The current implementation is an offline feedforward retargeter.

    During the pre-grasp phase:

    * Human image motion provides the basic forward approach signal.
    * approach_x_bias provides an empirical forward correction.
    * Y remains zero because the current GlassLift setup is aligned with
      the Panda Y axis.
    * Z descent starts after a configurable part of the approach.

    Actual EEF and tumbler positions are measured later in
    replay_trajectory.py.
    """

    def __init__(
        self,
        camera_tilt_deg: float = 60.0,
        position_gain: float = 8.0,
        smoothing_alpha: float = 0.4,
        max_action_magnitude: float = 0.8,
        eef_to_pad_offset_x: float = 0.043,
        approach_x_bias: float = 0.123,
        approach_bias_frames: int = 5,
        distance_gain_saturation: float = 0.15,
        minimum_approach_rate: float = 0.002,
        descent_delay_frac: float = 0.45,
        descent_rate: float = -0.28,
        descent_full_phase: float = 0.85,
        descent_stop_phase: float = 0.95,
        grasp_settle_frames: int = 14,
        release_settle_frames: int = 12,
        lift_gain: float = 7.5,
        max_lift_action: float = 0.45,
        retract_x_action: float = -0.06,
        retract_z_action: float = 0.15,
        fallback_entry_frame: int = 28,
        hand_motion_threshold: float = 0.005,
    ):
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1].")

        if max_action_magnitude <= 0.0:
            raise ValueError("max_action_magnitude must be positive.")

        if approach_bias_frames < 1:
            raise ValueError("approach_bias_frames must be at least 1.")

        if not 0.0 <= descent_delay_frac < 1.0:
            raise ValueError("descent_delay_frac must be in [0, 1).")

        if not 0.0 < descent_full_phase < descent_stop_phase <= 1.0:
            raise ValueError(
                "Require 0 < descent_full_phase "
                "< descent_stop_phase <= 1."
            )

        self.camera_tilt_deg = camera_tilt_deg
        self.camera_tilt_rad = float(np.radians(camera_tilt_deg))
        self.sin_theta = float(np.sin(self.camera_tilt_rad))

        self.position_gain = position_gain
        self.smoothing_alpha = smoothing_alpha
        self.max_action_magnitude = max_action_magnitude

        # Physical gripper geometry. Do not tune this just to move farther.
        self.eef_to_pad_offset_x = eef_to_pad_offset_x

        # Empirical feedforward X correction. This is intentionally separate
        # from the physical EEF-to-pad geometry.
        self.approach_x_bias = approach_x_bias
        self.approach_bias_frames = approach_bias_frames

        self.distance_gain_saturation = distance_gain_saturation
        self.minimum_approach_rate = minimum_approach_rate

        self.descent_delay_frac = descent_delay_frac
        self.descent_rate = descent_rate
        self.descent_full_phase = descent_full_phase
        self.descent_stop_phase = descent_stop_phase

        self.grasp_settle_frames = grasp_settle_frames
        self.release_settle_frames = release_settle_frames

        self.lift_gain = lift_gain
        self.max_lift_action = max_lift_action

        self.retract_x_action = retract_x_action
        self.retract_z_action = retract_z_action

        self.fallback_entry_frame = fallback_entry_frame
        self.hand_motion_threshold = hand_motion_threshold

    @staticmethod
    def _validate_input_arrays(
        wrist_deltas: np.ndarray,
        gripper_states: np.ndarray,
        timestamps: np.ndarray,
    ) -> None:
        if wrist_deltas.ndim != 2 or wrist_deltas.shape[1] < 3:
            raise ValueError(
                "wrist_deltas must have shape (T, 3) or larger."
            )

        number_of_frames = len(wrist_deltas)

        if len(gripper_states) != number_of_frames:
            raise ValueError(
                "gripper_states length does not match wrist_deltas."
            )

        if len(timestamps) != number_of_frames:
            raise ValueError(
                "timestamps length does not match wrist_deltas."
            )

    @staticmethod
    def _smooth_step(progress: float) -> float:
        """
        Cubic smooth-step interpolation.

        Returns:
            0 at progress=0
            1 at progress=1

        The first derivative is zero at both ends.
        """
        progress = float(np.clip(progress, 0.0, 1.0))
        return progress * progress * (3.0 - 2.0 * progress)

    def _find_entry_frame(
        self,
        data: np.lib.npyio.NpzFile,
        wrist_deltas: np.ndarray,
        grasp_frame: int,
    ) -> int:
        if "t_hand" in data:
            stored_entry = int(np.asarray(data["t_hand"]).item())
            if 0 < stored_entry < grasp_frame:
                return stored_entry

        entry_frame = min(self.fallback_entry_frame, max(0, grasp_frame - 1))

        for frame in range(grasp_frame):
            if (
                np.linalg.norm(wrist_deltas[frame])
                > self.hand_motion_threshold
            ):
                entry_frame = frame
                break

        return entry_frame

    def _load_normalized_centroids(
        self,
        data: np.lib.npyio.NpzFile,
        wrist_positions: np.ndarray,
        number_of_frames: int,
    ):
        has_tracks = (
            "glass_tracks" in data
            and "hand_tracks" in data
        )

        if has_tracks:
            glass_tracks = np.asarray(
                data["glass_tracks"],
                dtype=np.float32,
            )
            hand_tracks = np.asarray(
                data["hand_tracks"],
                dtype=np.float32,
            )

            if len(glass_tracks) != number_of_frames:
                raise ValueError(
                    "glass_tracks length does not match wrist_deltas."
                )

            if len(hand_tracks) != number_of_frames:
                raise ValueError(
                    "hand_tracks length does not match wrist_deltas."
                )

            glass_centroids = (
                glass_tracks.mean(axis=1) / TARGET_FRAME_SIZE
            )
            hand_centroids = (
                hand_tracks.mean(axis=1) / TARGET_FRAME_SIZE
            )

            return (
                True,
                glass_centroids.astype(np.float32),
                hand_centroids.astype(np.float32),
            )

        # This is only a fallback. It does not provide reliable image-space
        # object-relative targeting.
        glass_centroids = np.tile(
            np.array([0.5, 0.5], dtype=np.float32),
            (number_of_frames, 1),
        )

        hand_centroids = np.zeros(
            (number_of_frames, 2),
            dtype=np.float32,
        )

        if wrist_positions.ndim == 2 and wrist_positions.shape[1] >= 2:
            candidate = wrist_positions[:, :2].astype(np.float32)

            # Use the values directly only if they already appear normalized.
            if (
                np.all(np.isfinite(candidate))
                and np.min(candidate) >= 0.0
                and np.max(candidate) <= 1.0
            ):
                hand_centroids = candidate
            else:
                hand_centroids[:] = 0.5
        else:
            hand_centroids[:] = 0.5

        return False, glass_centroids, hand_centroids

    def _build_lift_profile(
        self,
        data: np.lib.npyio.NpzFile,
        has_tracks: bool,
        grasp_frame: int,
        number_of_frames: int,
    ):
        if not has_tracks or grasp_frame >= number_of_frames:
            return_frame = min(
                number_of_frames - 1,
                grasp_frame + 40,
            )
            lift_velocity = np.zeros(
                number_of_frames,
                dtype=np.float32,
            )
            return return_frame, lift_velocity

        glass_y = np.asarray(
            data["glass_tracks"],
            dtype=np.float32,
        ).mean(axis=1)[:, 1]

        pre_grasp_end = max(1, grasp_frame)
        rest_y = float(np.median(glass_y[:pre_grasp_end]))

        lift_pixels = np.maximum(0.0, rest_y - glass_y)

        apex_frame = (
            int(np.argmax(lift_pixels[grasp_frame:])) + grasp_frame
        )
        maximum_lift_pixels = float(lift_pixels[apex_frame])

        normalized_lift = np.zeros(
            number_of_frames,
            dtype=np.float32,
        )

        if maximum_lift_pixels > 5.0:
            normalized_lift[grasp_frame:] = (
                lift_pixels[grasp_frame:] / maximum_lift_pixels
            )
        else:
            fallback_length = min(
                25,
                max(10, (number_of_frames - grasp_frame) // 3),
            )
            fallback_start = grasp_frame + self.grasp_settle_frames
            fallback_end = min(
                number_of_frames,
                fallback_start + 2 * fallback_length,
            )

            for frame in range(fallback_start, fallback_end):
                progress = (
                    frame - fallback_start
                ) / float(2 * fallback_length)

                normalized_lift[frame] = np.sin(progress * np.pi)

        smoothing_kernel = np.ones(5, dtype=np.float32) / 5.0
        smoothed_lift = np.convolve(
            normalized_lift,
            smoothing_kernel,
            mode="same",
        )
        smoothed_lift = np.clip(smoothed_lift, 0.0, 1.0)

        after_apex = np.arange(number_of_frames) > apex_frame
        returned_mask = after_apex & (smoothed_lift < 0.03)

        if np.any(returned_mask):
            return_frame = int(np.where(returned_mask)[0][0])
        else:
            return_frame = min(
                max(0, number_of_frames - 10),
                apex_frame + 20,
            )

        lift_velocity = np.gradient(smoothed_lift).astype(np.float32)

        return return_frame, lift_velocity

    def _get_approach_actions(
        self,
        frame: int,
        entry_frame: int,
        grasp_frame: int,
        wrist_delta: np.ndarray,
        hand_centroid: np.ndarray,
        glass_centroid: np.ndarray,
    ):
        """
        Generate feedforward pre-grasp actions.

        This is not closed-loop Cartesian position control because this method
        does not receive the actual MuJoCo EEF pose.
        """
        reach_span = max(1, grasp_frame - entry_frame)
        frames_since_entry = frame - entry_frame

        approach_progress = float(
            np.clip(frames_since_entry / reach_span, 0.0, 1.0)
        )

        vector_to_glass = glass_centroid - hand_centroid
        image_distance = float(np.linalg.norm(vector_to_glass))

        if image_distance > 1e-6:
            reach_direction = vector_to_glass / image_distance
        else:
            reach_direction = np.zeros(2, dtype=np.float32)

        hand_image_velocity = np.asarray(
            wrist_delta[:2],
            dtype=np.float32,
        )

        velocity_along_reach = float(
            np.dot(hand_image_velocity, reach_direction)
        )

        distance_gain = float(
            np.clip(
                image_distance / self.distance_gain_saturation,
                0.0,
                1.0,
            )
        )

        approach_rate = (
            max(self.minimum_approach_rate, velocity_along_reach)
            * distance_gain
        )

        raw_dx = (
            approach_rate
            / max(0.1, self.sin_theta)
            * self.position_gain
        )

        # Separate empirical correction from physical pad geometry.
        if frames_since_entry < self.approach_bias_frames:
            raw_dx += (
                self.approach_x_bias
                / self.approach_bias_frames
            )

        # Current experiment is already aligned in Y.
        raw_dy = 0.0

        descent_start_frame = entry_frame + int(
            self.descent_delay_frac * reach_span
        )

        if frame <= descent_start_frame:
            descent_progress = 0.0
            raw_dz = 0.0
        else:
            descent_span = max(
                1,
                grasp_frame - descent_start_frame,
            )

            descent_progress = float(
                np.clip(
                    (frame - descent_start_frame) / descent_span,
                    0.0,
                    1.0,
                )
            )

            if descent_progress < self.descent_full_phase:
                raw_dz = self.descent_rate

            elif descent_progress < self.descent_stop_phase:
                taper_progress = (
                    descent_progress - self.descent_full_phase
                ) / (
                    self.descent_stop_phase
                    - self.descent_full_phase
                )

                taper = 1.0 - self._smooth_step(taper_progress)
                raw_dz = self.descent_rate * taper

            else:
                raw_dz = 0.0

        return {
            "raw_dx": float(raw_dx),
            "raw_dy": float(raw_dy),
            "raw_dz": float(raw_dz),
            "image_distance": image_distance,
            "approach_progress": approach_progress,
            "descent_progress": descent_progress,
        }

    def retarget_trajectory(
        self,
        npz_trajectory_file: str,
        output_file: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        path = Path(npz_trajectory_file)

        if not path.exists():
            raise FileNotFoundError(
                f"Trajectory file not found: {path}"
            )

        with np.load(path, allow_pickle=False) as data:
            required_keys = {
                "wrist_deltas",
                "gripper_states",
                "timestamps",
            }

            missing_keys = required_keys.difference(data.files)

            if missing_keys:
                raise KeyError(
                    f"Missing required NPZ keys: {sorted(missing_keys)}"
                )

            wrist_deltas = np.asarray(
                data["wrist_deltas"],
                dtype=np.float32,
            )
            gripper_states = np.asarray(
                data["gripper_states"],
                dtype=np.float32,
            )
            timestamps = np.asarray(
                data["timestamps"],
                dtype=np.float64,
            )

            self._validate_input_arrays(
                wrist_deltas,
                gripper_states,
                timestamps,
            )

            number_of_frames = len(wrist_deltas)

            if "wrist_positions" in data:
                wrist_positions = np.asarray(
                    data["wrist_positions"],
                    dtype=np.float32,
                )
            else:
                wrist_positions = np.zeros(
                    (number_of_frames, 3),
                    dtype=np.float32,
                )

            has_tracks, glass_centroids, hand_centroids = (
                self._load_normalized_centroids(
                    data,
                    wrist_positions,
                    number_of_frames,
                )
            )

            positive_gripper_frames = np.where(
                gripper_states > 0
            )[0]

            if len(positive_gripper_frames) > 0:
                grasp_frame = int(positive_gripper_frames[0])
            else:
                grasp_frame = number_of_frames

            entry_frame = self._find_entry_frame(
                data,
                wrist_deltas,
                grasp_frame,
            )

            return_frame, lift_velocity = self._build_lift_profile(
                data,
                has_tracks,
                grasp_frame,
                number_of_frames,
            )

            actions_7d = np.zeros(
                (number_of_frames, 7),
                dtype=np.float32,
            )

            smoothed_translation_action = np.zeros(
                3,
                dtype=np.float32,
            )

            log_rows = []

            for frame in range(number_of_frames):
                image_distance = float(
                    np.linalg.norm(
                        glass_centroids[frame]
                        - hand_centroids[frame]
                    )
                )

                approach_progress = 0.0
                descent_progress = 0.0

                if frame < grasp_frame:
                    gripper_action = -1.0

                    if frame < entry_frame:
                        phase = "wait"
                        raw_dx = 0.0
                        raw_dy = 0.0
                        raw_dz = 0.0

                    else:
                        phase = "approach"

                        approach = self._get_approach_actions(
                            frame=frame,
                            entry_frame=entry_frame,
                            grasp_frame=grasp_frame,
                            wrist_delta=wrist_deltas[frame],
                            hand_centroid=hand_centroids[frame],
                            glass_centroid=glass_centroids[frame],
                        )

                        raw_dx = approach["raw_dx"]
                        raw_dy = approach["raw_dy"]
                        raw_dz = approach["raw_dz"]

                        image_distance = approach["image_distance"]
                        approach_progress = approach[
                            "approach_progress"
                        ]
                        descent_progress = approach[
                            "descent_progress"
                        ]

                elif frame < grasp_frame + self.grasp_settle_frames:
                    phase = "grasp_settle"
                    raw_dx = 0.0
                    raw_dy = 0.0
                    raw_dz = 0.0
                    gripper_action = 1.0

                elif frame <= return_frame:
                    phase = "lift_return"
                    raw_dx = 0.0
                    raw_dy = 0.0
                    raw_dz = float(
                        np.clip(
                            lift_velocity[frame] * self.lift_gain,
                            -self.max_lift_action,
                            self.max_lift_action,
                        )
                    )
                    gripper_action = 1.0

                elif frame <= (
                    return_frame + self.release_settle_frames
                ):
                    phase = "release"
                    raw_dx = 0.0
                    raw_dy = 0.0
                    raw_dz = 0.0
                    gripper_action = -1.0

                else:
                    phase = "retract"
                    raw_dx = self.retract_x_action
                    raw_dy = 0.0
                    raw_dz = self.retract_z_action
                    gripper_action = -1.0

                raw_translation_action = np.array(
                    [raw_dx, raw_dy, raw_dz],
                    dtype=np.float32,
                )

                if frame == 0:
                    smoothed_translation_action = (
                        raw_translation_action.copy()
                    )
                else:
                    smoothed_translation_action = (
                        self.smoothing_alpha
                        * raw_translation_action
                        + (1.0 - self.smoothing_alpha)
                        * smoothed_translation_action
                    )

                clamped_action = np.clip(
                    smoothed_translation_action,
                    -self.max_action_magnitude,
                    self.max_action_magnitude,
                )

                actions_7d[frame, 0:3] = clamped_action

                # Zero rotational delta. This does not, by itself, guarantee
                # an absolute top-down orientation.
                actions_7d[frame, 3:6] = 0.0
                actions_7d[frame, 6] = gripper_action

                log_rows.append(
                    {
                        "frame": frame,
                        "timestamp": float(timestamps[frame]),
                        "phase": phase,
                        "hand_x": float(hand_centroids[frame, 0]),
                        "hand_y": float(hand_centroids[frame, 1]),
                        "tumbler_x": float(
                            glass_centroids[frame, 0]
                        ),
                        "tumbler_y": float(
                            glass_centroids[frame, 1]
                        ),
                        "image_distance": image_distance,
                        "raw_dx": float(raw_translation_action[0]),
                        "raw_dy": float(raw_translation_action[1]),
                        "raw_dz": float(raw_translation_action[2]),
                        "smoothed_dx": float(
                            smoothed_translation_action[0]
                        ),
                        "smoothed_dy": float(
                            smoothed_translation_action[1]
                        ),
                        "smoothed_dz": float(
                            smoothed_translation_action[2]
                        ),
                        "command_dx": float(clamped_action[0]),
                        "command_dy": float(clamped_action[1]),
                        "command_dz": float(clamped_action[2]),
                        "gripper": float(gripper_action),
                        "source_gripper": float(
                            gripper_states[frame]
                        ),
                        "approach_progress": approach_progress,
                        "descent_progress": descent_progress,
                    }
                )

            hand_centroid_log = np.array(
                [
                    [row["hand_x"], row["hand_y"]]
                    for row in log_rows
                ],
                dtype=np.float32,
            )

            tumbler_centroid_log = np.array(
                [
                    [row["tumbler_x"], row["tumbler_y"]]
                    for row in log_rows
                ],
                dtype=np.float32,
            )

            xyz_commands = np.array(
                [
                    [
                        row["command_dx"],
                        row["command_dy"],
                        row["command_dz"],
                    ]
                    for row in log_rows
                ],
                dtype=np.float32,
            )

            raw_xyz_commands = np.array(
                [
                    [
                        row["raw_dx"],
                        row["raw_dy"],
                        row["raw_dz"],
                    ]
                    for row in log_rows
                ],
                dtype=np.float32,
            )

            result = {
                "actions": actions_7d,
                "timestamps": timestamps,
                "hand_centroids": hand_centroid_log,
                "tumbler_centroids": tumbler_centroid_log,
                "xyz_commands": xyz_commands,
                "raw_xyz_commands": raw_xyz_commands,
                "source_gripper_states": gripper_states,
                "entry_frame": np.int32(entry_frame),
                "grasp_frame": np.int32(grasp_frame),
                "return_frame": np.int32(return_frame),
                "eef_to_pad_offset_x": np.float32(
                    self.eef_to_pad_offset_x
                ),
                "approach_x_bias": np.float32(
                    self.approach_x_bias
                ),
                "source_file": np.array(str(path)),
            }

            if "rgb_frames" in data:
                result["rgb_frames"] = data["rgb_frames"]

        if output_file is None:
            output_file = (
                path.parent
                / f"{path.stem.replace('_trajectory', '')}_actions.npz"
            )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(output_path, **result)

        log_path = (
            output_path.parent
            / f"{output_path.stem}_log.csv"
        )

        self._write_log(log_path, log_rows)

        print(f"[+] Retargeted {number_of_frames} steps")
        print(f"[+] Entry frame: {entry_frame}")
        print(f"[+] Source grasp frame: {grasp_frame}")
        print(f"[+] Return frame: {return_frame}")
        print(
            "[+] Physical EEF-to-pad X offset: "
            f"{self.eef_to_pad_offset_x:.4f} m"
        )
        print(
            "[+] Empirical approach X bias: "
            f"{self.approach_x_bias:.4f}"
        )
        print(f"[+] Actions saved to: {output_path}")
        print(f"[+] Log saved to: {log_path}")

        return result

    @staticmethod
    def _write_log(log_path: Path, rows) -> None:
        columns = [
            "frame",
            "timestamp",
            "phase",
            "hand_x",
            "hand_y",
            "tumbler_x",
            "tumbler_y",
            "image_distance",
            "raw_dx",
            "raw_dy",
            "raw_dz",
            "smoothed_dx",
            "smoothed_dy",
            "smoothed_dz",
            "command_dx",
            "command_dy",
            "command_dz",
            "gripper",
            "source_gripper",
            "approach_progress",
            "descent_progress",
        ]

        with open(log_path, "w", encoding="utf-8") as file:
            file.write(",".join(columns) + "\n")

            for row in rows:
                values = []

                for column in columns:
                    value = row[column]

                    if isinstance(value, float):
                        values.append(f"{value:.8f}")
                    else:
                        values.append(str(value))

                file.write(",".join(values) + "\n")


def retarget_all_cotracker_trajectories(
    directory: str = "datasets/cotracker_trajectories",
    output_directory: str = (
        "datasets/cotracker_iterations/iteration_09_x_bias"
    ),
) -> None:
    input_directory = Path(directory)
    output_directory_path = Path(output_directory)
    output_directory_path.mkdir(parents=True, exist_ok=True)

    trajectory_files = sorted(
        input_directory.glob("*_ct_trajectory.npz")
    )

    if not trajectory_files:
        print(
            f"[-] No CoTracker trajectories found in "
            f"{input_directory}"
        )
        return

    retargeter = CoTrackerToPandaRetargeter()

    for trajectory_file in trajectory_files:
        output_file = (
            output_directory_path
            / (
                trajectory_file.stem.replace(
                    "_trajectory",
                    "",
                )
                + "_actions.npz"
            )
        )

        retargeter.retarget_trajectory(
            str(trajectory_file),
            output_file=str(output_file),
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Retarget a CoTracker trajectory to Panda OSC actions."
        )
    )

    parser.add_argument(
        "trajectory",
        help="Input *_ct_trajectory.npz file.",
    )

    parser.add_argument(
        "output_folder",
        nargs="?",
        default=(
            "datasets/cotracker_iterations/"
            "iteration_09_x_bias"
        ),
        help="Folder in which to save actions and logs.",
    )

    parser.add_argument(
        "--approach-x-bias",
        type=float,
        default=0.123,
        help=(
            "Empirical forward approach correction. "
            "This is separate from physical EEF-to-pad geometry."
        ),
    )

    parser.add_argument(
        "--descent-delay-frac",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--descent-rate",
        type=float,
        default=-0.28,
    )

    args = parser.parse_args()

    input_path = Path(args.trajectory)
    output_directory = Path(args.output_folder)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_directory
        / f"{input_path.stem.replace('_trajectory', '')}_actions.npz"
    )

    retargeter = CoTrackerToPandaRetargeter(
        eef_to_pad_offset_x=0.043,
        approach_x_bias=args.approach_x_bias,
        descent_delay_frac=args.descent_delay_frac,
        descent_rate=args.descent_rate,
    )

    retargeter.retarget_trajectory(
        str(input_path),
        output_file=str(output_file),
    )