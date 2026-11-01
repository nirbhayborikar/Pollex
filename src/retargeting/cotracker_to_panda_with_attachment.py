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

python3 src/retargeting/cotracker_to_panda_with_attachment.py \
  datasets/cotracker_trajectories/cup_lifting_01_ct_trajectory.npz \
  datasets/cotracker_iterations/iteration_21_attachment \
  --approach-x-bias 0.123 \
  --descent-delay-frac 0.45 \
  --descent-rate -0.28 \
  --attachment-threshold 0.65 \
  --attachment-min-frames 3 \
  --attachment-smoothing-frames 5
  
"""
"""
CoTracker3 trajectory retargeting to Franka Panda OSC actions with
hand-object attachment detection.

Action:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]

Attachment evidence:
    - The tracked tumbler is moving.
    - Hand-tumbler relative motion is small.
    - The condition remains valid for consecutive frames.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np


TARGET_FRAME_SIZE = 512


class CoTrackerToPandaRetargeter:
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
        attachment_threshold: float = 0.65,
        attachment_min_frames: int = 3,
        attachment_smoothing_frames: int = 5,
    ):
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1].")
        if max_action_magnitude <= 0.0:
            raise ValueError("max_action_magnitude must be positive.")
        if approach_bias_frames < 1:
            raise ValueError("approach_bias_frames must be >= 1.")
        if not 0.0 <= descent_delay_frac < 1.0:
            raise ValueError("descent_delay_frac must be in [0, 1).")
        if not 0.0 < descent_full_phase < descent_stop_phase <= 1.0:
            raise ValueError("Require 0 < descent_full_phase < descent_stop_phase <= 1.")
        if not 0.0 <= attachment_threshold <= 1.0:
            raise ValueError("attachment_threshold must be in [0, 1].")
        if attachment_min_frames < 1 or attachment_smoothing_frames < 1:
            raise ValueError("Attachment frame counts must be positive.")

        self.camera_tilt_rad = float(np.radians(camera_tilt_deg))
        self.sin_theta = float(np.sin(self.camera_tilt_rad))
        self.position_gain = position_gain
        self.smoothing_alpha = smoothing_alpha
        self.max_action_magnitude = max_action_magnitude
        self.eef_to_pad_offset_x = eef_to_pad_offset_x
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
        self.attachment_threshold = attachment_threshold
        self.attachment_min_frames = attachment_min_frames
        self.attachment_smoothing_frames = attachment_smoothing_frames

    @staticmethod
    def _validate_input_arrays(wrist_deltas, gripper_states, timestamps):
        if wrist_deltas.ndim != 2 or wrist_deltas.shape[1] < 3:
            raise ValueError("wrist_deltas must have shape (T, 3) or larger.")
        if len(gripper_states) != len(wrist_deltas):
            raise ValueError("gripper_states length does not match wrist_deltas.")
        if len(timestamps) != len(wrist_deltas):
            raise ValueError("timestamps length does not match wrist_deltas.")

    @staticmethod
    def _smooth_step(progress: float) -> float:
        progress = float(np.clip(progress, 0.0, 1.0))
        return progress * progress * (3.0 - 2.0 * progress)

    @staticmethod
    def _centroids_from_tracks(tracks: np.ndarray) -> np.ndarray:
        tracks = np.asarray(tracks, dtype=np.float32)
        if tracks.ndim == 2 and tracks.shape[1] >= 2:
            return tracks[:, :2].copy()
        if tracks.ndim == 3 and tracks.shape[2] >= 2:
            valid = np.all(np.isfinite(tracks[:, :, :2]), axis=2)
            centroids = np.full((len(tracks), 2), np.nan, dtype=np.float32)
            for frame in range(len(tracks)):
                if np.any(valid[frame]):
                    centroids[frame] = np.mean(tracks[frame, valid[frame], :2], axis=0)
            return centroids
        raise ValueError(f"Expected tracks shape (T,2+) or (T,N,2+), got {tracks.shape}")

    @staticmethod
    def _fill_invalid_centroids(centroids: np.ndarray) -> np.ndarray:
        result = np.asarray(centroids, dtype=np.float32).copy()
        valid = np.all(np.isfinite(result), axis=1)
        if not np.any(valid):
            return np.full_like(result, 0.5)
        indices = np.arange(len(result))
        for axis in range(2):
            result[:, axis] = np.interp(indices, indices[valid], result[valid, axis])
        return result

    @staticmethod
    def _velocity_from_centroids(centroids: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        velocity = np.zeros_like(centroids, dtype=np.float32)
        if len(centroids) < 2:
            return velocity
        dt = np.diff(timestamps)
        valid_dt = dt[np.isfinite(dt) & (dt > 0)]
        fallback_dt = float(np.median(valid_dt)) if len(valid_dt) else 1.0
        for frame in range(1, len(centroids)):
            frame_dt = float(dt[frame - 1]) if np.isfinite(dt[frame - 1]) and dt[frame - 1] > 0 else fallback_dt
            velocity[frame] = (centroids[frame] - centroids[frame - 1]) / max(frame_dt, 1e-6)
        return velocity

    @staticmethod
    def _robust_motion_threshold(speed: np.ndarray) -> float:
        finite = speed[np.isfinite(speed)]
        if not len(finite):
            return 0.0
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        return max(1e-6, median + 3.0 * 1.4826 * mad)

    def _compute_attachment_signal(self, hand_centroids, glass_centroids, timestamps):
        hand_velocity = self._velocity_from_centroids(hand_centroids, timestamps)
        glass_velocity = self._velocity_from_centroids(glass_centroids, timestamps)
        relative_position = hand_centroids - glass_centroids
        relative_velocity = self._velocity_from_centroids(relative_position, timestamps)

        hand_speed = np.linalg.norm(hand_velocity, axis=1)
        glass_speed = np.linalg.norm(glass_velocity, axis=1)
        relative_speed = np.linalg.norm(relative_velocity, axis=1)
        object_motion_threshold = self._robust_motion_threshold(glass_speed)

        relative_stability = glass_speed / (glass_speed + relative_speed + 1e-6)
        direction_denominator = hand_speed * glass_speed
        direction_alignment = np.zeros(len(hand_speed), dtype=np.float32)
        valid = direction_denominator > 1e-8
        direction_alignment[valid] = np.sum(hand_velocity[valid] * glass_velocity[valid], axis=1) / direction_denominator[valid]
        direction_alignment = np.clip((direction_alignment + 1.0) * 0.5, 0.0, 1.0)

        object_motion_score = np.clip(glass_speed / max(object_motion_threshold, 1e-6), 0.0, 1.0)
        attachment_score = object_motion_score * relative_stability * direction_alignment
        attachment_score[glass_speed <= object_motion_threshold] = 0.0

        if self.attachment_smoothing_frames > 1:
            kernel = np.ones(self.attachment_smoothing_frames, dtype=np.float32) / self.attachment_smoothing_frames
            attachment_score = np.convolve(attachment_score, kernel, mode="same").astype(np.float32)

        return (
            np.clip(attachment_score, 0.0, 1.0),
            hand_velocity,
            glass_velocity,
            relative_velocity,
            hand_speed.astype(np.float32),
            glass_speed.astype(np.float32),
            relative_speed.astype(np.float32),
            np.float32(object_motion_threshold),
        )

    def _find_attachment_latch(self, score, glass_speed, motion_threshold, start_frame=0):
        count = 0
        for frame in range(max(0, start_frame), len(score)):
            attached = glass_speed[frame] > motion_threshold and score[frame] >= self.attachment_threshold
            count = count + 1 if attached else 0
            if count >= self.attachment_min_frames:
                return frame - self.attachment_min_frames + 1
        return -1

    def _find_entry_frame(self, data, wrist_deltas, grasp_frame):
        if "t_hand" in data:
            stored_entry = int(np.asarray(data["t_hand"]).item())
            if 0 < stored_entry < grasp_frame:
                return stored_entry
        fallback = min(self.fallback_entry_frame, max(0, grasp_frame - 1))
        for frame in range(grasp_frame):
            if np.linalg.norm(wrist_deltas[frame]) > self.hand_motion_threshold:
                return frame
        return fallback

    def _load_normalized_centroids(self, data, wrist_positions, number_of_frames):
        if "glass_tracks" in data and "hand_tracks" in data:
            glass = self._fill_invalid_centroids(self._centroids_from_tracks(data["glass_tracks"]))
            hand = self._fill_invalid_centroids(self._centroids_from_tracks(data["hand_tracks"]))
            if len(glass) != number_of_frames or len(hand) != number_of_frames:
                raise ValueError("Track length does not match wrist_deltas.")
            return True, (glass / TARGET_FRAME_SIZE).astype(np.float32), (hand / TARGET_FRAME_SIZE).astype(np.float32)

        glass = np.full((number_of_frames, 2), 0.5, dtype=np.float32)
        hand = np.full((number_of_frames, 2), 0.5, dtype=np.float32)
        if wrist_positions.ndim == 2 and wrist_positions.shape[1] >= 2:
            candidate = wrist_positions[:, :2].astype(np.float32)
            if np.all(np.isfinite(candidate)) and np.min(candidate) >= 0 and np.max(candidate) <= 1:
                hand = candidate
        return False, glass, hand

    def _build_lift_profile(self, data, has_tracks, grasp_frame, number_of_frames):
        if not has_tracks or grasp_frame >= number_of_frames:
            return min(number_of_frames - 1, grasp_frame + 40), np.zeros(number_of_frames, dtype=np.float32)

        glass_y = self._centroids_from_tracks(data["glass_tracks"])[:, 1]
        glass_y = self._fill_invalid_centroids(np.column_stack([glass_y, glass_y]))[:, 0]
        rest_y = float(np.median(glass_y[:max(1, grasp_frame)]))
        lift_pixels = np.maximum(0.0, rest_y - glass_y)
        apex_frame = int(np.argmax(lift_pixels[grasp_frame:])) + grasp_frame
        max_lift = float(lift_pixels[apex_frame])
        normalized = np.zeros(number_of_frames, dtype=np.float32)

        if max_lift > 5.0:
            normalized[grasp_frame:] = lift_pixels[grasp_frame:] / max_lift
        else:
            length = min(25, max(10, (number_of_frames - grasp_frame) // 3))
            start, end = grasp_frame + self.grasp_settle_frames, min(number_of_frames, grasp_frame + self.grasp_settle_frames + 2 * length)
            for frame in range(start, end):
                normalized[frame] = np.sin(((frame - start) / float(2 * length)) * np.pi)

        smoothed = np.convolve(normalized, np.ones(5, dtype=np.float32) / 5.0, mode="same")
        smoothed = np.clip(smoothed, 0.0, 1.0)
        returned = (np.arange(number_of_frames) > apex_frame) & (smoothed < 0.03)
        return_frame = int(np.where(returned)[0][0]) if np.any(returned) else min(max(0, number_of_frames - 10), apex_frame + 20)
        return return_frame, np.gradient(smoothed).astype(np.float32)

    def _get_approach_actions(self, frame, entry_frame, grasp_frame, wrist_delta, hand_centroid, glass_centroid):
        reach_span = max(1, grasp_frame - entry_frame)
        frames_since_entry = frame - entry_frame
        approach_progress = float(np.clip(frames_since_entry / reach_span, 0.0, 1.0))
        vector_to_glass = glass_centroid - hand_centroid
        image_distance = float(np.linalg.norm(vector_to_glass))
        reach_direction = vector_to_glass / image_distance if image_distance > 1e-6 else np.zeros(2, dtype=np.float32)
        velocity_along_reach = float(np.dot(np.asarray(wrist_delta[:2], dtype=np.float32), reach_direction))
        distance_gain = float(np.clip(image_distance / self.distance_gain_saturation, 0.0, 1.0))
        approach_rate = max(self.minimum_approach_rate, velocity_along_reach) * distance_gain
        raw_dx = approach_rate / max(0.1, self.sin_theta) * self.position_gain

        if frames_since_entry < self.approach_bias_frames:
            raw_dx += self.approach_x_bias / self.approach_bias_frames

        descent_start = entry_frame + int(self.descent_delay_frac * reach_span)
        if frame <= descent_start:
            descent_progress, raw_dz = 0.0, 0.0
        else:
            descent_progress = float(np.clip((frame - descent_start) / max(1, grasp_frame - descent_start), 0.0, 1.0))
            if descent_progress < self.descent_full_phase:
                raw_dz = self.descent_rate
            elif descent_progress < self.descent_stop_phase:
                taper = (descent_progress - self.descent_full_phase) / (self.descent_stop_phase - self.descent_full_phase)
                raw_dz = self.descent_rate * (1.0 - self._smooth_step(taper))
            else:
                raw_dz = 0.0

        return float(raw_dx), 0.0, float(raw_dz), image_distance, approach_progress, descent_progress

    def retarget_trajectory(self, npz_trajectory_file: str, output_file: Optional[str] = None) -> Dict[str, np.ndarray]:
        path = Path(npz_trajectory_file)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            required = {"wrist_deltas", "gripper_states", "timestamps"}
            missing = required.difference(data.files)
            if missing:
                raise KeyError(f"Missing required NPZ keys: {sorted(missing)}")

            wrist_deltas = np.asarray(data["wrist_deltas"], dtype=np.float32)
            gripper_states = np.asarray(data["gripper_states"], dtype=np.float32)
            timestamps = np.asarray(data["timestamps"], dtype=np.float64)
            self._validate_input_arrays(wrist_deltas, gripper_states, timestamps)

            number_of_frames = len(wrist_deltas)
            wrist_positions = np.asarray(data["wrist_positions"], dtype=np.float32) if "wrist_positions" in data else np.zeros((number_of_frames, 3), dtype=np.float32)
            has_tracks, glass_centroids, hand_centroids = self._load_normalized_centroids(data, wrist_positions, number_of_frames)

            positive_gripper_frames = np.where(gripper_states > 0)[0]
            grasp_frame = int(positive_gripper_frames[0]) if len(positive_gripper_frames) else number_of_frames
            entry_frame = self._find_entry_frame(data, wrist_deltas, grasp_frame)
            return_frame, lift_velocity = self._build_lift_profile(data, has_tracks, grasp_frame, number_of_frames)

            if has_tracks:
                attachment_score, hand_velocity, glass_velocity, relative_velocity, hand_speed, glass_speed, relative_speed, motion_threshold = self._compute_attachment_signal(hand_centroids, glass_centroids, timestamps)
                attachment_latch_frame = self._find_attachment_latch(attachment_score, glass_speed, motion_threshold, start_frame=entry_frame)
            else:
                attachment_score = np.zeros(number_of_frames, dtype=np.float32)
                hand_velocity = glass_velocity = relative_velocity = np.zeros((number_of_frames, 2), dtype=np.float32)
                hand_speed = glass_speed = relative_speed = np.zeros(number_of_frames, dtype=np.float32)
                motion_threshold, attachment_latch_frame = np.float32(0.0), -1

            actions = np.zeros((number_of_frames, 7), dtype=np.float32)
            smoothed_translation = np.zeros(3, dtype=np.float32)
            rows = []

            for frame in range(number_of_frames):
                image_distance = float(np.linalg.norm(glass_centroids[frame] - hand_centroids[frame]))
                approach_progress = descent_progress = 0.0

                if frame < grasp_frame:
                    gripper = -1.0
                    if frame < entry_frame:
                        phase, raw_dx, raw_dy, raw_dz = "wait", 0.0, 0.0, 0.0
                    else:
                        phase = "approach"
                        raw_dx, raw_dy, raw_dz, image_distance, approach_progress, descent_progress = self._get_approach_actions(
                            frame, entry_frame, grasp_frame, wrist_deltas[frame], hand_centroids[frame], glass_centroids[frame]
                        )
                elif frame < grasp_frame + self.grasp_settle_frames:
                    phase, raw_dx, raw_dy, raw_dz, gripper = "grasp_settle", 0.0, 0.0, 0.0, 1.0
                elif frame <= return_frame:
                    phase, raw_dx, raw_dy = "lift_return", 0.0, 0.0
                    raw_dz = float(np.clip(lift_velocity[frame] * self.lift_gain, -self.max_lift_action, self.max_lift_action))
                    gripper = 1.0
                elif frame <= return_frame + self.release_settle_frames:
                    phase, raw_dx, raw_dy, raw_dz, gripper = "release", 0.0, 0.0, 0.0, -1.0
                else:
                    phase, raw_dx, raw_dy, raw_dz, gripper = "retract", self.retract_x_action, 0.0, self.retract_z_action, -1.0

                raw_translation = np.array([raw_dx, raw_dy, raw_dz], dtype=np.float32)
                smoothed_translation = raw_translation.copy() if frame == 0 else self.smoothing_alpha * raw_translation + (1.0 - self.smoothing_alpha) * smoothed_translation
                command = np.clip(smoothed_translation, -self.max_action_magnitude, self.max_action_magnitude)
                actions[frame, :3], actions[frame, 3:6], actions[frame, 6] = command, 0.0, gripper

                rows.append({
                    "frame": frame, "timestamp": float(timestamps[frame]), "phase": phase,
                    "hand_x": float(hand_centroids[frame, 0]), "hand_y": float(hand_centroids[frame, 1]),
                    "tumbler_x": float(glass_centroids[frame, 0]), "tumbler_y": float(glass_centroids[frame, 1]),
                    "hand_vx": float(hand_velocity[frame, 0]), "hand_vy": float(hand_velocity[frame, 1]),
                    "tumbler_vx": float(glass_velocity[frame, 0]), "tumbler_vy": float(glass_velocity[frame, 1]),
                    "relative_vx": float(relative_velocity[frame, 0]), "relative_vy": float(relative_velocity[frame, 1]),
                    "hand_speed": float(hand_speed[frame]), "glass_speed": float(glass_speed[frame]),
                    "relative_speed": float(relative_speed[frame]), "attachment_score": float(attachment_score[frame]),
                    "attachment_latched": int(attachment_latch_frame >= 0 and frame >= attachment_latch_frame),
                    "image_distance": image_distance, "raw_dx": raw_dx, "raw_dy": raw_dy, "raw_dz": raw_dz,
                    "command_dx": float(command[0]), "command_dy": float(command[1]), "command_dz": float(command[2]),
                    "gripper": gripper, "source_gripper": float(gripper_states[frame]),
                    "approach_progress": approach_progress, "descent_progress": descent_progress,
                })

            result = {
                "actions": actions,
                "timestamps": timestamps,
                "hand_centroids": hand_centroids,
                "tumbler_centroids": glass_centroids,
                "hand_velocity": hand_velocity,
                "tumbler_velocity": glass_velocity,
                "relative_velocity": relative_velocity,
                "hand_speed": hand_speed,
                "glass_speed": glass_speed,
                "relative_speed": relative_speed,
                "attachment_score": attachment_score,
                "attachment_latch_frame": np.int32(attachment_latch_frame),
                "attachment_threshold": np.float32(self.attachment_threshold),
                "attachment_min_frames": np.int32(self.attachment_min_frames),
                "object_motion_threshold": np.float32(motion_threshold),
                "xyz_commands": actions[:, :3].copy(),
                "source_gripper_states": gripper_states,
                "entry_frame": np.int32(entry_frame),
                "grasp_frame": np.int32(grasp_frame),
                "return_frame": np.int32(return_frame),
                "eef_to_pad_offset_x": np.float32(self.eef_to_pad_offset_x),
                "approach_x_bias": np.float32(self.approach_x_bias),
                "source_file": np.array(str(path)),
            }

            if "rgb_frames" in data:
                result["rgb_frames"] = np.asarray(data["rgb_frames"], dtype=np.uint8)

        output_path = Path(output_file) if output_file else path.parent / f"{path.stem.replace('_trajectory', '')}_actions.npz"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **result)

        log_path = output_path.parent / f"{output_path.stem}_log.csv"
        self._write_log(log_path, rows)

        print(f"[+] Retargeted {number_of_frames} steps")
        print(f"[+] Entry frame: {entry_frame}")
        print(f"[+] Source grasp frame: {grasp_frame}")
        print(f"[+] Attachment latch frame: {attachment_latch_frame}")
        print(f"[+] Attachment threshold: {self.attachment_threshold:.2f}")
        print(f"[+] Object-motion threshold: {float(motion_threshold):.6f}")
        print(f"[+] Actions saved to: {output_path}")
        print(f"[+] Log saved to: {log_path}")
        return result

    @staticmethod
    def _write_log(log_path: Path, rows):
        columns = [
            "frame", "timestamp", "phase", "hand_x", "hand_y",
            "tumbler_x", "tumbler_y", "hand_vx", "hand_vy",
            "tumbler_vx", "tumbler_vy", "relative_vx", "relative_vy",
            "hand_speed", "glass_speed", "relative_speed",
            "attachment_score", "attachment_latched", "image_distance",
            "raw_dx", "raw_dy", "raw_dz", "command_dx", "command_dy",
            "command_dz", "gripper", "source_gripper",
            "approach_progress", "descent_progress",
        ]

        with open(log_path, "w", encoding="utf-8") as file:
            file.write(",".join(columns) + "\n")

            for row in rows:
                values = []

                for column in columns:
                    value = row[column]

                    if isinstance(value, (float, np.floating)):
                        values.append(f"{value:.8f}")
                    else:
                        values.append(str(value))

                file.write(",".join(values) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retarget CoTracker trajectory with attachment detection.")
    parser.add_argument("trajectory")
    parser.add_argument("output_folder", nargs="?", default="datasets/cotracker_iterations/iteration_21_attachment")
    parser.add_argument("--approach-x-bias", type=float, default=0.123)
    parser.add_argument("--descent-delay-frac", type=float, default=0.45)
    parser.add_argument("--descent-rate", type=float, default=-0.28)
    parser.add_argument("--attachment-threshold", type=float, default=0.65)
    parser.add_argument("--attachment-min-frames", type=int, default=3)
    parser.add_argument("--attachment-smoothing-frames", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.trajectory)
    output_directory = Path(args.output_folder)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = output_directory / f"{input_path.stem.replace('_trajectory', '')}_actions.npz"

    retargeter = CoTrackerToPandaRetargeter(
        approach_x_bias=args.approach_x_bias,
        descent_delay_frac=args.descent_delay_frac,
        descent_rate=args.descent_rate,
        attachment_threshold=args.attachment_threshold,
        attachment_min_frames=args.attachment_min_frames,
        attachment_smoothing_frames=args.attachment_smoothing_frames,
    )
    retargeter.retarget_trajectory(str(input_path), str(output_file))