"""
Hand Landmark Tracker & Action Extractor (MediaPipe Tasks API)
Extracts 3D hand keypoints from real egocentric phone videos,
computes 3D wrist displacements, and determines gripper grasp/lift actions.
"""

import os
import cv2
import argparse
import urllib.request
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_MODEL_PATH = "models/hand_landmarker.task"


def ensure_model_exists(model_path: str = DEFAULT_MODEL_PATH):
    """Downloads the official MediaPipe hand landmarker model if missing."""
    p = Path(model_path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"[*] Downloading MediaPipe HandLandmarker model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, str(p))
        print("[+] Download complete.")


class HandTrajectoryExtractor:
    """
    Extracts 3D hand landmarks, wrist deltas, and pinch/grasp state
    using MediaPipe Tasks API.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        grasp_threshold: float = 0.24,
        target_size: Tuple[int, int] = (256, 256),
    ):
        ensure_model_exists(model_path)
        self.model_path = model_path
        self.grasp_threshold = grasp_threshold
        self.target_size = target_size

    def _create_landmarker(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        return vision.HandLandmarker.create_from_options(options)

    def process_video(
        self,
        video_path: str,
        output_dir: str,
        save_annotated_video: bool = True,
    ) -> Optional[Dict[str, np.ndarray]]:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[*] Processing {video_path.name}: {total_frames} frames @ {fps:.1f} FPS ({width}x{height})")
        landmarker = self._create_landmarker()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        annotated_writer = None
        if save_annotated_video:
            annotated_video_path = output_path / f"{video_path.stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            annotated_writer = cv2.VideoWriter(
                str(annotated_video_path), fourcc, fps, self.target_size
            )

        frames_rgb = []
        wrist_positions = []
        pinch_distances = []
        gripper_states = []
        raw_landmarks_all = []
        timestamps = []

        last_valid_wrist = None
        last_valid_dist = 0.35
        last_valid_idx_tip = None
        current_gripper_state = -1.0  # start OPEN
        detected_indices = []

        close_threshold = 0.19
        open_threshold = 0.26
        jump_threshold = 0.12  # in normalized units

        max_dim = max(width, height)
        pad_top = (max_dim - height) // 2
        pad_bottom = max_dim - height - pad_top
        pad_left = (max_dim - width) // 2
        pad_right = max_dim - width - pad_left

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Letterbox pad to preserve 100% of the field of view without clipping
            padded = cv2.copyMakeBorder(
                frame, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[30, 30, 30]
            )
            resized = cv2.resize(padded, self.target_size)
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            t = frame_idx / fps
            timestamps.append(t)

            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_idx * 1000 / fps)

            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            wrist_pos = None
            pinch_dist = last_valid_dist
            lms_21 = np.zeros((21, 3), dtype=np.float32)
            used_landmark_name = "L8 (TIP)"
            active_target_pt = None

            if detection_result.hand_landmarks:
                detected_indices.append(frame_idx)
                lms = detection_result.hand_landmarks[0]
                for idx, lm in enumerate(lms):
                    lms_21[idx] = [lm.x, lm.y, lm.z]

                wrist_pos = lms_21[0].copy()
                last_valid_wrist = wrist_pos.copy()

                # Landmarks: 4=Thumb Tip, 8=Index Tip, 7=Index DIP, 6=Index PIP, 5=Index MCP
                thumb_tip = lms_21[4]
                idx_tip = lms_21[8]
                idx_dip = lms_21[7]
                idx_pip = lms_21[6]

                # Check if index tip abruptly moved or went near image edge
                is_idx_abrupt = False
                if last_valid_idx_tip is not None:
                    step_displacement = np.linalg.norm(idx_tip - last_valid_idx_tip)
                    if step_displacement > jump_threshold:
                        is_idx_abrupt = True

                # Also flag if landmark 8 is jammed near the edge
                if idx_tip[1] < 0.04 or idx_tip[1] > 0.96 or idx_tip[0] < 0.04 or idx_tip[0] > 0.96:
                    is_idx_abrupt = True

                if not is_idx_abrupt:
                    last_valid_idx_tip = idx_tip.copy()
                    # Tip is trustworthy: use minimum distance to tip or DIP
                    d_tip = float(np.linalg.norm(thumb_tip - idx_tip))
                    d_dip = float(np.linalg.norm(thumb_tip - idx_dip))
                    if d_tip <= d_dip:
                        pinch_dist = d_tip
                        active_target_pt = idx_tip
                        used_landmark_name = "L8 (TIP)"
                    else:
                        pinch_dist = d_dip
                        active_target_pt = idx_dip
                        used_landmark_name = "L7 (DIP)"
                else:
                    # Index tip is occluded/abrupt; fallback to DIP (L7) or PIP (L6)
                    d_dip = float(np.linalg.norm(thumb_tip - idx_dip))
                    d_pip = float(np.linalg.norm(thumb_tip - idx_pip))
                    if d_dip <= d_pip:
                        pinch_dist = d_dip
                        active_target_pt = idx_dip
                        used_landmark_name = "L7 (DIP fallback)"
                    else:
                        pinch_dist = d_pip
                        active_target_pt = idx_pip
                        used_landmark_name = "L6 (PIP fallback)"

                last_valid_dist = pinch_dist

            elif last_valid_wrist is not None:
                # Maintain last known position
                wrist_pos = last_valid_wrist.copy()
            else:
                # Hand not yet in view
                wrist_pos = np.array([0.5, 0.9, 0.0], dtype=np.float32)

            # Hysteresis trigger for gripper state:
            # Prevents chatter / flickering around threshold boundary
            if current_gripper_state < 0:  # currently OPEN
                if pinch_dist < close_threshold:
                    current_gripper_state = 1.0  # CLOSED
            else:  # currently CLOSED
                if pinch_dist > open_threshold:
                    current_gripper_state = -1.0  # OPEN

            gripper_cmd = current_gripper_state

            if save_annotated_video and annotated_writer:
                vis_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                if detection_result.hand_landmarks:
                    # Draw keypoints
                    for pt in lms_21:
                        px, py = int(pt[0] * self.target_size[0]), int(pt[1] * self.target_size[1])
                        cv2.circle(vis_bgr, (px, py), 3, (0, 255, 0), -1)

                    # Highlight pinch line to the active landmark
                    if active_target_pt is not None:
                        p_thumb = (int(lms_21[4, 0] * self.target_size[0]), int(lms_21[4, 1] * self.target_size[1]))
                        p_target = (int(active_target_pt[0] * self.target_size[0]), int(active_target_pt[1] * self.target_size[1]))
                        line_color = (0, 0, 255) if gripper_cmd > 0 else (0, 255, 0)
                        cv2.line(vis_bgr, p_thumb, p_target, line_color, 2)

                status_text = f"GRIPPER: {'CLOSED (HOLDING GLASS)' if gripper_cmd > 0 else 'OPEN (REACHING)'}"
                color = (0, 0, 255) if gripper_cmd > 0 else (0, 255, 0)
                cv2.putText(vis_bgr, status_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2)
                cv2.putText(vis_bgr, f"Dist: {pinch_dist:.3f} [{used_landmark_name}]", (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                annotated_writer.write(vis_bgr)

            frames_rgb.append(rgb_frame)
            wrist_positions.append(wrist_pos)
            pinch_distances.append(pinch_dist)
            gripper_states.append(gripper_cmd)
            raw_landmarks_all.append(lms_21)

        cap.release()
        if annotated_writer:
            annotated_writer.release()

        wrist_positions = np.array(wrist_positions, dtype=np.float32)
        gripper_states = np.array(gripper_states, dtype=np.float32)

        # Compute wrist deltas: delta[t] = pos[t] - pos[t-1]
        wrist_deltas = np.zeros_like(wrist_positions)
        if len(wrist_positions) > 1:
            wrist_deltas[1:] = wrist_positions[1:] - wrist_positions[:-1]

        trajectory_data = {
            "timestamps": np.array(timestamps, dtype=np.float32),
            "rgb_frames": np.array(frames_rgb, dtype=np.uint8),
            "wrist_positions": wrist_positions,
            "wrist_deltas": wrist_deltas,
            "pinch_distances": np.array(pinch_distances, dtype=np.float32),
            "gripper_states": gripper_states,
            "landmarks_3d": np.array(raw_landmarks_all, dtype=np.float32),
            "fps": np.float32(fps),
            "source_video": str(video_path),
        }

        save_file = output_path / f"{video_path.stem}_trajectory.npz"
        np.savez_compressed(save_file, **trajectory_data)
        print(f"[+] Extracted {len(frames_rgb)} steps (Hand detected in {len(detected_indices)} frames) -> {save_file}")
        return trajectory_data


def process_all_raw_videos(raw_dir: str, output_dir: str):
    """Batch processes all mp4/mov videos in raw_dir."""
    raw_path = Path(raw_dir)
    videos = sorted(list(raw_path.glob("Vid_*.mp4")) + list(raw_path.glob("demo_*.mp4")) + list(raw_path.glob("*.mp4")))
    # Deduplicate
    unique_videos = []
    seen = set()
    for v in videos:
        if v.name not in seen and not v.name.endswith("_annotated.mp4") and not v.name.startswith("synthetic"):
            seen.add(v.name)
            unique_videos.append(v)

    if not unique_videos:
        print(f"[-] No raw video files found in {raw_dir}")
        return

    print(f"[*] Found {len(unique_videos)} videos to process in {raw_dir}")
    extractor = HandTrajectoryExtractor()
    for video in unique_videos:
        extractor.process_video(str(video), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract hand landmarks and actions from video")
    parser.add_argument("--video", type=str, default=None, help="Path to single video file")
    parser.add_argument("--raw_dir", type=str, default="media", help="Directory of raw videos")
    parser.add_argument("--output_dir", type=str, default="data/processed_trajectories", help="Output directory")
    args = parser.parse_args()

    if args.video:
        extractor = HandTrajectoryExtractor()
        extractor.process_video(args.video, args.output_dir)
    else:
        process_all_raw_videos(args.raw_dir, args.output_dir)
