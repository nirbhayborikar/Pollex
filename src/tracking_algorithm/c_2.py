"""
CoTracker3 Dense Point Tracker (NVIDIA Jetson AGX Thor / CUDA)
- Tracks dense point trajectories on the human hand and target tumbler throughout the video.
- Estimates wrist motion from tracked point displacements and identifies grasp and release events
using hand-object co-motion correlation (rho_t).
"""

import os
import cv2
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = _select_device()

# Tracking grid configurations
HAND_GRID_SHAPE = (6, 6)   # 36 tracking points on the hand
GLASS_GRID_SHAPE = (4, 4)  # 16 tracking points on the steel glass


# Minimum hand/tumbler movement in pixels per frame required
# before cosine similarity is considered reliable.
MIN_VELOCITY_THRESH = 0.4     # minimum velocity magnitude to compute correlation (suppress noise)

# Do not allow grasp detection immediately when the hand first appears.
# With a 30 FPS video, 15 frames correspond to 0.5 seconds.
GRASP_FRAME_DELAY = 15

# ============================================================
# GRASP PARAMETERS
# ============================================================

# Close when hand and tumbler move in approximately the same direction.
GRASP_CORR_THRESHOLD = 0.85

# Close only when normalized hand-tumbler centroid distance is below 0.22.
GRASP_PROXIMITY_THRESH = 0.22


# ============================================================
# RELEASE PARAMETERS
# ============================================================

# Release when hand and tumbler no longer move together.
RELEASE_CORR_THRESHOLD = 0.20

# At the true release moment, the hand can still be close to the tumbler.
# Your observed release distance was approximately 0.19.
RELEASE_PROXIMITY_THRESH = 0.25

# Safety release when hand and tumbler are clearly far apart.
RELEASE_DIST_THRESH = 0.40

# ============================================================
# DEBOUNCE — prevents single-frame tremor from toggling state (We used option 2)
# ============================================================
# The trigger condition must hold for this many CONSECUTIVE
# frames before the state actually transitions.
# At 30 FPS, 5 frames ≈ 170 ms — fast enough for real events,


# slow enough to filter hand tremor.
CONFIRM_FRAMES = 5



class CoTrackerTrajectoryExtractor:
    """Extracts 6D rigid hand motion and object co-motion grasp states using CoTracker3 dense point tracking."""
    def __init__(self, target_size: Tuple[int, int] = (512, 512),model_name: str = "cotracker3_offline", device: str = DEVICE): # old 384 target size
        self.target_size = target_size
        self.device = device
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """loads CoTracker3 onto the target device."""
        if self._model is None:
            print(f"[*] Loading CoTracker3 ({self.model_name}) on {self.device}...")
            # Load CoTracker3 from the Facebook Research CoTracker
            # repository using PyTorch Hub.
            self._model = torch.hub.load("facebookresearch/co-tracker",self.model_name,force_reload=False).to(self.device)

            # Put CoTracker3 into inference/evaluation mode.
            """
            We are using the network to track points, not training it.
            "eval()" disables training-specific behavior in applicable
            neural-network layers.
            """
            self._model.eval()
            print("[+] CoTracker3 loaded successfully.")

        # Return the loaded model.
        """ If the "model" property is called again, self._model is no longer None, 
        so the expensive torch.hub.load() step above is skipped and the same model instance is returned. 
        """
        return self._model

    def _detect_hand_and_tembler_rois(self,frames: List[np.ndarray]) -> Tuple[int, Tuple[int, int, int, int], Optional[np.ndarray]]:
        """
        Finds the initial points needed by CoTracker3:
        1. Tumbler ROI: Defines the rectangular image region containing the tumbler. On table at frame 0 (x0, y0, x1, y1), strictly on the metal cup.
         IMPORTANT: The current implementation does NOT detect the tumbler automatically. Its position is manually defined using percentages of the image width and height.
        
        2. Hand entry frame:
        Searches the beginning of the video using MediaPipe and finds the first frame where a hand can be detected.
        
        3. Initial hand tracking points:
        MediaPipe provides 21 anatomical hand landmarks on the first detected hand frame. 
        These points are later given to CoTracker3 as query points.

        Hand initial points (21 landmark points) at entry frame t_hand.
        """
        T = len(frames)
        H, W = self.target_size

        # Tight glass ROI positioned strictly on cylindrical metal body.
        # With letterboxing, portrait video is centered with dark bars on left/right.
        # These percentages must cover ONLY the metal cup body — not the table.
        # Tune these 4 values by checking the first frame of your annotated video.
        """
        Left = decrease x
        Right = increase x
        Up = decrease y
        Down = increase y
        """
        glass_roi = (
            int(W * 0.38),  # x0 = LEFT edge. 0.40 -> 0.38 # left 
            int(H * 0.48),  # y0 = TOP edge # # 0.38 -> 0.40 (down)
            int(W * 0.52),  # x1 = RIGHT edge # # 0.56 -> 0.54 # left 
            int(H * 0.66),  # y1 = BOTTOM edge # # 0.58 -> 0.60 (down)
            )

        t_entry = None
        hand_pts_init = None

        try:
            model_path = Path("models/hand_landmarker.task")
            if not model_path.exists():
                try:
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"[*] Downloading MediaPipe Hand Landmarker model to {model_path}...")
                    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",str(model_path))
                    print("[+] MediaPipe Hand Landmarker model downloaded successfully.")
                except Exception as dl_err:
                    print(f"[!] Could not auto-download MediaPipe model: {dl_err}")

            if model_path.exists():
                base_options = python.BaseOptions(model_asset_path=str(model_path))
                """
                vision.RunningMode.VIDEO -> This allows MediaPipe to use tracking between frames.
                hand detection confidence : 0.4   Lower: more detections and potentially more false positives
                tracking_confidence=0.4 , Lower: tracking may survive difficult frames better, but can become less reliable
                """
                options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_hands=1, min_hand_detection_confidence=0.3)
                landmarker = vision.HandLandmarker.create_from_options(options)

                # Scan frames to find first reliable hand detection
                # T = total number of video frames.
                for t in range(0, min(T, 100), 2): #check every second frame, MediaPipe actually processes about 50 images during this search.
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frames[t])
                    res = landmarker.detect(mp_img)#  Run hand detection
                    if res.hand_landmarks:
                        lms = res.hand_landmarks[0]
                        # Extract all 21 keypoints directly on hand anatomy
                        pts = []  # Prepare a list for the 21 points
                        for lm in lms:
                            """
                            MediaPipe does NOT return x and y directly in 512x512 pixel coordinates.
                            It gives normalized coordinates. For example: lm.x = 0.50, lm.y = 0.70
                             means: x = 50% across the image, y = 70% down the image. 
                            """
                            # Clamp within valid video area (avoid black letterbox padding)
                            px = np.clip(lm.x * W, W * 0.22 + 5, W * 0.78 - 5)# Convert Normalized X to Pixel X
                            py = np.clip(lm.y * H, 5, H - 5)
                            pts.append([px, py])
                        pts = np.array(pts, dtype=np.float32)
                        t_entry = t # Remember when the hand appeared
                        hand_pts_init = pts
                        break
        except Exception as e:
            print(f"[!] MediaPipe scan note: {e}")

        if t_entry is None:
            t_entry = 30
            # Fallback 21 points tightly focused on the approaching hand region:
            hxs = np.linspace(W * 0.65, W * 0.74, 5)
            hys = np.linspace(H * 0.72, H * 0.82, 5)
            gx, gy = np.meshgrid(hxs, hys)
            hand_pts_init = np.stack([gx.reshape(-1)[:21], gy.reshape(-1)[:21]], axis=1).astype(np.float32)

        print(f"[*] Detected Tight Glass ROI at t=0: {glass_roi}")
        print(f"[*] Detected Hand entry at t={t_entry} with {len(hand_pts_init)} anatomical keypoints")
        return t_entry, glass_roi, hand_pts_init


    def _build_queries(self, t_hand: int, glass_roi: Tuple[int, int, int, int],hand_pts_init: np.ndarray) -> torch.Tensor:
        """Constructs query tensor of shape (1, N_glass + N_hand, 3) with [t, x, y]."""
        # Glass points seeded at t=0
        gx0, gy0, gx1, gy1 = glass_roi
        gxs = torch.linspace(gx0, gx1, GLASS_GRID_SHAPE[0], device=self.device)
        gys = torch.linspace(gy0, gy1, GLASS_GRID_SHAPE[1], device=self.device)
        gx_grid, gy_grid = torch.meshgrid(gxs, gys, indexing="xy")
        g_pts = torch.stack([gx_grid.reshape(-1), gy_grid.reshape(-1)], dim=1)
        g_queries = torch.cat([torch.zeros(len(g_pts), 1, device=self.device), g_pts], dim=1)

        # Hand points seeded directly at the 21 anatomical keypoints at t=t_hand
        h_pts = torch.from_numpy(hand_pts_init).float().to(self.device)
        h_queries = torch.cat([torch.full((len(h_pts), 1), float(t_hand), device=self.device), h_pts], dim=1)

        all_queries = torch.cat([g_queries, h_queries], dim=0).unsqueeze(0)  # (1, N, 3)
        return all_queries

    def process_video(self, video_path: str, output_dir: str = "datasets/cotracker_trajectories", save_annotated_video: bool = True) -> Dict[str, np.ndarray]:
        """
        Runs dense tracking on video, extracts hand trajectories,
        computes co-motion correlation, and saves trajectory dataset.
        """
        vpath = Path(video_path)
        if not vpath.exists():
            raise FileNotFoundError(f"Video not found: {vpath}")

        cap = cv2.VideoCapture(str(vpath))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"[*] Processing {vpath.name}: {total_frames} frames @ {fps:.1f} FPS ({orig_w}x{orig_h})")

        # Letterbox into square frame (preserve aspect ratio)
        max_dim = max(orig_w, orig_h)
        pad_top = (max_dim - orig_h) // 2
        pad_bottom = max_dim - orig_h - pad_top
        pad_left = (max_dim - orig_w) // 2
        pad_right = max_dim - orig_w - pad_left

        frames_rgb: List[np.ndarray] = []
        timestamps: List[float] = []
        for f_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            padded = cv2.copyMakeBorder(
                frame, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[30, 30, 30],
            )
            resized = cv2.resize(padded, self.target_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames_rgb.append(rgb)
            timestamps.append(f_idx / fps)
        cap.release()

        T = len(frames_rgb)
        if T == 0:
            raise ValueError(f"No frames read from {vpath}")

        # 1. Detect ROIs & Keypoints
        t_hand, glass_roi, hand_pts_init = self._detect_hand_and_tembler_rois(frames_rgb)
        N_glass = GLASS_GRID_SHAPE[0] * GLASS_GRID_SHAPE[1]
        N_hand = len(hand_pts_init)

        # 2. Build queries & run CoTracker3 forward pass
        video_tensor = (
            torch.from_numpy(np.stack(frames_rgb))
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )
        queries = self._build_queries(t_hand, glass_roi, hand_pts_init)

        print(f"[*] Executing CoTracker3 tracking ({N_glass} glass pts + {N_hand} hand pts on {self.device})...")
        with torch.no_grad():
            tracks, vis = self.model(video_tensor, queries=queries)

        tracks = tracks.squeeze(0).cpu().numpy()   # (T, N_total, 2)
        vis = vis.squeeze(0).cpu().numpy()          # (T, N_total)

        glass_tracks = tracks[:, :N_glass, :]       # (T, N_glass, 2)
        hand_tracks = tracks[:, N_glass:, :]        # (T, N_hand, 2)

        # 3. Compute Centroids & Normalized Trajectories
        H, W = self.target_size
        glass_centroids = glass_tracks.mean(axis=1) / np.array([W, H], dtype=np.float32)  # (T, 2)
        hand_centroids = hand_tracks.mean(axis=1) / np.array([W, H], dtype=np.float32)    # (T, 2)

        for t in range(t_hand):
            hand_centroids[t] = hand_centroids[t_hand]

        # Wrist 3D proxy: [centroid_x, centroid_y, dist_to_glass]
        wrist_pos_3d = np.zeros((T, 3), dtype=np.float32)
        for t in range(T):
            dist_to_glass = float(np.linalg.norm(hand_centroids[t] - glass_centroids[t]))
            wrist_pos_3d[t, 0] = hand_centroids[t, 0]
            wrist_pos_3d[t, 1] = hand_centroids[t, 1]
            wrist_pos_3d[t, 2] = dist_to_glass

        wrist_deltas = np.zeros_like(wrist_pos_3d)
        if T > 1:
            wrist_deltas[1:] = wrist_pos_3d[1:] - wrist_pos_3d[:-1]
        wrist_deltas[:t_hand] = 0.0

        # 4. Co-motion Grasp Correlation (cosine similarity of velocity vectors)
        v_glass = glass_tracks[1:] - glass_tracks[:-1]  # (T-1, N_glass, 2)
        v_hand = hand_tracks[1:] - hand_tracks[:-1]     # (T-1, N_hand, 2)
        mean_vg = v_glass.mean(axis=1)                   # (T-1, 2)
        mean_vh = v_hand.mean(axis=1)                    # (T-1, 2)

        # ============================================================
        # CO-MOTION GRASP AND RELEASE DETECTION
        # ============================================================

        corrs = np.zeros(T, dtype=np.float32)
        # -1.0 = OPEN, +1.0 = CLOSED
        gripper_states = np.full(T,-1.0,dtype=np.float32)
        current_gripper = -1.0

        # Debounce counters: how many consecutive frames the
        # pending transition condition has been true.
        grasp_confirm_count = 0
        release_confirm_count = 0

        for t in range(1, T):
            # Mean tumbler and hand displacement between frames t-1 and t.
            vg = mean_vg[t - 1]
            vh = mean_vh[t - 1]
            # Velocity/displacement magnitudes in pixels per frame.
            norm_g = float(np.linalg.norm(vg))
            norm_h = float(np.linalg.norm(vh))
            # --------------------------------------------------------
            # Calculate hand-tumbler directional similarity
            # --------------------------------------------------------

            # Only calculate cosine similarity when BOTH the hand and
            # tumbler have enough motion to give a meaningful direction.
            if (norm_g > MIN_VELOCITY_THRESH and norm_h > MIN_VELOCITY_THRESH):
                cos_sim = float(np.dot(vg, vh) / (norm_g * norm_h))
                # Protect against small floating-point errors.
                cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
            else:
                # If either object is nearly stationary, its direction
                # is not reliable. Treat co-motion correlation as zero.
                cos_sim = 0.0

            # Save correlation for visualization and dataset generation.
            corrs[t] = cos_sim
            # --------------------------------------------------------
            # Calculate normalized hand-tumbler distance
            # --------------------------------------------------------
            dist_to_glass = float(np.linalg.norm(hand_centroids[t] - glass_centroids[t]))

            # Named Boolean conditions make the transition logic clearer.
            is_near_glass = (dist_to_glass < GRASP_PROXIMITY_THRESH)
            is_in_release_proximity = (dist_to_glass < RELEASE_PROXIMITY_THRESH)
            is_far_from_glass = (dist_to_glass > RELEASE_DIST_THRESH)
            has_strong_co_motion = (cos_sim > GRASP_CORR_THRESHOLD)
            has_lost_co_motion = (cos_sim < RELEASE_CORR_THRESHOLD)
            grasp_delay_passed = (t >= t_hand + GRASP_FRAME_DELAY)


            # ========================================================
            # CURRENT STATE: OPEN
            # ========================================================

            if current_gripper < 0:
                grasp_want = (has_strong_co_motion and is_near_glass and grasp_delay_passed)

                if grasp_want:
                    grasp_confirm_count += 1
                else:
                    grasp_confirm_count = 0

                if grasp_confirm_count >= CONFIRM_FRAMES:
                    current_gripper = 1.0
                    grasp_confirm_count = 0
                    release_confirm_count = 0
                    print(
                        f"    [GRASP]   "
                        f"frame {t} "
                        f"(t={t / fps:.2f}s) "
                        f"rho={cos_sim:.3f} "
                        f"dist={dist_to_glass:.3f} "
                        f"glass_speed={norm_g:.3f} "
                        f"hand_speed={norm_h:.3f}"
                    )


            # ========================================================
            # CURRENT STATE: CLOSED
            # ========================================================

            else:

                # Primary release condition:
                #
                # The hand and tumbler have stopped moving together,
                # but the hand is still close to the tumbler.
                #
                # IMPORTANT: only fire this when at least one object is
                # actually moving. When both are nearly stationary,
                # cos_sim is forced to 0.0 by MIN_VELOCITY_THRESH —
                # that's "undefined co-motion", not "lost co-motion".
                # Without this guard, any pause mid-lift triggers a
                # false release.
                either_moving = (norm_g > MIN_VELOCITY_THRESH or norm_h > MIN_VELOCITY_THRESH)
                release_by_lost_co_motion = (has_lost_co_motion and is_in_release_proximity and either_moving)

                # Safety release:
                #
                # If the hand has moved very far from the tumbler,
                # it cannot reasonably still be holding it.
                release_by_separation = is_far_from_glass

                release_want = (release_by_lost_co_motion or release_by_separation)

                if release_want:
                    release_confirm_count += 1
                else:
                    release_confirm_count = 0

                if release_confirm_count >= CONFIRM_FRAMES:
                    current_gripper = -1.0
                    release_confirm_count = 0
                    grasp_confirm_count = 0
                    if release_by_lost_co_motion:
                        release_reason = "lost_co_motion"
                    else:
                        release_reason = "large_separation"

                    print(
                        f"    [RELEASE] "
                        f"frame {t} "
                        f"(t={t / fps:.2f}s) "
                        f"rho={cos_sim:.3f} "
                        f"dist={dist_to_glass:.3f} "
                        f"glass_speed={norm_g:.3f} "
                        f"hand_speed={norm_h:.3f} "
                        f"reason={release_reason}"
                    )


            # Save the resulting state for this frame.
            gripper_states[t] = current_gripper

        # 5. Save Annotated Video
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if save_annotated_video:
            ann_path = out_dir / f"{vpath.stem}_ct_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(ann_path), fourcc, fps, (W, H))

            for t in range(T):
                vis_bgr = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR)

                for pt in glass_tracks[t]:
                    cv2.circle(vis_bgr, (int(pt[0]), int(pt[1])), 3, (255, 120, 0), -1)

                hand_color = (0, 0, 255) if gripper_states[t] > 0 else (0, 255, 0)
                if t >= t_hand:
                    for pt in hand_tracks[t]:
                        cv2.circle(vis_bgr, (int(pt[0]), int(pt[1])), 3, hand_color, -1)

                    gc = (int(glass_centroids[t, 0] * W), int(glass_centroids[t, 1] * H))
                    hc = (int(hand_centroids[t, 0] * W), int(hand_centroids[t, 1] * H))
                    cv2.line(vis_bgr, gc, hc, hand_color, 2)

                state_text = "GRASPED (LIFTING)" if gripper_states[t] > 0 else "APPROACHING"
                dist_t = float(np.linalg.norm(hand_centroids[t] - glass_centroids[t]))
                cv2.putText(vis_bgr, f"CoTracker3: {state_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)
                cv2.putText(vis_bgr, f"Co-motion rho: {corrs[t]:.2f}  dist: {dist_t:.2f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                writer.write(vis_bgr)
            writer.release()
            print(f"[+] Saved annotated video -> {ann_path}")

        # 6. Save Trajectory Dataset
        output_file = out_dir / f"{vpath.stem}_ct_trajectory.npz"
        traj_data = {
            "timestamps": np.array(timestamps, dtype=np.float32),
            "rgb_frames": np.array(frames_rgb, dtype=np.uint8),
            "wrist_positions": wrist_pos_3d,
            "wrist_deltas": wrist_deltas,
            "gripper_states": gripper_states,
            "correlations": corrs,
            "glass_tracks": glass_tracks.astype(np.float32),
            "hand_tracks": hand_tracks.astype(np.float32),
            "t_hand": np.int32(t_hand),
            "fps": np.float32(fps),
            "source_video": str(vpath),
        }
        np.savez_compressed(str(output_file), **traj_data)
        print(f"[+] Saved CoTracker3 trajectory -> {output_file}")
        return traj_data


def process_all_raw_videos(raw_dir: str, output_dir: str):
    """
    Process every supported raw video in a directory using CoTracker3.

    Annotated and synthetic videos are excluded to prevent processing
    generated outputs as new inputs.
    """
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw video directory was not found: {raw_path}")

    candidates = []
    for pattern in ("*.mp4", "*.MP4", "*.mov", "*.MOV"):
        candidates.extend(raw_path.glob(pattern))

    unique_videos = []
    seen_paths: set = set()

    for video_path in sorted(candidates):
        resolved_path = str(video_path.resolve())

        if resolved_path in seen_paths:
            continue

        if video_path.stem.endswith(("_annotated", "_ct_annotated")):
            continue

        if video_path.name.startswith("synthetic"):
            continue

        seen_paths.add(resolved_path)
        unique_videos.append(video_path)

    if not unique_videos:
        print(f"[-] No raw MP4 or MOV videos found in {raw_dir}")
        return

    print(f"[*] Found {len(unique_videos)} videos to process in {raw_dir}")

    extractor = CoTrackerTrajectoryExtractor(target_size=(512, 512))

    for index, video_path in enumerate(unique_videos, start=1):
        print(f"\n[*] Video {index}/{len(unique_videos)}: {video_path.name}")

        try:
            extractor.process_video(
                video_path=str(video_path),
                output_dir=output_dir,
                save_annotated_video=True,
            )
        except Exception as error:
            print(f"[!] Failed to process {video_path.name}: {error}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Extract dense hand and tumbler trajectories "
            "from video using CoTracker3"))

    parser.add_argument("--video", type=str, default=None, help="Path to one video file")
    parser.add_argument("--raw_dir", type=str, default="media", help="Directory containing raw videos")
    parser.add_argument("--output_dir", type=str, default="datasets/cotracker_trajectories", help="Directory for trajectory datasets and annotated videos")
    parser.add_argument("--no_annotated_video", action="store_true", help="Do not save the annotated visualization video")
    args = parser.parse_args()

    if args.video:
        extractor = CoTrackerTrajectoryExtractor(target_size=(512, 512))
        extractor.process_video(
            video_path=args.video,
            output_dir=args.output_dir,
            save_annotated_video=(not args.no_annotated_video),
        )
    else:
        process_all_raw_videos(raw_dir=args.raw_dir, output_dir=args.output_dir)


""" 
python3 src/tracking_algorithm/cotracker_dense_point_tracking.py \
    --video datasets/egocentric_video/cup_lifting_01.mp4 \
    --output_dir datasets/cotracker_trajectories
"""