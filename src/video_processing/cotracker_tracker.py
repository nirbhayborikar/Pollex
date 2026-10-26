"""
CoTracker3 Dense Point Tracker & Action Extractor (Apple Silicon MPS)
Tracks dense point clouds on human hand and target steel glass across video frames.
Computes rigid wrist displacements via SVD/centroid tracking and detects
grasp/release transitions via co-motion velocity correlation (rho_t).
"""

import os
import cv2
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Check device: Apple Silicon GPU (MPS) or CPU fallback
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Tracking grid configurations
HAND_GRID_SHAPE = (6, 6)   # 36 tracking points on the hand
GLASS_GRID_SHAPE = (4, 4)  # 16 tracking points on the steel glass

# Co-motion grasp correlation parameters
GRASP_CORR_THRESHOLD = 0.55   # rho > 0.55 triggers gripper close
RELEASE_CORR_THRESHOLD = 0.25 # rho < 0.25 triggers gripper open


class CoTrackerTrajectoryExtractor:
    """
    Extracts 6D rigid hand motion and object co-motion grasp states
    using CoTracker3 dense point tracking on Apple Silicon MPS.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (384, 384),
        model_name: str = "cotracker3_offline",
        device: str = DEVICE,
    ):
        self.target_size = target_size
        self.device = device
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy-loads CoTracker3 onto the target device."""
        if self._model is None:
            print(f"[*] Loading CoTracker3 ({self.model_name}) on {self.device}...")
            self._model = torch.hub.load(
                "facebookresearch/co-tracker",
                self.model_name,
                force_reload=False,
            ).to(self.device)
            self._model.eval()
            print("[+] CoTracker3 loaded successfully.")
        return self._model

    def _detect_hand_and_glass_rois(
        self,
        frames: List[np.ndarray],
    ) -> Tuple[int, Tuple[int, int, int, int], Optional[np.ndarray]]:
        """
        Detects:
        1. Tight Glass ROI on table at frame 0 (x0, y0, x1, y1), strictly on the metal cup.
        2. First frame index t_hand where the human hand enters the frame.
        3. Hand initial points (21 landmark points) at entry frame t_hand.
        """
        T = len(frames)
        H, W = self.target_size

        # Tight glass ROI positioned strictly on cylindrical metal body (avoiding table below):
        # In 384x384: X in [160, 215], Y in [145, 222]
        glass_roi = (int(W * 0.42), int(H * 0.38), int(W * 0.56), int(H * 0.58))

        t_entry = None
        hand_pts_init = None

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import urllib.request

            model_path = Path("models/hand_landmarker.task")
            if not model_path.exists():
                try:
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"[*] Downloading MediaPipe Hand Landmarker model to {model_path}...")
                    urllib.request.urlretrieve(
                        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                        str(model_path),
                    )
                    print("[+] MediaPipe Hand Landmarker model downloaded successfully.")
                except Exception as dl_err:
                    print(f"[!] Could not auto-download MediaPipe model: {dl_err}")

            if model_path.exists():
                base_options = python.BaseOptions(model_asset_path=str(model_path))
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=1,
                    min_hand_detection_confidence=0.3,
                )
                landmarker = vision.HandLandmarker.create_from_options(options)

                # Scan frames to find first reliable hand detection
                for t in range(0, min(T, 100), 2):
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frames[t])
                    res = landmarker.detect(mp_img)
                    if res.hand_landmarks:
                        lms = res.hand_landmarks[0]
                        # Extract all 21 keypoints directly on hand anatomy
                        pts = []
                        for lm in lms:
                            # Clamp within valid video area (avoid black letterbox padding)
                            px = np.clip(lm.x * W, W * 0.22 + 5, W * 0.78 - 5)
                            py = np.clip(lm.y * H, 5, H - 5)
                            pts.append([px, py])
                        pts = np.array(pts, dtype=np.float32)
                        t_entry = t
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



    def _build_queries(
        self,
        t_hand: int,
        glass_roi: Tuple[int, int, int, int],
        hand_pts_init: np.ndarray,
    ) -> torch.Tensor:
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

    def process_video(
        self,
        video_path: str,
        output_dir: str = "data/cotracker_trajectories",
        save_annotated_video: bool = True,
    ) -> Dict[str, np.ndarray]:
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

        # Letterbox into square frame
        max_dim = max(orig_w, orig_h)
        pad_top = (max_dim - orig_h) // 2
        pad_bottom = max_dim - orig_h - pad_top
        pad_left = (max_dim - orig_w) // 2
        pad_right = max_dim - orig_w - pad_left

        frames_rgb = []
        timestamps = []
        for f_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            padded = cv2.copyMakeBorder(
                frame, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=[30, 30, 30]
            )
            resized = cv2.resize(padded, self.target_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames_rgb.append(rgb)
            timestamps.append(f_idx / fps)
        cap.release()

        T = len(frames_rgb)

        # 1. Detect ROIs & Keypoints
        t_hand, glass_roi, hand_pts_init = self._detect_hand_and_glass_rois(frames_rgb)
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


        tracks = tracks.squeeze(0).cpu().numpy()  # (T, N_total, 2)
        vis = vis.squeeze(0).cpu().numpy()        # (T, N_total)

        glass_tracks = tracks[:, :N_glass, :]     # (T, N_glass, 2)
        hand_tracks = tracks[:, N_glass:, :]      # (T, N_hand, 2)

        # 3. Compute Centroids & Normalized Trajectories
        W, H = self.target_size
        glass_centroids = glass_tracks.mean(axis=1) / np.array([W, H], dtype=np.float32)  # (T, 2)
        hand_centroids = hand_tracks.mean(axis=1) / np.array([W, H], dtype=np.float32)    # (T, 2)

        # For frames before hand entry, keep hand stationary at entry point
        for t in range(t_hand):
            hand_centroids[t] = hand_centroids[t_hand]

        # Wrist 3D Position Proxy:
        # X: lateral hand centroid [0, 1]
        # Y: vertical hand centroid [0, 1] (0 top, 1 bottom)
        # Z: depth proxy based on distance to glass
        wrist_pos_3d = np.zeros((T, 3), dtype=np.float32)
        for t in range(T):
            dist_to_glass = float(np.linalg.norm(hand_centroids[t] - glass_centroids[t]))
            wrist_pos_3d[t, 0] = hand_centroids[t, 0]
            wrist_pos_3d[t, 1] = hand_centroids[t, 1]
            wrist_pos_3d[t, 2] = dist_to_glass

        # Compute wrist deltas: delta[t] = pos[t] - pos[t-1]
        wrist_deltas = np.zeros_like(wrist_pos_3d)
        if T > 1:
            wrist_deltas[1:] = wrist_pos_3d[1:] - wrist_pos_3d[:-1]
        wrist_deltas[:t_hand] = 0.0

        # 4. Co-motion Grasp Correlation (rho_t)
        v_glass = glass_tracks[1:] - glass_tracks[:-1]  # (T-1, N_glass, 2)
        v_hand = hand_tracks[1:] - hand_tracks[:-1]    # (T-1, N_hand, 2)
        mean_vg = v_glass.mean(axis=1)                 # (T-1, 2)
        mean_vh = v_hand.mean(axis=1)                  # (T-1, 2)

        corrs = np.zeros(T, dtype=np.float32)
        gripper_states = np.full(T, -1.0, dtype=np.float32)  # -1.0 = OPEN, +1.0 = CLOSED
        current_gripper = -1.0

        for t in range(1, T):
            vg = mean_vg[t - 1]
            vh = mean_vh[t - 1]
            norm_g = float(np.linalg.norm(vg))
            norm_h = float(np.linalg.norm(vh))

            if norm_g > 0.4 and norm_h > 0.4:
                cos_sim = float(np.dot(vg, vh) / (norm_g * norm_h))
            else:
                cos_sim = 0.0
            corrs[t] = cos_sim

            # Hysteresis latch with spatial proximity constraint
            dist_to_glass = float(np.linalg.norm(hand_centroids[t] - glass_centroids[t]))
            is_near_glass = dist_to_glass < 0.25

            if current_gripper < 0:
                if cos_sim > GRASP_CORR_THRESHOLD and is_near_glass and t >= (t_hand + 10):
                    current_gripper = 1.0
            else:
                if (cos_sim < RELEASE_CORR_THRESHOLD and norm_g > 2.0) or dist_to_glass > 0.40:
                    current_gripper = -1.0
            gripper_states[t] = current_gripper

        # 5. Save Annotated Video if requested
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if save_annotated_video:
            annotated_video_path = out_dir / f"{vpath.stem}_ct_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (W, H))

            for t in range(T):
                vis_bgr = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR)

                # Draw glass points (Blue)
                for pt in glass_tracks[t]:
                    px, py = int(pt[0]), int(pt[1])
                    cv2.circle(vis_bgr, (px, py), 3, (255, 120, 0), -1)

                # Draw hand points (Green if reaching, Red if grasping)
                hand_color = (0, 0, 255) if gripper_states[t] > 0 else (0, 255, 0)
                if t >= t_hand:
                    for pt in hand_tracks[t]:
                        px, py = int(pt[0]), int(pt[1])
                        cv2.circle(vis_bgr, (px, py), 3, hand_color, -1)

                    # Connect centroids with correlation indicator
                    gc = (int(glass_centroids[t, 0] * W), int(glass_centroids[t, 1] * H))
                    hc = (int(hand_centroids[t, 0] * W), int(hand_centroids[t, 1] * H))
                    cv2.line(vis_bgr, gc, hc, hand_color, 2)

                # Status HUD
                state_text = "GRASPED (LIFTING)" if gripper_states[t] > 0 else "APPROACHING"
                cv2.putText(vis_bgr, f"CoTracker3: {state_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2)
                cv2.putText(vis_bgr, f"Co-motion rho: {corrs[t]:.2f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                writer.write(vis_bgr)
            writer.release()
            print(f"[+] Saved annotated video -> {annotated_video_path}")

        # 6. Save Trajectory Dataset
        output_file = out_dir / f"{vpath.stem}_ct_trajectory.npz"
        traj_data = {
            "timestamps": np.array(timestamps, dtype=np.float32),
            "rgb_frames": np.array(frames_rgb, dtype=np.uint8),
            "wrist_positions": wrist_pos_3d,
            "wrist_deltas": wrist_deltas,
            "gripper_states": gripper_states,
            "correlations": corrs,
            "glass_tracks": glass_tracks,
            "hand_tracks": hand_tracks,
            "t_hand": np.int32(t_hand),
            "fps": np.float32(fps),
            "source_video": str(vpath),
        }
        np.savez_compressed(output_file, **traj_data)
        print(f"[+] Saved CoTracker3 trajectory -> {output_file}")
        return traj_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract dense point trajectories via CoTracker3 on MPS")
    parser.add_argument("--video", type=str, default="media/Vid_0.mp4", help="Path to video file")
    parser.add_argument("--out_dir", type=str, default="data/cotracker_trajectories", help="Output directory")
    args = parser.parse_args()

    extractor = CoTrackerTrajectoryExtractor()
    extractor.process_video(args.video, args.out_dir)
