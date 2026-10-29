"""
=============================================================================
HAND LANDMARK TRACKER & ACTION EXTRACTOR
=============================================================================

PURPOSE
-------
This script reads a phone video frame-by-frame and uses MediaPipe
HandLandmarker to detect one hand.

For every frame it produces:

    1. RGB image resized to target_size
    2. 21 MediaPipe hand landmarks
    3. Wrist position
    4. Thumb-index pinch distance
    5. OPEN / CLOSED gripper state
    6. Wrist movement between frames
    7. Annotated visualization video

At the end everything is stored in an "NPZ file".

IMPORTANT COORDINATE NOTE
-------------------------
This script uses:

    detection_result.hand_landmarks

These landmarks are NORMALIZED IMAGE LANDMARKS.

Typical meaning:

    x = horizontal image coordinate
    y = vertical image coordinate
    z = MediaPipe relative depth

Therefore these are NOT calibrated robot XYZ coordinates in meters.

The output currently called:

    wrist_positions

should therefore be understood as:

    normalized / MediaPipe image-space wrist positions

and "NOT":

    Panda XYZ in meters

For our newer human-to-Panda pipeline, MediaPipe world landmarks and
table/robot coordinate calibration are preferable for actual trajectory
retargeting.

=============================================================================
"""

import cv2
import argparse # for passing argument during call.
import urllib.request # Used to download the MediaPipe model automatically.
import numpy as np # for operation such as Numerical arrays, distances, subtraction, etc.
from pathlib import Path # filesystem path handling.
from typing import Dict, Tuple, Optional

import mediapipe as mp # Main MediaPipe package.
from mediapipe.tasks import python # MediaPipe Tasks Python API.
from mediapipe.tasks.python import vision # MediaPipe computer-vision Tasks API.

# URL of the pretrained MediaPipe Hand Landmarker model. The script downloads this if the model doesn't already exist locally.
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_MODEL_PATH = "models/hand_landmarker.task"

# MODEL DOWNLOAD. If not available
def ensure_model_exists(model_path: str = DEFAULT_MODEL_PATH):
    # Convert string path into pathlib Path.
    p = Path(model_path)
    if not p.exists():
        # Create parent folders if necessary.
        p.parent.mkdir(parents=True, exist_ok=True)
        print(f"[*] Downloading MediaPipe HandLandmarker model to {model_path}...")
        urllib.request.urlretrieve(MODEL_URL, str(p))
        print("[+] Download successfull.")

# =============================================================================
# MAIN EXTRACTOR CLASS
# =============================================================================

class HandTrajectoryExtractor:
    """
    Extract hand information from a video.
    Current extracted signals:
        landmarks
        wrist position
        wrist displacement
        pinch distance
        gripper OPEN/CLOSED

    IMPORTANT: Current landmarks come from MediaPipe normalized image landmarks, not metric robot coordinates.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, target_size: Tuple[int, int] = (512, 512)): # 256, 256 (baseline)
        ensure_model_exists(model_path)
        self.model_path = model_path
        self.target_size = target_size

    # =========================================================================
    # CREATE MEDIAPIPE HAND DETECTOR
    # =========================================================================

    def _create_landmarker(self):
        # Tell MediaPipe where the trained model is.
        base_options = python.BaseOptions(model_asset_path=self.model_path)
       
        """
        vision.RunningMode.VIDEO -> This allows MediaPipe to use tracking between frames.
        hand detection confidence : 0.4   Lower: more detections and potentially more false positives
        tracking_confidence=0.4 , Lower: tracking may survive difficult frames better, but can become less reliable
        """
        options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_hands=1,min_hand_detection_confidence=0.4,min_tracking_confidence=0.4)
        return vision.HandLandmarker.create_from_options(options) # Build and return the actual detector.

    # PROCESS ONE VIDEO
    def process_video(self, video_path: str, output_dir: str,save_annotated_video: bool = True) -> Optional[Dict[str, np.ndarray]]:
        """
        Process one complete input video. Returns a dictionary containing arrays such as:
            timestamps
            RGB frames
            wrist positions
            wrist deltas
            pinch distances
            gripper states
            all 21 landmarks

        It also writes an NPZ file. If save_annotated_video=True, it additionally writes an MP4 visualization.
        """

        video_path = Path(video_path) # Convert video filename to Path.
        if not video_path.exists():
            raise FileNotFoundError(f"Video file is not found: {video_path}")

        # Open video through OpenCV.
        cap = cv2.VideoCapture(str(video_path))
        # Check that OpenCV successfully opened it.
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        # Read source-video FPS. If OpenCV gives 0 / False-ish value, use 30 FPS as fallback.
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Original frame width.
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Original frame height.
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Number of video frames reported by OpenCV.

        print(f"[*] Processing {video_path.name}: {total_frames} frames @ {fps:.1f} FPS ({width}x{height})")
        
        # Create MediaPipe detector.
        landmarker = self._create_landmarker()

        output_path = Path(output_dir) # Prepare output directory.
        output_path.mkdir(parents=True, exist_ok=True)

        # OPTIONAL ANNOTATED the VIDEO file name
        annotated_writer = None
        if save_annotated_video:
            annotated_video_path = output_path / f"{video_path.stem}_annotated.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v") # MPEG-4 video codec.
            # Create video writer.
            annotated_writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, self.target_size)

        # output storage

        frames_rgb = [] # Processed 256x256 / 512 x 512 RGB image for every frame
        wrist_positions = [] # Wrist position for every frame: [x, y, z]
        pinch_distances = [] # Thumb-to-index/DIP/PIP distance used as pinch signal.
        gripper_states = [] # -1 = OPEN,  +1 = CLOSED
        raw_landmarks_all = [] # Full 21 x 3 MediaPipe landmark array per frame.
        timestamps = [] # Video timestamp of each frame.

        # =====================================================================
        # TEMPORAL STATE
        # =====================================================================

        last_valid_wrist = None # Last successfully measured wrist. Used when detection is temporarily lost.
        last_valid_dist = 0.35 # Starts at 0.35, which represents an initially separated, thumb/index assumption.
        last_valid_idx_tip = None # Previous valid index fingertip. Used to identify abrupt jumps.
        current_gripper_state = -1.0  # start OPEN
        detected_indices = [] # Store indices where MediaPipe really detected a hand.

        # GRIPPER / TRACKING PARAMETERS
        close_threshold = 0.19 # If pinch distance becomes smaller than 0.19: close
        open_threshold = 0.26 # Once closed, hand must open beyond 0.26: open
        # Maximum acceptable frame-to-frame index-tip displacement, before that tip is considered suspicious.
        jump_threshold = 0.12  # in normalized units, 
       
        # Now padding 
        max_dim = max(width, height)
        pad_top = (max_dim - height) // 2 # Compute padding required to turn image into a square, without cropping its field of view.
        pad_bottom = max_dim - height - pad_top
        pad_left = (max_dim - width) // 2
        pad_right = max_dim - width - pad_left

        # PROCESS VIDEO FRAME BY FRAME
        for frame_idx in range(total_frames):
            ret, frame = cap.read() # Read one frame.
            if not ret: # End if OpenCV has no more frames.
                break

            # Add borders instead of directly stretching/cropping the original aspect ratio.
            # Padding color is dark gray:[30,30,30]
            padded = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right,cv2.BORDER_CONSTANT, value=[30, 30, 30])
            resized = cv2.resize(padded, self.target_size)
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)# OpenCV uses BGR. While MediaPipe expects RGB/SRGB.

            # TIMESTAMP
            t = frame_idx / fps
            timestamps.append(t)

            # Convert to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_idx * 1000 / fps)

            # Run MediaPipe HandLandmarker.
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            wrist_pos = None
            pinch_dist = last_valid_dist # If hand detection fails, this effectively holds pinch state.
            lms_21 = np.zeros((21, 3), dtype=np.float32) # Allocate 21 MediaPipe landmarks:
            used_landmark_name = "Index Fingertip"
            active_target_pt = None # Point that thumb is currently compared against.

            # HAND DETECTED

            if detection_result.hand_landmarks:
                detected_indices.append(frame_idx) # Record successful detection.
                lms = detection_result.hand_landmarks[0] # Get first detected hand.
                for idx, lm in enumerate(lms): # Convert MediaPipe landmark objects into NumPy array.
                    lms_21[idx] = [lm.x, lm.y, lm.z]

                wrist_pos = lms_21[0].copy() # Landmark 0 = wrist.
                last_valid_wrist = wrist_pos.copy() # Store most recent valid wrist.

                # Landmarks: 4=Thumb Tip, 8=Index Tip, 7=Index DIP, 6=Index PIP, 5=Index MCP
                thumb_tip = lms_21[4]
                idx_tip = lms_21[8]
                idx_dip = lms_21[7]
                idx_pip = lms_21[6]

                # Check if index tip abruptly moved or went near image edge, # SUSPICIOUS INDEX-TIP TRACKING
                is_idx_abrupt = False
                if last_valid_idx_tip is not None:
                    # Distance between current and previous index-tip position.
                    step_displacement = np.linalg.norm(idx_tip - last_valid_idx_tip)

                    # If fingertip apparently jumps more than 0.12 normalized units in one frame, reject it.
                    if step_displacement > jump_threshold:
                        is_idx_abrupt = True
                

                # =============================================================
                # EDGE CHECK
                # =============================================================

                # MediaPipe x/y are approximately normalized to [0,1]. If fingertip is within 4% of an image edge,mark it suspicious.
                # Also flag landmark 8 
                if idx_tip[1] < 0.04 or idx_tip[1] > 0.96 or idx_tip[0] < 0.04 or idx_tip[0] > 0.96:
                    is_idx_abrupt = True

                # =============================================================
                # GOOD INDEX TIP (Pinch target)
                # =============================================================

                if not is_idx_abrupt:

                    # Update previous trustworthy tip.
                    last_valid_idx_tip = idx_tip.copy()
                    # Distance: thumb tip -> index tip
                    # Tip is trustworthy: use minimum distance to tip or DIP
                    d_tip = float(np.linalg.norm(thumb_tip - idx_tip))
                    d_dip = float(np.linalg.norm(thumb_tip - idx_dip))

                    # Whichever is closer is used as pinch target.
                    if d_tip <= d_dip:
                        pinch_dist = d_tip
                        active_target_pt = idx_tip
                        used_landmark_name = "Index Fingertip"
                    else:
                        pinch_dist = d_dip
                        active_target_pt = idx_dip
                        used_landmark_name = "Index DIP Joint"

                # SUSPICIOUS INDEX TIP
                else:
                    # If index fingertip looks unreliable,don't use landmark 8.
                    # Instead fallback to DIP (L7) or PIP (L6)
                    d_dip = float(np.linalg.norm(thumb_tip - idx_dip))
                    d_pip = float(np.linalg.norm(thumb_tip - idx_pip))
                    if d_dip <= d_pip:
                        pinch_dist = d_dip
                        active_target_pt = idx_dip
                        used_landmark_name = "Index DIP Fallback"
                    else:
                        pinch_dist = d_pip
                        active_target_pt = idx_pip
                        used_landmark_name = "Index PIP Fallback"

                # Save valid pinch result for possible future temporarily missing-hand frames.
                last_valid_dist = pinch_dist

            elif last_valid_wrist is not None:
                # Maintain last known position. This makes missing frames look like the wrist stayed still.
                wrist_pos = last_valid_wrist.copy()
            
            # HAND HAS NEVER BEEN DETECTED
            else:
                # Hand not yet in view
                wrist_pos = np.array([0.5, 0.9, 0.0], dtype=np.float32)

            # GRIPPER HYSTERESIS trigger for gripper state:
            # Prevents chatter / flickering around threshold boundary
            if current_gripper_state < 0:  # currently OPEN
                if pinch_dist < close_threshold: # Only close once pinch becomes sufficiently small.
                    current_gripper_state = 1.0  # CLOSED
            
            # Otherwise currently CLOSED.
            else:  # currently CLOSED
                if pinch_dist > open_threshold:
                    current_gripper_state = -1.0  # OPEN

            # Final gripper command for this frame.
            gripper_cmd = current_gripper_state

            # =================================================================
            # ANNOTATED VIDEO
            # =================================================================
            #
            # Visualization convention:
            #
            #   Normal hand landmarks -> GREEN small circles
            #   Thumb tip             -> MAGENTA large circle
            #   Active pinch target   -> YELLOW large circle
            #
            #   Gripper OPEN:
            #       pinch line        -> CYAN
            #       status text       -> CYAN
            #       indicator         -> hollow CYAN circle
            #
            #   Gripper CLOSED:
            #       pinch line        -> RED
            #       status text       -> RED
            #       indicator         -> filled RED circle
            #
            # OpenCV uses BGR instead of RGB:
            #
            #   GREEN   = (0, 255, 0)
            #   RED     = (0, 0, 255)
            #   CYAN    = (255, 255, 0)
            #   YELLOW  = (0, 255, 255)
            #   MAGENTA = (255, 0, 255)
            #   WHITE   = (255, 255, 255)
            #
            # =================================================================

            if save_annotated_video and annotated_writer:
                # Define visualization colors
                LANDMARK_COLOR = (0,255,0)  # Green
                THUMB_COLOR = (255,0,255)  # Magenta
                TARGET_COLOR = (0,255,255)  # Yellow
                OPEN_COLOR = (255,255,0)  # Cyan
                CLOSED_COLOR = (0,0,255)  # Red
                TEXT_COLOR = (255,255,255)  # White

                # MediaPipe processing uses RGB. OpenCV: drawingVideoWriter use BGR.
                visual_bgr = cv2.cvtColor(rgb_frame,cv2.COLOR_RGB2BGR) # Convert RGB -> BGR

                # Only draw hand features if a hand was detected in this actual frame.
                if detection_result.hand_landmarks:
                    # ---------------------------------------------------------
                    # Draw all 21 MediaPipe landmarks
                    # ---------------------------------------------------------
                    # MediaPipe x/y coordinates are normalized:
                    #     x = 0 ... 1
                    #     y = 0 ... 1
                    # To draw them on the image we convert them to pixels:
                    #     pixel_x = x * image_width
                    #     pixel_y = y * image_height
                    for pt in lms_21:
                        px = int(pt[0]* self.target_size[0])
                        py = int(pt[1]* self.target_size[1])
                        # Small green filled circle. radius = 3, thickness = -1 means filled.
                        cv2.circle(visual_bgr,(px,py,),3,LANDMARK_COLOR,-1)

                    # ---------------------------------------------------------
                    # Calculate thumb-tip pixel location
                    # ---------------------------------------------------------
                    # MediaPipe landmark: L4 = thumb fingertip
                    p_thumb = (int(lms_21[4, 0]* self.target_size[0]),int(lms_21[4, 1]* self.target_size[1]),)



                    # Highlight thumb tip.
                    # Use a larger MAGENTA circle so it is easy to distinguish from the other 20 landmarks.
                    cv2.circle(visual_bgr,p_thumb,6,THUMB_COLOR,-1,)


                    # Draw the currently selected pinch target active_target_pt can be:
                    #     L8 -> Index fingertip
                    #     L7 -> Index DIP
                    #     L6 -> Index PIP fallback
                    # depending on tracking quality.
                    if active_target_pt is not None:
                        p_target = (int(active_target_pt[0]* self.target_size[0]), int(active_target_pt[1]* self.target_size[1]))

                        # Highlight active target
                        # Yellow distinguishes the active index landmark from:
                        #     normal landmarks -> green
                        #     thumb           -> magenta
                        cv2.circle(visual_bgr,p_target,6,TARGET_COLOR,-1)



                        # Gripper-dependent pinch-line color: CLOSED:use RED, OPEN: use CYAN
                        line_color = (
                            CLOSED_COLOR
                            if gripper_cmd > 0
                            else OPEN_COLOR)

                        # Draw thumb -> active target line
                        cv2.line(visual_bgr,p_thumb,p_target,line_color,2)


                # -------------------------------------------------------------
                # Human-readable gripper status
                # -------------------------------------------------------------
                status_text = (
                    "GRIPPER: CLOSED"
                    if gripper_cmd > 0
                    else "GRIPPER: OPEN")

                # Match text color to state.
                status_color = (
                    CLOSED_COLOR
                    if gripper_cmd > 0
                    else OPEN_COLOR
                )


                indicator_center = (self.target_size[0] - 18, 18)
                if gripper_cmd > 0:
                    cv2.circle(visual_bgr, indicator_center, 8, CLOSED_COLOR, -1)
                else:
                    cv2.circle(visual_bgr, indicator_center, 8, OPEN_COLOR, 2)


                cv2.putText(visual_bgr, status_text, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 2)

                pinch_text = f"Pinch: {pinch_dist:.3f} [{used_landmark_name}]"
                cv2.putText(visual_bgr, pinch_text, (10, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_COLOR, 1)

                annotated_writer.write(visual_bgr)

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

        print(f"[+] Extracted {len(frames_rgb)} steps "
              f"(Hand detected in {len(detected_indices)} frames) -> {save_file}")

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
    parser.add_argument("--output_dir", type=str, default="datasets/processed_trajectories", help="Output directory")
    args = parser.parse_args()

    if args.video:
        extractor = HandTrajectoryExtractor()
        extractor.process_video(args.video, args.output_dir)
    else:
        process_all_raw_videos(args.raw_dir, args.output_dir)

# .venv 
# cd Pollex
# python3 src/tracking_video_processing/hand_tracker.py --video datasets/egocentric_video/cup_lifting_01.mp4
