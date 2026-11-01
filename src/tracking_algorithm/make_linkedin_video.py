#!/usr/bin/env python3
"""
Generate a complete LinkedIn highlight reel video for the Pollex pipeline.
Stitches all iterations with title cards, text overlays, and a 3-panel finale.
"""

import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT_W, OUT_H = 768, 432
TARGET_FPS = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def make_title_card(text_lines, duration_s=3.0, bg_color=(20, 20, 20)):
    """Create a title card with centered text lines."""
    frames = []
    n_frames = int(duration_s * TARGET_FPS)
    for _ in range(n_frames):
        frame = np.full((OUT_H, OUT_W, 3), bg_color, dtype=np.uint8)
        total_height = len(text_lines) * 50
        y_start = (OUT_H - total_height) // 2 + 30
        for i, (text, scale, color, thickness) in enumerate(text_lines):
            size = cv2.getTextSize(text, FONT_BOLD, scale, thickness)[0]
            x = (OUT_W - size[0]) // 2
            y = y_start + i * 50
            cv2.putText(frame, text, (x, y), FONT_BOLD, scale, color, thickness, cv2.LINE_AA)
        frames.append(frame)
    return frames


def read_video_frames(path, speed=1.0):
    """Read all frames from a video, resampled at speed factor."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    if not all_frames:
        return []

    # Resample for speed
    src_duration = len(all_frames) / src_fps
    out_duration = src_duration / speed
    n_out = int(out_duration * TARGET_FPS)
    frames = []
    for i in range(n_out):
        t = i / TARGET_FPS * speed
        src_idx = min(int(t * src_fps), len(all_frames) - 1)
        frames.append(all_frames[src_idx])
    return frames


def fit_to_canvas(frame, canvas_w=OUT_W, canvas_h=OUT_H):
    """Resize frame to fit canvas while maintaining aspect ratio, with black bars."""
    h, w = frame.shape[:2]
    scale = min(canvas_w / w, canvas_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    y_off = (canvas_h - new_h) // 2
    x_off = (canvas_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def add_overlay(frame, top_text, bottom_text=None, top_color=(0, 255, 255),
                bottom_color=(200, 200, 200)):
    """Add text overlay with semi-transparent background."""
    out = frame.copy()
    # Top bar
    cv2.rectangle(out, (0, 0), (OUT_W, 38), (0, 0, 0), -1)
    out[0:38] = cv2.addWeighted(out[0:38], 0.7, frame[0:38], 0.3, 0)
    size = cv2.getTextSize(top_text, FONT, 0.55, 1)[0]
    x = (OUT_W - size[0]) // 2
    cv2.putText(out, top_text, (x, 26), FONT, 0.55, top_color, 1, cv2.LINE_AA)

    # Bottom bar
    if bottom_text:
        y_start = OUT_H - 34
        cv2.rectangle(out, (0, y_start), (OUT_W, OUT_H), (0, 0, 0), -1)
        out[y_start:OUT_H] = cv2.addWeighted(out[y_start:OUT_H], 0.7, frame[y_start:OUT_H], 0.3, 0)
        size2 = cv2.getTextSize(bottom_text, FONT, 0.45, 1)[0]
        x2 = (OUT_W - size2[0]) // 2
        cv2.putText(out, bottom_text, (x2, OUT_H - 10), FONT, 0.45, bottom_color, 1, cv2.LINE_AA)

    return out


def make_tri_panel(ct_frames, expert_frames, smolvla_frames):
    """Create 3-panel comparison frames."""
    PANEL = 256
    HEADER = 32
    canvas_h = PANEL + HEADER

    labels = [
        ("1. iPhone + CoTracker3", (255, 255, 0)),
        ("2. Expert Retargeted", (0, 255, 0)),
        ("3. SmolVLA Policy (98/100)", (0, 255, 255)),
    ]

    max_len = max(len(ct_frames), len(expert_frames), len(smolvla_frames))
    # Resample all to same length
    all_sources = [ct_frames, expert_frames, smolvla_frames]
    out_frames = []

    for i in range(max_len):
        canvas = np.zeros((canvas_h, PANEL * 3, 3), dtype=np.uint8)
        for p, (src, (label, color)) in enumerate(zip(all_sources, labels)):
            idx = min(int(i * len(src) / max_len), len(src) - 1)
            panel = cv2.resize(src[idx], (PANEL, PANEL), interpolation=cv2.INTER_AREA)
            x_off = p * PANEL
            canvas[0:HEADER, x_off:x_off + PANEL] = (30, 30, 30)
            ts = cv2.getTextSize(label, FONT, 0.38, 1)[0]
            tx = x_off + (PANEL - ts[0]) // 2
            cv2.putText(canvas, label, (tx, HEADER - 8), FONT, 0.38, color, 1, cv2.LINE_AA)
            canvas[HEADER:HEADER + PANEL, x_off:x_off + PANEL] = panel
        out_frames.append(canvas)

    return out_frames


def main():
    print("[*] Building Pollex LinkedIn highlight reel...")

    all_output_frames = []

    # === TITLE CARD ===
    print("  [1/12] Title card")
    all_output_frames.extend(make_title_card([
        ("POLLEX", 1.2, (0, 200, 255), 2),
        ("iPhone Video to Robot Policy", 0.65, (220, 220, 220), 1),
        ("SmolVLA  |  CoTracker3  |  RoboSuite", 0.5, (150, 150, 150), 1),
    ], duration_s=3.0))

    # === SECTION 1: INPUT VIDEO ===
    print("  [2/12] iPhone input video")
    all_output_frames.extend(make_title_card([
        ("Part 1: Real-World Input", 0.8, (100, 255, 100), 2),
        ("11-second iPhone video of human pick-and-place", 0.5, (180, 180, 180), 1),
    ], duration_s=2.0))

    iphone_frames = read_video_frames(BASE / "datasets/egocentric_video/cup_lifting_01.mp4", speed=2.5)
    for f in iphone_frames:
        all_output_frames.append(add_overlay(
            fit_to_canvas(f),
            "Real-World iPhone Video (1080p, 30fps)",
            "Egocentric view: human picks up a glass from a table"
        ))

    # === SECTION 2: COTRACKER ===
    print("  [3/12] CoTracker3 tracking")
    ct_frames_raw = read_video_frames(
        BASE / "datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4", speed=2.5
    )
    for f in ct_frames_raw:
        all_output_frames.append(add_overlay(
            fit_to_canvas(f),
            "CoTracker3: 37 Dense Points (21 hand + 16 glass)",
            "Co-motion correlation rho > 0.55 detects grasp/release"
        ))

    # === SECTION 3: RETARGETING ITERATIONS ===
    print("  [4/12] Retargeting iterations title")
    all_output_frames.extend(make_title_card([
        ("Part 2: Retargeting Iterations", 0.8, (100, 255, 100), 2),
        ("7 iterations to solve video-to-robot transfer", 0.5, (180, 180, 180), 1),
    ], duration_s=2.0))

    iterations = [
        ("iteration_01_raw/cup_lifting_01_iteration_1.mp4",
         "Iter 1: Raw CoTracker3 Baseline",
         "0 deg camera | 35cm motion but 14cm miss | NO CONTACT", (0, 100, 255)),
        ("iteration_02_camera_roi/cup_lifting_01_iteration_2.mp4",
         "Iter 2: 60 deg Camera Calibration",
         "Min distance: 13.97cm -> 3.80cm (73% improvement) | NO GRASP", (0, 100, 255)),
        ("iteration_03_physics/cup_lifting_01_iteration_3.mp4",
         "Iter 3: Grasp Geometry Diagnostic",
         "4.35cm pad-to-axis offset identified | NO CONTACT", (0, 100, 255)),
        ("iteration_04_gripper/cup_lifting_01_iteration_4.mp4",
         "Iter 4: EEF-to-Pad Offset (diluted)",
         "FIRST CONTACT (frame 201, retract phase) | NO GRASP", (0, 200, 255)),
        ("iteration_05_goal_directed/cup_lifting_01_iteration_5.mp4",
         "Iter 5: Goal-Directed Pad Targeting",
         "FIRST GRASP + LIFT: +8.22cm!", (0, 255, 0)),
        ("iteration_06_wrist_squared/cup_lifting_01_iteration_6.mp4",
         "Iter 6: Wrist Squaring (q7=0)",
         "Lift: +8.27cm | Marginal improvement, side-grip persists", (0, 200, 255)),
        ("iteration_07_blended_feedback/cup_lifting_01_iteration_07.mp4",
         "Iter 7: Object-Relative X/Z Feedback",
         "PICK & PLACE: +8.17cm [SUCCESS] -- Expert demonstration", (0, 255, 0)),
    ]

    for i, (path, title, desc, color) in enumerate(iterations):
        print(f"  [{5+i}/12] {title}")
        frames = read_video_frames(BASE / "datasets/cotracker_iterations" / path, speed=2.0)
        for f in frames:
            all_output_frames.append(add_overlay(
                fit_to_canvas(f), title, desc, top_color=color
            ))

    # === SECTION 4: SmolVLA ===
    print("  [11/12] SmolVLA section")
    all_output_frames.extend(make_title_card([
        ("Part 3: SmolVLA Neural Policy", 0.8, (100, 255, 100), 2),
        ("450M parameter Vision-Language-Action model", 0.5, (180, 180, 180), 1),
        ("learns from a single demonstration", 0.5, (180, 180, 180), 1),
    ], duration_s=2.5))

    # SmolVLA baseline
    baseline_frames = read_video_frames(
        BASE / "datasets/cotracker_iterations/iteration_10_smolvla_expert/cup_lifting_01_iteration_10.mp4",
        speed=2.0
    )
    for f in baseline_frames:
        all_output_frames.append(add_overlay(
            fit_to_canvas(f),
            "SmolVLA Baseline (H=50, K=5, no gain)",
            "Contact + Grasp + Hold... but lift only +1.17cm (needs 3cm)",
            top_color=(0, 150, 255)
        ))

    # SmolVLA calibrated
    calib_frames = read_video_frames(
        BASE / "media/smolvla_gain_calibrated_best.mp4", speed=1.5
    )
    for f in calib_frames:
        all_output_frames.append(add_overlay(
            fit_to_canvas(f),
            "SmolVLA Calibrated: gamma_z=1.25, gamma_lift=2.0 + Release Guard",
            "FULL-TASK SUCCESS: +4.31cm lift, Score 98.07/100",
            top_color=(0, 255, 0), bottom_color=(0, 255, 100)
        ))

    # === SECTION 5: TRI-PANEL FINALE ===
    print("  [12/12] 3-panel finale")
    all_output_frames.extend(make_title_card([
        ("End-to-End Pipeline Comparison", 0.8, (0, 200, 255), 2),
        ("iPhone Video  |  Expert Retarget  |  SmolVLA Policy", 0.55, (220, 220, 220), 1),
    ], duration_s=2.0))

    ct_full = read_video_frames(
        BASE / "datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4", speed=1.0
    )
    expert_full = read_video_frames(
        BASE / "datasets/cotracker_iterations/iteration_11_full_mujoco/cup_lifting_01_iteration_11.mp4",
        speed=1.0
    )
    smolvla_full = read_video_frames(BASE / "media/smolvla_gain_calibrated_best.mp4", speed=1.0)

    tri_frames = make_tri_panel(ct_full, expert_full, smolvla_full)
    for f in tri_frames:
        all_output_frames.append(fit_to_canvas(f))

    # === END CARD ===
    all_output_frames.extend(make_title_card([
        ("FULL-TASK SUCCESS", 1.0, (0, 255, 0), 2),
        ("Score: 98.07 / 100", 0.7, (0, 200, 255), 2),
        ("1 iPhone video -> 1 demonstration -> autonomous policy", 0.45, (180, 180, 180), 1),
        ("github.com/pollex", 0.45, (120, 120, 120), 1),
    ], duration_s=4.0))

    # === WRITE OUTPUT ===
    out_path = str(BASE / "media/pollex_full_pipeline_linkedin.mp4")
    print(f"\n[*] Writing {len(all_output_frames)} frames to {out_path}")
    print(f"    Duration: {len(all_output_frames)/TARGET_FPS:.1f}s at {TARGET_FPS}fps")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, TARGET_FPS, (OUT_W, OUT_H))
    if not writer.isOpened():
        print("[-] Failed to open VideoWriter")
        return

    for frame in all_output_frames:
        if frame.shape[:2] != (OUT_H, OUT_W):
            frame = cv2.resize(frame, (OUT_W, OUT_H))
        writer.write(frame)

    writer.release()
    total_s = len(all_output_frames) / TARGET_FPS
    print(f"[+] Done! {out_path}")
    print(f"    {len(all_output_frames)} frames, {total_s:.1f}s, {OUT_W}x{OUT_H} @ {TARGET_FPS}fps")
    print(f"    LinkedIn max: 10 min. This video: {int(total_s//60)}m {int(total_s%60)}s")


if __name__ == "__main__":
    main()
