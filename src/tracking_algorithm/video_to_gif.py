"""Convert MP4 videos to GIF using OpenCV + Pillow."""

import argparse
import cv2
from pathlib import Path
from PIL import Image


def video_to_gif(video_path: str, output_path: str = None,
                 fps: int = 10, scale: float = 1.0, max_width: int = 560):
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = video_path.with_suffix(".gif")
    else:
        output_path = Path(output_path)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = max(1, int(src_fps / fps))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            w, h = img.size
            new_w = min(int(w * scale), max_width)
            new_h = int(h * (new_w / w))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            frames.append(img)
        idx += 1
    cap.release()

    if not frames:
        print(f"[-] No frames extracted from {video_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = int(1000 / fps)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    print(f"[+] Saved GIF: {output_path} ({len(frames)} frames, {new_w}x{new_h})")
    return str(output_path)


def batch_convert(input_dir: str, output_dir: str = None, **kwargs):
    input_path = Path(input_dir)
    videos = sorted(input_path.glob("*.mp4"))
    if not videos:
        print(f"[-] No MP4 files found in {input_dir}")
        return

    print(f"[*] Converting {len(videos)} videos to GIF")
    for video in videos:
        out = None
        if output_dir:
            out = str(Path(output_dir) / video.with_suffix(".gif").name)
        video_to_gif(str(video), out, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MP4 videos to GIF")
    parser.add_argument("--video", type=str, default=None, help="Path to a single video")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory of videos to batch convert")
    parser.add_argument("--output", type=str, default=None, help="Output path (file or directory)")
    parser.add_argument("--fps", type=int, default=10, help="GIF frame rate (default: 10)")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor (default: 1.0)")
    parser.add_argument("--max_width", type=int, default=560, help="Max GIF width in pixels (default: 560)")
    args = parser.parse_args()

    if args.video:
        video_to_gif(args.video, args.output, fps=args.fps,
                     scale=args.scale, max_width=args.max_width)
    elif args.input_dir:
        batch_convert(args.input_dir, args.output, fps=args.fps,
                      scale=args.scale, max_width=args.max_width)
    else:
        print("Provide --video or --input_dir")

"""
python3 src/tracking_algorithm/video_to_gif.py --video datasets/cotracker_trajectories/cup_lifting_01_ct_annotated.mp4 --output media/cup_lifting_01_ct_annotated.gif --fps 10 --scale 0.5 --max_width 400

"""