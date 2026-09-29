import os
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "renders" / "demo_data"

def extract_stereo_pair(video_path: str, out_dir: str = DEFAULT_OUT_DIR, frame_idx: int = 0):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = min(frame_idx, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError(f"Can't read frame {frame_idx} from {video_path}")

    height, total_width, _ = frame.shape
    one_view_width = total_width // 3
    left_img = frame[:, :one_view_width]
    right_img = frame[:, one_view_width : 2 * one_view_width]

    left_path = os.path.join(out_dir, "left.png")
    right_path = os.path.join(out_dir, "right.png")
    
    cv2.imwrite(left_path, left_img)
    cv2.imwrite(right_path, right_img)
    
    print(f"\n>>> Succesfully extracted frame {frame_idx}/{total_frames}:")
    print(f"  - Resolution: {one_view_width}x{height}")
    print(f"  - Saved: {left_path}")
    print(f"  - Saved: {right_path}")

    # generate intrinsics for D405 with curr resolution
    fx = one_view_width / (2.0 * np.tan(np.radians(84.0 / 2.0)))
    fy = fx
    cx = one_view_width / 2.0
    cy = height / 2.0
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)
    
    k_path = os.path.join(out_dir, "K.txt")
    np.savetxt(k_path, K, fmt="%.6f")
    print(f"  - Saved K.txt: {k_path}")
    
    return left_path, right_path, k_path


if __name__ == "__main__":
    video_path = input(">>> Enter video path for left and right png extraction: ")
    extract_stereo_pair(video_path)