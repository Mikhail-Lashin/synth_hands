import sys
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import torch
import tyro

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import HandPoseIPR
from src.utils.visualisation import colorize_depth, draw_skeleton, project_3d_to_pixel


def main(
    npz_path: Path,
    weights_path: Path = Path("best_IPR.pth"),
    output_video: Path = Path("renders/pred_overlay.mp4"),
    fps: int = 30,
    h_fov: float = 84.0,       # по умолчанию для RealSense D405
    min_depth: float = 0.05,
    max_depth: float = 0.60,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    if not npz_path.exists():
        raise FileNotFoundError(f"Clip not found: {npz_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    # loading model
    output_video.parent.mkdir(parents=True, exist_ok=True)
    print(f">>> Loading model weights from: {weights_path} (on {device})...")
    model = HandPoseIPR()
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # loading npz clips
    data = np.load(npz_path)
    depths = data["depth"].astype(np.float32)      # (N, H, W)
    joints_gt = data["joints_3d"] if "joints_3d" in data else None  # (N, 21, 3)

    num_frames, H, W = depths.shape
    print(f">>> Processing {num_frames} frames from {npz_path.name}...")

    # VideoWriter init
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (W, H))

    with torch.no_grad():
        for frame_idx in range(num_frames):
            if frame_idx % 50 == 0:
                print(f">>> Processed {frame_idx} clips")
            
            depth_map = depths[frame_idx]

            input_tensor = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0).to(device)
            pred_joints = model(input_tensor)[0].cpu().numpy()  # (21, 3)
            canvas = colorize_depth(depth_map, min_dist=min_depth, max_dist=max_depth)
            joints_2d = project_3d_to_pixel(pred_joints, width=W, height=H, h_fov=h_fov)
            draw_skeleton(canvas, joints_2d)
            z_min, z_max = pred_joints[:, 2].min(), pred_joints[:, 2].max()
            text = f"{npz_path.name} | Frame: {frame_idx:02d}/{num_frames} | Pred Z: [{z_min:.2f}m..{z_max:.2f}m]"

            # compute MPJPE if GT exists in npz
            if joints_gt is not None:
                frame_gt = joints_gt[frame_idx]
                err_mm = np.linalg.norm(pred_joints - frame_gt, axis=-1).mean() * 1000.0
                text += f" | MPJPE: {err_mm:.1f}mm"

            cv2.putText(
                canvas,
                text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            video_writer.write(canvas)

    video_writer.release()
    print(f"\n>>> Video successfully saved to: {output_video}")
    
    if shutil.which("ffmpeg") is not None:
        temp_file = output_video.with_suffix(".tmp.mp4")
        output_video.rename(temp_file)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_file),
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            str(output_video)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        temp_file.unlink(missing_ok=True)

    print(f"\n>>> Video successfully saved and converted to: {output_video}")


if __name__ == "__main__":
    tyro.cli(main)
