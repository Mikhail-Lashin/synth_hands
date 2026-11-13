import sys
from pathlib import Path

import cv2
import numpy as np
import tyro

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.visualisation import colorize_depth, draw_skeleton, project_3d_to_pixel


def main(npz_path: Path,
         frame_idx: int = 0,
         output_image: Path = Path("renders/debug_joint_overlay.png"),
         h_fov: float = 84.0, # default for RealSense D405 (approximately, needs fix)
         min_depth: float = 0.05,
         max_depth: float = 0.60,
         bbox_padding: float = 1.30,
):
    
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path)
    depths = data["depth"].astype(np.float32)
    joints_3d = data["joints_3d"]  # (60, 21, 3)
    bboxes = data["bbox"]

    num_frames = depths.shape[0]
    frame_idx = max(0, min(frame_idx, num_frames - 1))

    depth_map = depths[frame_idx]
    joints_frame = joints_3d[frame_idx]
    bbox = bboxes[frame_idx]

    H, W = depth_map.shape

    # depth map in turbo
    canvas = colorize_depth(depth_map, min_dist=min_depth, max_dist=max_depth)

    # joints projection
    # fx = width / (2.0 * np.tan(np.radians(h_fov / 2.0)))
    # TODO: replace h_fov, add focal_length (m) and compute fx using focal_length & resolution
    fx = 435.97  # correct only for (848x480)
    joints_2d = project_3d_to_pixel(joints_frame, width=W, height=H, fx=fx)

    # skeleton rendering
    draw_skeleton(canvas, joints_2d)
            
    # bbox rendering
    cx_box, cy_box, box_size = bbox
    box_size *= bbox_padding
    x1 = cx_box - box_size / 2
    y1 = cy_box - box_size / 2
    x2 = cx_box + box_size / 2
    y2 = cy_box + box_size / 2
    cv2.rectangle(canvas,
                  (int(x1), int(y1)),
                  (int(x2), int(y2)),
                  (0, 255, 0),
                  thickness=2)
                  

    # text metadata
    z_min, z_max = joints_frame[:, 2].min(), joints_frame[:, 2].max()
    cv2.putText(
        canvas,
        f"{npz_path.name} | Frame: {frame_idx:02d} | 21 Joints | Depth: [{z_min:.2f}m .. {z_max:.2f}m]",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.imwrite(str(output_image), canvas)
    print(f"\n>>> Saved to: {output_image}")


if __name__ == "__main__":
    tyro.cli(main)