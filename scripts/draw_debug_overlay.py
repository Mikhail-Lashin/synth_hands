from pathlib import Path
import cv2
import numpy as np
import tyro

MANO_21_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # THUMB
    (0, 5), (5, 6), (6, 7), (7, 8),         # INDEX
    (0, 9), (9, 10), (10, 11), (11, 12),    # MIDDLE
    (0, 13), (13, 14), (14, 15), (15, 16),  # RING
    (0, 17), (17, 18), (18, 19), (19, 20)   # PINKY
]

FINGER_COLORS = [
    (0, 255, 255),  # THUMB
    (0, 255, 0),    # INDEX
    (255, 255, 0),  # MIDDLE
    (255, 0, 0),    # RING
    (255, 0, 255),  # PINKY
]

def project_3d_to_pixel(joints_cam: np.ndarray, width: int, height: int, h_fov: float = 84.0) -> np.ndarray:
    fx = width / (2.0 * np.tan(np.radians(h_fov / 2.0)))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    X = joints_cam[:, 0]
    Y = joints_cam[:, 1]
    Z = joints_cam[:, 2]

    Z_safe = np.maximum(Z, 1e-4)

    u = (fx * X / Z_safe) + cx
    v = (fy * Y / Z_safe) + cy

    return np.stack([u, v], axis=-1)

def main(npz_path: Path,
         frame_idx: int = 0,
         output_image: Path = Path("renders/debug_joint_overlay.png"),
         h_fov: float = 84.0, # default for RealSense D405
         min_depth: float = 0.05,
         max_depth: float = 0.60
):
    
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")

    data = np.load(npz_path)
    depths = data["depth"].astype(np.float32)
    joints_3d = data["joints_3d"]  # (60, 21, 3)

    num_frames = depths.shape[0]
    frame_idx = max(0, min(frame_idx, num_frames - 1))

    depth_map = depths[frame_idx]
    joints_frame = joints_3d[frame_idx]

    H, W = depth_map.shape

    # depth map in turbo
    clean_depth = np.nan_to_num(depth_map, nan=max_depth, posinf=max_depth, neginf=min_depth)
    depth_norm = np.clip((clean_depth - min_depth) / (max_depth - min_depth), 0.0, 1.0)
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)
    canvas = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)

    # joints projection
    joints_2d = project_3d_to_pixel(joints_frame, width=W, height=H, h_fov=h_fov)

    # bones rendering
    for bone_idx, (j1, j2) in enumerate(MANO_21_BONES):
        finger_id = bone_idx // 4  # 4 segments for 1 finger
        color = FINGER_COLORS[finger_id]
        
        pt1 = tuple(np.round(joints_2d[j1]).astype(int))
        pt2 = tuple(np.round(joints_2d[j2]).astype(int))

        if 0 <= pt1[0] < W and 0 <= pt1[1] < H and 0 <= pt2[0] < W and 0 <= pt2[1] < H:
            cv2.line(canvas, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)

    # joints rendering
    tip_indices = {4, 8, 12, 16, 20}
    for idx, pt in enumerate(joints_2d):
        p = tuple(np.round(pt).astype(int))
        if 0 <= p[0] < W and 0 <= p[1] < H:
            if idx == 0:
                color = (255, 255, 255)
                radius = 5
            elif idx in tip_indices:
                color = (0, 255, 0)
                radius = 4
            else:
                color = (0, 0, 255)
                radius = 3

            cv2.circle(canvas, p, radius=radius+1, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

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