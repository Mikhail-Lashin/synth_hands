from __future__ import annotations

import cv2
import numpy as np

# MANO 21-joint skeleton connectivity (finger segment pairs)
MANO_21_BONES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # THUMB
    (0, 5), (5, 6), (6, 7), (7, 8),         # INDEX
    (0, 9), (9, 10), (10, 11), (11, 12),    # MIDDLE
    (0, 13), (13, 14), (14, 15), (15, 16),  # RING
    (0, 17), (17, 18), (18, 19), (19, 20),  # PINKY
]

# Finger colors in BGR format
FINGER_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 255),  # THUMB  (yellow)
    (0, 255, 0),    # INDEX  (green)
    (255, 255, 0),  # MIDDLE (cyan)
    (255, 0, 0),    # RING   (blue)
    (255, 0, 255),  # PINKY  (magenta)
]


def project_3d_to_pixel(
    joints_cam: np.ndarray,
    width: int,
    height: int,
    h_fov: float | None = 84.0,
    fx: float | None = None,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
) -> np.ndarray:
    """
    Project 3D camera coordinates (X, Y, Z) to 2D pixel coordinates (u, v).

    Parameters:
        joints_cam: (N, 3) or (..., 3) array of 3D points in meters.
        width: Image width in pixels.
        height: Image height in pixels.
        h_fov: Horizontal field of view in degrees (used if fx is None).
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point offsets in pixels.
    """
    if fx is None:
        if width == 848 and height == 480:
            # fx = (w * focal_length) / horizontal_aperture = (848 * 2.0030) / 3.896 = 435.97125
            # TODO: replace h_fov, add focal_length (m) and compute fx using focal_length & resolution
            fx = 435.97  # correct only for (848x480)
        elif h_fov is not None:
            # fx = width / (2.0 * np.tan(np.radians(h_fov / 2.0)))
            fx = width / (2.0 * np.tan(np.radians(h_fov / 2.0)))
        else:
            raise ValueError("Either fx, h_fov or (848x480) resolution must be provided.")
    if fy is None:
        fy = fx
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0

    X = joints_cam[..., 0]
    Y = joints_cam[..., 1]
    Z = joints_cam[..., 2]

    Z_safe = np.maximum(Z, 1e-4)

    u = (fx * X / Z_safe) + cx
    v = (fy * Y / Z_safe) + cy

    return np.stack([u, v], axis=-1)


def draw_skeleton(
    canvas: np.ndarray,
    joints_2d: np.ndarray,
    bones: list[tuple[int, int]] = MANO_21_BONES,
    colors: list[tuple[int, int, int]] = FINGER_COLORS,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw 21-joint MANO hand skeleton on canvas (modifies canvas in-place and returns it).

    Parameters:
        canvas: (H, W, 3) image array (BGR or RGB).
        joints_2d: (21, 2) pixel coordinates.
        bones: list of (joint_idx1, joint_idx2) bone segments.
        colors: list of colors for each finger.
        thickness: line thickness.
    """
    H, W = canvas.shape[:2]

    # 1. Draw bones
    for bone_idx, (j1, j2) in enumerate(bones):
        finger_id = bone_idx // 4  # 4 segments per finger
        color = colors[finger_id % len(colors)]

        pt1 = tuple(np.round(joints_2d[j1]).astype(int))
        pt2 = tuple(np.round(joints_2d[j2]).astype(int))

        if 0 <= pt1[0] < W and 0 <= pt1[1] < H and 0 <= pt2[0] < W and 0 <= pt2[1] < H:
            cv2.line(canvas, pt1, pt2, color, thickness=thickness, lineType=cv2.LINE_AA)

    # 2. Draw joints
    tip_indices = {4, 8, 12, 16, 20}
    for idx, pt in enumerate(joints_2d):
        p = tuple(np.round(pt).astype(int))
        if 0 <= p[0] < W and 0 <= p[1] < H:
            if idx == 0:
                color = (255, 255, 255)  # Wrist: white
                radius = 5
            elif idx in tip_indices:
                color = (0, 255, 0)      # Fingertips: green
                radius = 4
            else:
                color = (0, 0, 255)      # Internal joints: red
                radius = 3

            # Black contour + colored circle
            cv2.circle(canvas, p, radius=radius + 1, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p, radius=radius, color=color, thickness=-1, lineType=cv2.LINE_AA)

    return canvas


def colorize_depth(
    depth_meters: np.ndarray,
    min_dist: float = 0.05,
    max_dist: float = 0.60,
    colormap: int = cv2.COLORMAP_TURBO,
    rgb: bool = False,
) -> np.ndarray:
    """
    Normalize and colorize depth map using an OpenCV colormap.

    Parameters:
        depth_meters: (H, W) depth in meters.
        min_dist: Near clipping distance in meters.
        max_dist: Far clipping distance in meters.
        colormap: OpenCV colormap (default COLORMAP_TURBO).
        rgb: If True, convert output from BGR to RGB (e.g. for Isaac Sim / raw video).
    """
    clean_depth = np.nan_to_num(depth_meters, nan=max_dist, posinf=max_dist, neginf=min_dist)
    depth_norm = np.clip((clean_depth - min_dist) / (max_dist - min_dist), 0.0, 1.0)
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)

    colored = cv2.applyColorMap(depth_uint8, colormap)
    if rgb:
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored
