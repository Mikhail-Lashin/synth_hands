import sys
import json
import shutil
import inspect
import warnings
import subprocess
from pathlib import Path

# Compatibility patches for Python 3.12 and NumPy 2.0
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
import numpy as np
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    for alias, target in [("bool", bool), ("int", int), ("float", float),
                          ("complex", complex), ("object", object),
                          ("unicode", str), ("str", str)]:
        if not hasattr(np, alias):
            setattr(np, alias, target)
try:
    import scipy.sparse
    sys.modules["scipy.sparse.csc"] = scipy.sparse
except ImportError:
    pass

import cv2
import pyrealsense2 as rs
import torch
import tyro

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import HandPoseIPR
from src.utils.visualisation import colorize_depth, draw_skeleton


def get_bbox_from_depth(depth: np.ndarray, min_d: float, max_d: float, W: int, H: int) -> tuple[float, float, float]:
    """Computes bounding box via center of mass and span of the valid mask pixels."""
    mask = (depth >= min_d) & (depth <= max_d) & ~np.isnan(depth) & ~np.isinf(depth)
    ys, xs = np.where(mask)
    
    if len(xs) > 50:
        # Center of mass of the hand mask
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        
        # Bounding box span based on coordinate spread
        w_span = float(xs.max() - xs.min())
        h_span = float(ys.max() - ys.min())
        box_size = max(w_span, h_span, 180.0) # minimum safety size
        box_size = float(np.clip(box_size, 180.0, 320.0))
        
        return cx, cy, box_size

    return W / 2.0, H * 0.45, 260.0

def load_cached_depth(file_path: Path) -> np.ndarray:
    if file_path.suffix == ".npz":
        with np.load(file_path) as data:
            return data["d"].astype(np.float32)
    return np.load(file_path).astype(np.float32)


def load_binary_mask(mask_path: Path | None, H: int, W: int) -> np.ndarray | None:
    """Loads binary mask where True / >0 indicates regions to reset to background."""
    if mask_path is None or not mask_path.exists():
        return None
    if mask_path.suffix == ".npy":
        mask = np.load(mask_path)
    elif mask_path.suffix == ".npz":
        with np.load(mask_path) as data:
            mask = data[list(data.files)[0]]
    else:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def ensure_ir_cache(bag_path: Path, cache_bag_dir: Path, target_count: int):
    """Extracts left IR frames from the bag file if they were not cached previously."""
    ir_files = sorted(cache_bag_dir.glob("ir_*.jpg"))
    if len(ir_files) >= target_count:
        return

    print(f">>> Caching left IR frames from {bag_path} into {cache_bag_dir}...")
    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)

    frame_idx = 0
    try:
        while frame_idx < target_count:            
            try:
                frames = pipeline.wait_for_frames(timeout_ms=3000)
            except RuntimeError:
                break
            ir1, ir2 = frames.get_infrared_frame(1), frames.get_infrared_frame(2)
            if not ir1 or not ir2:
                continue

            ir_file = cache_bag_dir / f"ir_{frame_idx:06d}.jpg"
            if not ir_file.exists():
                img_l = np.asanyarray(ir1.get_data())
                cv2.imwrite(str(ir_file), img_l, [cv2.IMWRITE_JPEG_QUALITY, 95])
            frame_idx += 1
    finally:
        pipeline.stop()


def prepare_ffs_cache(
    bag_path: Path,
    cache_bag_dir: Path,
    ffs_weights: Path,
    ffs_dir: Path,
    valid_iters: int,
    max_disp: int,
    max_frames: int | None,
    device: str
) -> tuple[dict, list[Path]]:
    meta_path = cache_bag_dir / "meta.json"
    cache_bag_dir.mkdir(parents=True, exist_ok=True)

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        depth_files = sorted(list(cache_bag_dir.glob("depth_*.npz")) + list(cache_bag_dir.glob("depth_*.npy")))
        if depth_files and (max_frames is None or len(depth_files) >= max_frames):
            return meta, depth_files[:max_frames] if max_frames else depth_files

    sys.path.insert(0, str(ffs_dir.resolve()))
    import core.foundation_stereo  # noqa: F401
    from core.utils.utils import InputPadder

    ffs_model = torch.load(str(ffs_weights), map_location="cpu", weights_only=False)
    ffs_model.args.valid_iters = valid_iters
    ffs_model.args.max_disp = max_disp
    ffs_model.to(device).eval()

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    profile = pipeline.start(config)
    profile.get_device().as_playback().set_real_time(False)

    fx, fy, cx, cy, baseline, W, H = None, None, None, None, 0.0182, None, None
    frame_idx = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        while max_frames is None or frame_idx < max_frames:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=3000)
            except RuntimeError:
                break
            ir1, ir2 = frames.get_infrared_frame(1), frames.get_infrared_frame(2)
            if not ir1 or not ir2:
                continue

            if fx is None:
                prof1 = ir1.get_profile().as_video_stream_profile()
                intr = prof1.get_intrinsics()
                fx, fy, cx, cy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)
                W, H = int(intr.width), int(intr.height)
                extrin = prof1.get_extrinsics_to(ir2.get_profile().as_video_stream_profile())
                if abs(extrin.translation[0]) > 1e-4:
                    baseline = float(abs(extrin.translation[0]))

            img_l = np.asanyarray(ir1.get_data())
            ir_file = cache_bag_dir / f"ir_{frame_idx:06d}.jpg"
            if not ir_file.exists():
                cv2.imwrite(str(ir_file), img_l, [cv2.IMWRITE_JPEG_QUALITY, 95])

            frame_file = cache_bag_dir / f"depth_{frame_idx:06d}.npz"
            if frame_file.exists():
                frame_idx += 1
                continue

            img_r = np.asanyarray(ir2.get_data())
            rgb_l = cv2.cvtColor(img_l, cv2.COLOR_GRAY2RGB) if img_l.ndim == 2 else cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
            rgb_r = cv2.cvtColor(img_r, cv2.COLOR_GRAY2RGB) if img_r.ndim == 2 else cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

            img0 = torch.as_tensor(rgb_l, device=device).float()[None].permute(0, 3, 1, 2)
            img1 = torch.as_tensor(rgb_r, device=device).float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(img0.shape, divis_by=32, force_square=False)
            img0_pad, img1_pad = padder.pad(img0, img1)

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda"), dtype=amp_dtype):
                disp = ffs_model.forward(img0_pad, img1_pad, iters=valid_iters, test_mode=True, optimize_build_volume="pytorch1")

            disp = padder.unpad(disp.float()).squeeze().cpu().numpy().clip(0.1, None)
            depth_meters = (fx * baseline) / disp
            depth_meters[depth_meters > 1.0] = 1.35
            np.savez_compressed(frame_file, d=depth_meters.astype(np.float16))
            frame_idx += 1
    finally:
        pipeline.stop()
        del ffs_model
        if device == "cuda":
            torch.cuda.empty_cache()

    meta = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "baseline": baseline, "W": W, "H": H}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    depth_files = sorted(cache_bag_dir.glob("depth_*.npz"))
    return meta, depth_files[:max_frames] if max_frames else depth_files


def main(
    bag_path: Path = Path("test.bag"),
    hand_weights: Path = Path("src/best_hand_model.pth"),
    ffs_weights: Path = Path("libs/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth"),
    ffs_dir: Path = Path("libs/Fast-FoundationStereo"),
    cache_dir: Path = Path("cache/ffs_depth"),
    mask_path: Path | None = Path("cache/bg_mask.png"),
    output_video: Path = Path("renders/real_pred_overlay.mp4"),
    bbox: int = 1,
    bbox_padding: float = 1.25,
    overlay_ir: bool = True,
    depth_threshold: float = 0.35,
    min_depth: float = 0.05,
    bg_depth_value: float = 1.35,
    crop_size: int = 224,
    valid_iters: int = 4,
    max_disp: int = 128,
    max_frames: int | None = None,
    fps: int = 120,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Run the hand pose inference pipeline on a RealSense (.bag) recording.

    Pipeline steps:
    1. Extract stereo frames (IR1 / IR2) and reconstruct depth maps using Fast-FoundationStereo.
    2. Cache generated depth maps and IR frames on disk to speed up subsequent runs.
    3. Filter the depth map by distance and apply a background mask.
    4. Detect or track the hand bounding box (modes 0, 1, or 2).
    5. Extract and resize the hand crop (crop_size x crop_size) and pass it to HandPoseIPR.
    6. Project the predicted 3D joints back into the frame coordinate system.
    7. Render the hand skeleton (21 MANO joints) onto the IR frame or depth map and save an H.264 video.

    Args:
        bag_path: Path to the input RealSense .bag recording.
        hand_weights: Path to the trained HandPoseIPR model weights (.pth).
        ffs_weights: Path to the Fast-FoundationStereo model weights.
        ffs_dir: Directory containing the Fast-FoundationStereo source code.
        cache_dir: Base directory for caching depth maps and metadata.
        mask_path: Path to an external binary background mask (PNG/NPY/NPZ) for noise filtering.
        output_video: Path to the output video file (MP4).
        bbox: Hand region detection mode (0: fixed min(H, W) center square without adaptive tracking;
            1: adaptive tracking with exponential moving average (EMA) smoothing across frames;
            2: independent per-frame depth-based bbox detection).
        bbox_padding: Expansion (padding) factor around the detected hand region.
        overlay_ir: If True, draw the skeleton over the left IR frame; if False, draw it over the colorized depth map.
        depth_threshold: Maximum depth in meters (objects farther away are treated as background).
        min_depth: Minimum depth in meters to clip near-range sensor artifacts.
        bg_depth_value: Depth value (in meters) assigned to background pixels inside the crop.
        crop_size: Side length (in pixels) of the square tensor fed to the neural network.
        valid_iters: Number of disparity refinement iterations in Fast-FoundationStereo.
        max_disp: Maximum disparity range for stereo matching.
        max_frames: Maximum number of frames to process (None processes the entire file).
        fps: Frame rate of the output video.
        device: Compute device ('cuda' or 'cpu').
    """
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cache_bag_dir = cache_dir / f"{bag_path.stem}_iters{valid_iters}_disp{max_disp}"
    meta, depth_files = prepare_ffs_cache(bag_path, cache_bag_dir, ffs_weights, ffs_dir, valid_iters, max_disp, max_frames, device)

    if overlay_ir:
        ensure_ir_cache(bag_path, cache_bag_dir, len(depth_files))

    fx, fy, W, H = meta["fx"], meta["fy"], meta["W"], meta["H"]
    bg_binary_mask = load_binary_mask(mask_path, H, W)

    # Central 480x480 boundaries
    crop_size_c = min(H, W)
    x_min_c = (W - crop_size_c) // 2
    x_max_c = x_min_c + crop_size_c
    center_crop = crop_size / 2.0

    hand_model = HandPoseIPR()
    hand_model.load_state_dict(torch.load(hand_weights, map_location=device, weights_only=True))
    hand_model.to(device).eval()

    video_writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    current_bbox = None

    try:
        for frame_idx, depth_file in enumerate(depth_files):
            if frame_idx % 50 == 0:
                print(f">>> Processed {frame_idx} clips")
            
            depth_filtered = load_cached_depth(depth_file)

            # Filter out invalid depth and outside of central square
            bg_mask = (
                (depth_filtered > depth_threshold)
                | (depth_filtered < min_depth)
                | np.isnan(depth_filtered)
                | np.isinf(depth_filtered)
            )
            bg_mask[:, :x_min_c] = True
            bg_mask[:, x_max_c:] = True

            # Apply external binary mask if provided
            if bg_binary_mask is not None:
                bg_mask |= bg_binary_mask

            depth_filtered[bg_mask] = bg_depth_value

            # Determine crop coordinates based on bbox mode (0, 1, or 2)
            if bbox == 0:
                pad_size = float(crop_size_c)
                x1, x2 = x_min_c, x_max_c
                y1, y2 = 0, H
            elif bbox == 2:
                # Mode 2: Compute bounding box purely from depth on every single frame
                box_cx, box_cy, box_size = get_bbox_from_depth(depth_filtered, min_depth, depth_threshold, W, H)
                box_cx = float(np.clip(box_cx, x_min_c + 70, x_max_c - 70))
                box_size = float(np.clip(box_size, 200.0, 320.0))
                pad_size = box_size * bbox_padding
                half = pad_size / 2.0

                x1 = int(np.clip(round(box_cx - half), 0, W - 2))
                x2 = int(np.clip(round(box_cx + half), x1 + 1, W))
                y1 = int(np.clip(round(box_cy - half), 0, H - 2))
                y2 = int(np.clip(round(box_cy + half), y1 + 1, H))
            else:
                # Mode 1: Standard adaptive tracker with EMA smoothing across frames
                if current_bbox is None:
                    box_cx, box_cy, box_size = get_bbox_from_depth(depth_filtered, min_depth, depth_threshold, W, H)
                else:
                    box_cx, box_cy, box_size = current_bbox

                box_cx = float(np.clip(box_cx, x_min_c + 70, x_max_c - 70))
                box_size = float(np.clip(box_size, 200.0, 320.0))
                pad_size = box_size * bbox_padding
                half = pad_size / 2.0

                x1 = int(np.clip(round(box_cx - half), 0, W - 2))
                x2 = int(np.clip(round(box_cx + half), x1 + 1, W))
                y1 = int(np.clip(round(box_cy - half), 0, H - 2))
                y2 = int(np.clip(round(box_cy + half), y1 + 1, H))

            # Crop and resize to model input size
            crop = depth_filtered[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                current_bbox = None
                crop_resized = np.full((crop_size, crop_size), bg_depth_value, dtype=np.float32)
            else:
                crop_resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)

            hand_input = torch.from_numpy(crop_resized).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_rel = hand_model(hand_input)[0].cpu().numpy()

            # Project 3D predictions back to full image pixel coordinates
            pred_z = np.clip(pred_rel[:, 2], 0.12, 0.60)
            scale_factor = crop_size / float(pad_size)
            fx_crop = fx * scale_factor
            fy_crop = fy * scale_factor

            u_crop = (pred_rel[:, 0] * fx_crop / pred_z) + center_crop
            v_crop = (pred_rel[:, 1] * fy_crop / pred_z) + center_crop

            u_full = (u_crop / scale_factor) + x1
            v_full = (v_crop / scale_factor) + y1
            joints_2d = np.stack([u_full, v_full], axis=-1)

            # Update tracker if mode 1 is active
            if bbox == 1:
                u_min_next, u_max_next = u_full.min(), u_full.max()
                v_min_next, v_max_next = v_full.min(), v_full.max()
                valid_joints_span = (u_max_next - u_min_next > 60) and (v_max_next - v_min_next > 80)
                in_frame = (x_min_c <= u_full.mean() <= x_max_c) and (0 <= v_full.mean() <= H)

                if valid_joints_span and in_frame:
                    next_cx = (u_min_next + u_max_next) / 2.0
                    next_cy = (v_min_next + v_max_next) / 2.0
                    current_depth = float(np.median(pred_z))
                    next_size = float(np.clip((fx * 0.19) / max(current_depth, 0.15), 200.0, 320.0))

                    current_bbox = (
                        0.6 * next_cx + 0.4 * box_cx,
                        0.6 * next_cy + 0.4 * box_cy,
                        0.7 * next_size + 0.3 * box_size
                    )
                else:
                    current_bbox = None

            # Select background canvas: Left IR grayscale or Depth Turbo colormap
            ir_file = cache_bag_dir / f"ir_{frame_idx:06d}.jpg"
            if overlay_ir and ir_file.exists():
                ir_gray = cv2.imread(str(ir_file), cv2.IMREAD_GRAYSCALE)
                canvas = cv2.cvtColor(ir_gray, cv2.COLOR_GRAY2BGR)
            else:
                canvas = colorize_depth(depth_filtered, min_dist=min_depth, max_dist=0.60)

            cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
            draw_skeleton(canvas, joints_2d)

            info_text = f"Frame {frame_idx:03d} | Bbox:{bbox} | Z: [{pred_z.min():.2f}m..{pred_z.max():.2f}m]"
            cv2.putText(canvas, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            video_writer.write(canvas)
    finally:
        video_writer.release()

    if shutil.which("ffmpeg") is not None and output_video.exists():
        tmp = output_video.with_suffix(".tmp.mp4")
        output_video.rename(tmp)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp), "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(output_video)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        tmp.unlink(missing_ok=True)

    print(f">>> Saved result to: {output_video}")


if __name__ == "__main__":
    tyro.cli(main)
