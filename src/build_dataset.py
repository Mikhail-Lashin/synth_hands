import inspect
import re
import sys
from pathlib import Path
if not hasattr(inspect, "getargspec"): inspect.getargspec = inspect.getfullargspec
import numpy as np
for alias, target in [("bool", bool), ("int", int), ("float", float),
                      ("complex", complex), ("object", object),
                      ("unicode", str), ("str", str),]:
    if not hasattr(np, alias): setattr(np, alias, target)
import scipy.sparse; sys.modules["scipy.sparse.csc"] = scipy.sparse
import cv2
import smplx
import torch
import tyro
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FFS_DIR = PROJECT_ROOT / "libs" / "Fast-FoundationStereo"
sys.path.insert(0, str(FFS_DIR))

import core.foundation_stereo
from core.utils.utils import InputPadder

MANO_TIPS_VERT_INDICES = [744, 320, 443, 555, 672]

R_HAND = np.array([
        [ 0.0, 0.0, 1.0],
        [ 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ], dtype=np.float32)

MIN_DISP = 0.1

def get_clip_id(filename: str) -> int:
    match = re.search(r"clip_(\d+)", filename)
    if not match:
        raise ValueError(f"Can't determine clip ID: {filename}")
    return int(match.group(1))

def get_joints_21(joints_16: np.ndarray, verts_778: np.ndarray) -> np.ndarray:
    """
    16 MANO joints + 5 additional tip joints -> 21 joints
    (similar to HaMeR)
    """
    tips = verts_778[MANO_TIPS_VERT_INDICES]  # (5, 3)

    joints_21 = np.zeros((21, 3), dtype=np.float32)
    
    # WRIST
    joints_21[0] = joints_16[0]

    # THUMB
    joints_21[1:4] = joints_16[13:16] 
    joints_21[4] = tips[0]

    # INDEX
    joints_21[5:8] = joints_16[1:4]
    joints_21[8] = tips[1]

    # MIDDLE
    joints_21[9:12] = joints_16[4:7]
    joints_21[12] = tips[2]

    # RING
    joints_21[13:16] = joints_16[10:13]
    joints_21[16] = tips[3]

    # PINKY
    joints_21[17:20] = joints_16[7:10]
    joints_21[20] = tips[4]

    return joints_21

def load_model(weights_path: Path, valid_iters: int, max_disp: int, device: str):
    print(f">>> Loading FFS weights from {weights_path}...")
    model = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model.args.valid_iters = valid_iters
    model.args.max_disp = max_disp
    model.to(device).eval()
    return model

def main(input_dir: Path,
         output_dir: Path,
         ground_truth_path: Path = Path("handx/train_mano.npz"),
         mano_models_dir: Path = Path("assets/mano/originals"),
         weights_path: Path = Path("libs/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth"),

         valid_iters: int = 8,
         max_disp: int = 192,
         device: str = "cuda",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth = np.load(ground_truth_path, allow_pickle=True)

    mano_layer = smplx.create(
        model_path=str(mano_models_dir),
        model_type="MANO_RIGHT.pkl",
        is_rhand=True,
        use_pca=False,
        flat_hand_mean=False
    ).to(device)

    ffs_model = load_model(weights_path,
                           valid_iters,
                           max_disp,
                           device)
    video_files = sorted(list(input_dir.glob("*.mp4")))
    print(f">>> Processing clips: {len(video_files)}")

    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # camera params
    cam_config_path = Path(input_dir) / "cam_config.json"
    with open(cam_config_path, "r", encoding="utf-8") as f:
        cam_config = json.load(f)
        
    fx = cam_config["intrinsics"][0][0]
    baseline = cam_config["baseline"]
    center = cam_config["position"]
    cam_left_pos = np.array([center[0]-baseline / 2.0,
                             center[1],
                             center[2]],
                            dtype=np.float32)
    clip_range = tuple(cam_config["clip_range"])
    

    for vid_path in video_files:
        clip_id = get_clip_id(vid_path.name)
        save_path = output_dir / f"clip_{clip_id:04d}.npz"

        if save_path.exists():
            print(f">>> Skip (already exists): {save_path.name}")
            continue

        print(f"\n>>> Processing clip {clip_id:04d} ({vid_path.name})...")
        clip_key = str(clip_id)
        if clip_key not in ground_truth:
            continue

        clip_dict = ground_truth[clip_key].item()
        poses_all = torch.tensor(clip_dict["right_pose"], dtype=torch.float32, device=device)
        shapes_all = torch.tensor(clip_dict["right_shape"], dtype=torch.float32, device=device)

        cap = cv2.VideoCapture(str(vid_path))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        depth_clip = []
        joints_clip = []

        with torch.no_grad():
            for frame_idx in range(num_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                height, total_width, _ = frame.shape
                single_width = total_width // 3

                left_bgr = frame[:, :single_width]
                right_bgr = frame[:, single_width : 2 * single_width]

                left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
                right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

                img0 = torch.as_tensor(left_rgb, device=device).float()[None].permute(0, 3, 1, 2)
                img1 = torch.as_tensor(right_rgb, device=device).float()[None].permute(0, 3, 1, 2)

                padder = InputPadder(img0.shape, divis_by=32, force_square=False)
                img0_pad, img1_pad = padder.pad(img0, img1)

                with torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
                    disp = ffs_model.forward(
                        img0_pad, img1_pad,
                        iters=valid_iters,
                        test_mode=True,
                        optimize_build_volume="pytorch1"
                    )

                disp = padder.unpad(disp.float())
                disp = disp.data.cpu().numpy().reshape(height, single_width).clip(0, None)
                disp = np.maximum(disp, MIN_DISP)
                depth_meters = (fx * baseline) / disp
                depth_meters = np.clip(depth_meters, clip_range[0], clip_range[1]).astype(np.float16)
                depth_clip.append(depth_meters)

                # compute 21 joints
                pose_f = poses_all[frame_idx : frame_idx + 1]
                shape_f = shapes_all[frame_idx : frame_idx + 1]

                output = mano_layer(hand_pose=pose_f[:, 3:], betas=shape_f)
                j16_world = output.joints[0].cpu().numpy()     # (16, 3)
                verts_world = output.vertices[0].cpu().numpy() # (778, 3)
                joints_world_21 = get_joints_21(j16_world, verts_world)
                joints_world_21 = joints_world_21 @ R_HAND.T

                # go to left cam optical frame (opencv: X-right, Y-down, Z-forward)
                delta = joints_world_21 - cam_left_pos
                joints_cam = np.zeros_like(delta)
                joints_cam[:, 0] = delta[:, 0]
                joints_cam[:, 1] = -delta[:, 2]
                joints_cam[:, 2] = delta[:, 1]

                joints_clip.append(joints_cam.astype(np.float32))

        cap.release()

        depth_array = np.stack(depth_clip, axis=0)    # (60, H, W), float16
        joints_array = np.stack(joints_clip, axis=0)  # (60, 21, 3), float32

        np.savez_compressed(save_path, depth=depth_array, joints_3d=joints_array)
        print(f">>> Done: {save_path.name} | Depth: {depth_array.shape} | Joints: {joints_array.shape}")


if __name__ == "__main__":
    tyro.cli(main)