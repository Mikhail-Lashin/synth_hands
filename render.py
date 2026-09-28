import os
import sys
import inspect
from PIL import Image
import subprocess
import tempfile
from PIL import Image
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# compatibility patches for python 3.12
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
try:
    import scipy.sparse
    sys.modules["scipy.sparse.csc"] = scipy.sparse
except ImportError:
    pass

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "renderer": "RealTimePathTracing"})

import omni.usd

import config as cfg
from entities import Hand, StereoCamera, SceneManager


def _colorize_depth(depth_meters: np.ndarray, min_dist: float = 0.05, max_dist: float = 0.6) -> np.ndarray:
    clean_depth = np.nan_to_num(depth_meters, nan=max_dist, posinf=max_dist, neginf=min_dist)

    depth_norm = np.clip((clean_depth - min_dist) / (max_dist - min_dist), 0.0, 1.0)
    depth_uint8 = (depth_norm * 255.0).astype(np.uint8)

    if HAS_CV2:
        depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    else:
        return np.repeat(depth_uint8[:, :, np.newaxis], 3, axis=-1)


def render_video(
    clip_idx: int,
    hand,
    camera,
    simulation_app,
    num_frames: int = 60,
    fps: int = 30,
    output_dir: str = cfg.OUTPUT_DIR,
    num_subframes: int = cfg.RENDER_SUBFRAMES,
    depth_range: tuple = (0.05, 0.6)
):
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"clip_{clip_idx:04d}_stereo.mp4")

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"\n>>> Rendering clip {clip_idx:04d} ({num_frames} frames)...")

        for frame_idx in range(num_frames):
            hand.upd_verts(clip_idx=clip_idx, frame_idx=frame_idx)
            hand.rot_hand()

            simulation_app.update()

            data = camera.render(num_subframes=num_subframes)
            left_rgb = data["left_rgb"]
            right_rgb = data["right_rgb"]
            left_depth = data["left_depth"]
            
            depth_vis = _colorize_depth(left_depth, min_dist=depth_range[0], max_dist=depth_range[1])
            combined_frame = np.hstack([left_rgb, right_rgb, depth_vis])

            frame_path = os.path.join(tmp_dir, f"frame_{frame_idx:04d}.png")
            Image.fromarray(combined_frame).save(frame_path)

        # build video
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            video_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    print(f">>> Видео готово: {video_path}")

def main():
    obj_path = os.path.abspath(cfg.OBJ_PATH)
    tex_path = os.path.abspath(cfg.TEXTURE_PATH)
    for p in (obj_path, tex_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    # hand
    stage = omni.usd.get_context().get_stage()
    hand = Hand()
    hand.load_obj(obj_path)
    hand.rot_hand()
    hand.create_usd_mesh(stage)

    # backdrop plane, lightning
    scene = SceneManager(textures_dir="assets/plane_materials")

    # camera
    camera = StereoCamera(
        position=cfg.CAM_POSITION,
        look_at=cfg.CAM_LOOK_AT,
        resolution=cfg.RESOLUTION
    )

    # warmup
    for _ in range(cfg.WARMUP_STEPS):
        simulation_app.update()

    # video render
    for texture_idx, clip_idx in enumerate(range(704, 1177)):
        scene.set_backdrop_by_idx(texture_idx)
        scene.randomize_lighting()
        simulation_app.update()

        render_video(
            clip_idx=clip_idx,
            hand=hand,
            camera=camera,
            simulation_app=simulation_app,
            num_frames=60,
            fps=30
        )

if __name__ == "__main__":
    main()
    simulation_app.close()