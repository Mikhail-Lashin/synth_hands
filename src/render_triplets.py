import inspect
import os
import subprocess
import sys
import numpy as np

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
from entities import NpzHand, StereoCamera, SceneManager
from transforms import get_clip_transform
try:
    from src.utils.visualisation import colorize_depth
except ImportError:
    from utils.visualisation import colorize_depth


def render_video(
    clip_idx: int,
    hand,
    camera,
    simulation_app,
    num_frames: int = 120,
    fps: int = 60,
    output_dir: str = cfg.OUTPUT_DIR,
    num_subframes: int = cfg.RENDER_SUBFRAMES,
    depth_range: tuple = (0.05, 0.6)
):
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"clip_{clip_idx:04d}_stereo.mp4")
    print(f"\n>>> Rendering clip {clip_idx:04d} ({num_frames} frames)...")

    w, h = camera.resolution
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w*3}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        video_path
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    R_clip, t_clip = get_clip_transform(clip_idx)
    for frame_idx in range(num_frames):
        hand.upd_verts(clip_idx=clip_idx, frame_idx=frame_idx)
        hand.transform_hand(R=R_clip, T=t_clip)

        data = camera.render(num_subframes=num_subframes)
        left_rgb = data["left_rgb"]
        right_rgb = data["right_rgb"]
        left_depth = data["left_depth"]
        
        depth_vis = colorize_depth(left_depth, min_dist=depth_range[0], max_dist=depth_range[1], rgb=True)
        combined_frame = np.hstack([left_rgb, right_rgb, depth_vis])
        process.stdin.write(combined_frame.tobytes())
        
    process.stdin.close()
    process.wait()
    print(f">>> Video ready: {video_path}")


def main():
    obj_path = os.path.abspath(cfg.OBJ_PATH)
    tex_path = os.path.abspath(cfg.TEXTURE_PATH)
    for p in (obj_path, tex_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    # hand
    stage = omni.usd.get_context().get_stage()
    hand = NpzHand()
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
    cam_config_path = str(cfg.OUTPUT_DIR + "/cam_config.json")
    camera.save_config(cam_config_path)

    # warmup
    for _ in range(cfg.WARMUP_STEPS):
        simulation_app.update()

    # video render
    for texture_idx, clip_idx in enumerate(range(23, 597)): # handx: 704 - 1177
        scene.set_backdrop_by_idx(texture_idx)
        scene.randomize_lighting()
        simulation_app.update()

        render_video(
            clip_idx=clip_idx,
            hand=hand,
            camera=camera,
            simulation_app=simulation_app,
            num_frames=120,
            fps=60
        )

if __name__ == "__main__":
    main()
    simulation_app.close()