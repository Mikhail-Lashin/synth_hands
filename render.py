import os
import sys
import inspect
import datetime
from PIL import Image
import subprocess
import tempfile
from PIL import Image
from pathlib import Path

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

import carb
import omni.usd
import omni.replicator.core as rep

import config as cfg
from hand import Hand

def setup_render_pipeline(camera, resolution=cfg.RESOLUTION):
    render_product = rep.create.render_product(camera, resolution)
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])
    
    print(">>> Warming up...")
    for _ in range(cfg.WARMUP_STEPS):
        simulation_app.update()
        
    return rgb_annotator

def render(rgb_annotator, 
           output_dir=cfg.OUTPUT_DIR, 
           num_subframes=cfg.RENDER_SUBFRAMES,
           filename=None):
    os.makedirs(output_dir, exist_ok=True)
    rep.orchestrator.step(rt_subframes=num_subframes, pause_timeline=True)
    data = rgb_annotator.get_data()
    
    if filename is None:
        filename = datetime.datetime.now().strftime("%H%M%S_%f_%d%m%Y.png")
        
    save_path = os.path.join(output_dir, filename)
    
    img = Image.fromarray(data[:, :, :3]) # without metadata
    img.save(save_path)
    print(f">>> Rendered: {save_path}")
    return save_path

def render_video(
    clip_idx: int,
    hand,
    rgb_annotator,
    simulation_app,
    num_frames: int = 60,
    fps: int = 30,
    output_dir: str = cfg.OUTPUT_DIR,
    num_subframes: int = cfg.RENDER_SUBFRAMES
):
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, f"clip_{clip_idx:04d}.mp4")

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"\n>>> Rendering clip {clip_idx:04d} ({num_frames} frames)...")
        
        for frame_idx in range(num_frames):
            hand.upd_verts(clip_idx=clip_idx, frame_idx=frame_idx)
            hand.rot_hand()
            
            simulation_app.update()
            
            rep.orchestrator.step(rt_subframes=num_subframes, pause_timeline=True)
            data = rgb_annotator.get_data()
            
            frame_path = os.path.join(tmp_dir, f"frame_{frame_idx:04d}.png")
            Image.fromarray(data[:, :, :3]).save(frame_path)
            
        print(f">>> Building MP4 with ffmpeg...")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            video_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        
    print(f">>> Video saved: {video_path}")

def main():
    # .obj & texture files
    obj_path = os.path.abspath(cfg.OBJ_PATH)
    tex_path = os.path.abspath(cfg.TEXTURE_PATH)
    for p in (obj_path, tex_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found at: {p}")
        
    # hand
    hand = Hand()
    hand.load_obj(obj_path)
    hand.rot_hand()
    stage = omni.usd.get_context().get_stage()
    hand.create_usd_mesh(stage)
    
    # plane
    textures_dir = Path("assets/plane_materials")
    texture_files = sorted(list(textures_dir.glob("*_Color.jpg")))
    plane = rep.create.plane(
        position=(0.0, 1.1, 0.0),
        rotation=(90, 0, 0),
        scale=1.5
    )

    # camera
    camera = rep.create.camera(
        position=cfg.CAM_POSITION,
        look_at=cfg.CAM_LOOK_AT,
        clipping_range=(0.01, 100.0)
    )

    # lights
    rep.create.light(light_type="dome", intensity=300)
    rep.create.light(light_type="distant",
                     intensity=700,
                     rotation=(40, 30, 0)
    )

    # rendering options
    carb.settings.get_settings().set("/rtx/post/aa/op", 2)      # anti-aliasing (0 - off, 1 - fxaa, 2 - taa)
    carb.settings.get_settings().set("/omni/replicator/backends/disk/root_dir"
                                     , os.path.abspath(".")     # root dir = project dir
    )

    # render hand
    rgb_annotator = setup_render_pipeline(camera)
    for texture_idx, clip_idx in enumerate(range(704, 714)):
        tex_path = os.path.abspath(texture_files[texture_idx])
        tex_name = texture_files[texture_idx].name.split("_")[0]

        plane_mat = rep.create.material_omnipbr(
            diffuse_texture=tex_path,
            roughness=0.85
        )
        with plane:
            rep.modify.material(plane_mat)
            
        simulation_app.update()
        
        render_video(
            clip_idx=clip_idx,
            hand=hand,
            rgb_annotator=rgb_annotator,
            simulation_app=simulation_app,
            num_frames=60,
            fps=30
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()