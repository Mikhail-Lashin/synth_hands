import os
import sys
import inspect
import numpy as np
import datetime
from PIL import Image
import smplx

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
    for i in range(100, 150):
        for j in range(0, 60, 20):
            hand.upd_verts(clip_idx=i, frame_idx=j)
            simulation_app.update()
            hand.rot_hand()
            render(rgb_annotator)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()