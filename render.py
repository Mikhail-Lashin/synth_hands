import os
import sys
import inspect
import numpy as np
import datetime
from PIL import Image

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

def rot_hand(verts: np.ndarray) -> np.ndarray:
    R = np.array([
        [ 0.0, 0.0, 1.0],
        [ 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0]
    ], dtype=np.float32)

    assert np.isclose(np.linalg.det(R), 1.0), "Rotation matrix must have det = +1"
    return verts @ R.T

def main():
    # .obj & texture files
    obj_path = os.path.abspath(cfg.OBJ_MODEL_PATH)
    tex_path = os.path.abspath(cfg.TEXTURE_PATH)
    for p in (obj_path, tex_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required file not found at: {p}")
        
    # hand
    hand = Hand()
    hand.load_obj(obj_path)
    hand.verts = rot_hand(hand.verts)
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

    # replicator pipeline
    render_product = rep.create.render_product(camera, resolution=cfg.RESOLUTION)
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])
    
    for _ in range(cfg.WARMUP_STEPS):
        simulation_app.update()
        
    rep.orchestrator.step(rt_subframes=cfg.RENDER_SUBFRAMES, pause_timeline=True)
    data = rgb_annotator.get_data()
    filename = datetime.datetime.now().strftime("%H%M%S_%d%m%Y.png")
    save_path = os.path.join(cfg.OUTPUT_DIR, filename)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    img = Image.fromarray(data[:, :, :3]) # without metadata
    img.save(save_path)
    print(f">>> Rendered to: {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()