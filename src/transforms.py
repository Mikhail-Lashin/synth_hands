import numpy as np
from scipy.spatial.transform import Rotation as R

R_HAND = np.array([                     # default hand rot (corrects hand pos in front of the camera)
                [ 0.0, 0.0, 1.0],
                [ 0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0]
            ], dtype=np.float32)

# poor dependency: synchronisation required for render_triplets.py & build_dataset.py
# TODO: add (R, t) saving in json, remove double seed randomisation
def get_clip_transform(clip_idx: int) -> tuple:
    """Deterministic pseudorandom hand rotation & translation for clip."""
    rng = np.random.RandomState(seed=clip_idx)
    
    # rotation (deg)
    pitch = rng.uniform(-10.0, 10.0)
    yaw   = rng.uniform(-10.0, 10.0)
    roll  = rng.uniform(-15.0, 15.0)
    
    R_rand = R.from_euler('xyz', [pitch, yaw, roll], degrees=True).as_matrix().astype(np.float32)
    
    # translation (m)
    t_rand = rng.uniform(-0.025, 0.025, size=3).astype(np.float32)
    
    return R_rand @ R_HAND, t_rand
