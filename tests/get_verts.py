import numpy as np
import torch
import smplx
import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg

# compatibility patches for python 3.12
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

data = np.load(cfg.NPZ_PATH, allow_pickle=True)

clip = data["0"].item()
frame_idx = 0
pose_frame = torch.tensor(clip["right_pose"][frame_idx : frame_idx + 1], dtype=torch.float32)   # (1, 48)
trans_frame = torch.tensor(clip["right_trans"][frame_idx : frame_idx + 1], dtype=torch.float32) # (1, 3)
shape_frame = torch.tensor(clip["right_shape"][frame_idx : frame_idx + 1], dtype=torch.float32) # (1, 10)

mano_layer = smplx.create(
    model_path=str(cfg.MANO_DIR),
    model_type="MANO_RIGHT.pkl",
    is_rhand=True,
    use_pca=False,
    flat_hand_mean=False
)

output = mano_layer(
    global_orient=pose_frame[:, :3],
    hand_pose=pose_frame[:, 3:],
    betas=shape_frame,
    transl=trans_frame
)

verts = output.vertices[0].detach().cpu().numpy()
print(">>> Verts:", verts.shape)