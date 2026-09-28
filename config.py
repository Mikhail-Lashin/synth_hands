from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# hand model & texture
NPZ_PATH = PROJECT_ROOT / "handx" / "train_mano.npz"
MANO_DIR = PROJECT_ROOT / "assets" / "mano" / "originals"
OBJ_PATH = PROJECT_ROOT / "assets" / "mano" / "hand.obj"
TEXTURE_PATH = PROJECT_ROOT / "assets" / "mano" / "hand_texture.png"

# camera
CAM_POSITION = (0.0, -0.42, 0.0)
CAM_LOOK_AT = (0.0, 0.0, 0.0)

# render
RESOLUTION = (1024, 1024)
WARMUP_STEPS = 15
RENDER_SUBFRAMES = 32
OUTPUT_DIR = "renders"