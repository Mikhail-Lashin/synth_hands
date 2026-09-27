import os

# hand model & texture
ASSET_DIR = "./mano"
OBJ_MODEL_PATH = os.path.join(ASSET_DIR, "hand.obj")
TEXTURE_PATH = os.path.join(ASSET_DIR, "hand_texture.png")

# camera
CAM_POSITION = (0.0, -0.42, 0.0)
CAM_LOOK_AT = (0.0, 0.0, 0.0)

# render
RESOLUTION = (1024, 1024)
WARMUP_STEPS = 15
RENDER_SUBFRAMES = 32
OUTPUT_DIR = "renders"