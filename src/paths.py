"""Project paths used by the renderer and viewer."""


from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OBJ_FILES_DIR = PROJECT_ROOT / "obj_files"
SAVED_IMAGES_DIR = PROJECT_ROOT / "saved_images"

ORIGINAL_RENDER_PATH = SAVED_IMAGES_DIR / "original_render.png"
DENOISED_RENDER_PATH = SAVED_IMAGES_DIR / "denoised_render.png"
