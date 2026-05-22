"""Configuration settings for the path tracer."""


# Screen width, height.
screen_dimensions = 1280, 720

# Field Of View (degrees), 0 < FOV < 180.
camera_fov = 50

# In world units.
camera_focal_length = 1

# Position in world units (x, y, z).
camera_world_pos = 0, 0, -5

# Local rotation (pitch, yaw, roll).
camera_rotation = 0, 0, 0

# Brightness factor.
exposure = 1

# Float RGB values.
background_color = 0.2, 0.2, 0.2

# Number of rays per pixel.
ray_samples = 5

# Number of times to accumulate frames.
accumulation_samples = 30

# Number of ray bounces per ray.
bounces = 5

# Denoise with Intel's Open Image Denoise while rendering chunks (reduces performance).
denoise = False

# Compare between noisy and denoised images after render.
comparison_slider = True