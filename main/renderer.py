"""
Main rendering module.

Renders the scene using the path tracer and outputs a denoised image and its original image.

Refer to docs/coordinate_spaces.md for information.
Refer to docs/rendering.md for information.
"""


import contextlib
import numpy as np
import time
from numba import njit
from numba import prange
from PIL import Image
import threading

# Remove pygame message.
with contextlib.redirect_stdout(None):
    import pygame

import settings
from path_tracer import path_trace
from denoiser import oidn_denoise
import math_utils
import input_check


WIDTH, HEIGHT = settings.screen_dimensions

ASPECT_RATIO = WIDTH / HEIGHT

CAMERA_FOV = np.radians(settings.camera_fov)
CAMERA_WORLD_POS = np.array((
    settings.camera_world_pos[0],
    settings.camera_world_pos[1],
    -settings.camera_world_pos[2],
    ),
    dtype=np.float64
)
CAMERA_ROTATION = np.array((
    # Invert to correct for display y coordinates.
    -np.radians(settings.camera_rotation[0]), # Pitch
    np.radians(settings.camera_rotation[1]), # Yaw
    np.radians(settings.camera_rotation[2]) # Roll
))

EXPOSURE = settings.exposure

# Distance from camera to the projection plane.
PROJ_DIST = -settings.camera_focal_length
# Calculate half-size of the projection plane.
PROJ_HALF_HEIGHT = -PROJ_DIST * np.tan(CAMERA_FOV / 2)
PROJ_HALF_WIDTH = PROJ_HALF_HEIGHT * ASPECT_RATIO

RAY_SAMPLES = settings.ray_samples
ACCUM_SAMPLES = settings.accumulation_samples
BOUNCES = settings.bounces

DENOISE = settings.denoise

input_check.check_settings()

# Split pixels into chunks so the screen updates incrementally.
NUM_CHUNKS = 16
PIXELS = np.array([(i, j) for j in range(HEIGHT) for i in range(WIDTH)])
CHUNKS = np.array_split(PIXELS, NUM_CHUNKS)

total_accum_light = np.zeros((WIDTH, HEIGHT, 3))

sample_count = 0
finished_chunks = 0

avg_light = None
original_render = None
denoised_render = None

render_start_time = None
sample_start_time = None


@njit(cache=True)
def screen_to_world(pixel):
    """
    Convert a screen-space pixel to world space through the projection plane.

    Notes
    -----
    Refer to docs/rendering.md for more information.

    Parameters
    ----------
    pixel : np.ndarray
        Coords of the screen pixel, shape (2,).

    Returns
    -------
    np.ndarray
        The normalized world-space vector transformed from the screen pixel coords.

    References
    ----------
    https://gabrielgambetta.com/computer-graphics-from-scratch/02-basic-raytracing.html
    """

    display_x, display_y = pixel
    world_z = PROJ_DIST

    # Set the middle of the screen as the origin (0, 0).
    screen_x = display_x - WIDTH / 2
    screen_y = display_y - HEIGHT / 2

    # Calculate screen space position (from -1 to 1) then convert to world space.
    world_x = screen_x / (WIDTH / 2) * PROJ_HALF_WIDTH
    # Make y negative because y-direction in screen space is inverted.
    world_y = -screen_y / (HEIGHT / 2) * PROJ_HALF_HEIGHT

    return math_utils.normalize(np.array((world_x, world_y, world_z)))


@njit(cache=True)
def rotate_direction(ray_dir, rotation_angles):
    """
    Transform the ray direction by rotation transformations (pitch, yaw, roll).

    Parameters
    ----------
    ray_dir : np.ndarray
        The ray direction vector, shape (3,).
    rotation_angles : np.ndarray
        The pitch, yaw, roll values to rotate the ray direction by, shape (3,).

    Returns
    -------
    np.ndarray
        The rotated ray direction after the transformations, shape (3,).

    References
    ----------
    https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    """

    pitch, yaw, roll = rotation_angles
    
    rotation_x = np.array((
        (1, 0, 0),
        (0, np.cos(pitch), -np.sin(pitch)),
        (0, np.sin(pitch), np.cos(pitch))
    ))
    rotation_y = np.array((
        (np.cos(yaw), 0, np.sin(yaw)),
        (0, 1, 0),
        (-np.sin(yaw), 0, np.cos(yaw))
    ))
    rotation_z = np.array((
        (np.cos(roll), -np.sin(roll), 0),
        (np.sin(roll), np.cos(roll), 0),
        (0, 0, 1)
    ))

    # Rotate x (pitch), then y (yaw), then z (roll).
    rotation_matrix = rotation_x @ rotation_y @ rotation_z

    return ray_dir @ rotation_matrix


@njit(cache=True)
def jitter():
    """
    Random pixel jitter for anti-aliasing. Generates a random value between -1 and 1.

    Returns
    -------
    np.ndarray
        Random 2D pixel offset, shape (2,).

    References
    ----------
    https://www.youtube.com/watch?v=Qz0KTGYJtUk
    """

    return np.random.uniform(-1, 1, 2)


@njit(cache=True)
def linear_to_srgb(render):
    """
    Gamma correct from linear RGB to sRGB.

    Notes
    -----
    Refer to docs/rendering.md for more information.

    Parameters
    ----------
    render : np.ndarray
        Output after tonemapping and/or denoising, shape (width, height, 3).

    Returns
    -------
    np.ndarray
        Output int RGB after gamma correction, shape (width, height, 3), dtype uint8.

    References
    ----------
    https://learnopengl.com/Advanced-Lighting/Gamma-Correction
    """

    float_rgb = np.clip(render, 0, 1)
    # Gamma correct to sRGB color.
    srgb = float_rgb ** (1.0 / 2.2)
    int_rgb = (srgb * 255).astype(np.uint8)
    return int_rgb


@njit(cache=True)
def aces_film(render):
    """
    Apply tonemapping with the ACES method.

    Notes
    -----
    Refer to docs/rendering.md for more information.

    Parameters
    ----------
    render : np.ndarray
        Output after path tracing, shape (width, height, 3).

    Returns
    -------
    np.ndarray
        Output tonemapped image, shape (width, height, 3).

    References
    ----------
    https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    """

    # ACES tonemapping graphing curve.
    return (render*(2.51*render+0.03))/(render*(2.43*render+0.59)+0.14)


@njit(cache=True, parallel=True)
def render_chunk(scene, chunk, ray_samples, bounces):
    """
    Path trace a chunk of pixels.

    Uses parallel CPU threads with Numba parallelization to calculate multiple pixels at a time.

    Parameters
    ----------
    scene : numba.typed.List(SceneObject)
        Numba typed list of SceneObject numba jitclasses.
        Each SceneObject requires data used for path tracing.
    chunk : np.ndarray
        A chunk of pixels, shape (width, height, 3).
    ray_samples : int
        Number of samples per pixel.
    bounces : int
        How many times a ray should bounce in the scene.

    Returns
    -------
    np.ndarray
        Rendered chunk after path tracing, shape (width, height, 3).
    """

    chunk_render = np.zeros((WIDTH, HEIGHT, 3))
    
    for i in prange(len(chunk)):
        for _ in range(ray_samples):
            pixel = chunk[i]
            ray_dir = screen_to_world(pixel + jitter())
            ray_dir = rotate_direction(ray_dir, CAMERA_ROTATION)
            color = path_trace(scene, CAMERA_WORLD_POS, ray_dir, bounces)
            chunk_render[pixel[0], pixel[1]] += color
    
    return chunk_render


def return_render(chunk_render):
    """
    Render on pygame screen after path tracing.

    Calculates and prints output data: chunk number, percent completed, elapsed time, and estimated time of arrival.

    Parameters
    ----------
    screen : pygame.display
        The initialized pygame screen used.
    chunk_render : np.ndarray
        A chunk of pixels, shape (width, height, 3).
    """

    global total_accum_light
    global finished_chunks
    global original_render
    global denoised_render
    global avg_light
    
    total_accum_light += chunk_render
    finished_chunks += 1

    avg_light = (total_accum_light / sample_count * EXPOSURE).astype(np.float32)
    original_render = linear_to_srgb(aces_film(avg_light))
    if DENOISE:
        denoised_render = linear_to_srgb(aces_film(oidn_denoise(avg_light)))
    
    total_elapsed_time = time.perf_counter() - render_start_time
    sample_elapsed_time = time.perf_counter() - sample_start_time
    percent_done = (finished_chunks / NUM_CHUNKS) * 100
    avg_time = sample_elapsed_time / finished_chunks
    remaining_time = avg_time * (NUM_CHUNKS - finished_chunks)

    if remaining_time > 0:
        eta_text = f"{remaining_time:.2f}s"
    else:
        eta_text = "Done"

    print(f"\rChunk {finished_chunks}/{NUM_CHUNKS} ({percent_done:.1f}%) | "
          f"Elapsed: {total_elapsed_time:.2f}s | "
          f"ETA: {eta_text}"
          # Add padding to completely overwrite previous print without problems.
          " " * 100,
          # Prevent creating a newline.
          end="",
          # Immediately output to prevent glitches.
          flush=True)


def render(scene):
    """
    Main rendering function.

    Parameters
    ----------
    scene : numba.typed.List(SceneObject)
        Numba typed list of SceneObject numba jitclasses.
        Each SceneObject requires data used for path tracing.
    """

    global render_start_time
    global sample_start_time
    global sample_count
    global finished_chunks
    global original_render
    global denoised_render

    render_start_time = time.perf_counter()

    sample_count = 0
    for i in range(ACCUM_SAMPLES):
        print(f"\n\n--- Sample {i+1} ---\n")

        finished_chunks = 0
        sample_count += RAY_SAMPLES
        
        sample_start_time = time.perf_counter()

        for chunk in CHUNKS:
            chunk_render = render_chunk(scene, chunk, RAY_SAMPLES, BOUNCES)
            return_render(chunk_render)

    print("\nTotal render time: {:.2f} seconds".format(time.perf_counter() - render_start_time))


def start():
    """
    Main function to start the render process.

    1) Compile the scene and njit functions (if not compiled).
    2) Render for all pixels and all samples with path tracing.
    3) Save noisy and denoised renders to saved_images/.
    """

    global denoised_render
    
    compilation_start_time = time.perf_counter()
    print("\n\nCompiling...")

    # Build the scene.
    from scene_builder import scene as SCENE

    # Force njit compilation of all njit functions with one call (if not stored with cache).
    compilation_chunk = np.array([[0, 0]])
    render_chunk(SCENE, compilation_chunk, 1, 1)

    compilation_time = time.perf_counter() - compilation_start_time
    print("\nCompilation finished: {:.2f}s".format(compilation_time))

    pygame.init()
    SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Path Tracer")

    # Start threading the render.
    # This allows the main thread to stay alive and prevent pygame freezing.
    thread = threading.Thread(target=render, args=[SCENE])

    # Close the threading process if the user exits.
    thread.daemon = True

    thread.start()

    clock = pygame.time.Clock()
    
    running = True
    # While waiting for the render, keep pygame responsive.
    while running:
        # Limit FPS of the display screen to save CPU utilization.
        clock.tick(15)

        # Check if the thread finished the process.
        if not thread.is_alive():
            running = False
            break
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        if original_render is not None:
            if DENOISE:
                if denoised_render is not None:
                    pygame.surfarray.blit_array(SCREEN, denoised_render)
            else:
                pygame.surfarray.blit_array(SCREEN, original_render)

            pygame.display.update()

    pygame.quit()

    # Required denoising for image saving won't previously update if the denoise setting is false.
    if not DENOISE:
        denoised_render = linear_to_srgb(aces_film(oidn_denoise(avg_light)))

    # Check if the thread finished the process.
    if not thread.is_alive():
        # Create and save images after complete render.
        # Swap axes because Pillow stores images as [height, width].
        original_render_image = Image.fromarray(np.swapaxes(original_render, 0, 1))
        denoised_render_image = Image.fromarray(np.swapaxes(denoised_render, 0, 1))

        original_render_image.save("main/saved_images/original_render.png")
        denoised_render_image.save("main/saved_images/denoised_render.png")

if __name__ == "__main__":
    start()