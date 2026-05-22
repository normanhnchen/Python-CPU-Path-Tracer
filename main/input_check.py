"""Check if the settings are invalid."""


import settings


settings_error = "SETTINGS ERROR:"


def check_settings():
    if type(settings.screen_dimensions) not in (tuple, list):
        raise TypeError(f"{settings_error} Screen dimensions must be a tuple or list.")

    if len(settings.screen_dimensions) != 2:
        raise ValueError(f"{settings_error} Screen dimensions must be 2d.")

    if type(settings.camera_fov) not in (float, int):
        raise TypeError(f"{settings_error} Camera FOV must be a real number.")
    
    if not (0 < settings.camera_fov < 180):
        raise ValueError(f"{settings_error} Camera FOV must be between 0 and 180 degrees (exclusive).")
    
    if type(settings.camera_focal_length) not in (int, float):
        raise TypeError(f"{settings_error} Camera focal length must be a positive number.")
    
    if settings.camera_focal_length <= 0:
        raise ValueError(f"{settings_error} Camera focal length must be positive.")

    if type(settings.camera_world_pos) not in (tuple, list):
        raise TypeError(f"{settings_error} Camera world position must be a tuple or list.")

    if len(settings.camera_world_pos) != 3:
        raise ValueError(f"{settings_error} Camera world position must be a 3d coordinate.")

    if type(settings.camera_rotation) not in (tuple, list) or len(settings.camera_rotation) != 3:
        raise TypeError(f"{settings_error} Camera rotation must be a tuple or list of 3 angles (pitch, yaw, roll).")
    
    if len(settings.camera_rotation) != 3:
        raise ValueError(f"{settings_error} Camera rotation must be of 3 angles (pitch, yaw, roll).")

    if type(settings.exposure) not in (int, float):
        raise ValueError(f"{settings_error} Exposure must be a positive real number.")
    
    if settings.exposure < 0:
        raise ValueError(f"{settings_error} Exposure must be a positive.")

    if type(settings.background_color) not in (tuple, list):
        raise TypeError(f"{settings_error} Background color must be a tuple or list of 3 float RGB values.")
    
    if len(settings.background_color) != 3:
        raise ValueError(f"{settings_error} Background color must be of 3 float RGB values.")

    for channel in settings.background_color:
        if not (0 <= channel <= 1.0):
            raise ValueError(f"{settings_error} Background color (float RGB) should be between 0 and 1.")

    if settings.ray_samples < 1:
        raise ValueError(f"{settings_error} Ray samples cannot be less than 1.")

    if settings.accumulation_samples < 1:
        raise ValueError(f"{settings_error} Accumulation samples cannot be less than 1.")

    if settings.bounces < 0:
        raise ValueError(f"{settings_error} Ray bounces cannot be less than 0.")

    if type(settings.denoise) != bool:
        raise TypeError(f"{settings_error} Denoise must be a boolean.")

    if type(settings.comparison_slider) != bool:
        raise TypeError(f"{settings_error} Comparison slider must be a boolean.")


if __name__ == "__main__":
    check_settings()