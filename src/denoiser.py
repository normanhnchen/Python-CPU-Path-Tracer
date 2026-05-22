"""Denoises the path tracer's output buffer using Intel Open Image Denoise."""


import numpy as np
import pyoidn


def oidn_denoise(render):
    """
    Denoise an image using Intel Open Image Denoise (OIDN).

    Parameters
    ----------
    render : np.ndarray
        RGB image buffer, shape (height, width, 3), dtype float32.

    Returns
    -------
    np.ndarray
        Denoised image buffer, shape (height, width, 3), dtype float32.

    References
    ----------
    https://pypi.org/project/pyoidn/
    https://www.openimagedenoise.org/
    """

    # Output buffer for the denoised image (float32 required by OIDN).
    denoised_render = np.zeros_like(render, dtype=np.float32)

    # Initialize and commit an OIDN device (CPU-based by default).
    device = pyoidn.Device()
    device.commit()

    # Create a ray-tracing denoising filter.
    filter = pyoidn.Filter(device, "RT")
    # Set the noisy input image.
    filter.set_image(pyoidn.OIDN_IMAGE_COLOR, render, pyoidn.OIDN_FORMAT_FLOAT3)
    # Set the output buffer.
    filter.set_image(pyoidn.OIDN_IMAGE_OUTPUT, denoised_render, pyoidn.OIDN_FORMAT_FLOAT3)

    # Finalize and execute denoising.
    filter.commit()
    filter.execute()

    # Release OIDN resources.
    filter.release()
    device.release()
    
    return denoised_render