"""
Used by the path tracer to calculate physically-based rendering (PBR).

The module updates color and light accumulation stored by the path tracer and computes the bounce
ray direction based on the object's material.

Refer to docs/pbr.md for more information
"""


import numpy as np
from numba import njit

import main.math_utils as math_utils


@njit(cache=True)
def diffuse_reflect(surface_normal):
    """
    Generate a random diffuse reflection direction.

    Sample a random direction from a normal distribution and normalize it. If the resulting
    direction points inward (opposite the surface normal), it is flipped to stay above the surface.

    Parameters
    ----------
    surface_normal : np.ndarray
        The surface normal vector, shape (3,).

    Returns
    -------
    np.ndarray
        A random bounce direction.
    
    References
    ----------
    https://learnopengl.com/PBR/Lighting
    """

    # Sample a normalizd random direction for diffuse bounce.
    bounce_dir = math_utils.normalize(np.random.normal(0, 1, size=3))

    # Prevent self-intersection by ensuring the bounce points outward.
    if np.dot(surface_normal, bounce_dir) < 0:
        bounce_dir *= -1
    
    return bounce_dir


@njit(cache=True)
def specular_reflect(ray_dir, surface_normal, roughness):
    """
    Find the specular reflection ray direcion.

    The specular direction is a perfect, mirror-like reflection of the incident ray direction. The
    output direction is influenced by the material's surface roughness which adds a random offset
    to the specular ray direction, creating a blur effect.

    Parameters
    ----------
    ray_dir : np.ndarray
        The ray direction vector, shape (3,).
    surface_normal : np.ndarray
        The surface normal vector, shape (3,).
    roughness : float
        The material's roughness, controlling the specular reflection's blur.

    Returns
    -------
    np.ndarray
        The bounce direction influenced by the material roughness.
    
    References
    ----------
    https://www.bluebill.net/2021/vector_reflection.html
    """

    # Find the perfect mirror reflection direction.
    specular_dir = ray_dir - 2 * np.dot(ray_dir, surface_normal) * surface_normal
    
    # Calculate blurred reflection depending on surface roughness.
    roughness_dir = math_utils.normalize(specular_dir + np.random.normal(0, 1, size=3) * roughness)

    # Prevent self-intersection by ensuring the bounce points outward.
    if np.dot(surface_normal, roughness_dir) < 0:
        roughness_dir *= -1
    
    return roughness_dir


@njit(cache=True)
def fresnel_schlick(ray_dir, surface_normal, base_reflectivity):
    """
    Calculate an object's tendency to reflect (reflectance) using Schlick's approximation of how
    reflectance behaves in real-world physics.

    Calculate Schlick's approximation using a material's base reflectivity, it's reflective
    qualities without any lighting influence.

    Parameters
    ----------
    ray_dir : np.ndarray
        The ray direction vector, shape (3,).
    surface_normal : np.ndarray
        The surface normal vector, shape (3,).
    base_reflectivity : np.ndarray
        The base reflectivity of a material, shape (3,).

    Returns
    -------
    np.ndarray
        The RGB material reflectance value, shape (3,).

    Notes
    -----
    Check docs/derivations.md section "Fresnel-Schlick" for more information.
    
    References
    ----------
    https://learnopengl.com/PBR/Lighting
    """
    
    # Schlick approximation for material reflectance.
    return base_reflectivity + (1 - base_reflectivity) * min(max((1 - np.dot(surface_normal, -ray_dir)), 0), 1) ** 5


@njit(cache=True)
def calculate_pbr(ray_dir, surface_normal, object, ray_color, accum_light):
    """
    Physically-based rendering is a process which calculates how lighting interacts with a
    material and closely matches how light behaves in real life.

    Calculates influence from diffuse/specular reflection, Fresnel reflectance, and updates ray
    color and total light accumulation.

    Parameters
    ----------
    ray_dir : np.ndarray
        The ray direction vector, shape (3,).
    surface_normal : np.ndarray
        The surface normal vector, shape (3,).
    object : SceneObject
        Numba jitclass SceneObject, containing data used for PBR.
    ray_color : np.ndarray
        The initial ray RGB color, shape (3,).
    accum_light : np.ndarray
        The inital total accumulated RGB light after hitting the
        material, shape (3,).
    
    Returns
    -------
    ray_dir : np.ndarray
        The bounce ray direction vector, shape (3,).
    ray_color : np.ndarray
        The ray added RGB color after hitting the material, shape (3,).
    accum_light : np.ndarray
        The total accumulated RGB light after hitting the material,
        shape (3,).

    References
    ----------
    https://pema.dev/obsidian/math/light-transport/cosine-weighted-sampling.html
    """

    object_material = object.material
    base_color = object_material.base_color
    roughness = object_material.roughness
    metalness = object_material.metalness
    emissive_strength = object_material.emissive_strength
    radiance = emissive_strength / object.surface_area
    emission = radiance * base_color

    if emissive_strength > 0:
        accum_light += emission * ray_color
        return ray_dir, ray_color, accum_light
    
    # Metals reflect the base color, dielectrics reflect ~0.04.
    base_reflectivity = math_utils.lerp(np.full(3, 0.04), base_color, metalness)

    reflectance = fresnel_schlick(ray_dir, surface_normal, base_reflectivity)
    reflectance_probability = np.mean(reflectance)
    # How much light doesn't get reflected (transmitted).
    transmission = 1 - reflectance

    # Linear interpolate between a non-dialectric white reflection and a metal tinted reflection.
    specular_color = math_utils.lerp(np.ones(3), base_color, metalness)
    reflection_color = reflectance * specular_color
    transmission_color = transmission * base_color
    
    if np.random.rand() < reflectance_probability:
        ray_dir = specular_reflect(ray_dir, surface_normal, roughness)
    else:
        # Shortcut trick to distributing diffuse bounce rays.
        ray_dir = math_utils.normalize(surface_normal + diffuse_reflect(surface_normal))
    
    ray_color *= reflection_color + transmission_color

    accum_light += emission * ray_color

    return ray_dir, ray_color, accum_light
