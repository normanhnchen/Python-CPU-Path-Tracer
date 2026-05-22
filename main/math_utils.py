"""Includes common math functions."""


import numpy as np
from numba import njit


# Used to prevent self-intersection artifacts by accommodating for floating point errors and division by zero.
EPSILON = 1e-6


@njit(cache=True)
def normalize(vector):
    """Normalize a vector to unit length."""

    magnitude = np.linalg.norm(vector)
    return vector / magnitude


@njit(cache=True)
def lerp(start, end, factor):
    """Linearly interpolate between start and end by a factor."""
    
    return start + factor * (end - start)