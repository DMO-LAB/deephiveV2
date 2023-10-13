import numpy as np
from typing import Tuple


def sphere_function(params: np.ndarray) -> np.ndarray:
    """
    Sphere function.
    Args:
        params (np.ndarray): Parameters for the sphere function.
    Returns:
        np.ndarray: Function values.
    """
    return np.sum(params**2, axis=1)


def cosine_mixture(params: np.ndarray) -> np.ndarray:
    """
    Cosine mixture function.
    Args:
        params (np.ndarray): Parameters for the cosine mixture function.
    Returns:
        np.ndarray: Function values.
    """
    
    return 0.1 * np.sum(np.cos(5 * np.pi * params), axis=1) - np.sum(params**2, axis=1)

