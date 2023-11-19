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

# Rosenbrock Function
def rosenbrock_function(params: np.ndarray) -> np.ndarray:
    return -np.sum(100.0 * (params[:, 1:] - params[:, :-1]**2.0)**2.0 + (1 - params[:, :-1])**2.0, axis=1)

def ackley_function(params: np.ndarray) -> np.ndarray:
    first_term = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(params**2, axis=1)))
    second_term = -np.exp(np.mean(np.cos(2.0 * np.pi * params), axis=1))
    return -(first_term + second_term + 20 + np.e)

def gaussian_peak(params: np.ndarray) -> np.ndarray:
    """
    Another test function with distinct local maxima and a clear global maximum.
    Args:
        params (np.ndarray): Parameters for the test function.
    Returns:
        np.ndarray: Function values.
    """
    x, y = params[:, 0], params[:, 1]
    return (
        np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) +
        4 * np.exp(-((x - 0.8)**2 + (y - 0.8)**2)) +  # Global maximum
        3 * np.exp(-((x + 0.8)**2 + (y + 0.8)**2)) +  # Local maximum
        3 * np.exp(-((x - 0.8)**2 + (y + 0.8)**2)) +  # Local maximum
        3 * np.exp(-((x + 0.8)**2 + (y - 0.8)**2))    # Local maximum
    ) / (1 + 0.3 * (x**2 + y**2))

