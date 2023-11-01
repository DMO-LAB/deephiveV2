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
