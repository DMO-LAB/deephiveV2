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
    # reshape params to 2D array
    # params = params.reshape(-1, 1)
    
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

def schaffers_f7_function(params: np.ndarray, f_opt: float = 0) -> np.ndarray:
    """
    Schaffer's F7 Function.
    
    Args:
        params (np.ndarray): Parameters for the Schaffer's F7 function.
        f_opt (float): Optimal function value shift.
    
    Returns:
        np.ndarray: Function values.
    """
    D = params.shape[1]  # Dimensionality of the input
    
    # Calculate s_i
    z = params  # Assuming z = x for simplicity
    s_i = np.sqrt(z[:, :-1]**2 + z[:, 1:]**2)
    
    # Calculate the function value
    term = (np.sqrt(s_i) + np.sqrt(s_i) * np.sin(50 * s_i**0.2)**2)**2
    result = np.mean(term, axis=1) + f_opt  # Assuming f_opt includes f_pen(x) if necessary
    
    return -result

def bent_cigar_function(params: np.ndarray, f_opt: float = 0) -> np.ndarray:
    """
    Bent Cigar Function.
    
    Args:
        params (np.ndarray): Parameters for the Bent Cigar function.
        f_opt (float): Optimal function value shift.
    
    Returns:
        np.ndarray: Function values.
    """
    # Assuming z = x - x_opt and x_opt = 0 for simplicity
    z = params  # This would be different if rotation and asymmetry were applied
    
    # Calculate the function value
    term1 = z[:, 0]**2
    term2 = 10**6 * np.sum(z[:, 1:]**2, axis=1)
    result = term1 + term2 + f_opt
    
    return -result

def rosenbrock_function_modified(params: np.ndarray, x_opt: np.ndarray = None, f_opt: float = 0) -> np.ndarray:
    if x_opt is None:
        x_opt = np.ones(params.shape[1])  # Assuming x_opt = 1 for all dimensions
    
    D = params.shape[1]

    # Compute the scaling factor
    scaling_factor = max(1, np.sqrt(D / 8))
    
    # Transform x to z
    z = scaling_factor * (params - x_opt) + 1
    
    # Apply the Rosenbrock formula
    summands = 100.0 * (z[:, :-1]**2 - z[:, 1:])**2 + (z[:, :-1] - 1)**2
    result = np.sum(summands, axis=1)
    
    # Since you're maximizing, negate the function (subtract from f_opt if used)
    return -(result + f_opt)
