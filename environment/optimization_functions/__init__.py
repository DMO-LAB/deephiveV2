import numpy as np
from environment.commons.heat_exchanger import HeatExchanger
from typing import Tuple

class OptimizationFunctionBase:
    """ Base class for optimization functions. """
    def __init__(self):
        pass
    
    def evaluate(self, params: np.ndarray) -> np.ndarray:
        """
        Evaluate the optimization function.
        Args:
            params (np.ndarray): Parameters for the optimization function.
        Returns:
            np.ndarray: Function values.
        """
        raise NotImplementedError("The `evaluate` method should be implemented in subclasses.")
        
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the bounds of the optimization function.
        Returns:
            Tuple[np.ndarray, np.ndarray]: The lower and upper bounds.
        """
        raise NotImplementedError("The `bounds` method should be implemented in subclasses.")

    def optimal_value(self) -> float:
        """
        Return the optimal value of the optimization function.
        Returns:
            float: The optimal value.
        """
        raise NotImplementedError("The `optimal_value` method should be implemented in subclasses.")