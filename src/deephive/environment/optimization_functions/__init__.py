import numpy as np
from deephive.environment.commons.heat_exchanger import HeatExchanger
from typing import Tuple

class OptimizationFunctionBase:
    """ Base class for optimization functions. """
    def __init__(self, log_dir: str = None, name: str = None, minimize: bool = False):
        self.tracer = EvaluationTracker(log_dir, name, minimize)
    
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
    
    
class EvaluationTracker:
    """ A class to track the evaluation of an optimization function. """
    def __init__(self, log_dir: str = None, name: str = None, minimize: bool = False):
        """
        Args:
            log_dir (str): The directory to save the log.
            name (str): The name of the optimization function.
            minimize (bool, optional): Whether to minimize the function. Defaults to True.
        """
        
        self.log_dir = log_dir if log_dir is not None else "logs/"
        self.name = name if name is not None else "optimization_function"
        self.minimize = minimize 
        self.reset()
        
    def update(self, params: np.ndarray, values: np.ndarray):
        """
        Update the evaluation tracker.
        Args:
            params (np.ndarray): The parameters.
            values (np.ndarray): The values.
        """
        self.evaluations["evaluation_count"].append(len(self.evaluations["evaluation_count"]))
        self.evaluations["params"].append(params)
        self.evaluations["values"].append(values)
        if self.minimize:
            if values < self.best_value:
                self.best_value = values
                self.best_params = params
        else:
            if values > self.best_value:
                self.best_value = values
                self.best_params = params
                
    def save(self):
        """ Save the evaluation tracker. """
        np.savez_compressed(self.log_dir + self.name + ".npz", **self.evaluations)
        
    def reset(self):
        """ Reset the evaluation tracker. """
        self.evaluations = {
            "evaluation_count":[],
            "params": [],
            "values": []
        }
        self.best_value = np.inf if self.minimize else -np.inf
        self.best_params = None
        
        
        
    
    
        