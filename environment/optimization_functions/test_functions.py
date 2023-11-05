from environment.optimization_functions import OptimizationFunctionBase
import numpy as np
from environment.commons.heat_exchanger import HeatExchanger, params
from environment.commons.objective_functions import sphere_function, cosine_mixture, ackley_function, rosenbrock_function
from typing import Tuple, Callable, Dict, Any


    
class HeatExchangerFunction(OptimizationFunctionBase):
    def __init__(self):
        pass 
    
    def _get_model(self, params: np.ndarray) -> HeatExchanger:
        return HeatExchanger(params=params, n_dim=3)

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        self._get_model(params)
        return self.model.objective_function(params)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([0.015, 0.1, 0.05]), np.array([0.051, 1.5, 0.5])
    
    def optimal_value(self, dim) -> float:
        return None
    
    
class SphereFunction(OptimizationFunctionBase):
    def __init__(self):
        pass

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        return sphere_function(params)

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-5 for _ in range(dim)]), np.array([5 for _ in range(dim)])
    
    def optimal_value(self, dim) -> float:
        return 0 * dim
    

class CosineMixtureFunction(OptimizationFunctionBase):
    def __init__(self):
        pass

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        return cosine_mixture(params)

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-1 for _ in range(dim)]), np.array([1 for _ in range(dim)])
    
    def optimal_value(self, dim) -> float:
        return 0.1 * dim
    
class AckleyFunction(OptimizationFunctionBase):
    def __init__(self):
        pass

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        return ackley_function(params)

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-5 for _ in range(dim)]), np.array([5 for _ in range(dim)])
    
    def optimal_value(self, dim) -> float:
        return 0 * dim
    

class RosenbrockFunction(OptimizationFunctionBase):
    def __init__(self):
        pass

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        return rosenbrock_function(params)

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-5 for _ in range(dim)]), np.array([5 for _ in range(dim)])
    
    def optimal_value(self, dim) -> float:
        return 0 * dim