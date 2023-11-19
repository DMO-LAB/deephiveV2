from environment.optimization_functions import OptimizationFunctionBase
import numpy as np
from environment.commons.heat_exchanger import HeatExchanger, params
from environment.commons.objective_functions import sphere_function, cosine_mixture, ackley_function, rosenbrock_function, gaussian_peak
from typing import Tuple, Callable, Dict, Any


    
class HeatExchangerFunction(OptimizationFunctionBase):
    def __init__(self):
        self.params = {
                'tube': {
                    'mt': 68.90,  # mass flow rate of tube side fluid (kg/s)
                    'Tci': 25,  # cold fluid inlet temperature (C)
                    'Tco': 40,  # cold fluid outlet temperature (C)
                    'rhot': 995.0,  # density of tube side fluid (kg/m^3)
                    'mut': 0.0008,  # viscosity of tube side fluid (kg/m.s)
                    'muwt': 0.00052,# viscosity of tube side fluid at wall temperature (kg/m.s)
                    'cpt': 4.2 * 1000,  # specific heat capacity of tube side fluid (kJ/kg.K)
                    'kt': 0.59,  # thermal conductivity of tube side fluid (W/m.K)
                    'Rft': 0.0002,  # fouling resistance of tube side fluid (m^2.K/W)
                },
                'shell': {
                    'ms': 27.8,  # mass flow rate of shell side fluid (kg/s)
                    'Thi': 95,  # hot fluid inlet temperature (C)
                    'Tho': 40,  # hot fluid outlet temperature (C)
                    'rhos': 750,  # density of shell side fluid (kg/m^3)
                    'cps': 2.84 * 1000,  # specific heat capacity of shell side fluid (kJ/kg.K)
                    'mews': 0.00034,  # viscosity of shell side fluid (kg/m.s)
                    'mewws': 0.00038, # viscosity of shell side fluid at wall temperature (kg/m.s)
                    'ks': 0.19,  # thermal conductivity of shell side fluid (W/m.K)
                    'Rfs': 0.00033,  # fouling resistance of shell side fluid (m^2.K/W)
                },
                'constants': {
                    'a1':8000,  # numerical constant
                    'a2':259.2,  # numerical constant
                    'a3':0.91,  # numerical constant
                    'p': 4,  # constant
                    'i': 0.1, # constant
                    'C': 0.319, # constant
                    'n1': 2.142, # constant
                },
                'n': 2,  # number of passes
                'eff': 0.8,  # pump efficiency
                'L': 3.115,  # tube length (m)
                'ny': 10, # number of years
                'H': 7000, # number of hours
                'ce': 0.00012 # cost of electricity per kWh

            } 
    
    def _get_model(self) -> HeatExchanger:
        return HeatExchanger(params=self.params)

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        model = self._get_model()
        return model.objective_function(params)

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([0.015, 0.1, 0.05]), np.array([0.051, 1.5, 0.5])
    
    def optimal_value(self, dim) -> float:
        return None
    
    
class SphereFunction(OptimizationFunctionBase):
    def __init__(self):
        pass

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        return sphere_function(params)

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-100 for _ in range(dim)]), np.array([100 for _ in range(dim)])
    
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
    
class ShiftedCosineMixtureFunction:
    def __init__(self, shift: np.ndarray= None):
        if shift is None:
            # set random shift
            self.shift = np.round(np.random.uniform(-0.5, 0.5), 2)
        else:
            self.shift = shift

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        params = params - self.shift
        return self.cosine_mixture(params)

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        # Assuming the bounds should also be shifted
        lower_bound = np.array([-1 for _ in range(dim)]) + self.shift
        upper_bound = np.array([1 for _ in range(dim)]) + self.shift
        return lower_bound, upper_bound
    
    def optimal_value(self, dim) -> float:
        # Assuming the optimal value is still 0.1 * dim after the shift
        return 0.1 * dim
    
    def cosine_mixture(self, params: np.ndarray) -> np.ndarray:
        """
        Shifted cosine mixture function.
        Args:
            params (np.ndarray): Parameters for the cosine mixture function.
        Returns:
            np.ndarray: Function values.
        """
        # Shift the parameters
        shifted_params = params - self.shift
        return 0.1 * np.sum(np.cos(5 * np.pi * shifted_params), axis=1) - np.sum(shifted_params**2, axis=1)

    
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
    
class GaussianPeakFunction(OptimizationFunctionBase):
    def __init__(self):
        pass

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        return gaussian_peak(params)

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-1 for _ in range(dim)]), np.array([1 for _ in range(dim)])
    
    def optimal_value(self, dim) -> float:
        return 4.808