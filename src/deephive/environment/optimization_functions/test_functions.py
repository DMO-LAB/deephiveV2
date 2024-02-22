from deephive.environment.optimization_functions import OptimizationFunctionBase
import numpy as np
from deephive.environment.commons.heat_exchanger import HeatExchanger, params
from deephive.environment.commons.objective_functions import sphere_function, cosine_mixture, ackley_function, rosenbrock_function, gaussian_peak
from typing import Tuple, Callable, Dict, Any
import sys 
from deephive.environment.optimization_functions.cec2017 import functions

all_functions = functions.all_functions


class Tracker():
    """ A class to track the number of times a function is called. """
    def __init__(self):
        self.count = 0
        self.function_values = []
        
    def __call__(self, z):
        self.count += 1
        self.function_values.append(z)
    
    def reset(self):
        self.count = 0
        self.function_values = []
        
    
class HeatExchangerFunction(OptimizationFunctionBase):
    def __init__(self, **kwargs):
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
        
        self.tracker = Tracker()
    
    def _get_model(self) -> HeatExchanger:
        return HeatExchanger(params=self.params)

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        model = self._get_model()
        z = model.objective_function(params)
        self.tracker(z)
        return z

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([0.015, 0.1, 0.05]), np.array([0.051, 1.5, 0.5])
    
    def optimal_value(self, dim) -> float:
        return None
    
    def __str__(self):
        return "Heat Exchanger Function"
    
class SphereFunction(OptimizationFunctionBase):
    def __init__(self, **kwargs):
        self.tracker = Tracker()

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        z = sphere_function(params)
        self.tracker(z)
        return z

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-100 for _ in range(dim)]), np.array([100 for _ in range(dim)])
    
    def optimal_value(self, dim) -> float:
        return 0 * dim
    
    def __str__(self):
        return "Sphere Function"
    

class CosineMixtureFunction(OptimizationFunctionBase):
    def __init__(self, **kwargs):
        self.tracker = Tracker()

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        z = cosine_mixture(params)
        self.tracker(z)
        return z

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-1 for _ in range(dim)]), np.array([1 for _ in range(dim)])
    
    def optimal_value(self, dim) -> float:
        return 0.1 * dim
    
    def __str__(self):
        return "Cosine Mixture Function"
    
class ShiftedCosineMixtureFunction:
    def __init__(self, shift: np.ndarray= None, **kwargs):
        if shift is None:
            # set random shift
            self.shift = np.round(np.random.uniform(-0.5, 0.5), 2)
        else:
            self.shift = shift
            
        self.tracker = Tracker()

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        params = params - self.shift
        z = cosine_mixture(params)
        self.tracker(z)
        return z

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

    def __str__(self):
        return "Shifted Cosine Mixture Function"

    
class AckleyFunction(OptimizationFunctionBase):
    def __init__(self, **kwargs):
        self.tracker = Tracker()

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        z = ackley_function(params)
        self.tracker(z)
        return z

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-5 for _ in range(dim)]), np.array([5 for _ in range(dim)])
    
    def optimal_value(self, dim) -> float:
        return 0 * dim
    
    def __str__(self):
        return "Ackley Function"

class RosenbrockFunction(OptimizationFunctionBase):
    def __init__(self, **kwargs):
        self.tracker = Tracker()
        

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        z = rosenbrock_function(params)
        self.tracker(z)
        return z

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-5 for _ in range(dim)]), np.array([5 for _ in range(dim)])
    
    def optimal_value(self, dim) -> float:
        return 0 * dim
    
    def __str__(self):
        return "Rosenbrock Function"
    
class GaussianPeakFunction(OptimizationFunctionBase):
    def __init__(self, mininize: bool = False, **kwargs):
        super().__init__(minimize=mininize)
        self.minimize = mininize
        self.tracker = Tracker()

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        z = gaussian_peak(params)
        self.tracker(z)
        return z

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-1 for _ in range(dim)]), np.array([1 for _ in range(dim)])
    
    def optimal_value(self, dim) -> float:
        return 4.808
    
    def __str__(self):
        return "Gaussian Peak Function"
    
    
class CEC17(OptimizationFunctionBase):
    def __init__(self, function_id: int, negative: bool = True, **kwargs):
        self.tracker = Tracker()
        self.function_id = function_id
        self.negative = negative

    def evaluate(self, params: np.ndarray) -> np.ndarray:
        z = all_functions[self.function_id](params, negative=self.negative)
        self.tracker(z)
        return z

    def bounds(self, dim) -> Tuple[np.ndarray, np.ndarray]:
        return np.array([-100 for _ in range(dim)]), np.array([100 for _ in range(dim)])
    
    def optimal_value(self, dim) -> float:
        return -100 * (self.function_id+1)  # 100 * (i+1) where i is the function_id
    
    def __str__(self):
        return f"CEC17 Function {self.function_id}"