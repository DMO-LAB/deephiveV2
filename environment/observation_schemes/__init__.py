from abc import ABC, abstractmethod
import numpy as np


class ObservationScheme(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def generate_observation(self, *args, **kwargs):
        pass