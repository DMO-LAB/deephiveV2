from abc import ABC, abstractmethod

class RewardScheme(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def compute_reward(self, *args, **kwargs):
        pass
    
    
