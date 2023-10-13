from environment.reward_schemes import RewardScheme
import numpy as np

class CustomRewardScheme(RewardScheme):
    def __init__(self, env, exploration_bonus=0.1, optimal_bonus=1.0, stagnation_penalty=-0.1, boundary_penalty=-0.5, global_improvement_bonus=0.5):
        super().__init__(env)
        self.exploration_bonus = exploration_bonus
        self.optimal_bonus = optimal_bonus
        self.stagnation_penalty = stagnation_penalty
        self.boundary_penalty = boundary_penalty
        self.global_improvement_bonus = global_improvement_bonus

    def compute_reward(self):
        rewards = np.zeros(self.env.n_agents)
        
        # Improvement Reward
        if self.env.optimization_type == "minimize":
            improvement_reward = self.env.prev_obj_values - self.env.obj_values
        else:
            improvement_reward = self.env.obj_values - self.env.prev_obj_values
        rewards += improvement_reward
        
        # Exploration Bonus
        new_area_explored = self._check_new_area_explored()
        rewards += self.exploration_bonus * new_area_explored.astype(int)
        
        # Optimal Solution Bonus
        if self.env.optimization_type == "minimize":
            optimal_condition = self.env.obj_values <= self.env.opt_value
        else:
            optimal_condition = self.env.obj_values >= self.env.opt_value
        optimal_bonus = self.optimal_bonus * optimal_condition.astype(int)
        rewards += optimal_bonus
        
        # Stagnation Penalty
        agent_stagnated = self._check_agent_stagnation()
        rewards += self.stagnation_penalty * agent_stagnated.astype(int)
        
        # Boundaries Penalty
        outside_boundaries = self.env._check_boundary_violations()
        rewards += self.boundary_penalty * outside_boundaries.astype(int)
        
        # Global Improvement Reward
        improved_global_best = self._check_global_improvement()
        #print(improved_global_best, self.global_improvement_bonus)
        rewards += self.global_improvement_bonus * improved_global_best.astype(int)
        
        return rewards

    def _check_global_improvement(self):
        # check if the global best improved
        previous_gbest = self.env.gbest_history[-2, -1]
        current_gbest = self.env.gbest_history[-1, -1]
        
        if self.env.optimization_type == "minimize":
            return (current_gbest < previous_gbest).astype(int)
        else:
            return (current_gbest > previous_gbest).astype(int)

    
    def _check_new_area_explored(self, threshold=0.1):
        # check if the distance between the current state and the previous state is greater than a threshold
        previous_state = self.env.state_history[:, -2, :-1]
        current_state = self.env.state_history[:, -1, :-1]
        
        # measure the euclidean distance between the two states for each agent
        distances = np.linalg.norm(previous_state - current_state, axis=1)
        
        # return a boolean array indicating whether the distance is greater than the threshold
        return distances > threshold
    
    def _check_agent_stagnation(self, threshold=2):
        # check if the agents are stuck
        stuck_agents = self.env._get_stuck_agents(threshold)
        
        # return a boolean array indicating whether the agent is in the stuck_agents list or not
        return np.isin(np.arange(self.env.n_agents), stuck_agents)
    
        
        
        