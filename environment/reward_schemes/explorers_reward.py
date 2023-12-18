import numpy as np
from environment.reward_schemes import RewardScheme

class ExplorersRewardScheme(RewardScheme):
    def __init__(self, env, stuck_reward=-2, optimal_reward=10, frozen_best_reward=0):
        super().__init__(env)
        self.stuck_reward = stuck_reward
        self.optimal_reward = optimal_reward
        self.frozen_best_reward = frozen_best_reward

    def compute_reward(self):
        """
        Computes rewards based on the environment's objective values.
        Rewards are negated if the optimization type is minimize.
        Calls _post_process_rewards for further processing based on agent statuses.
        """
        # Ensure required attributes and methods are present in env
        required_attrs = [ 'surrogate']
        required_methods = ['_get_stuck_agents', '_get_optimal_agents']
        for attr in required_attrs:
            assert hasattr(self.env, attr), f"Environment missing required attribute: {attr}"
        for method in required_methods:
            assert callable(getattr(self.env, method, None)), f"Environment missing required method: {method}"

        surrogate_reward = self.env.surrogate.cal_reward()
        # propagate the surrogate reward to the environment
        reward = [surrogate_reward for _ in range(self.env.n_agents)]
        reward = self._post_process_rewards(np.array(reward))
        # reward += -np.log(1 - self.env.state[:, -1] + 0.001)
        return reward
    
    def _post_process_rewards(self, reward):
        """
        Post processes rewards based on certain conditions:
        - Identifies stuck and optimal agents to assign specific rewards.
        - If the environment is frozen, assigns a specific reward to the best agent.
        """
        # Check if agents are stuck
        stuck_agents = self.env._get_stuck_agents(threshold=2)
        for agent in stuck_agents:
            if self.env.prev_state[agent, -1] == self.env.state[agent, -1]:
                reward[agent] = self.stuck_reward
            else:
                pass

        if self.env.current_step >= self.env.ep_length:
            # get percentage of agent swith high std from surrogate
            reward -= (self.env.surrogate.percent_high_std / 100) * 2
         
        return reward


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
    
        
        
        