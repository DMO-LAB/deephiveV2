from deephive.environment.reward_schemes import RewardScheme
import numpy as np

class SparseRewardScheme(RewardScheme):
    def compute_reward(self):
        reward = np.zeros(len(self.env.state))
        self._post_process_rewards(reward)
        return reward

    def _post_process_rewards(self, reward):
        # check if agents are stuck
        stuck_agents = self.env._get_stuck_agents(threshold=2)
        if len(stuck_agents) > 0:
            reward[stuck_agents] = -1
        # check if agents are optimal
        if np.any(self.env.state[:, -1] >= self.env.opt_bound):
            reward[self.env.state[:, -1] >= self.env.opt_bound] += 10 * \
                self.env.state[self.env.state[:, -1] >= self.env.opt_bound, -1]
        if self.env.done:
            reward[self.env.state[:, -1] >= self.env.opt_bound] += 10
        # freeze best agent
        if self.env.freeze:
            reward[self.env.bestAgent] = 0
        return reward
    
    