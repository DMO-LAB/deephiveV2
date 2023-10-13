import unittest
import numpy as np
from environment.optimization_environment import OptimizationEnv
from environment.reward_schemes.full_reward import FullRewardScheme


class TestOptimizationEnv(unittest.TestCase):

    def setUp(self):
        self.optimizer = OptimizationEnv("config/env_config.json")
        self.reward_scheme = FullRewardScheme(self.optimizer)

    def test_reset(self):
        state, obs = self.optimizer.reset()
        self.assertEqual(state.shape, (self.optimizer.n_agents, self.optimizer.n_dim + 1))
        self.assertTrue((state >= 0).all() and (state <= 1).all())  # Assuming state values are normalized to [0, 1]

    def test_step(self):
        self.optimizer.reset()
        actions = np.random.uniform(low=-0.5, high=0.5, size=(self.optimizer.n_agents, self.optimizer.n_dim))
        observation, reward, done, info = self.optimizer.step(actions)
        # Ensure that the shapes of returned arrays are correct
        self.assertEqual(reward.shape, (self.optimizer.n_agents,))

    def test_pbest_gbest_update(self):
        self.optimizer.reset()
        for _ in range(10):
            actions = np.random.uniform(low=-2, high=2, size=(self.optimizer.n_agents, self.optimizer.n_dim))
            self.optimizer.step(actions)
        pbest = self.optimizer.pbest
        pbest_actual = self.optimizer._get_actual_state(pbest)
        gbest = self.optimizer.gbest
        gbest_actual = self.optimizer._get_actual_state(gbest.reshape(1, -1))
        
        self.assertEqual(pbest.shape, (self.optimizer.n_agents, self.optimizer.n_dim + 1))
        self.assertEqual(gbest.shape, (self.optimizer.n_dim + 1,))
        # Ensure pbest and gbest values are within bounds
        self.assertTrue((pbest >= 0).all() and (pbest <= 1).all())
        self.assertTrue((gbest >= 0).all() and (gbest <= 1).all())
        
        # Ensure that pbest and gbest values are updated correctly
        if self.optimizer.optimization_type == "minimize":
            self.assertTrue((pbest_actual[:, -1] <= self.optimizer._get_actual_state()[:, -1]).all())
            self.assertTrue((gbest_actual[:, -1] <= self.optimizer._get_actual_state()[:, -1]).all())
            print(pbest_actual[:, -1]), print(self.optimizer._get_actual_state()[:, -1])
            print(gbest_actual[:, -1]), print(self.optimizer._get_actual_state()[:, -1])
        else:
            self.assertTrue((pbest_actual[:, -1] >= self.optimizer._get_actual_state()[:, -1]).all())
            self.assertTrue((gbest_actual[:, -1] >= self.optimizer._get_actual_state()[:, -1]).all())
            print(pbest_actual[:, -1]), print(self.optimizer._get_actual_state()[:, -1])
            print(gbest_actual[:, -1]), print(self.optimizer._get_actual_state()[:, -1])
            
    def test_stuck_agents(self):
        # Manually set state_history and obj_values to simulate a scenario
        self.optimizer.reset()
        self.optimizer.current_step = 5
        self.optimizer.state_history[:, -2, -1] = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.optimizer.state_history[:, -1, -1] = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.optimizer.obj_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        self.optimizer.best_agent = 0  # Assuming the first agent is the best
        stuck_agents = self.optimizer._get_stuck_agents()
        self.assertListEqual(stuck_agents, [1, 2, 3, 4])  # All agents except the best one should be stuck

        # Reset and simulate no stuck agents scenario
        self.optimizer.reset()
        self.optimizer.current_step = 2
        stuck_agents = self.optimizer._get_stuck_agents()
        self.assertListEqual(stuck_agents, [])  # No agents should be stuck


    def test_compute_reward(self):
        # Set up a scenario in the environment
        self.optimizer.reset()
        self.optimizer.step(np.random.rand(self.optimizer.n_agents, self.optimizer.n_dim))

        # Call the method under test
        reward = self.reward_scheme.compute_reward()

        # Check the type and shape of the reward
        self.assertIsInstance(reward, np.ndarray)
        self.assertEqual(reward.shape, (self.optimizer.n_agents,))

        # Further tests could include checking the values of the reward,
        # especially for known scenarios such as when agents are stuck or optimal.

    def test_post_process_rewards(self):
        # Assume a basic reward array
        initial_reward = np.zeros(self.optimizer.n_agents)

        # Mocking methods to simulate conditions (you could set up real scenarios instead)
        self.optimizer._get_stuck_agents = lambda threshold: [0, 1]
        self.optimizer._get_optimal_agents = lambda: [2, 3]
        self.optimizer.freeze = True
        self.optimizer.best_agent = 4

        # Call the method under test
        processed_reward = self.reward_scheme._post_process_rewards(initial_reward)

        # Check the rewards for various conditions
        self.assertEqual(processed_reward[0], self.reward_scheme.stuck_reward)
        self.assertEqual(processed_reward[1], self.reward_scheme.stuck_reward)
        self.assertEqual(processed_reward[2], self.reward_scheme.optimal_reward)
        self.assertEqual(processed_reward[3], self.reward_scheme.optimal_reward)
        self.assertEqual(processed_reward[4], self.reward_scheme.frozen_best_reward)

if __name__ == '__main__':
    unittest.main()


        


if __name__ == "__main__":
    unittest.main()
