import numpy as np
from policies.mappo import MAPPO
from environment.optimization_environment import OptimizationEnv

def initialize(config_path, mode="train", **kwargs):
    env = OptimizationEnv(config_path)
    agent_policy = MAPPO(config_path)
    if mode == "test" or mode == "benchmark":
        model_path = kwargs.get("model_path", None)
        if model_path is None:
            raise ValueError("Model path must be provided for testing")
        agent_policy.load(model_path)
    return env, agent_policy

def print_items(**kwargs):
    for key, value in kwargs.items():
        print(key, value)
        
def get_action(observation_info, agent_policy, env):
    observation, observation_std = observation_info
    actions = np.zeros((env.n_agents, env.n_dim))
    for dim in range(env.n_dim):
        observation[dim] = observation[dim].astype(np.float32)
        observation_std[dim] = observation_std[dim].astype(np.float32)
        action = agent_policy.select_action(observation[dim], observation_std[dim])
        actions[:, dim] = action
    return actions

def get_informed_action(env):
    # let the action be the distance it takes for the agents to get to the a random point in the high std points
    actions = np.zeros((env.n_agents, env.n_dim))
    high_std_points, _ = env._get_unexplored_area()
    taken_points_index = []
    for agents in range(env.n_agents):
        while True:
            index = np.random.randint(0, high_std_points.shape[0])
            if index not in taken_points_index:
                taken_points_index.append(index)
                break
        agent_target = high_std_points[index]
        # get the difference between the agent's current position and the target
        diff = agent_target - env.state[agents][:env.n_dim]
        # add noise to the action
        actions[agents] = diff #+ np.random.normal(-0.1, 0.1, size=env.n_dim)

    return actions
        