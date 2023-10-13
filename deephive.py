from environment.optimization_environment import OptimizationEnv
print("Imported OptimizationEnv")   
from policies import mappo
print("Imported MAPPO")
import numpy as np

n_episodes = 100
n_steps = env.ep_length
average_returns = []
total_time = 0
timesteps = 0
update_timestep = 100
decay_timestep = 100
decay_rate = 0.9


def train_agent(agent, env, n_episodes, n_steps, update_timestep, decay_timestep, decay_rate, config):
    average_returns = []
    total_time = 0
    timesteps = 0
    for episode in range(n_episodes):
        observation_info = env.reset()
        done = False
        episode_return = np.zeros(env.n_agents)
        for step in range(n_steps):
            # get action
            actions = get_action(observation_info, agent)
            # Go to the next state
            observation_info, reward, done, info = env.step(actions)
            # add reward to agent buffer
            for ag in range(env.n_agents):
                agent.buffer.rewards += [reward[ag]] * env.n_dim
                agent.buffer.is_terminals += [done[ag]] * env.n_dim
            
            episode_return += reward
            
            if step == env.ep_length - 1:
                average_returns.append(np.mean(episode_return))
                
            timesteps += 1
            if timesteps % update_timestep == 0:
                print("Update agent")
                agent.update()
                
            if timesteps % decay_timestep == 0:
                print("Decay std")
                agent.decay_action_std(decay_rate, config["std_min"], config["learn_std"])
                
    return average_returns, agent, env

def get_action(observation_info, agent):
    observation, observation_std = observation_info
    print(observation_std)
    actions = np.zeros((env.n_agents, env.n_dim))
    for dim in range(env.n_dim):
        # convert observation and std to float32
        observation[dim] = observation[dim].astype(np.float32)
        observation_std[dim] = observation_std[dim].astype(np.float32)
        action = agent.select_action(observation[dim], observation_std[dim])
        #print(action)
        actions[:, dim] = action
    return actions
