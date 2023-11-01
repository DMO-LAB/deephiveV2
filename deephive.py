import os
import pandas as pd
from policies.mappo import MAPPO
from environment.optimization_environment import OptimizationEnv
from environment.utils import parse_config
from datetime import datetime
import numpy as np
import seaborn as sns

def initialize(config_path):
    env = OptimizationEnv(config_path)
    agent_policy = MAPPO(config_path)
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



config_path = 'config/config.json'
env, agent_policy = initialize(config_path)
config = parse_config(config_path)


# Training
n_episodes = 100
n_steps = env.ep_length
average_returns = []
total_time = 0
timesteps = 0
update_timestep = 100
decay_timestep = 100
decay_rate = 0.9



def train_agent(env, agent, n_episodes=2000, update_timestep=25, decay_rate=0.01, log_interval=200, min_action_std=0.1,
                decay_interval=1000, save_interval=2000):
    average_returns = []
    training_run_title = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    save_path = f"training_runs/{training_run_title}/"
    os.mkdir(save_path)
    timesteps = 0

    for episode in range(n_episodes):
        observation_info = env.reset()
        episode_return = np.zeros(env.n_agents)
        for step in range(env.ep_length):
            actions = get_action(observation_info, agent, env)
            observation_info, reward, done, info = env.step(actions)
            # add reward to agent buffer
            for ag in range(env.n_agents):
                agent.buffer.rewards += [reward[ag]] * env.n_dim
                agent.buffer.is_terminals += [done[ag]] * env.n_dim
            
            episode_return += reward

            if step == env.ep_length - 1:
                average_returns.append(np.mean(episode_return))
                running_average_rewards = np.mean(average_returns)
                
            timesteps += 1
        if timesteps % update_timestep == 0:
            agent.update()
    
        if episode % log_interval == 0 and timesteps > 0:
            print_items(
                    episode = episode,
                    average_returns = average_returns[-1],
                    timesteps = timesteps,
                )
            env.render(file_path=f"{save_path}{episode}.gif", type="history")
            
        if timesteps % decay_interval == 0:
            agent.decay_action_std(decay_rate, min_action_std=min_action_std)
            
        if timesteps % save_interval == 0 and timesteps > 0:
            if average_returns[-1] > running_average_rewards:
                print(f"Average return: {average_returns[-1]}, running average: {running_average_rewards}")
                agent.save(save_path, episode=timesteps)

    return average_returns, agent, env


# def train_agent(agent, env, n_episodes, n_steps, update_timestep, decay_timestep, decay_rate, config):
#     average_returns = []
#     total_time = 0
#     timesteps = 0
#     for episode in range(n_episodes):
#         observation_info = env.reset()
#         done = False
#         episode_return = np.zeros(env.n_agents)
#         for step in range(n_steps):
#             # get action
#             actions = get_action(observation_info, agent)
#             # Go to the next state
#             observation_info, reward, done, info = env.step(actions)
#             # add reward to agent buffer
#             for ag in range(env.n_agents):
#                 agent.buffer.rewards += [reward[ag]] * env.n_dim
#                 agent.buffer.is_terminals += [done[ag]] * env.n_dim
            
#             episode_return += reward
            
#             if step == env.ep_length - 1:
#                 average_returns.append(np.mean(episode_return))
                
#             timesteps += 1
#             if timesteps % update_timestep == 0:
#                 print("Update agent")
#                 agent.update()
                
#             if timesteps % decay_timestep == 0:
#                 print("Decay std")
#                 agent.decay_action_std(decay_rate, config["std_min"], config["learn_std"])
                
#     return average_returns, agent, env
