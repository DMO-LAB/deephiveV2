
from deephive.environment.utils import parse_config, num_function_evaluation, plot_individual_function_evaluation
from datetime import datetime
import os 
import neptune 
import numpy as np
# from deephive.environment.deephive_utils import *
import torch
from deephive.environment.optimization_environment import OptimizationEnv
from deephive.policies.mappo import MAPPO
# Set print options
np.set_printoptions(suppress=True, precision=4)
from dotenv import load_dotenv
load_dotenv()

def initialize_logger(api_token, tags, config, mode="train"):
        run = neptune.init_run(
        project="DMO-LAB/DeepHive-V2",
        source_files=["../deephive/environment/optimization_environment.py", "../deephive/policies/mappo.py"
                      "../deephive/environment/utils.py", "../deephive/environment/deephive_utils.py"
                      "../config/exp_config.json"],
        api_token=api_token,
        tags=[tags, mode, config["objective_function"], str(config["layer_size"])]
        )
        return run
        
def initialize(config, mode="train", **kwargs):
    env = OptimizationEnv(config)
    agent_policy = MAPPO(config)
    if mode == "test" or mode == "benchmark":
        model_path = kwargs.get("model_path", None)
        if model_path is None:
            raise ValueError("Model path must be provided for testing")
        # check if model path is a list of paths
        if isinstance(model_path, list):
            agent_policies = []
            for path in model_path:
                agent_policy = MAPPO(config)
                agent_policy.load(path)
                agent_policies.append(agent_policy)
            return env, agent_policies
        else:
            agent_policy.load(model_path)
            return env, agent_policy
    else:
        return env, agent_policy


def get_action(observation, agent_policy, env, observation_std=None, random=False, env_random=False, pso_action=False, **kwargs):
    # Ensure observation_info is a numpy array
    
    if env_random:
        current_state =  env.state[:, :2]
        env._generate_init_state(count=True)
        new_state = env.state[:, :2]
        action = new_state - current_state
        return action
    
    if random:
        # Generate random actions
        action = np.random.uniform(-1, 1, size=(env.n_agents, env.n_dim))
        return action
        
    
    if not isinstance(observation, np.ndarray):
        observation = np.array(observation)
        assert observation.shape[0] == env.n_dim, "Observation must have the same number of dimensions as the environment"

    if pso_action:
        # grab the pso parameters from kwargs
        w = kwargs.get("w", 0.5)
        c1 = kwargs.get("c1", 1.5)
        c2 = kwargs.get("c2", 1.0)
        print(f"PSO action with w={w}, c1={c1}, c2={c2}")
        # grab the second, fourth and eight element of the observation for each dimension
        # that represents the previous velocity, best position and global best position
        # for each particle
        p_vel = observation[:, :,1]
        p_best = observation[:,:,3]
        g_best = observation[:,:,9]
        
        rp = np.random.uniform(-1, 1,size=env.n_dim)
        rg = np.random.uniform(-1, 1,size=env.n_dim)
        
        v = w * p_vel.T + c1 * rp * p_best.T + c2 * rg * g_best.T
        return v
    # Initialize observation_std with zeros or use provided std, ensuring it matches the shape of observation
    if observation_std is None:
        observation_std = np.zeros_like(observation)
    else:
        observation_std = np.array(observation_std)

    # Flatten the observation and std arrays
    observation_flat = observation.reshape(env.n_dim * env.n_agents, -1)  # Flatten to 1D array
    observation_std_flat = observation_std.reshape(-1)  # Flatten to 1D array
    # Pass the entire flattened observation and std arrays to select_action
    action_flat = agent_policy.select_action(observation_flat, observation_std_flat)

    # Reshape the flattened action array back to the original (n_agents, n_dim) shape
    actions = action_flat.reshape(env.n_dim, env.n_agents).T  # Reshape to (n_agents, n_dim

    return actions  # Return the action


def get_direct_action(obs, obs_std, agent_policy):
    torch_obs = torch.FloatTensor(obs)
    torch_obs_std = torch.FloatTensor(obs_std)
    action = agent_policy.policy.act(torch_obs, torch_obs_std)
    return action


def train(env, agent_policy, config, title="experiment_1", **kwargs):
    neptune_logger = kwargs.get("neptune_logger", None)
        
    save_path = kwargs.get("save_path", "training_results/")
    save_path = os.path.join(save_path, title)
    # make directory if it does not exist
    os.makedirs(save_path, exist_ok=True)
    n_episodes = config["n_episodes"]
    average_returns = []  
    timestep = 0    
    for i in range(n_episodes):
        #print(f"Episode {i} started, timestep {timestep}")
        obs, obs_std = env.reset()
        episode_return = np.zeros(env.n_agents)
        for step in range(env.ep_length):
            #print(f"Episode {i}, step {step}, timestep {timestep}")
            actions = get_action(obs, agent_policy, env, obs_std)
            obs, reward, done, info = env.step(actions)
            for ag in range(env.n_agents):
                agent_policy.buffer.rewards += [reward[ag]] * env.n_dim
                agent_policy.buffer.is_terminals += [done[ag]] * env.n_dim
            episode_return += reward
            obs, obs_std = obs
            timestep += 1
            if step == env.ep_length - 1:
                average_returns.append(np.mean(episode_return))
                if neptune_logger is not None:
                    neptune_logger["train/average_return"].log(average_returns[-1])
                running_average_rewards = np.mean(average_returns)

        if i % config["update_timestep"] == 0 and timestep > 0:
            #print(f"Updating policy at episode {i}")
            agent_policy.update()
        if i % config["log_interval"] == 0 and timestep > 0 :
            print(f"Episode {i} completed")
            print(f"Average return: {running_average_rewards}")
            if env.n_dim == 2:
                env.render(type="history", file_path=f"{save_path}/episode_{i}.gif")  
                if neptune_logger:
                    neptune_logger[f"train/gifs/{i}.gif"].upload(f"{save_path}/episode_{i}.gif")
        if i % config["decay_interval"] == 0 and timestep > 0:
            agent_policy.decay_action_std(config["decay_rate"], min_action_std=config["min_action_std"], debug=False)
        if i % config["save_interval"] == 0 and timestep > 0:
            if average_returns[-1] > running_average_rewards:
                print(f"Saving model at episode {i} with average return {average_returns[-1]} and running average {running_average_rewards}")
                agent_policy.save(save_path, episode=i)
        
    return agent_policy

def test(env, policy, config, n_episodes=100, ep_length=100, **kwargs):
    debug = kwargs.get("debug", False)
    show_interval = kwargs.get("show_interval", ep_length+1)
    all_gbest_values = []
    optimal_positions = kwargs.get("optimal_positions", None)
    restart_interval = kwargs.get("restart_interval", ep_length+1)
    # Retrieve action_std and min_action_std from kwargs with defaults from config
    action_std = kwargs.get("action_std", config["action_std"])
    min_action_std = kwargs.get("min_action_std", config["min_action_std"])
    # New decay parameters
    decay_rate = kwargs.get("decay_rate", 0.9)  # Default decay rate if not provided
    decay_start = kwargs.get("decay_start", 0)  # Default start time if not provided
    
    save_path = kwargs.get("save_path", "test_results/")
    randomize = kwargs.get("randomize", False)
    env.ep_length = ep_length
    # print all the parameters
    print(f"Parameters: n_episodes={n_episodes}, ep_length={env.ep_length}, show_interval={show_interval}, restart_interval={restart_interval}, action_std={action_std}, min_action_std={min_action_std}, decay_rate={decay_rate}, decay_start={decay_start}, save_path={save_path}, randomize={randomize}")
    for episode in range(n_episodes):
        gbest_values = []
        obs, obs_std = env.reset()
        current_action_std = action_std  # Reset action_std for each episode
        for step in range(ep_length):
            policy.set_action_std(current_action_std)   
            if (step <= decay_start or step % restart_interval == 0) and randomize==True:
                actions = get_action(obs, policy, env, obs_std, env_random=True, pso_action=False, w=0.0, c1=0.0, c2=2)
            else:
                if step <= decay_start or step % restart_interval == 0:
                    #print(f"Randomizing at step {step}")
                    actions = get_action(obs, policy, env, obs_std, random=True) 
                actions = get_action(obs, policy, env, obs_std)
                obs, _, _, _ = env.step(actions)
                obs, obs_std = obs
            gbest_values.append(env.gbest[-1])
            # Update action_std with decay only after a certain step is reached
            if step >= decay_start:
                # Decay the std uniformly from the max to the min std over the specified rate
                current_action_std = max(min_action_std, current_action_std * decay_rate)
                policy.set_action_std(current_action_std)   
                if debug:
                    print(f"Step {step}, Decaying action std to {current_action_std}")
                    print(f"Policy action variance: {policy.policy.action_var}")
            if step == ep_length - 1:
                if debug:
                    print(f"Episode {episode} completed with gbest value {env.gbest[-1]}")
        if config["plot_gif"]:            
            if (episode % show_interval == 0 or episode == n_episodes - 1) and env.n_dim == 2:
                print(f"Rendering episode {episode}")
                env.render(type="history", file_path=f"{save_path}/episode_{episode}.gif", fps=kwargs.get("fps", 4),optimal_positions=optimal_positions)
        print(f"Number of function Evaluations: {env.objective_function.tracker.count}, Episode {episode} completed with gbest value {env.gbest[-1]}")
        all_gbest_values.append(gbest_values)
    
    if config['plot_gbest']:
        num_function_evaluation(fopt=np.array(all_gbest_values) ,n_agents=env.n_agents, save_dir=save_path + "num_function_evaluations.png", opt_value=env.best_obj_value,
                                log_scale=False, plot_error_bounds=True, title=f"CEC FUNCTION {env.function_id}")
        
        plot_individual_function_evaluation(fopt=np.array(all_gbest_values) ,n_agents=env.n_agents, save_dir=save_path + "num_function_evaluations2.png", opt_value=env.best_obj_value,
                                log_scale=False, title=f"CEC FUNCTION {env.function_id}")
        
        num_function_evaluation(fopt=np.array(all_gbest_values) ,n_agents=env.n_agents, save_dir=save_path + "num_function_evaluations_log.png", opt_value=env.best_obj_value,
                                log_scale=True, plot_error_bounds=True, title=f"CEC FUNCTION {env.function_id}")
        
        plot_individual_function_evaluation(fopt=np.array(all_gbest_values) ,n_agents=env.n_agents, save_dir=save_path + "num_function_evaluations2_log.png", opt_value=env.best_obj_value,
                                log_scale=True, title=f"CEC FUNCTION {env.function_id}")
    # save gbest values to save_path
    np.save(f"{save_path}/gbest_values.npy", np.array(all_gbest_values))
    return all_gbest_values


# gbest_values = test(env, agent_policy, config, n_episodes=100, mode="test")
import scipy.stats
def mean_confidence_interval(data, confidence=0.95):
    # check if data is just a single array and reshape it to a 2D array
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis = 0), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# gbest_values = test(env, agent_policy, config, n_episodes=100, mode="test")



def mutate_and_crossover(self, positions, lbest_positions):
    # Mutation and Crossover
    for i in range(self.swarm_size):
        if np.random.rand() < self.mutation_probability:
            trial_positions = self.mutate(positions, lbest_positions)
        else:
            self.permute_lbest(i)
            trial_positions = self.mutate(positions, lbest_positions[i])

        new_positions = self.crossover(positions, trial_positions)
        new_fitness = self.fitness_function(new_positions)

        # Selection-II: Update the personal and global bests
        better_mask = new_fitness < self.fitness_values
        positions[better_mask] = new_positions[better_mask]
        self.pbest_positions[better_mask] = new_positions[better_mask]
        self.fitness_values[better_mask] = new_fitness[better_mask]

        if np.min(new_fitness) < self.fitness_function(self.gbest_position.reshape(1, -1)):
            self.gbest_position = new_positions[np.argmin(new_fitness)]
    return positions

def mutate_and_crossover_(self, positions, lbest_positions):
        # Initialize an array to hold the fitness values of the new positions
        new_fitness_values = np.empty(self.swarm_size)
        
        # Track whether any position has resulted in a new global best
        new_global_best_found = False
        new_gbest_position = None
        new_gbest_value = float('inf')
        
        for i in range(self.swarm_size):
            if np.random.rand() < self.mutation_probability:
                trial_positions = self.mutate(positions, lbest_positions)
            else:
                self.permute_lbest(i)
                trial_positions = self.mutate(positions, lbest_positions[i])

            new_positions = self.crossover(positions, trial_positions)
        
            # Evaluate the fitness of the new positions only once
            new_fitness = self.fitness_function(new_positions[i].reshape(1, -1))
            new_fitness_values[i] = new_fitness
            
            # Selection-II: Update the personal and global bests based on the new fitness
            if new_fitness < self.fitness_values[i]:
                positions[i] = new_positions[i]
                self.pbest_positions[i] = new_positions[i]
                self.fitness_values[i] = new_fitness

                # Check for a new global best
                if new_fitness < new_gbest_value:
                    new_global_best_found = True
                    new_gbest_position = new_positions[i]
                    new_gbest_value = new_fitness
        
        # Update the global best position after checking all particles
        if new_global_best_found:
            self.gbest_position = new_gbest_position

        return positions
    
    
    
    

# def get_action(observation, agent_policy, env, observation_std=None, **kwargs):
#     # Ensure observation_info is a numpy array
    
#     if not isinstance(observation, np.ndarray):
#         observation = np.array(observation)
#         assert observation.shape[0] == env.dimension, "Observation must have the same number of dimensions as the environment"
#     # Initialize observation_std with zeros or use provided std, ensuring it matches the shape of observation
#     if observation_std is None:
#         observation_std = np.zeros_like(observation)
#     else:
#         observation_std = np.array(observation_std)

#     # Flatten the observation and std arrays
#     observation_flat = observation.reshape(env.dimension * env.swarm_size, -1)  # Flatten to 1D array
#     observation_std_flat = observation_std.reshape(-1)  # Flatten to 1D array
#     # Pass the entire flattened observation and std arrays to select_action
#     action_flat = agent_policy.select_action(observation_flat, observation_std_flat)

#     # Reshape the flattened action array back to the original (swarm_size, dimension) shape
#     actions = action_flat.reshape(env.dimension, env.swarm_size).T  # Reshape to (n_agents, n_dim

#     return actions  