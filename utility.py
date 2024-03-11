
from deephive.environment.utils import parse_config
from datetime import datetime
import os 
import numpy as np
import neptune
import torch
from plot_utils import *
from deephive.environment.optimization_environment import OptimizationEnv
from deephive.environment.optimization_functions.benchmark_functions import FunctionSelector
from deephive.environment.utils import mean_confidence_interval
import pandas as pd
function_selector = FunctionSelector()
from deephive.policies.mappo import MAPPO
np.set_printoptions(suppress=True, precision=4)
from dotenv import load_dotenv
api_token = os.environ.get("NEPTUNE_API_TOKEN")
load_dotenv()

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
    

def get_action(observation, agent_policy, env, observation_std=None, **kwargs):
    # Ensure observation_info is a numpy array
    
    if not isinstance(observation, np.ndarray):
        observation = np.array(observation)
        assert observation.shape[0] == env.n_dim, "Observation must have the same number of dimensions as the environment"

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

def initialize_logger(api_token, tags, config, mode="train"):
        run = neptune.init_run(
        project="DMO-LAB/DeepHive-V2",
        # source files = all python files in the current directory,
        source_files=["*.py"],
        api_token=api_token,
        tags=[tags, mode, config["objective_function"], str(config["layer_size"])]
        )
        return run

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
                
            if neptune_logger and i % config["log_interval"] == 0:
                neptune_logger[f"Episode_{i}/gbest_values"].log(env.gbest[-1])

        if i % config["update_timestep"] == 0 and timestep > 0:
            #print(f"Updating policy at episode {i}")
            agent_policy.update()
        if i % config["log_interval"] == 0 and timestep > 0:
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

def test(env, agent_policy, iters, decay_start=100, decay_rate=0.995, 
         min_action_std=0.001, max_action_std=0.5, 
         save_gif = False, save_path = "test_results", function_id=0, **kwargs):
    neptune_logger = kwargs.get("neptune_logger", None)
    os.makedirs(save_path, exist_ok=True)
    all_gbest_vals = []
    print(f"Testing function {function_id} with {env.n_agents} agents and {env.n_dim} dimensions")
    for iter in range(iters):
        gbest_vals = []
        obs = env.reset()[0]
        agent_policy.set_action_std(max_action_std)
        current_action_std = agent_policy.action_std
        for step in range(env.ep_length):
            action = get_action(obs, agent_policy, env)
            obs, _ , _ , _ = env.step(action)
            obs = obs[0]
            gbest_vals.append(env.gbest[-1])
            if neptune_logger:
                neptune_logger[f"test/{function_id}/Episode_{iter}/gbest_values"].log(env.gbest[-1])
            if step >= decay_start:
                #print(f"Decaying action std at step {step}")
                # Decay the std uniformly from the max to the min std over the specified rate
                current_action_std = max(min_action_std, current_action_std * decay_rate)
                agent_policy.set_action_std(current_action_std)   
                #print(f"Current action std: {current_action_std}")
        if env.n_dim == 2 and save_gif:
            env.render(type="history", file_path=f"{save_path}/episode_{iter}.gif") 
            if neptune_logger:
                neptune_logger[f"test/{function_id}/gifs/{iter}.gif"].upload(f"{save_path}/episode_{iter}.gif")
        all_gbest_vals.append(np.array(gbest_vals))
        print(f"Final gbest value: {env.gbest[-1]} at iteration {iter}")
    
    np.save(f"{save_path}/{function_id}_gbest_history.npy", np.array(all_gbest_vals))
    return all_gbest_vals

def analyze_results(base_path, model_lists, model_path_list, successful_functions, function_selector, **kwargs):
    neptune_logger = kwargs.get("neptune_logger", None)
    # Define a multi-level column structure: each optimizer has the same sub-columns
    columns = pd.MultiIndex.from_product([model_lists, 
                                          ["mean", "lower", "upper", "optimum", "error"]],
                                         names=['optimizer', 'metric'])
    # Initialize an empty DataFrame with these columns
    df = pd.DataFrame(columns=columns)
    
    for function_id in successful_functions:
        row_data = {}
        function_info = function_selector.get_function(function_id)
        function_opt_val = function_info["global_min"]
        
        # Loop through each optimizer
        for optimizer, model_path in zip(model_lists, model_path_list):
            data_path = model_path + f"deephive/{function_id}/{function_id}_gbest_history.npy"
            try:
                # Attempt to load the optimizer's result and compute metrics
                gbest_values = np.load(data_path) * -1
                mean_val, lower_val, upper_val = mean_confidence_interval(gbest_values)
                error_val = abs(mean_val[-1] - function_opt_val)
                row_data[(optimizer, 'mean')] = mean_val[-1]
                row_data[(optimizer, 'lower')] = lower_val[-1]
                row_data[(optimizer, 'upper')] = upper_val[-1]
                row_data[(optimizer, 'optimum')] = function_opt_val
                row_data[(optimizer, 'error')] = error_val
                
                # Additional logging or plotting can be added here as needed
                
            except Exception as e:
                print(f"Error processing {optimizer} for function {function_id}: {e}")
                # Fill missing values if any error occurs
                for metric in ["mean", "lower", "upper", "optimum", "error"]:
                    row_data[(optimizer, metric)] = np.nan
        
        # After collecting data for all optimizers, add the row to the DataFrame
        #df = df.append(pd.Series(row_data, name=function_id))
        df.loc[function_id] = row_data
    
    # Save the DataFrame to CSV
    df.to_csv(f"{base_path}/results.csv")
    
    # Log the DataFrame if Neptune logger is provided
    if neptune_logger:
        neptune_logger["test/results"].upload(f"{base_path}/results.csv")
        
    return df

def run_test(function_ids, iters, save_dir, model_path, config, **kwargs):
    neptune_logger = kwargs.get("neptune_logger", None)
    successful_functions = []
    for function_id in function_ids:
        try:
            config["function_id"] = function_id
            function_dim = function_selector.get_function(function_id)["dimension"]
            if config["n_dim"] > function_dim:
                print(f"function {function_id} has {function_dim} dimensions, setting n_dim to {function_dim}")
                config["n_dim"] = function_dim
            env, agent_policy = initialize(config, mode="test", model_path=model_path)
            _ = env.reset()[0]
            agent_policy.load(model_path)
            save_path = f"{save_dir}/{function_id}"
            try:
                all_gbest_vals = test(env, agent_policy, iters, decay_start=config["test_decay_start"], decay_rate=config["test_decay_rate"],
                                min_action_std=config["min_action_std"], max_action_std=config["max_action_std"],
                                save_gif = False, save_path = save_path, function_id=function_id, neptune_logger=neptune_logger)
                successful_functions.append(function_id)
            except Exception as e:
                print(f"Function {function_id} failed with error {e}")
        except Exception as e:
            print(f"Function {function_id} failed with error {e}")
        
    #df = analyze_results(save_dir, successful_functions, function_selector, neptune_logger=neptune_logger)
    if neptune_logger: 
        neptune_logger.stop()
    return successful_functions, save_dir, env, agent_policy


def run_test_deephive(function_ids, iters, save_dir, model_path, config, **kwargs):
    dimension = config["n_dim"]
    neptune_logger = kwargs.get("neptune_logger", None)
    successful_functions = []
    for function_id in function_ids:
        try:
            config["function_id"] = function_id
            function_dim = function_selector.get_function(function_id)["dimension"]
            if config["n_dim"] > function_dim:
                print(f"function {function_id} has {function_dim} dimensions, setting n_dim to {function_dim}")
                config["n_dim"] = function_dim
            else:
                config["n_dim"] = dimension
            
            
            env, agent_policy = initialize(config, mode="test", model_path=model_path)
            _ = env.reset()[0]
            agent_policy.load(model_path)
            
            try:
                test_save_path = f"{save_dir}/deephive/{function_id}"
                all_gbest_vals = test(env, agent_policy, iters, decay_start=config["test_decay_start"], decay_rate=config["test_decay_rate"],
                                min_action_std=config["min_action_std"], max_action_std=config["max_action_std"],
                                save_gif = False, save_path = test_save_path, function_id=function_id, neptune_logger=neptune_logger)
                plot_individual_function_evaluation(all_gbest_vals, config["n_agents"], f"{test_save_path}/{function_id}_individual.png", log_scale=config["log_scale"])
                num_function_evaluation(all_gbest_vals, config["n_agents"], f"{test_save_path}/{function_id}_num_evaluations.png", log_scale=config["log_scale"])
                if neptune_logger:
                    neptune_logger[f"deephive/{function_id}/individual"].upload(f"{test_save_path}/{function_id}_individual.png")
                    neptune_logger[f"deephive/{function_id}/num_evaluations"].upload(f"{test_save_path}/{function_id}_num_evaluations.png")
            except Exception as e:
                print(f"Function {function_id} failed with error {e}")
            successful_functions.append(function_id)
        except Exception as e:
            print(f"Function {function_id} failed with error {e}")
        
    #df = analyze_results(save_dir, successful_functions, function_selector, neptune_logger=neptune_logger)
    if neptune_logger: 
        neptune_logger.stop()
    return successful_functions, save_dir



