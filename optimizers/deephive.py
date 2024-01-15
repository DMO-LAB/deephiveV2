import os
import pandas as pd
from deephive.policies.mappo import MAPPO
from deephive.environment.optimization_environment import OptimizationEnv
from deephive.environment.utils import parse_config
from datetime import datetime
import numpy as np
import seaborn as sns
import os
import neptune
from neptune.types import File
import argparse 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy
from deephive.other_algorithms.pso import ParticleSwarmOptimizer
import time

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

def train_agent(env, agent, neptune_logger=None, **kwargs):
    # Setting default values if not provided
    n_episodes = kwargs.get('n_episodes', 2000)
    update_timestep = kwargs.get('update_timestep', 25)
    decay_rate = kwargs.get('decay_rate', 0.01)
    log_interval = kwargs.get('log_interval', 200)
    decay_interval = kwargs.get('decay_interval', 1000)
    save_interval = kwargs.get('save_interval', 2500)
    min_action_std = kwargs.get('std_min', 0.02)
    
    average_returns = []
    training_run_title = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    save_path = f"training_runs/{training_run_title}/"
    os.makedirs(save_path, exist_ok=True)
    timesteps = 0
    for episode in range(0, n_episodes+1):
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
            if neptune_logger and episode % log_interval == 0:
                # log global best agent value
                neptune_logger[f"train/global_best_value/episode{episode}"].log(float(info["gbest"][-1]))
                # neptune_logger[f"train/global_best_value/episod_best_pos{episode}"].upload(File.as_image(np.array(info["gbest"])))

            if step == env.ep_length - 1:
                average_returns.append(np.mean(episode_return))
                running_average_rewards = np.mean(average_returns)
                if neptune_logger:
                    neptune_logger["train/average_return"].log(average_returns[-1])

                
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
            if neptune_logger:
                neptune_logger[f"train/gifs/{episode}.gif"].upload(f"{save_path}{episode}.gif")

            
        if timesteps % decay_interval == 0:
            agent.decay_action_std(decay_rate, min_action_std=min_action_std)
            
        if timesteps % save_interval == 0 and timesteps > 0:
            if average_returns[-1] > running_average_rewards:
                print(f"Average return: {average_returns[-1]}, running average: {running_average_rewards}")
                agent.save(save_path, episode=timesteps)
                if neptune_logger:
                    neptune_logger[f"train/checkpoints/timesteps-{timesteps}"].upload(f"{save_path}/policy-{timesteps}.pth")
        
    return average_returns, agent, env

def test_policy(agent, env, n_iterations, neptune_logger=None, log_interval=5, save_dir=None):
    global_best_values = []
    optimal_positions = []
    if save_dir is None:
        test_run_title = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        save_path = f"testing_runs/{test_run_title}/"
    else:
        save_path = save_dir
    
    os.makedirs(save_path, exist_ok=True)

    print(agent.action_std, agent.policy.action_var)

    duration = 0
    for i in range(n_iterations+1):
        global_best_value = []
        observation_info = env.reset()
        start_time = time.time()
        for step in range(env.ep_length):
            actions = get_action(observation_info, agent, env)
            observation_info, _, _, info = env.step(actions)
            global_best_value.append(info["gbest"][-1])
            if neptune_logger:
                neptune_logger[f"test/global_best_value/iteration_{i}"].log(float(info["gbest"][-1]))
        end_time = time.time()
        duration += end_time - start_time
        global_best_values.append(global_best_value)
        optimal_positions.append(env.gbest)
        # if neptune_logger:
        #     neptune_logger[f"test/global_best_value/iteration_{i}"].upload(File.as_image(np.array(info["gbest"])))
        if i % log_interval == 0:
            env.render(file_path=f"{save_path}{i}.gif", type="history")
            if neptune_logger:
                neptune_logger[f"test/gifs/{i}.gif"].upload(f"{save_path}{i}.gif")
        
    # plot the global best values
    save_dir = f"{save_path}num_function_evaluations.png"
    #plot_num_function_evaluation([global_best_values], env.n_agents, save_dir, opt_value=env.objective_function.optimal_value(),  show_std=True)
    num_function_evaluation(global_best_values, env.n_agents, save_dir, env.objective_function.optimal_value())
    if neptune_logger:
        neptune_logger[f"test/num_function_evaluations"].upload(save_dir)
        
    return global_best_values, optimal_positions, duration

def benchmark_algorithms(env, agent, n_iterations, config, neptune_logger=None, log_interval=5, net=None):
    # initailize the PSO algorithm
    save_dir = f"benchmarking_runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
    os.makedirs(save_dir, exist_ok=True)

    pso_optimizer = ParticleSwarmOptimizer(func=env.objective_function, lb=env.bounds[0], ub=env.bounds[1], 
                                           swarmsize=env.n_agents, omega=config["pso_omega"], phip=config["pso_phip"],
                                             phig=config["pso_phig"], maxiter=env.ep_length, minstep=config["pso_minstep"],
                                                minfunc=config["pso_minfunc"], debug=False, minimize=config["pso_minimize"],
                                                normalize=config["pso_normalize"], use_net=config["pso_use_net"],
                                                net=net)
    
    
    # MAPPO algorithm
    dh_global_best_values, optimal_positions, dh_duration = test_policy(agent, env, args.iters, neptune_logger=run, log_interval=log_interval, save_dir=save_dir)

    # PSO algorithm
    pso_gbest_values, duration  = pso_optimizer.multiple_run(n_iterations, plot_particles=False, plot_history=True, fps=2, save_dir=save_dir, neptune_logger=neptune_logger, log_interval=log_interval)

    # plot the global best values
    opts = [pso_gbest_values, dh_global_best_values]
    labels = ["PSO", "DeepHive"]
    colors = ["#3F7F4C", "#CC4F1B"]
    plot_num_function_evaluation(opts, env.n_agents, f"{save_dir}num_function_evaluations_benchmark.png", label_list=labels, color_list=colors, opt_value=env.objective_function.optimal_value(), symbol_list=['-', '--'])
    if neptune_logger:
        neptune_logger[f"test/num_function_evaluations"].upload(f"{save_dir}num_function_evaluations_benchmark.png")

    # compare the time taken to reach the best fitness value
    print(f"Time taken for PSO: {duration}, Time taken for DeepHive: {dh_duration}")
    if neptune_logger:
        neptune_logger["test/time_taken"].log(duration)
        neptune_logger["test/time_taken"].log(dh_duration)
                               

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis = 0), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def num_function_evaluation(fopt, n_agents, save_dir, opt_value, label="TEST OPT"):
    # convert fopt to numpy array if it is not already
    fopt = np.array(fopt)
    mf1 = np.mean(fopt, axis = 0)
    err = np.std(fopt, axis = 0)
    mf1, ml1, mh1 = mean_confidence_interval(fopt,0.95)

    fig = plt.figure(figsize=(6,4), dpi=200)
    plt.rcParams["figure.figsize"] = [6, 4]
    plt.rcParams["figure.autolayout"] = True
    plt.fill_between((np.arange(len(mf1))+1)*n_agents, ml1, mh1, alpha=0.1, edgecolor='#3F7F4C', facecolor='#7EFF99')
    plt.plot((np.arange(len(mf1))+1)*n_agents, mf1, linewidth=2.0, label = label, color='#3F7F4C')
    plt.plot((np.arange(len(mf1))+1)*n_agents, np.ones(len(mf1))*opt_value, linewidth=1.0, label = 'True OPT', color='#CC4F1B')

    plt.xlabel('number of function evaluations', fontsize = 14)
    plt.ylabel('best fitness value', fontsize = 14)

    plt.legend(fontsize = 14, frameon=False)
    plt.xscale('log')
    plt.yticks(fontsize = 14)
    plt.savefig(save_dir)
    # close the figure
    plt.close(fig)


def plot_num_function_evaluation(fopt, n_agents, save_dir, symbol_list=None, color_list=None, label_list=None, opt_value=None, show_std=False):
    fig = plt.figure(figsize=(6, 4), dpi=200)
    plt.rcParams["figure.figsize"] = [6, 4]
    plt.rcParams["figure.autolayout"] = True

    if symbol_list is None:
        symbol_list = ['-']
    if color_list is None:
        color_list = ['#3F7F4C']
    if label_list is None:
        label_list = ['DeepHive']

    print(f"Number of function evaluations: {len(fopt[0])}")
    print(f"Number of algorithms: {len(fopt)}")

    if len(fopt) == 1:
        print("Single algorithm")
        num_function_evaluation(fopt[0], n_agents, save_dir, opt_value, label=label_list[0])
    else:
        for i in range(len(fopt)):
            
            mf1, ml1, mh1 = mean_confidence_interval(fopt[i], 0.95)
            if show_std:
                plt.errorbar((np.arange(len(mf1)) + 1) * n_agents, mf1, yerr=mh1 - ml1, linewidth=2.0,
                             label=label_list[i],
                             color=color_list[i])
            # plt.fill_between((np.arange(len(mf1)) + 1) * n_agents, ml1, mh1, alpha=0.1, edgecolor='#3F7F4C',
            #                  facecolor=color_list[i])
            plt.plot((np.arange(len(mf1)) + 1) * n_agents, mf1, symbol_list[i], linewidth=2.0, label=label_list[i],
                     color=color_list[i])

    if opt_value is not None:
        plt.plot((np.arange(len(mf1))+1)*n_agents, np.ones(len(mf1))*opt_value, linewidth=1.0, label = 'True OPT', color='#CC4F1B')

    plt.xlabel('number of function evaluations', fontsize=14)
    plt.ylabel('best fitness value', fontsize=14)
    plt.legend(fontsize=8, frameon=False, loc="lower right")
    plt.xscale('log')
    plt.yticks(fontsize=14)
    plt.savefig(save_dir)
    #plt.show()
    # close the figure
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an agent to optimize a function')
    # add tag argument 
    parser.add_argument('--tag', type=str, default="default", help='tag for the experiment')
    parser.add_argument('--mode', type=str, default="train", help='train or test or benchmark')
    parser.add_argument('--iters', type=int, default=20, help='number of iterations')
    parser.add_argument('--model_path', type=str, default="models/policy.pth", help='path to model for testing')
    parser.add_argument('--log', type=bool, default=False, help='log to neptune')
    
    args = parser.parse_args()
    config_path = 'config/config.json'
    env, agent_policy = initialize(config_path, mode=args.mode, model_path=args.model_path)
    config = parse_config(config_path)
    config["n_episode"] = 2000

    # Initialize Neptune logger
    if bool(args.log):
        run = neptune.init_run(
        project="DMO-LAB/DeepHive-V2",
        source_files=["environment", "policies", "deephive.py", "config"],
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YTk1NWE5Ny1iMzhmLTRlMjMtYTQwYi05MTlmYTFjOTNhMTEifQ==",
        tags=[args.tag, args.mode, config["objective_function"], str(config["layer_size"])]
        )
    else:
        run = None

    # # add config file to neptune
    # run["parameters"] = config

    if args.mode == "train":
        average_returns, agent_policy, env = train_agent(env, agent_policy, neptune_logger=run)
    elif args.mode == "test":
        agent_policy.set_action_std(config["test_action_std"])
        global_best_values, optimal_positions, duration = test_policy(agent_policy, env, args.iters, neptune_logger=run, log_interval=5)
    elif args.mode == "benchmark":
        agent_policy.set_action_std(config["test_action_std"])
        benchmark_algorithms(env, agent_policy, args.iters, config, neptune_logger=run, log_interval=5)
    else:
        raise ValueError("Mode must be either train or test")
    if run:
        run.stop()
