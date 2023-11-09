
import os
import pandas as pd
from policies.mappo import MAPPO
from environment.optimization_environment import OptimizationEnv
from environment.utils import parse_config
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
from other_algorithms.pso import ParticleSwarmOptimizer
import time
from dotenv import load_dotenv
load_dotenv()

class OptimizationTrainer:
    def __init__(self, config_path, mode="train", model_path=None):
        self.env = OptimizationEnv(config_path)
        self.agent_policy = MAPPO(config_path)
        self.mode = mode
        self.config = parse_config(config_path)
        if mode in ["test", "benchmark"] and model_path is None:
            raise ValueError("Model path must be provided for testing or benchmarking")
        if model_path and mode in ["test", "benchmark"]:
            print(f"Loading model from {model_path}")
            self.agent_policy.load(model_path)
            self.agent_policy.set_action_std(self.config["test_action_std"])
        # Initialize Neptune logger if necessary
        self.neptune_logger = None  # Replace with actual initialization if needed

    def initialize_logger(self, api_token, tags):
        run = neptune.init_run(
        project="DMO-LAB/DeepHive-V2",
        source_files=["environment", "policies", "deephive.py", "config"],
        api_token=api_token,
        tags=[tags, self.mode, self.config["objective_function"], str(self.config["layer_size"])]
        )
        self.neptune_logger = run

    def get_action(self, observation_info):
        actions = np.zeros((self.env.n_agents, self.env.n_dim))
        observation, observation_std = observation_info
        for dim in range(self.env.n_dim):
            observation[dim] = observation[dim].astype(np.float32)
            observation_std[dim] = observation_std[dim].astype(np.float32)
            action = self.agent_policy.select_action(observation[dim], observation_std[dim])
            actions[:, dim] = action
        return actions

    def print_items(self, **kwargs):
        for key, value in kwargs.items():
            print(key, value)

    def train_agent(self, n_episodes=None, update_timestep=None, decay_rate=None, log_interval=None, decay_interval=None, save_interval=None, min_action_std=None):
        # if parameters are not provided, use the ones from the config file
        if update_timestep is None:
            update_timestep = self.config["update_timestep"]
        if decay_rate is None:
            decay_rate = self.config["decay_rate"]
        if log_interval is None:
            log_interval = self.config["log_interval"]
        if decay_interval is None:
            decay_interval = self.config["decay_interval"]
        if save_interval is None:
            save_interval = self.config["save_interval"]
        if min_action_std is None:
            min_action_std = self.config["min_action_std"]
        if n_episodes is None:
            n_episodes = self.config["n_episodes"]

        average_returns = []
        training_run_title = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        save_path = f"training_runs/{training_run_title}/"
        os.makedirs(save_path, exist_ok=True)
        timesteps = 0
        for episode in range(0, n_episodes+1):
            observation_info = self.env.reset()
            episode_return = np.zeros(self.env.n_agents)
            for step in range(self.env.ep_length):
                actions = self.get_action(observation_info)
                observation_info, reward, done, info = self.env.step(actions)
                # add reward to agent buffer
                for ag in range(self.env.n_agents):
                    self.agent_policy.buffer.rewards += [reward[ag]] * self.env.n_dim
                    self.agent_policy.buffer.is_terminals += [done[ag]] * self.env.n_dim
                
                episode_return += reward
                if self.neptune_logger and episode % log_interval == 0:
                    # log global best agent value
                    self.neptune_logger[f"train/global_best_value/episode{episode}"].log(float(info["gbest"][-1]))
                    
                if step == self.env.ep_length - 1:
                    average_returns.append(np.mean(episode_return))
                    running_average_rewards = np.mean(average_returns)
                    if self.neptune_logger:
                        self.neptune_logger["train/average_return"].log(average_returns[-1])
                timesteps += 1
            if timesteps % update_timestep == 0:
                self.agent_policy.update()
        
            if episode % log_interval == 0 and timesteps > 0:
                self.print_items(
                        episode = episode,
                        average_returns = average_returns[-1],
                        timesteps = timesteps,
                    )
                if self.env.n_dim <= 2:
                    self.env.render(file_path=f"{save_path}{episode}.gif", type="history")
                    if self.neptune_logger:
                        self.neptune_logger[f"train/gifs/{episode}.gif"].upload(f"{save_path}{episode}.gif")
                    
            if timesteps % decay_interval == 0:
                self.agent_policy.decay_action_std(decay_rate, min_action_std=min_action_std)
                
            if timesteps % save_interval == 0 and timesteps > 0:
                if average_returns[-1] > running_average_rewards:
                    print(f"Average return: {average_returns[-1]}, running average: {running_average_rewards}")
                    self.agent_policy.save(save_path, episode=timesteps)
                    if self.neptune_logger:
                        self.neptune_logger[f"train/checkpoints/timesteps-{timesteps}"].upload(f"{save_path}/policy-{timesteps}.pth")

    def test_policy(self, n_iterations, log_interval=5, save_dir=None):
        global_best_values = []
        optimal_positions = []
        if save_dir is None:
            test_run_title = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            save_path = f"testing_runs/{test_run_title}/"
        else:
            save_path = save_dir
        
        os.makedirs(save_path, exist_ok=True)

        print(self.agent_policy.action_std, self.agent_policy.policy.action_var)

        duration = 0
        for i in range(n_iterations+1):
            global_best_value = []
            observation_info = self.env.reset()
            start_time = time.time()
            global_best_value.append(self.env.gbest[-1])
            for step in range(self.env.ep_length):
                actions = self.get_action(observation_info)
                observation_info, _, _, info = self.env.step(actions)
                global_best_value.append(info["gbest"][-1])
                if self.neptune_logger:
                    self.neptune_logger[f"test/global_best_value/iteration_{i}"].log(float(info["gbest"][-1]))
            end_time = time.time()
            duration += end_time - start_time
            global_best_values.append(global_best_value)
            optimal_positions.append(self.env.gbest)
            # if neptune_logger:
            #     neptune_logger[f"test/global_best_value/iteration_{i}"].upload(File.as_image(np.array(info["gbest"])))
            if i % log_interval == 0 and self.env.n_dim <= 2:
                self.env.render(file_path=f"{save_path}{i}.gif", type="history")
                if self.neptune_logger:
                    self.neptune_logger[f"test/gifs/{i}.gif"].upload(f"{save_path}{i}.gif")
            
        # plot the global best values
        save_dir = f"{save_path}num_function_evaluations.png"
        #plot_num_function_evaluation([global_best_values], env.n_agents, save_dir, opt_value=env.objective_function.optimal_value(),  show_std=True)
        self.num_function_evaluation(global_best_values, self.env.n_agents, save_dir, self.env.objective_function.optimal_value(self.env.n_dim))
        if self.neptune_logger:
            self.neptune_logger[f"test/num_function_evaluations"].upload(save_dir)
        return global_best_values, optimal_positions, duration

    def benchmark_algorithms(self, n_iterations, log_interval=5, net=None):
        save_dir = f"benchmarking_runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"
        os.makedirs(save_dir, exist_ok=True)
        print(self.env.bounds[0], self.env.bounds[1])
        pso_optimizer = ParticleSwarmOptimizer(func=self.env.objective_function, lb=self.env.bounds[0], ub=self.env.bounds[1], 
                                            swarmsize=self.env.n_agents, omega=self.config["pso_omega"], phip=self.config["pso_phip"],
                                                phig=self.config["pso_phig"], maxiter=self.env.ep_length, minstep=self.config["pso_minstep"],
                                                    minfunc=self.config["pso_minfunc"], debug=False, minimize=self.config["pso_minimize"],
                                                    normalize=self.config["pso_normalize"], use_net=self.config["pso_use_net"],
                                                    net=net)
        
        
        # MAPPO algorithm
        dh_global_best_values, _, dh_duration = self.test_policy( n_iterations, log_interval=log_interval, save_dir=save_dir)

        # PSO algorithm
        pso_gbest_values, duration  = pso_optimizer.multiple_run(n_iterations, plot_particles=False, plot_history=True, fps=2, save_dir=save_dir, 
                                                                 neptune_logger=self.neptune_logger, log_interval=log_interval)

        # plot the global best values
        opts = [pso_gbest_values, dh_global_best_values]
        labels = ["PSO", "DeepHive"]
        colors = ["#3F7F4C", "#CC4F1B"]
        self.plot_num_function_evaluation(opts, self.env.n_agents, f"{save_dir}num_function_evaluations_benchmark.png", label_list=labels, color_list=colors, 
                                          opt_value=self.env.objective_function.optimal_value(self.env.n_dim), symbol_list=['-', '--'])
        if self.neptune_logger:
            self.neptune_logger[f"test/num_function_evaluations"].upload(f"{save_dir}num_function_evaluations_benchmark.png")

        # compare the time taken to reach the best fitness value
        print(f"Time taken for PSO: {duration}, Time taken for DeepHive: {dh_duration}")
        if self.neptune_logger:
            self.neptune_logger["test/time_taken"].log(duration)
            self.neptune_logger["test/time_taken"].log(dh_duration)

    @staticmethod
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a, axis = 0), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h

    @staticmethod
    def num_function_evaluation(fopt, n_agents, save_dir, opt_value, label="TEST OPT"):
    # convert fopt to numpy array if it is not already
        fopt = np.array(fopt)
        mf1 = np.mean(fopt, axis = 0)
        err = np.std(fopt, axis = 0)
        mf1, ml1, mh1 = OptimizationTrainer.mean_confidence_interval(fopt,0.95)

        fig = plt.figure(figsize=(6,4), dpi=200)
        plt.rcParams["figure.figsize"] = [6, 4]
        plt.rcParams["figure.autolayout"] = True
        plt.fill_between((np.arange(len(mf1))+1)*n_agents, ml1, mh1, alpha=0.1, edgecolor='#3F7F4C', facecolor='#7EFF99')
        plt.plot((np.arange(len(mf1))+1)*n_agents, mf1, linewidth=2.0, label = label, color='#3F7F4C')
        if opt_value is not None:
            plt.plot((np.arange(len(mf1))+1)*n_agents, np.ones(len(mf1))*opt_value, linewidth=1.0, label = 'True OPT', color='#CC4F1B')

        plt.xlabel('number of function evaluations', fontsize = 14)
        plt.ylabel('best fitness value', fontsize = 14)

        plt.legend(fontsize = 14, frameon=False)
        plt.xscale('log')
        plt.yticks(fontsize = 14)
        plt.savefig(save_dir)
        # close the figure
        plt.close(fig)

    @staticmethod
    def plot_num_function_evaluation(fopt, n_agents, save_dir, opt_value, show_std=False, symbol_list=None, color_list=None, label_list=None):
        # The method implementation goes here
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
            OptimizationTrainer.num_function_evaluation(fopt[0], n_agents, save_dir, opt_value, label=label_list[0])
        else:
            for i in range(len(fopt)):
                
                mf1, ml1, mh1 = OptimizationTrainer.mean_confidence_interval(fopt[i], 0.95)
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
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/config.json")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument('--log', type=bool, default=False, help='log to neptune')
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tags", type=str, default="default run")
    parser.add_argument("--n_episodes", type=int, default=None)
    parser.add_argument("--update_timestep", type=int, default=None)
    parser.add_argument("--decay_rate", type=float, default=None)
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--decay_interval", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--min_action_std", type=float, default=None)
    parser.add_argument("--iters", type=int, default=10)

    args = parser.parse_args()
    api_token = os.environ.get("NEPTUNE_API_TOKEN")
    trainer = OptimizationTrainer(args.config_path, args.mode, args.model_path)
    if args.log:
        trainer.initialize_logger(api_token, args.tags)
    if args.tags is None:
        args.tags = f"{args.mode}_RUN"
    if args.mode == "train":
        trainer.train_agent(n_episodes=args.n_episodes, update_timestep=args.update_timestep, decay_rate=args.decay_rate, log_interval=args.log_interval, decay_interval=args.decay_interval, save_interval=args.save_interval, min_action_std=args.min_action_std)
    elif args.mode == "test":
        trainer.test_policy(n_iterations=args.iters, log_interval=args.log_interval)
    elif args.mode == "benchmark":
        trainer.benchmark_algorithms(n_iterations=args.iters, log_interval=args.log_interval, net=None)
    else:
        raise ValueError("Invalid mode - must be either train, test or benchmark")