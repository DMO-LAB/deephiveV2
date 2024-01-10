import sys 
sys.path.append('../')
from environment.deephive_utils import *
from environment.utils import *
from dotenv import load_dotenv
load_dotenv()
import os 
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def run_experiment(env, agent_policy, timesteps, iters, save_gif=False, result_path="experiment/", save_interval=10,
                   split_agents=True):
    gbest_values = []
    if save_gif:
        os.makedirs(result_path, exist_ok=True)
    for iter in range(iters):
        print("Iteration: ", iter)
        observation_info = env.reset()
        episode_gbVals = []
        for _ in range(timesteps):
            episode_gbVals.append(env.gbest[-1])
            exploiters_action =  get_action(observation_info, agent_policy, env)
            explorer_action = get_informed_action(env)
            # split the agents into two groups and let one group exploit and the other explore
            actions = np.zeros((env.n_agents, env.n_dim))
            if split_agents:
                actions[:env.n_agents//2] = exploiters_action[:env.n_agents//2]
                actions[env.n_agents//2:] = explorer_action[env.n_agents//2:]
            else:
                actions = exploiters_action
            observation_info, reward, done, info = env.step(actions)
        gbest_values.append(episode_gbVals)
        if save_gif and iter % save_interval == 0:
            _ = env.render(type="history", file_path=result_path + "iter_" + str(iter) + ".gif")
    return gbest_values

if __name__ == "__main__":
    import argparse 
    from str2bool import str2bool
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="../config/config.json")
    parser.add_argument("--model_path", type=str, default="../models/exploiting_model.pth")
    parser.add_argument("--exp_num", type=int, default=1)
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--save_gif", type=str, default=True)
    parser.add_argument("--split_agents", type=str, default=False)
    parser.add_argument("--run_summary", type=str, required=True, help="summary of the run")

    args = parser.parse_args()

    config_path = args.config_path
    model_path = args.model_path
    mode = args.mode
    env1, agent_policy1 = initialize(config_path, mode=mode, model_path=model_path)
    config = parse_config(config_path)

    iters = args.iters
    timesteps = args.timesteps
    exp_num = args.exp_num

    # ensure split_agents and save_gif are boolean
    args.split_agents = str2bool(args.split_agents)
    args.save_gif = str2bool(args.save_gif)
    
    experiments = [
        [env1, agent_policy1, f"experiment_{exp_num}", timesteps, iters, args.save_gif, "results/", args.split_agents],
    ]
    
    print("Experiment parameters: ", args)

    for env, agent_policy, exp_name, timesteps, iters, save_gif, result_path, split_agents in experiments:
        print("Running experiment: ", exp_name)
        result_path = 'experiments/results/' + exp_name + '/'
        gbest_values = run_experiment(env, agent_policy, timesteps, iters, save_gif=args.save_gif, result_path=result_path, split_agents=split_agents,
                                      save_interval=1)
        np.save(result_path + exp_name + ".npy", gbest_values)
        print("Experiment: ", exp_name, " completed.")

        # save run summary in a text file and in the results folder
        with open(result_path + "run_summary.txt", "w") as f:
            f.write(args.run_summary)