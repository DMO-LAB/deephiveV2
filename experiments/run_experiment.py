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
                   split_agents=True, threshold=0.1, save_surrogate_plots=False, sur_debug=False, number_of_points=6):
    gbest_values = []
    if save_gif:
        os.makedirs(result_path, exist_ok=True)
    for iter in range(iters):
        print("Iteration: ", iter)
        observation_info = env.reset()
        episode_gbVals = []
        gp_Info = []
        for i in range(timesteps):
            if sur_debug:
                env.surrogate.plot_surrogate(save_dir=result_path + "iter_" + str(iter) + "_time_" + str(i) + "_mean.png")
                env.surrogate.plot_variance(save_dir=result_path + "iter_" + str(iter) + "_time_" + str(i) + "_variance.png")
                env.surrogate.plot_checkpoints_state(save_dir=result_path + "iter_" + str(iter) + "_time_" + str(i) + "_checkpoints.png")
                env.render(type="state", file_path=result_path + "iter_" + str(iter) + "_time_" + str(i) + "_state_.png")
            episode_gbVals.append(env.gbest[-1])
            exploiters_action =  get_action(observation_info, agent_policy, env)
            explorer_action, next_point = get_informed_action(env, number_of_points=number_of_points)
            actions = np.zeros((env.n_agents, env.n_dim))
            actions[:(env.n_agents - number_of_points)] = exploiters_action[:(env.n_agents - number_of_points)]
            actions[(env.n_agents - number_of_points):] = explorer_action
            observation_info, reward, done, info = env.step(actions)
            if split_agents == False and i == 4:
                agent_policy.set_action_std(0.02)
                print("Setting action std to 0.03")
            if sur_debug:
                plot_point(env.grid_points, env.evaluated_points, next_point, save_dir=result_path + "iter_" + str(iter) + "_time_" + str(i) + "_points.png")
            if sur_debug:
                gp_Info.append(env.surrogate.gp.kernel_)
        gbest_values.append(episode_gbVals)
        if save_gif and iter % save_interval == 0:
            _ = env.render(type="history", file_path=result_path + "iter_" + str(iter) + ".gif")
        if save_surrogate_plots and iter % save_interval == 0:
            env.surrogate.plot_surrogate(save_dir=result_path + "iter_" + str(iter) + ".png")
            env.surrogate.plot_variance(save_dir=result_path + "iter_" + str(iter) + "_variance.png")
            env.surrogate.plot_checkpoints_state(save_dir=result_path + "iter_" + str(iter) + "_checkpoints.png")
        # SAVE the surrogate state buffer to file
        # buffer = []
        # for buff in env.surrogate_states_buffer:
        #     for bu in buff:
        #         buffer.append(bu)
        # # reset the shape of the buffer
        # buffer = np.array(buffer)
        # #np.save(result_path + "iter_" + str(iter) + "_buffer.npy", buffer)           

    return gbest_values, gp_Info

if __name__ == "__main__":
    import argparse 
    from str2bool import str2bool
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="../config/config.json")
    parser.add_argument("--model_path", type=str, default="../models/exploiting_model.pth")
    parser.add_argument("--exp_num", type=int, default=1)
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--save_gif", type=str, default=True)
    parser.add_argument("--split_agents", type=str, default=False)
    parser.add_argument("--run_summary", type=str, required=True, help="summary of the run")
    parser.add_argument("--save_surrogate_plots", type=str, default=False)
    parser.add_argument("--sur_debug", type=str, default=False)

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
    args.save_surrogate_plots = str2bool(args.save_surrogate_plots)
    args.sur_debug = str2bool(args.sur_debug)
    
    experiments = [
        [env1, agent_policy1, f"experiment_{exp_num}", timesteps, iters, args.save_gif, "results/", args.split_agents],
    ]
    
    print("Experiment parameters: ", args)

    for env, agent_policy, exp_name, timesteps, iters, save_gif, result_path, split_agents in experiments:
        print("Running experiment: ", exp_name)
        
        result_path = 'experiments/results/' + exp_name + '/'
        gbest_values, gp_Info = run_experiment(env, agent_policy, timesteps, iters, save_gif=args.save_gif, result_path=result_path, split_agents=split_agents,
                                      save_interval=10, save_surrogate_plots=args.save_surrogate_plots, sur_debug=args.sur_debug)
        np.save(result_path + exp_name + ".npy", gbest_values)
        print("Experiment: ", exp_name, " completed.")

        # save run summary in a text file and in the results folder
        with open(result_path + "run_summary.txt", "w") as f:
            f.write(args.run_summary)

        with open(result_path + "run_summary.txt", "a") as f:
            f.write(args.run_summary)
            f.write("\n")
            f.write("Experiment: " + exp_name)
            f.write("\n")
            f.write("Experiment parameters: " + str(args))
            f.write("\n")
            f.write("\n")
            for i, info in enumerate(gp_Info):
                f.write(str(info) + '\t' + str(i))
                f.write("\n")
