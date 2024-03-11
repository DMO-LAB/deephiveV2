
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
from utility import *
import pandas as pd
function_selector = FunctionSelector()
from deephive.policies.mappo import MAPPO
np.set_printoptions(suppress=True, precision=4)
from dotenv import load_dotenv
api_token = os.environ.get("NEPTUNE_API_TOKEN")
load_dotenv()

config_path = "config/exp_config.json"
model_path = "models/pbest_unfreeze.pth"
model_path_2 = "models/gbest.pth"
config = parse_config(config_path)
config['use_gbest'] = False
config['use_lbest'] = False
config["use_optimal_value"] = True
config["log_scale"] = True
config["freeze"] = False
config["include_gbest"] = False
config["negative"] = True
if config["include_gbest"] or config["use_lbest"]:
    config["obs_dim"] = 5
else:
    config["obs_dim"] = 4
config["ep_length"] = 25

config["min_action_std"] = 0.001
config["action_std"] = 0.2
config["variable_std"] = False
config["update_timestep"] = 2
config["decay_rate"] = 0.99
config["log_interval"] = 500
config["decay_interval"] = 5
config["save_interval"] = 200
config["test_decay_rate"] = 0.9
config["test_decay_start"] = 0
config["reward_scheme"] = "FullRewardScheme2"
config["observation_scheme"]= "SimpleObservationScheme"
config["n_episodes"] = 5000
config["plot_gif"] = True
config["plot_gbest"] = True
config["test_ep_length"] = 100
config["n_agents"] = 12
config["n_dim"] = 2
config['objective_function'] = "BenchmarkFunctions" 
config["function_id"] = "f01"
config["neighborhood_size"] = 4
config["topology"] = "random"

mode = "train"

model_experiments = [102, 103, 104]
model_lists = ["gbest", "lbest", "pbest"]
obs_dim = [4, 5, 4]
function_ids = ["f01", "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19",
                "f20", "f21", "f22", "f23", "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31", "f32", "f33", "f34", "f35", "f36", "f37", "f38", "f39", "f40",
                "f41", "f42", "f43", "f44", "f45", "f46", "f47", "f48", "f49", "f50"]
#function_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
config["n_agents"] = 40
config["n_dim"] = 30
config['objective_function'] = "BenchmarkFunctions"
config["ep_length"] = 1000
config["log_scale"] = False
config["min_action_std"] = 0.0001
config["max_action_std"] = 0.5
config["test_decay_rate"] = 0.99
config["test_decay_start"] = 100
iters = 10
config["neighborhood_size"] = 10

base_save_dir = f"new_test_results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/" 
for i, model in enumerate(model_experiments):
    config["n_dim"] = 30
    config["n_agents"] = 40
    config["obs_dim"] = obs_dim[i]
    if i==1:
        config["use_lbest"] = True
    else:
        config["use_lbest"] = False
    MODEL_PATH = f"models/model_{model}_policy-4800.pth"

    title = f"experiment_{model}"
    save_dir = base_save_dir + f"model{model}/"
    tags = f"testing with 40 agents and log scale"
    neptune_logger = None#initialize_logger(api_token, title, config, mode="test")
    # function_ids, iters, save_dir, model_path, config, **kwargs
    successful_functions, save_dir = run_test_deephive(function_ids, iters, save_dir, MODEL_PATH, config, neptune_logger=neptune_logger)
    if neptune_logger:
        neptune_logger.stop()
    print(f"Experiment {model} completed")
    

