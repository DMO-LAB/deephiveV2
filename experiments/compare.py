import os 
import numpy as np

import sys
print(sys.path)

from deephive.environment.deephive_utils import *
from deephive.environment.utils import *

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--result_path", type=str, default="experiments/results/")
parser.add_argument("--exp_list", type=str, default="2,3")
parser.add_argument("--exp_numC", type=int, default=1)
parser.add_argument("--minimize", action='store_true')

args = parser.parse_args()

config_path = "config/exp_config.json"
# load config
config = parse_config(config_path)
n_agents = config["n_agents"]

minimize = args.minimize

def load_json(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def mean_confidence_interval(data, confidence=0.95):
    # check if data is just a single array and reshape it to a 2D array
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis = 0), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

all_symbols = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
all_colors = ["r", "g", "b", "k", "m", "c", "y", "r", "g", "b", "k", "m", "c", "y"]

result_path = f"experiments/results_{config['n_dim']}/"
exp_list = args.exp_list.split(",")
#print(exp_list)
# remove the first character 
# if exp_list[0][0] == "":
if exp_list[0] == "":
    exp_list = exp_list[1:]
# convert to list of ints
exp_list = [int(exp) for exp in exp_list]

# load the .npy file in each experiment folder in the result_path

symbol_list = []
color_list = []
label_list = []
gbest_values = []
opt_value = 0
result_comparison = []
for i, exp_num in enumerate(exp_list):
    try:
        exp_path = result_path + "exp_" + str(exp_num) + "/"
        # load the .npy file
        gbest = np.load(exp_path + f"gbestVals.npy") * -1 if minimize else np.load(exp_path + f"gbestVals.npy")
        gbest_values.append(gbest)
        mean, lower, upper = mean_confidence_interval(gbest)
        mid_mean, mid_lower, mid_upper = mean_confidence_interval(gbest[:, :len(gbest[0])//2])
        # load json
        with open(exp_path + "run_summary.json") as f:
            config = json.load(f)
        label_list.append(config["title"])
        result_comparison.append({
            "title": label_list[i],
            "mean": np.round(mean[-1], 4),
            "lower": np.round(lower[-1], 4),
            "upper": np.round(upper[-1], 4),
            "mid_mean": np.round(mid_mean[-1], 4),
            "mid_lower": np.round(mid_lower[-1], 4),
            "mid_upper": np.round(mid_upper[-1], 4),
        })
        # get the symbol
        symbol_list.append(all_symbols[i])
        # get the color
        color_list.append(all_colors[i])
    except:
        print(f"Experiment {exp_num} not found")
        continue
    
opt_value = None

import pandas 
result_comparison = pandas.DataFrame(result_comparison)

# save the result_comparison
result_comparison.to_csv(f"{result_path}result_comparison_function_{args.exp_numC}.csv")

# plot the gbest_values
plot_num_function_evaluation(fopt=gbest_values, label_list=label_list, symbol_list=symbol_list, color_list=color_list, save_dir=f"{result_path}comparison_function_{args.exp_numC}.png",
                             n_agents=n_agents, opt_value=opt_value, log_scale=False, minimize=minimize)

plot_num_function_evaluation(fopt=gbest_values, label_list=label_list, symbol_list=symbol_list, color_list=color_list, save_dir=f"{result_path}log_comparison_function_{args.exp_numC}.png",
                             n_agents=n_agents, opt_value=opt_value, log_scale=True, minimize=minimize)

print(f"Completed Comparison for experiment {args.exp_numC}")


