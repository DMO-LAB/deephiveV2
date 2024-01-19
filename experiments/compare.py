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

args = parser.parse_args()

config_path = "config/config.json"
# load config
config = parse_config(config_path)

all_symbols = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
all_colors = ["r", "g", "b", "k", "m", "c", "y", "r", "g", "b", "k", "m", "c", "y"]

result_path = args.result_path
exp_list = args.exp_list.split(",")
print(exp_list)
# remove the first character 
exp_list = exp_list[1:]
# convert to list of ints
exp_list = [int(exp) for exp in exp_list]

# load the .npy file in each experiment folder in the result_path

symbol_list = []
color_list = []
label_list = []
gbest_values = []
opt_value = 0
for i, exp_num in enumerate(exp_list):
    
    exp_path = result_path + "exp_" + str(exp_num) + "/"
    # load the .npy file
    gbest_values.append(np.load(exp_path + f"gbestVals.npy"))
    print(gbest_values[i].shape)
    # get the label
    # load json
    with open(exp_path + "run_summary.json") as f:
        config = json.load(f)
    label_list.append(config["title"])
    # get the symbol
    symbol_list.append(all_symbols[i])
    # get the color
    color_list.append(all_colors[i])
    
opt_value = config["opt_value"]

# plot the gbest_values
plot_num_function_evaluation(fopt=gbest_values, label_list=label_list, symbol_list=symbol_list, color_list=color_list, save_dir=f"experiments/results/comparison_{args.exp_numC}.png",
                             n_agents=10, opt_value=opt_value, log_scale=False)