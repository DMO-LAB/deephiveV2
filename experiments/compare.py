import os 
import numpy as np
from environment.deephive_utils import *
from environment.utils import *


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--result_path", type=str, default="experiments/results/")
parser.add_argument("--exp_list", type=str, default="2,3")
parser.add_argument("--exp_num", type=int, default=1)

args = parser.parse_args()

config_path = "config/config.json"
# load config
config = parse_config(config_path)

all_symbols = ["-", "--", "-.", ":"]
all_colors = ["r", "g", "b", "k"]

result_path = args.result_path
exp_list = args.exp_list
# convert to list of ints
exp_list = [int(exp) for exp in exp_list.split(",")]

# load the .npy file in each experiment folder in the result_path

symbol_list = []
color_list = []
label_list = []
gbest_values = []
for i, exp_num in enumerate(exp_list):
    exp_path = result_path + "experiment_" + str(exp_num) + "/"
    # load the .npy file
    gbest_values.append(np.load(exp_path + f"experiment_{exp_num}.npy"))
    # get the label
    with open(exp_path + "run_summary.txt", "r") as f:
        lines = f.readlines()
        label = lines[0].strip()
        label_list.append(label)
    # get the symbol
    symbol_list.append(all_symbols[i])
    # get the color
    color_list.append(all_colors[i])

# plot the gbest_values
plot_num_function_evaluation(fopt=np.array(gbest_values), label_list=label_list, symbol_list=symbol_list, color_list=color_list, save_dir=f"experiments/results/comparison_{exp_num}.png",
                             n_agents=10, opt_value=4.808)