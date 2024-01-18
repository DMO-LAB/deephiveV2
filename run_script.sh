#!/bin/bash

# Define variables
exp_num1=1030
exp_num2=$((exp_num1 + 1))

exp_numC=19

# Activate the virtual environment
workon mlEnv

# Run the first two Python scripts in parallel
# python experiments/run_experiment.py --config_path "config/config.json" --model_path "models/pbest_freeze.pth" --exp_num $exp_num1 --split_agents False --save_gif True --run_summary "NO SPLITTING" &
# python experiments/run_experiment.py --config_path "config/split_config.json" --model_path "models/gbest.pth" --exp_num $exp_num2 --split_agents True --save_gif True --run_summary "SPLITTING" &
python experiments/run_experiment.py --config_path config/exp_config.json --model_path models/pbest_unfreeze.pth --run_summary "No-Splitting-uniform-stds-with-decay - 100" --decay_std True --save_gif True --exp_num $exp_num1 --iter 100   &
python experiments/run_experiment.py --config_path config/exp_2_config.json --model_path models/gbest.pth --run_summary "Splitting-with grid" --decay_std False --save_gif True --exp_num $exp_num2 --split_agent True --iters 100
# Wait for the above scripts to finish
# wait

# # Then run the third script
# python experiments/compare.py --exp_list "$exp_num1,$exp_num2" --exp_numC $exp_numC


# python experiments/run_experiment.py --config_path config/exp_2_config.json --model_path models/gbest.pth --iters 10 --run_summary "No-Splitting-uniform-stds-with-decay" --decay_std False --save_gif True --exp_num 1009
