#!/bin/bash

# Define variables
exp_num1=32
exp_num2=$((exp_num1 + 1))

exp_numC=19

# Activate the virtual environment
workon DLEnv

# Run the first two Python scripts in parallel
python experiments/run_experiment.py --config_path "config/config.json" --model_path "models/pbest_freeze.pth" --exp_num $exp_num1 --split_agents False --save_gif True --run_summary "NO SPLITTING" &
python experiments/run_experiment.py --config_path "config/split_config.json" --model_path "models/gbest.pth" --exp_num $exp_num2 --split_agents True --save_gif True --run_summary "SPLITTING" &

# Wait for the above scripts to finish
wait

# Then run the third script
python experiments/compare.py --exp_list "$exp_num1,$exp_num2" --exp_numC $exp_numC


