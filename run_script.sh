#!/bin/bash

# Run the first two Python scripts in parallel
python experiments/run_experiment.py --config_path "config/config.json" --model_path "models/pbest_freeze.pth" --exp_num 4 --split_agents False --save_gif True --run_summary "NO SPLITTING" &
python experiments/run_experiment.py --config_path "config/split_config.json" --model_path "models/gbest.pth" --exp_num 5 --split_agents True --save_gif True --run_summary "SPLITTING" &

# Wait for the above scripts to finish
wait

# Then run the third script
python experiments/compare.py --exp_list "4,5" --exp_num 2
