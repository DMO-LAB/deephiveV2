#!/bin/bash

# Common variables
SCRIPT_NAME="experiments/run_experiment2.py"
ITERS=100
VARIABLE_STD=True  # Note: This variable is no longer needed as the flag will be used directly
ACTION_STD=0.02
DECAY_RATE=0.9
DECAY_START=0
PLOT_GBEST=True  # Note: This variable is no longer needed as the flag will be used directly

# Experiment 2: NO SPLITTING - DECAY STD - FREEZE
EXP_NUM=1010
TITLE="NO SPLITTING - DECAY STD - FREEZE"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --freeze --role_std_exploiters 0.3 --role_std_explorers 0.3 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --decay_std --decay_start $DECAY_START --split_type "use_stds" --plot_gbest --iters $ITERS --tol 0.99 


# Experiment 3: NO SPLITTING - WITH DECAYED STD - DELAYED DECAY
EXP_NUM=$((EXP_NUM+1))
TITLE="NO SPLITTING - WITH DECAYED STD - DELAYED DECAY - FREEZE"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --freeze --role_std_exploiters 0.3 --role_std_explorers 0.3 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --decay_std --decay_start 4 --split_type "use_stds" --plot_gbest --iters $ITERS --tol 0.99 

# Experiment 4: NO SPLITTING - WITH DECAYED STD - DELAYED DECAY - UNFREEZE
EXP_NUM=$((EXP_NUM+1))
TITLE="NO SPLITTING - WITH DECAYED STD - DELAYED DECAY - UNFREEZE"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.3 --role_std_explorers 0.3 --variable_std --action_std $ACTION_STD --decay_rate 0.95 --decay_std --decay_start 4 --split_type "use_stds" --plot_gbest --iters $ITERS --tol 0.99 

# Experiment 5: SPLITTING - WITH STDs
EXP_NUM=$((EXP_NUM+1))
TITLE="SPLITTING - WITH STDs"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.02 --role_std_explorers 0.2 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_stds" --plot_gbest --iters $ITERS --tol 0.99 

# Experiment 6: SPLITTING - USE GRID - pbest
EXP_NUM=$((EXP_NUM+1))
TITLE="SPLITTING - USE GRID - pbest"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.02 --role_std_explorers 0.02 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_grid" --plot_gbest --iters $ITERS --tol 0.99 --exploit_std 0.02 --policy_type "pbest" 


# Experiment 6: SPLITTING - USE GRID - gbest
EXP_NUM=$((EXP_NUM+1))
TITLE="SPLITTING - USE GRID - gbest"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --use_gbest --role_std_exploiters 0.02 --role_std_explorers 0.02 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_grid" --plot_gbest --iters $ITERS --tol 0.99 --exploit_std 0.02 --policy_type "gbest" 


# Experiment 7: SPLITTING - TWO POLICIES
EXP_NUM=$((EXP_NUM+1))
TITLE="SPLITTING - TWO POLICIES"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.4 --role_std_explorers 0.4 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_two_policies" --plot_gbest --iters $ITERS --tol 0.99 --exploit_std 0.02 

EXP_NUM=$((EXP_NUM+1))
TITLE="NO SPLITTING - DECAY STD - UNFREEZE"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.3 --role_std_explorers 0.3 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --decay_std --decay_start $DECAY_START --split_type "use_stds" --plot_gbest --iters $ITERS --tol 0.99 


# Exit
exit 0