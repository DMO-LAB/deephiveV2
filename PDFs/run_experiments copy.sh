#!/bin/bash

# Common variables
SCRIPT_NAME="experiments/run_experiment2.py"
ITERS=100
VARIABLE_STD=True  # Note: This variable is no longer needed as the flag will be used directly
ACTION_STD=0.02
DECAY_RATE=0.99
DECAY_START=0
PLOT_GBEST=True  # Note: This variable is no longer needed as the flag will be used directly
TOL=0.99
exp_list=""
W=0.5
C1=0.7
C2=0.4

start_exp_num=50
EXP_NUM=$start_exp_num

Experiment 2: NO SPLITTING - DECAY STD - FREEZE
EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="NO SPLITTING - FREEZE"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --freeze --role_std_exploiters 0.5 --role_std_explorers 0.5 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --decay_std --decay_start $DECAY_START --split_type "use_stds" --plot_gbest --iters $ITERS --tol $TOL


Experiment 3: NO SPLITTING - WITH DECAYED STD - DELAYED DECAY
EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="NO SPLITTING-DELAYED DECAY-FREEZE"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --freeze --role_std_exploiters 0.5 --role_std_explorers 0.5 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --decay_std --decay_start 10 --split_type "use_stds" --plot_gbest --iters $ITERS --tol $TOL

# Experiment 4: NO SPLITTING - WITH DECAYED STD - DELAYED DECAY - UNFREEZE
EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="NO SPLITTING-DELAYED DECAY-UNFREEZE"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.5 --role_std_explorers 0.5 --variable_std --action_std $ACTION_STD --decay_rate 0.9 --decay_std --decay_start 10 --split_type "use_stds" --plot_gbest --iters $ITERS --tol $TOL

# Experiment 5: SPLITTING - WITH STDs
EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="SPLITTING-WITH STDs"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.03 --role_std_explorers 0.4 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_stds" --plot_gbest --iters $ITERS --tol $TOL

# Experiment 6: SPLITTING - USE GRID - pbest
# EXP_NUM=$((EXP_NUM+1))
# TITLE="SPLITTING-USE GRID-pbest"
# echo "Running experiment $EXP_NUM: $TITLE"
# python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.02 --role_std_explorers 0.02 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_grid" --plot_gbest --iters $ITERS --tol $TOL --exploit_std 0.02 --policy_type "pbest" 


# # Experiment 6: SPLITTING - USE GRID - gbest
# EXP_NUM=$((EXP_NUM+1))
# exp_list="$exp_list,$EXP_NUM"
# TITLE="SPLITTING-USE GRID-gbest"
# echo "Running experiment $EXP_NUM: $TITLE"
# python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --use_gbest --role_std_exploiters 0.03 --role_std_explorers 0.03 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_grid" --plot_gbest --iters $ITERS --tol $TOL --exploit_std 0.03 --policy_type "gbest" 


# Experiment 7: SPLITTING - TWO POLICIES
EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="SPLITTING-TWO POLICIES"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.5 --role_std_explorers 0.5 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_two_policies" --plot_gbest --iters $ITERS --tol $TOL --exploit_std 0.03


EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="NO SPLITTING-UNFREEZE"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.5 --role_std_explorers 0.5 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --decay_std --decay_start $DECAY_START --split_type "use_stds" --plot_gbest --iters $ITERS --tol $TOL

EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="PSO"
echo "Running experiment $EXP_NUM: $TITLE"
python experiments/run_experiment2.py --title "$TITLE" --exp_num $EXP_NUM --algo "$TITLE" --iters $ITERS --plot_gbest --pso_w $W --pso_c1 $C1 --pso_c2 $C2


EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="GA"
echo "Running experiment $EXP_NUM: $TITLE"
python experiments/run_experiment2.py --title "$TITLE" --exp_num $EXP_NUM --algo "$TITLE" --iters $ITERS --plot_gbest

# Wait for the above scripts to finish
wait

# Then run the third script
python experiments/compare.py --exp_list "$exp_list" --exp_numC $start_exp_num
