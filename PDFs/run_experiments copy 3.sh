#!/bin/bash

# Common variables
SCRIPT_NAME="experiments/run_experiment3.py"
ITERS=100
ACTION_STD=0.02
DECAY_RATE=0.99
DECAY_START=0
TOL=0.99
exp_list=""
W=0.3
C1=0.4
C2=0.3
SPLIT_INTERVAL=3
function_id=0

start_exp_num=2000
EXP_NUM=$start_exp_num

Experiment 1: NO SPLITTING - UNFREEZE
exp_list="$exp_list,$EXP_NUM"
TITLE="NO SPLITTING-UNFREEZE"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.5 --role_std_explorers 0.5 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --decay_std --decay_start $DECAY_START --split_type "use_stds" --plot_gbest --iters $ITERS --tol $TOL --function_id $function_id


Experiment 3: NO SPLITTING - DELAYED DECAY
EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="NO SPLITTING-DELAYED DECAY-FREEZE"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --freeze --role_std_exploiters 0.5 --role_std_explorers 0.5 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --decay_std --decay_start 10 --split_type "use_stds" --plot_gbest --iters $ITERS --tol $TOL --function_id $function_id

Experiment 5: SPLITTING - WITH STDs
# EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="SPLITTING-WITH STDs"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.03 --role_std_explorers 0.4 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_stds" --plot_gbest --iters $ITERS --tol $TOL --function_id $function_id


# EXP_NUM=$((EXP_NUM+1))
EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="SPLITTING-WITH STDs - Dynamic-split"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --function_id $function_id --exp_num $EXP_NUM --use_gbest --freeze --role_std_exploiters 0.002 --role_std_explorers 0.4 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_stds" --plot_gbest --iters $ITERS --tol $TOL --split_interval $SPLIT_INTERVAL --use_split_interval


# Experiment 7: SPLITTING - TWO POLICIES
EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="SPLITTING-TWO POLICIES"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.4 --role_std_explorers 0.4 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_two_policies" --plot_gbest --iters $ITERS --tol $TOL --exploit_std 0.03 --function_id $function_id


EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="SPLITTING-TWO POLICIES - Dynamic-split"
echo "Running experiment $EXP_NUM: $TITLE"
python $SCRIPT_NAME  --function_id $function_id--title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.4 --role_std_explorers 0.4 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_two_policies" --plot_gbest --iters $ITERS --tol $TOL --exploit_std 0.03 --dynamic_split --split_interval $SPLIT_INTERVAL --use_split_interval


EXP_NUM=$((EXP_NUM+1))
exp_list="$exp_list,$EXP_NUM"
TITLE="PSO"
echo "Running experiment $EXP_NUM: $TITLE"
python experiments/run_experiment2.py  --function_id $function_id --title "$TITLE" --exp_num $EXP_NUM --algo "$TITLE" --iters $ITERS --plot_gbest --pso_w $W --pso_c1 $C1 --pso_c2 $C2

# Wait for the above scripts to finish
wait

# Then run the third script
python experiments/compare.py --exp_list "$exp_list" --exp_numC $start_exp_num --minimize &
wait
python experiments/compare.py --exp_list "$exp_list" --exp_numC $start_exp_num 
echo "Finished running all experiments"

