#!/bin/bash

# Common variables
SCRIPT_NAME="experiments/run_experiment3.py"
ITERS=20
ACTION_STD=0.005
DECAY_RATE=0.999
DECAY_START=1000
TOL=0.99
W=1
C1=2
C2=2
SPLIT_INTERVAL=250
function_end=29
n_dims=(2) # Changed to bash array syntax

# Loop over each dimension in n_dims
for n_dim in "${n_dims[@]}"
do
    echo "Running experiments for n_dim=$n_dim"
    source ../lit-llama/llmenv/bin/activate

    start_exp_num=0

    # Loop over function_id from 0 to 29
    for function_id in $(seq 0 $function_end)
    do
        echo "Running experiment from $start_exp_num to $((start_exp_num+6)) for function_id $function_id with n_dim=$n_dim"
        exp_list=""
        EXP_NUM=$start_exp_num

        # Experiment 1: NO SPLITTING - UNFREEZE
        exp_list="$exp_list,$EXP_NUM"
        TITLE="NO SPLITTING-FREEZE"
        echo "Running experiment $EXP_NUM: $TITLE for function_id $function_id"
        python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.5 --role_std_explorers 0.5 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --decay_std --decay_start $DECAY_START --split_type "use_stds" --plot_gbest --iters $ITERS --tol $TOL --function_id $function_id --n_dim $n_dim --freeze

        # Experiment 3: NO SPLITTING - DELAYED DECAY
        EXP_NUM=$((EXP_NUM+1))
        exp_list="$exp_list,$EXP_NUM"
        TITLE="NO SPLITTING-DELAYED DECAY-FREEZE"
        echo "Running experiment $EXP_NUM: $TITLE for function_id $function_id"
        python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --freeze --role_std_exploiters 0.5 --role_std_explorers 0.5 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --decay_std --decay_start 2500 --split_type "use_stds" --plot_gbest --iters $ITERS --tol $TOL --function_id $function_id --n_dim $n_dim --freeze

        Experiment 5: SPLITTING - WITH STDs
        EXP_NUM=$((EXP_NUM+1))
        exp_list="$exp_list,$EXP_NUM"
        TITLE="SPLITTING-WITH STDs"
        echo "Running experiment $EXP_NUM: $TITLE for function_id $function_id"
        python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.03 --role_std_explorers 0.5 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_stds" --plot_gbest --iters $ITERS --tol $TOL --function_id $function_id --n_dim $n_dim --freeze

        # # Experiment with dynamic split
        # EXP_NUM=$((EXP_NUM+1))
        # exp_list="$exp_list,$EXP_NUM"
        # TITLE="SPLITTING-WITH STDs - Dynamic-split"
        # echo "Running experiment $EXP_NUM: $TITLE for function_id $function_id"
        # python $SCRIPT_NAME --title "$TITLE" --function_id $function_id --exp_num $EXP_NUM --use_gbest --freeze --role_std_exploiters 0.02 --role_std_explorers 0.5 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_stds" --plot_gbest --iters $ITERS --tol $TOL --split_interval $SPLIT_INTERVAL --use_split_interval --n_dim $n_dim --dynamic_split

        # # Experiment 7: SPLITTING - TWO POLICIES
        # EXP_NUM=$((EXP_NUM+1))
        # exp_list="$exp_list,$EXP_NUM"
        # TITLE="SPLITTING-TWO POLICIES"
        # echo "Running experiment $EXP_NUM: $TITLE for function_id $function_id"
        # python $SCRIPT_NAME --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.4 --role_std_explorers 0.4 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_two_policies" --plot_gbest --iters $ITERS --tol $TOL --exploit_std 0.03 --function_id $function_id --n_dim $n_dim

        # # Dynamic split version of TWO POLICIES
        # EXP_NUM=$((EXP_NUM+1))
        # exp_list="$exp_list,$EXP_NUM"
        # TITLE="SPLITTING-TWO POLICIES - Dynamic-split"
        # echo "Running experiment $EXP_NUM: $TITLE for function_id $function_id"
        # python $SCRIPT_NAME --function_id $function_id --title "$TITLE" --exp_num $EXP_NUM --role_std_exploiters 0.4 --role_std_explorers 0.4 --variable_std --action_std $ACTION_STD --decay_rate $DECAY_RATE --split_agents --split_type "use_two_policies" --plot_gbest --iters $ITERS --tol $TOL --exploit_std 0.03 --dynamic_split --split_interval $SPLIT_INTERVAL --use_split_interval --n_dim $n_dim

        # PSO Experiment
        EXP_NUM=$((EXP_NUM+1))
        exp_list="$exp_list,$EXP_NUM"
        TITLE="PSO"
        echo "Running experiment $EXP_NUM: $TITLE for function_id $function_id"
        python experiments/run_experiment3.py --function_id $function_id --title "$TITLE" --exp_num $EXP_NUM --algo "$TITLE" --iters $ITERS --plot_gbest --pso_w $W --pso_c1 $C1 --pso_c2 $C2 --n_dim $n_dim

        wait 
        python experiments/compare.py --exp_list "$exp_list" --exp_numC $((function_id)) --minimize --n_dim $n_dim

        start_exp_num=$(((function_id+1)*10 + 1))

    done
    # Wait for all background jobs to finish
    wait
    # Run comparisons for each function_id (if needed, ensure this part also uses the current n_dim)
    python experiments/benchmark.py --n_dim $n_dim

    echo "Finished running all experiments for n_dim=$n_dim"
done

echo "Finished running experiments for all dimensions"
