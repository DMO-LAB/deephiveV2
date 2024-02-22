import sys 
sys.path.append('../')
import os
from deephive.environment.deephive_utils import *
from deephive.environment.utils import *
import numpy as np 
from deephive.exploration.gp_surrogate import GPSurrogateModule
from deephive.environment.utils import filter_points
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern
from sko.GA import GA
from sko.PSO import PSO
from sko.SA import SA
from dotenv import load_dotenv
load_dotenv()
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import argparse
import traceback
import time 
from concurrent.futures import ProcessPoolExecutor


def run_experiment_other_algorithm(algorithm:[GA, PSO, SA], env, config, exp_name, title=""):
    base_path = f'experiments/results_{env.n_dim}/'
    os.makedirs(base_path, exist_ok=True)
    result_path = base_path + exp_name + '/'
    os.makedirs(result_path, exist_ok=True)
    
    objective_function = lambda x: -env.objective_function.evaluate(np.array([x]).reshape(1, -1))[0]
    lower_bound, upper_bound = env.objective_function.bounds(dim=env.n_dim)[0].tolist(), env.objective_function.bounds(dim=env.n_dim)[1].tolist()
    
    gbestVals = []
    opt_value = env.objective_function.optimal_value(dim=env.n_dim)
    try:
        iteration_start_time = time.time()
        for _ in range(config['iters']):
            try:
                if isinstance(algorithm, GA):
                    algorithm = GA(func=objective_function, n_dim=env.n_dim, size_pop=config['n_agents'], max_iter=config['ep_length'], lb=lower_bound, ub=upper_bound, precision=[1e-7 for _ in range(env.n_dim)])
                    ga = algorithm
                    _ = ga.run()
                    gbVal = [ga.generation_best_Y[i] * -1 for i in range(len(ga.generation_best_Y))]
                    gbestVals.append(gbVal)
                elif isinstance(algorithm, PSO):
                    algorithm = PSO(func=objective_function, n_dim=env.n_dim, pop=config['n_agents'], max_iter=config['ep_length'], lb=lower_bound, ub=upper_bound,
                                    w = config['w'], c1 = config['c1'], c2 = config['c2'])
                    pso = algorithm
                    _ = pso.run()
                    gbVal = [pso.gbest_y_hist[i][0] * -1 for i in range(len(pso.gbest_y_hist))]
                    gbestVals.append(gbVal)
                elif isinstance(algorithm, SA):
                    algorithm = SA(func=objective_function, x0=lower_bound, T_max=config['T_max'], T_min=config['T_min'], L=config['L'], max_stay_counter=config['max_stay_counter'])
                    sa = algorithm
                    _ = sa.run()
                    gbestVals.append(sa.best_y_history * -1)
                else:
                    raise ValueError("algorithm must be either GA, PSO or SA")
            except Exception as e:
                raise e
        iteration_end_time = time.time()
        print(f"[PSO]: Time taken for {config['iters']} iterations: {iteration_end_time - iteration_start_time} seconds")
        
        gb = np.array(gbestVals)
        np.save(result_path + "gbestVals.npy", gb)
            
        if config['plot_gbest']:
                num_function_evaluation(fopt=gbestVals ,n_agents=env.n_agents, save_dir=result_path + "num_function_evaluations.png", opt_value=opt_value,
                                        log_scale=False, plot_error_bounds=True, title=title)
                plot_individual_function_evaluation(fopt=gbestVals ,n_agents=env.n_agents, save_dir=result_path + "num_function_evaluations2.png", opt_value=opt_value,
                                        log_scale=False, title=title)
                
        run_summary = {
        "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "title": title,
        "opt_value": opt_value,
    }
        
        with open(result_path + "run_summary.json", 'w') as f:
            json.dump(run_summary, f)
    
    except Exception as e:
        print(f"Error in experiment {exp_name}")
        # print(e)
        # traceback.print_exc()
        return None

def get_agent_actions(env, policy, obs, config, roles=None, split_agents=False,
                      split_type="use_stds", decay_std=False, split_ratio=1, dynamic_split=False, **kwargs):
    
    if decay_std:
        if isinstance(policy, list):
            policy[0].std_controller.decay_std()
            policy[1].std_controller.decay_std()
        else:
            policy.std_controller.decay_std()
        
    # Pre-compute common variables and conditions
    variable_std_enabled = config["variable_std"][0]
    if isinstance(policy, list):
        obs_std = policy[0].std_controller.get_all_std() if variable_std_enabled else None
    else:
        obs_std = policy.std_controller.get_all_std() if variable_std_enabled else None

    if not split_agents:
        actions = get_action(obs, policy, env, obs_std) if variable_std_enabled else get_action(obs, policy, env)
        return actions
    else:
        assert roles is not None, "Roles must be provided if split_agents is True"
        # Efficiently manage agent splitting
        if kwargs.get("exploiter_ids") is not None: 
            exploiters_id = kwargs.get("exploiter_ids")
            explorers_id = kwargs.get("explorer_ids")
        else:
            if not dynamic_split:
                exploiters_id = np.where(roles[0] == 1)[0]
                explorers_id = np.where(roles[0] == 0)[0]
            else:
                explorers_id = np.random.choice(env.n_agents, int(env.n_agents * split_ratio), replace=False)
                exploiters_id = np.setdiff1d(np.arange(env.n_agents), explorers_id)
                #print(f"Explorers: {explorers_id}, Exploiters: {exploiters_id}")

        #print(f"Explorers: {explorers_id}, Exploiters: {exploiters_id}")
        roles = np.zeros(env.n_agents)
        exploiters_id = np.array(exploiters_id, dtype=int)
        roles[exploiters_id] = 1
        roles = np.tile(roles, (env.n_dim, 1)).T  # Optimized roles computation
        env.state_history[:, env.current_step, -2] = roles[:, 0]  # Updated for vectorization

        roles = roles.reshape(env.n_dim, env.n_agents)
        if split_type == "use_stds":
            policy.std_controller.update_roles(roles)
            assert variable_std_enabled, "variable_std must be True in config"
            assert config['role_std']['explorer'] > config['role_std']['exploiter'], "explorers must have higher std than exploiters"
            actions = get_action(obs, policy, env, obs_std)
        elif split_type == "use_grid":
            policy.std_controller.update_roles(roles)
            explorer_actions, _ = get_informed_action(env, env.n_agents)
            exploiter_actions = get_action(obs, policy, env, obs_std)
            actions = np.zeros((env.n_agents, env.n_dim))
            actions[explorers_id] = explorer_actions[explorers_id]
            actions[exploiters_id] = exploiter_actions[exploiters_id]
        elif split_type == "use_two_policies":
            policy[0].std_controller.update_roles(roles)
            obs_std = policy[0].std_controller.get_all_std()
            obs_explore, _ = env.observation_schemes.generate_observation(pbest=env.pbest.copy(), use_gbest=False, ratio=env.split_ratio)
            obs_exploit, _ = env.observation_schemes.generate_observation(pbest=env.pbest.copy(), use_gbest=True, ratio=env.split_ratio)
            exploit_std_obs = policy[1].std_controller.get_all_std(std=config["exploit_std"])
            assert len(policy) == 2, "Two policies must be provided"
            explorer_actions = get_action(obs_explore, policy[0], env, obs_std)
            exploiter_actions = get_action(obs_exploit, policy[1], env, exploit_std_obs)
            actions = np.zeros((env.n_agents, env.n_dim))
            actions[explorers_id] = explorer_actions[explorers_id]
            actions[exploiters_id] = exploiter_actions[exploiters_id]

        return actions
        
        
def optimize(env, agent_policy, obs, roles, config):
    gbests = []
    split_interval = config['split_interval']
    if config['use_split_intervals']:
        exploiters_id = np.array([])
        explorers_id = np.arange(env.n_agents)
    else:
        exploiters_id = np.where(roles[0] == 1)[0]
        explorers_id = np.where(roles[0] == 0)[0]
    episode_start_time = time.time()
    for i in range(config['ep_length']):
        if i == config['decay_start']:
            decay_std_run = config['decay_std']
        else:
            decay_std_run = False
        if i % split_interval == 0 and config['use_split_intervals']:
            #env.render()
            if i != 0:
                if len(explorers_id) > 0:
                    # add one explorer to the exploiters
                    exploiters_id = np.append(exploiters_id, int(explorers_id[0]))
                    # remove the explorer from the explorers
                    explorers_id = explorers_id[1:]
            explorers_id = np.array(explorers_id).astype(int) if len(explorers_id) > 0 else np.array([])
            exploiters_id = np.array(exploiters_id).astype(int) if len(exploiters_id) > 0 else np.array([])

            
                # add one explorer to the exploiters
        actions = get_agent_actions(env, agent_policy, obs, config, roles=roles, split_agents=config["split_agents"], 
                                    split_type=config["split_type"], decay_std=decay_std_run, split_ratio=config["split_ratio"],
                                    dynamic_split=config["dynamic_split"], exploiter_ids=exploiters_id, explorer_ids=explorers_id)
        observation_info, _, _, _ = env.step(actions)
        obs, roles = observation_info
        gbests.append(env.gbest[-1])
        #print(f"Step {i} - Best value: {env.gbest[-1]}")
    episode_end_time = time.time()
    #print(f"Time taken for {config['ep_length']} iterations: {episode_end_time - episode_start_time} seconds")
    return gbests
        
        
def run_single_iteration(env, agent_policy, config, result_path, iter_num, save_gif=False):
    # This function is designed to run a single iteration of the experiment.
    # Since multiprocessing requires functions to be picklable, global variables and complex objects
    # passed to this function should be minimized or managed accordingly.

    env = env.deepcopy()
    #agent_policy = agent_policy.deepcopy()
    
    obs, roles = env.reset()
    if isinstance(agent_policy, list):
        agent_policy[0].std_controller.reset_std()
        agent_policy[1].std_controller.reset_std()
    else:
        agent_policy.std_controller.reset_std()
    episode_gbest = [env.gbest[-1]]
    try:
        gbests = optimize(env, agent_policy, obs, roles, config)
        episode_gbest.extend(gbests)
        #print(f"Episode {iter_num} - Best value: {env.gbest[-1]}")
    except Exception as e:
        traceback.print_exc()
        raise e
    
    if save_gif and iter_num % 100 == 0:
        env.render(type="history", file_path=result_path + "history_" + str(iter_num) + "_.gif")
    
    return episode_gbest

def run_experiment_concurrently(env, agent_policy, config, exp_name, save_gif=False, title=""):
    base_path = f"experiments/results_{env.n_dim}/"
    os.makedirs(base_path, exist_ok=True)
    result_path = base_path + exp_name + '/'
    os.makedirs(result_path, exist_ok=True)

    gbestVals = []

    start_time = time.time()
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_single_iteration, env, agent_policy, config, result_path, iter_num, save_gif)
                   for iter_num in range(config['iters'])]

        for future in futures:
            try:
                episode_gbest = future.result()
                gbestVals.append(episode_gbest)
            except Exception as e:
                traceback.print_exc()
                print(f"Error in iteration: {e}")

    np.save(result_path + "gbestVals.npy", np.array(gbestVals))
    if config['plot_gbest']:
            
                num_function_evaluation(fopt=gbestVals ,n_agents=env.n_agents, save_dir=result_path + "num_function_evaluations.png", opt_value=opt_value,
                                        log_scale=False, plot_error_bounds=True, title=title)
                
                plot_individual_function_evaluation(fopt=gbestVals ,n_agents=env.n_agents, save_dir=result_path + "num_function_evaluations2.png", opt_value=opt_value,
                                        log_scale=False, title=title)
                
                
    run_summary = {
    "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "title": title,
    "split_agents": config['split_agents'],
    "split_type": config['split_type'],
    "decay_std": config['decay_std'],
    "decay_start": config['decay_start'],
    "freeze": config['freeze'],
    "use_gbest": config['use_gbest'],
    "variable_std": config['variable_std'],
    "role_std": config['role_std'],
    "decay_std": config['decay_std'],
}
    
    with open(result_path + "run_summary.json", 'w') as f:
        json.dump(run_summary, f)
        
    end_time = time.time()
    print(f"[DEEPHIVE]: Time taken for {config['iters']} iterations: {end_time - start_time} seconds")
    
    
def run_experiment_sequentially(env, agent_policy, config, exp_name, save_gif=False, title=""):
    base_path = f"experiments/results_{env.n_dim}/"
    os.makedirs(base_path, exist_ok=True)
    result_path = base_path + exp_name + '/'
    os.makedirs(result_path, exist_ok=True)

    gbestVals = []

    start_time = time.time()
    for i in range(config['iters']):
        try:
            episode_gbest = run_single_iteration(env, agent_policy, config, result_path, i, save_gif)
            gbestVals.append(episode_gbest)
        except Exception as e:
            traceback.print_exc()
            print(f"Error in iteration: {e}")
            
    # end_time = time.time()
    np.save(result_path + "gbestVals.npy", np.array(gbestVals))
    if config['plot_gbest']:
        num_function_evaluation(fopt=gbestVals ,n_agents=env.n_agents, save_dir=result_path + "num_function_evaluations.png", opt_value=opt_value,
                                log_scale=False, plot_error_bounds=True, title=title)
        
        plot_individual_function_evaluation(fopt=gbestVals ,n_agents=env.n_agents, save_dir=result_path + "num_function_evaluations2.png", opt_value=opt_value,
                                log_scale=False, title=title)
                    
    run_summary = {
    "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "title": title,
    "split_agents": config['split_agents'],
    "split_type": config['split_type'],
    "decay_std": config['decay_std'],
    "decay_start": config['decay_start'],
    "freeze": config['freeze'],
    "use_gbest": config['use_gbest'],
    "variable_std": config['variable_std'],
    "role_std": config['role_std'],
    "decay_std": config['decay_std'],
}
    with open(result_path + "run_summary.json", 'w') as f:
        json.dump(run_summary, f)
        
    end_time = time.time()
    print(f"[DEEPHIVE]: Time taken for {config['iters']} iterations: {end_time - start_time} seconds")
    
    

        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str, default='test')
    parser.add_argument('--exp_num', type=int, default=1)
    parser.add_argument('--algo', type=str, default='Deephive', choices=["Deephive","GA", "PSO", "SA"])
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--use_gbest', action='store_true')
    parser.add_argument('--split_interval', type=int, default=2)
    parser.add_argument('--role_std_exploiters', type=float, default=0.2)
    parser.add_argument('--role_std_explorers', type=float, default=0.2)
    parser.add_argument('--variable_std', action='store_true')
    parser.add_argument('--action_std', type=float, default=0.02)
    parser.add_argument('--decay_rate', type=float, default=0.9)
    parser.add_argument('--decay_std', action='store_true')
    parser.add_argument('--decay_start', type=int, default=0)
    parser.add_argument('--split_agents', action='store_true')
    parser.add_argument('--split_type', type=str, default="use_stds", choices=["use_stds", "use_grid", "use_two_policies"])
    parser.add_argument('--plot_gbest', action='store_true')
    parser.add_argument('--tol', type=float, default=.99)
    parser.add_argument('--exploit_std', type=float, default=0.02)
    parser.add_argument('--policy_type', type=str, default="pbest", choices=["pbest", "gbest"])
    parser.add_argument('--pso_w', type=float, default=0.8)
    parser.add_argument('--pso_c1', type=float, default=2)
    parser.add_argument('--pso_c2', type=float, default=2)
    parser.add_argument('--dynamic_split', action='store_true')
    parser.add_argument('--use_split_intervals', action='store_true')
    parser.add_argument('--split_ratio', type=float, default=0.5)
    parser.add_argument('--function_id', type=int, default=0)
    parser.add_argument('--n_dim', type=int, default=2)

    


    args = parser.parse_args()

    config_path = "config/exp_config.json"
    model_path = "models/pbest_unfreeze.pth"
    model_path_2 = "models/gbest.pth"

    config = parse_config(config_path)

    config['n_dim'] = args.n_dim
    config['freeze'] = args.freeze
    config['use_gbest'] = args.use_gbest
    config['role_std'] = {'explorer': args.role_std_explorers, 'exploiter': args.role_std_exploiters}
    config['variable_std'] = args.variable_std,
    config['action_std'] = args.action_std
    config['decay_rate'] = args.decay_rate
    config['decay_std'] = args.decay_std
    config['decay_start'] = args.decay_start
    config['split_agents'] = args.split_agents
    config['split_type'] = args.split_type
    config['plot_gbest'] = args.plot_gbest
    config['iters'] = args.iters
    config['tol'] = args.tol
    config['exploit_std'] = args.exploit_std
    config['w'] = args.pso_w
    config['c1'] = args.pso_c1
    config['c2'] = args.pso_c2
    config['split_interval'] = args.split_interval
    config['dynamic_split'] = args.dynamic_split
    config['use_split_intervals'] = args.use_split_intervals
    config['split_ratio'] = args.split_ratio
    config["use_optimal_value"] = False
    config["function_id"] = args.function_id
    if args.split_type == "use_grid":
        config['use_grid']  = True

    mode = "test"
    
    hybrid_functions = ['f10','f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f28', 'f29', 'f30']
    if f"f{config['function_id']}" in range(11, 21) and config["n_dim"] <= 2:
        print(f"[ERROR] - Hybrid function {config['function_id']} is not supported for n_dim <= 2")
        sys.exit(1)
        
    # if config["n_dim"] == 2:
    #     config["n_agents"] = 10
    # elif config["n_dim"] == 50:
    #     config["n_agents"] = 60
    # elif config["n_dim"] == 100:
    #     config["n_agents"] = 100
    # else:
    #     config["n_agents"] = 50
        
    exp_name = "exp_" + str(args.exp_num)
    result_path = f'experiments/results_{config["n_dim"]}/' + str(exp_name) + '/' 
    os.makedirs(result_path, exist_ok=True)

    env, agent_policies = initialize(config, mode=mode, model_path=[model_path, model_path_2])
    pbest_policy = agent_policies[0]
    gbest_policy = agent_policies[1]
    if args.policy_type == "pbest":
        agent_policy = pbest_policy
    elif args.policy_type == "gbest":
        agent_policy = gbest_policy
    else:
        raise ValueError("policy_type must be either pbest or gbest")
    
    opt_value = env.objective_function.optimal_value(dim=env.n_dim)
    
    optimization_function_name = str(env.objective_function)
    print(f"[INFO] - Optimization function: {optimization_function_name}")
    
    title = f"{optimization_function_name} - {args.title}" 
    
    if config['use_split_intervals']:
        config['use_gbest'] = True
    
    if config["split_type"] == "use_two_policies":
        assert len(agent_policies) == 2, "Two policies must be provided - first policy is for explorers and second policy is for exploiters"
        assert config['split_agents'] == True, "split_agents must be True in config"
        assert config['split_type'] == "use_two_policies", "split_type must be use_two_policies in config"
        assert len(agent_policies) == 2, "Two policies must be provided - first policy is for explorers and second policy is for exploiters"
        agent_policy = agent_policies
    
    print(f"[INFO] - Running experiment {exp_name} ..." )
    print(f"[TITLE] - {title}")
    print(f"[CONFIG] - {args}")
    
    objective_function = lambda x: -env.objective_function.evaluate(np.array([x]).reshape(1, -1))[0]
    lower_bound, upper_bound = env.objective_function.bounds(dim=env.n_dim)[0].tolist(), env.objective_function.bounds(dim=env.n_dim)[1].tolist()

    try:
        if args.algo == "GA":
            algorithm = GA(func=objective_function, n_dim=env.n_dim, size_pop=config['n_agents'], max_iter=config['ep_length'], lb=lower_bound, ub=upper_bound, precision=[1e-7 for _ in range(env.n_dim)])
            run_experiment_other_algorithm(algorithm, env, config, exp_name, title=title)
        elif args.algo == "PSO":
            algorithm = PSO(func=objective_function, n_dim=env.n_dim, pop=config['n_agents'], max_iter=config['ep_length'], lb=lower_bound, ub=upper_bound,
                            w = config['w'], c1 = config['c1'], c2 = config['c2'])
            run_experiment_other_algorithm(algorithm, env, config, exp_name, title=title)
        elif args.algo == "SA":
            algorithm = SA(func=objective_function, x0=lower_bound, T_max=config['T_max'], T_min=config['T_min'], L=config['L'], max_stay_counter=config['max_stay_counter'])
            run_experiment_other_algorithm(algorithm, env, config, exp_name, title=title)
        elif args.algo == "Deephive":
            run_experiment_concurrently(env, agent_policy, config, exp_name, save_gif=False, title=title)
        else:
            raise ValueError("algo must be either GA, PSO, SA or Deephive")
    except Exception as e:
        print(f"Error in experiment {exp_name}")
        # print(e)
        # traceback.print_exc()
        sys.exit(1)
    
