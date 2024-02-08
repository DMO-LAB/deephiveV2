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

def get_agent_actions(env, policy, obs, config, roles=None, split_agents=False,
                      split_type="use_stds", decay_std=False):
    if decay_std:
        policy.std_controller.decay_std()
    if split_agents==False:
        if config["variable_std"][0] == True:
            obs_std = policy.std_controller.get_all_std()
            actions = get_action(obs, policy, env, obs_std)
        else:
            actions = get_action(obs, policy, env)
        return actions
    elif split_agents==True:
        assert roles is not None, "Roles must be provided if split_agents is True"
        exploiters_id = np.where(roles[0] == 1)[0]
        explorers_id = np.where(roles[0] == 0)[0]
        if split_type == "use_stds":
            policy.std_controller.update_roles(roles)
            obs_std = policy.std_controller.get_all_std()
            assert config['variable_std'][0] == True, "variable_std must be True in config"
            assert config['role_std']['explorer'] > config['role_std']['exploiter'], "explorers must have higher std than exploiters"
            actions = get_action(obs, policy, env, obs_std)
            return actions
        elif split_type == "use_grid":
            policy.std_controller.update_roles(roles)
            obs_std = policy.std_controller.get_all_std()
            explorer_actions, _ = get_informed_action(env, env.n_agents)
            exploiter_actions = get_action(obs, policy, env, obs_std)
            actions = np.zeros((env.n_agents, env.n_dim))
            actions[explorers_id] = explorer_actions[explorers_id]
            actions[exploiters_id] = exploiter_actions[exploiters_id]
            return actions
        elif split_type == "use_two_policies":
            policy[0].std_controller.update_roles(roles)
            obs_std = policy[0].std_controller.get_all_std()
            obs_explore, roles = env.observation_schemes.generate_observation(pbest=env.pbest.copy(), use_gbest=False, ratio=env.split_ratio)
            obs_exploit, roles = env.observation_schemes.generate_observation(pbest=env.pbest.copy(), use_gbest=True, ratio=env.split_ratio)
            exploit_std_obs = policy[1].std_controller.get_all_std(std=config["exploit_std"])
            assert len(policy) == 2, "Two policies must be provided - first policy is for explorers and second policy is for exploiters"
            explorer_actions = get_action(obs_explore, policy[0], env, obs_std)
            exploiter_actions = get_action(obs_exploit, policy[1], env, exploit_std_obs)
            actions = np.zeros((env.n_agents, env.n_dim))
            actions[explorers_id] = explorer_actions[explorers_id]
            actions[exploiters_id] = exploiter_actions[exploiters_id]
            return actions
        else:
            raise ValueError("split_type must be either use_stds, use_grid or use_two_policies")

def run_experiment_other_algorithm(algorithm:[GA, PSO, SA], env, config, exp_name, title=""):
    result_path = 'experiments/results/' + exp_name + '/'
    os.makedirs(result_path, exist_ok=True)
    
    objective_function = lambda x: -env.objective_function.evaluate(np.array([x]).reshape(1, -1))[0]
    lower_bound, upper_bound = env.objective_function.bounds(dim=env.n_dim)[0].tolist(), env.objective_function.bounds(dim=env.n_dim)[1].tolist()
    
    gbestVals = []
    opt_value = env.objective_function.optimal_value(dim=env.n_dim)
    for _ in range(config['iters']):
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
    
    gb = np.array(gbestVals)
    np.save(result_path + "gbestVals.npy", gb)
        
    if config['plot_gbest']:
            num_function_evaluation(fopt=gbestVals ,n_agents=env.n_agents, save_dir=result_path + "pso_num_function_evaluations.png", opt_value=opt_value,
                                    log_scale=False, plot_error_bounds=True, title=title)
            plot_individual_function_evaluation(fopt=gbestVals ,n_agents=env.n_agents, save_dir=result_path + "pso_num_function_evaluations2.png", opt_value=opt_value,
                                    log_scale=False, title=title)
            
    run_summary = {
    "time": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "title": title,
    "opt_value": opt_value,
}
    
    with open(result_path + "run_summary.json", 'w') as f:
        json.dump(run_summary, f)
        
        


def run_experiment(env, agent_policy, config, exp_name, save_gif=True, title=""):
    result_path = 'experiments/results/' + exp_name + '/'
    os.makedirs(result_path, exist_ok=True)
    animated_iters = []
    if isinstance(agent_policy, list):
        assert config['split_agents'] == True, "split_agents must be True in config"
        assert config['split_type'] == "use_two_policies", "split_type must be use_two_policies in config"
        assert len(agent_policy) == 2, "Two policies must be provided - first policy is for explorers and second policy is for exploiters"
    
    opt_value = env.objective_function.optimal_value(dim=env.n_dim)
    gbestVals = []
    for iter in range(config['iters']):
        obs, roles = env.reset()
        if isinstance(agent_policy, list):
            agent_policy[0].std_controller.reset_std()
            agent_policy[1].std_controller.reset_std()
        else:
            agent_policy.std_controller.reset_std()
        episode_gbest = []
        episode_gbest.append(env.gbest[-1])
        decay_std_run = False
        for i in range(config['ep_length']):
            if i == config['decay_start']:
                #print("Starting to decay std ...")
                decay_std_run = config['decay_std']
            actions = get_agent_actions(env, agent_policy, obs, config, roles=roles, split_agents=config["split_agents"], split_type=config["split_type"], decay_std=decay_std_run)
            observation_info, _, _, _ = env.step(actions)
            obs, roles = observation_info
            
            episode_gbest.append(env.gbest[-1])
        if config['tol'] != 0:
            tol = config['tol'] * opt_value
        elif config['tol'] == 0:
            tol = -0.01
        else:
            raise ValueError("tol must be positive")
        if env.gbest[-1] < tol:
            animated_iters.append(iter)
            print(f"[WARNING - iter {iter}] - Optimization failed to get to {config['tol']}% of optimal value - {opt_value}")
            print(f"Best value found: {env.gbest[-1]}")
            print("Rendering the episode history ...")
            print("------------------------------------------")
            if env.n_dim <=2:
                env.render(type="history", file_path=result_path + "error_history_" + str(iter) + "_.gif")
        if save_gif and env.n_dim <=2:
            if iter % 10 == 0 and iter not in animated_iters:
                env.render(type="history", file_path=result_path + "history_" + str(iter) + "_.gif")
        gbestVals.append(episode_gbest)
        
    # save gbestVals
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
    "opt_value": opt_value,
}
    
    with open(result_path + "run_summary.json", 'w') as f:
        json.dump(run_summary, f)
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str, default='test')
    parser.add_argument('--exp_num', type=int, default=1)
    parser.add_argument('--algo', type=str, default='Deephive', choices=["Deephive","GA", "PSO", "SA"])
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--use_gbest', action='store_true')
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
    


    args = parser.parse_args()

    config_path = "config/exp_config.json"
    model_path = "models/pbest_unfreeze.pth"
    model_path_2 = "models/gbest.pth"

    config = parse_config(config_path)

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
    mode = "test"

    exp_name = "exp_" + str(args.exp_num)
    result_path = 'experiments/results/' + str(exp_name) + '/' 
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
        run_experiment(env, agent_policy, config, exp_name, save_gif=True, title=title)
    else:
        raise ValueError("algo must be either GA, PSO, SA or Deephive")
    