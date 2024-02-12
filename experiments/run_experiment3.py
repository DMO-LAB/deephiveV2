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


def run_experiment_other_algorithm(algorithm:[GA, PSO, SA], env, config, exp_name, title=""):
    base_path = f'experiments/results_{env.n_dim}/'
    os.makedirs(base_path, exist_ok=True)
    result_path = base_path + exp_name + '/'
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
        
        

def get_agent_actions(env, policy, obs, config, roles=None, split_agents=False,
                      split_type="use_stds", decay_std=False, split_ratio=1, dynamic_split=False, **kwargs):
    
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
        if kwargs.get("exploiter_ids") is not None: 
            exploiters_id = kwargs.get("exploiter_ids")
            explorers_id = kwargs.get("explorer_ids")
        else:
            if not dynamic_split:
                exploiters_id = np.where(roles[0] == 1)[0]
                explorers_id = np.where(roles[0] == 0)[0]
            else:
                # use the split ratio to split the agents
                explorers_id = np.random.choice(env.n_agents, int(env.n_agents * split_ratio), replace=False)
                exploiters_id = np.setdiff1d(np.arange(env.n_agents), explorers_id)
        roles = np.zeros((env.n_agents))
        for i in range(len(exploiters_id)):
            roles[int(exploiters_id[i])] = 1
        for i in range(len(explorers_id)):
            roles[int(explorers_id[i])] = 0
            
        roles = [roles, roles]
        # update env state history roles to the correct exploration/exploitation roles
        env.state_history[:, env.current_step, -2] = roles[0]
        
        if split_type == "use_stds":
            policy.std_controller.update_roles(roles)
            obs_std = policy.std_controller.get_all_std()
            #print(obs_std)
            assert config['variable_std'][0] == True, "variable_std must be True in config"
            assert config['role_std']['explorer'] > config['role_std']['exploiter'], "explorers must have higher std than exploiters"
            # print(f"Explorers std: {config['role_std']['explorer']}")
            # print(f"Exploiters std: {config['role_std']['exploiter']}")
            # print(f"Obs std: {obs_std}")
            # print(f"Gbest: {env.gbest}")
            
            actions = get_action(obs, policy, env, obs_std)
            return actions
        elif split_type == "use_grid":
            policy.std_controller.update_roles(roles)
            obs_std = policy.std_controller.get_all_std()
            explorer_actions, _ = get_informed_action(env, env.n_agents)
            exploiter_actions = get_action(obs, policy, env, obs_std)
            actions = np.zeros((env.n_agents, env.n_dim))
            if explorers_id.size > 0:
                actions[explorers_id] = explorer_actions[explorers_id]
            if exploiters_id.size > 0:
                # print(exploiters_id.size)
                # print(exploiters_id)
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
            if explorers_id.size > 0:
                actions[explorers_id] = explorer_actions[explorers_id]
            if exploiters_id.size > 0:
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
    return gbests
        
def run_experiment(env, agent_policy, config, exp_name, save_gif=False, title=""):
    base_path = f"experiments/results_{env.n_dim}/"
    os.makedirs(base_path, exist_ok=True)
    result_path = base_path  + exp_name + '/'
    os.makedirs(result_path, exist_ok=True)
    
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
        
        gbests = optimize(env, agent_policy, obs, roles, config)
        # combine gbests to episode gbest
        episode_gbest.extend(gbests)
        #print(f"Episode gbests - {episode_gbest}")
    
        # if env.gbest[-1] < config['tol'] * opt_value:
        #     print(f"[WARNING - iter {iter}] - Optimization failed to get to {config['tol']}% of optimal value - {opt_value}")
        #     print(f"Best value found: {env.gbest[-1]}")
        #     print("Rendering the episode history ...")
        #     print("------------------------------------------")
        #     env.render(type="history", file_path=result_path + "error_history_" + str(iter) + "_.gif")
        if save_gif:
            if iter % 100 == 0:
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
        run_experiment(env, agent_policy, config, exp_name, save_gif=False, title=title)
    else:
        raise ValueError("algo must be either GA, PSO, SA or Deephive")
    
