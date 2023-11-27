import importlib
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from environment.optimization_functions import OptimizationFunctionBase
from environment.observation_schemes import ObservationScheme
from environment.reward_schemes import RewardScheme
from environment.utils import parse_config, ScalingHelper, Render
from exploration.gaussian_mixture import ExplorationModule
from exploration.gp_surrogate import GPSurrogateModule
    
    
class OptimizationEnv(gym.Env):
    def __init__(self, config_path: str):
        """
        A class representing the optimization environment.

        Attributes:
        -----------
        env_name : str
            The name of the environment.
        objective_function : function
            The objective function to be optimized.
        n_agents : int
            The number of agents in the environment.
        n_dim : int
            The number of dimensions in the search space.
        bounds : tuple
            The bounds of the search space.
        ep_length : int
            The maximum number of steps in an episode.
        init_state : ndarray
            The initial state of the environment.
        opt_bound : float
            The threshold for the objective function value to be considered optimal.
        reward_type : str
            The type of reward function to use.
        freeze : bool
            Whether to freeze the environment after the first episode.
        opt_value : float
            The optimal value of the objective function.
        observation_schemes : function
            The observation scheme to use.
        reward_schemes : function
            The reward scheme to use.
        """
        self.config = parse_config(config_path)
        self.setup_config()
        
    def setup_config(self):
        # Configuration code from the original __init__ method
        try:
            self.env_name = self.config["env_name"]
            self.objective_function:OptimizationFunctionBase = getattr(importlib.import_module(".barrel", "environment.optimization_functions"), self.config["objective_function"])()
            self.n_agents = self.config["n_agents"]
            self.n_dim = self.config["n_dim"]
            self.bounds:Tuple[np.ndarray] = self.objective_function.bounds(self.n_dim)
            self.ep_length = self.config["ep_length"]
            self.init_state = self.config["init_state"]
            self.opt_bound = self.config["opt_bound"]
            self.reward_type = self.config["reward_type"]
            self.freeze = self.config["freeze"]
            self.optimization_type: str = self.config["optimization_type"] # optimization type: "minimize" or "maximize"
            self.opt_value = self.objective_function.optimal_value(self.n_dim)
            self.observation_schemes: ObservationScheme = getattr(importlib.import_module(".barrel", "environment.observation_schemes"), self.config["observation_scheme"])(self)
            self.reward_schemes:RewardScheme = getattr(importlib.import_module(".barrel", "environment.reward_schemes"), self.config["reward_scheme"])(self)
            self.scaler_helper = ScalingHelper()
            self.render_helper = Render(self)
            self.use_gbest = self.config["use_gbest"]
            self.use_optimal_value = self.config["use_optimal_value"]
            self.enforce_good_actions = self.config["enforce_good_actions"]
            self.use_gmm = self.config["use_gmm"] if "use_gmm" in self.config else False
            self.use_surrogate = self.config["use_surrogate"] if "use_surrogate" in self.config else False
            self.use_variance = self.config["use_variance"] if "use_variance" in self.config else False
            self.variance_threshold = self.config["variance_threshold"] if "variance_threshold" in self.config else 0.001
            self.debug = self.config["debug"] if "debug" in self.config else False
        except KeyError as e:
            raise KeyError(f"Key {e} not found in config file.")

    def _reset_variables(self):
        self.state_history = np.zeros(
            (self.n_agents, self.ep_length+1, self.n_dim+3))
        self.ids_true_function_eval_history = np.array([])
        self.gbest_history = np.zeros((self.ep_length+1, self.n_dim+1))
        self.best_obj_value = np.inf if self.optimization_type == "minimize" else -np.inf
        if self.use_optimal_value:
            self.best_obj_value = self.opt_value
        self.worst_obj_value = -np.inf if self.optimization_type == "minimize" else np.inf
        self.min_pos, self.max_pos = self.bounds[0], self.bounds[1]
        self.lower_bound_actions = np.array(
            [-np.inf for _ in range(self.n_dim)], dtype=np.float64)
        self.upper_bound_actions = np.array(
            [np.inf for _ in range(self.n_dim)], dtype=np.float64)
        self.lower_bound_obs = np.append(np.array(
            [self.min_pos[i] for i in range(self.n_dim)], dtype=np.float64), -np.inf)
        self.upper_bound_obs = np.append(np.array(
            [self.max_pos[i] for i in range(self.n_dim)], dtype=np.float64), np.inf)

        self.low = np.array([self.lower_bound_obs.tolist()
                            for _ in range(self.n_agents)])
        self.high = np.array([self.upper_bound_obs.tolist()
                             for _ in range(self.n_agents)])
        self.action_low = np.array(
            [self.lower_bound_actions.tolist() for _ in range(self.n_agents)], dtype=np.float64)
        self.action_high = np.array(
            [self.upper_bound_actions.tolist() for _ in range(self.n_agents)], dtype=np.float64)

        self.action_space = spaces.Box(
            low=self.action_low, high=self.action_high, dtype=np.float64) 
        self.observation_space = spaces.Box(
            low=self.low, high=self.high, dtype=np.float64) 
        self.num_of_function_evals = 0

    def __str__(self):
        return f"OptimizationEnv: {self.env_name} with {self.n_agents} agents in {self.n_dim} dimensions"
    
    def reset(self):
        self.current_step = 0
        self._reset_variables()
        self.state = self._generate_init_state()
        self._update_env_state()
        self.prev_state = self.state.copy()
        self.pbest = self._get_actual_state()
        self.gbest = self.pbest[np.argmin(self.pbest[:, -1])] if self.optimization_type == "minimize" else self.pbest[np.argmax(self.pbest[:, -1])]
        self._update_pbest()
        actual_samples = self._get_actual_state()
        self.ids_true_function_eval = np.arange(self.n_agents)
        self.gmm = ExplorationModule(initial_samples=actual_samples[:, :-1], n_components=5, max_samples=None)
        if self.use_surrogate:
            self.surrogate = GPSurrogateModule(initial_samples=actual_samples[:, :-1], initial_values=actual_samples[:, -1], bounds=self.bounds)
        observation = self.observation_schemes.generate_observation(pbest=self.pbest.copy(), use_gbest=self.use_gbest)
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Run one time step of the environment's dynamics.
        """
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        self.prev_state = self.state.copy()
        self.prev_obj_values = self.obj_values.copy()
        self.best_agent = np.argmin(self.obj_values) if self.optimization_type == "minimize" else np.argmax(self.obj_values)
        
        # Apply the action to the state
        self.state[:, :-1] += action

        # Handle freezing of the best agent 
        if self.freeze: 
            self.state[self.best_agent, :-1] -= action[self.best_agent] 
        
        # Check and handle boundary violations
        self.boundary_violating_agents = self._check_boundary_violations() 
        self.boundary_violating_agents = np.where(self.boundary_violating_agents)[0]
        self.state = np.clip(self.state, np.zeros_like(self.state), np.ones_like(self.state))
        self.state = self._get_actual_state()
        self.gmm.update_distribution(self.state[:, :-1])

        # Evaluate novelty and divide agents
        if self.use_surrogate:
            if self.debug:
                surr_all, pred_var_all = self.surrogate.evaluate(self.state[:, :-1])
                obj_values_all = self.objective_function.evaluate(params=self.state[:, :-1])
                novelty_scores = self.gmm.assess_novelty(self.state[:, :-1])
                # write this to file for analysis
                with open("surrogate_predictions.txt", "a") as f:
                    f.write(f"Current step: {self.current_step} \n")
                    f.write(f"Surrogate prediction: {surr_all} \n variance: {pred_var_all} \n")
                    f.write(f"Objective values: {obj_values_all} \n")
                    f.write(f"Novelty scores: {novelty_scores} \n")
                    f.write(f"sorted novelty scores: {np.argsort(novelty_scores)} \n")
                    f.write(f"sorted variance: {np.argsort(pred_var_all)} \n")
            
            if self.use_variance:
                surr_all, pred_var_all = self.surrogate.evaluate(self.state[:, :-1])
                # less novel is agent with variance less than self.variance_threshold 
                less_novel_indices = np.where(pred_var_all < self.variance_threshold)[0]
                most_novel_indices = np.where(pred_var_all >= self.variance_threshold)[0]
                print(f"{len(less_novel_indices)} agents with variance less than {self.variance_threshold} \n")
                print(f"{len(most_novel_indices)} agents with variance greater than {self.variance_threshold} \n")
                if len(most_novel_indices) == 0 and self.current_step < 5:
                    print("No agents with variance greater than 0.001, early in the episode. Using all agents instead. \n")
                    most_novel_indices = np.argsort(pred_var_all)[-len(pred_var_all)//2:]
                    less_novel_indices = np.argsort(pred_var_all)[:len(pred_var_all)//2]
                
            else:
                novelty_scores = self.gmm.assess_novelty(self.state[:, :-1])
                print(f"Novelty scores: {novelty_scores} \n")
                most_novel_indices = np.argsort(novelty_scores)[-len(novelty_scores)//2:]  # Half with highest scores
                less_novel_indices = np.argsort(novelty_scores)[:len(novelty_scores)//2]  # Half with lowest scores

            # Evaluate most novel agents using the true function
            self.obj_values[most_novel_indices] = self.objective_function.evaluate(params=self.state[most_novel_indices, :-1])
            self.num_of_function_evals += len(most_novel_indices)

            # # Evaluate less novel agents using the surrogate
            if len(less_novel_indices) > 0:
                surr_pred, _ = self.surrogate.evaluate(self.state[less_novel_indices, :-1])
                #print(f"Surrogate prediction: {surr_pred} \n variance: {pred_var}")
                self.obj_values[less_novel_indices] = surr_pred
            
                # if the any values in the surr_pred is better than the gbest value, then evaluate the true function and add the id to the most_novel_indices
                better_than_gbest_relative = np.where(surr_pred < self.gbest[-1])[0] if self.optimization_type == "minimize" else np.where(surr_pred > self.gbest[-1])[0]
                # Map these indices back to the original state array
                better_than_gbest = less_novel_indices[better_than_gbest_relative]
                if len(better_than_gbest) > 0:
                    self.obj_values[better_than_gbest] = self.objective_function.evaluate(params=self.state[better_than_gbest, :-1])
                    self.num_of_function_evals += len(better_than_gbest)
                    most_novel_indices = np.append(most_novel_indices, better_than_gbest)
            
            # Update the surrogate with the data of the most novel agents
            if len(most_novel_indices) > 0:
                self.surrogate.update_model(self.state[most_novel_indices, :-1], self.obj_values[most_novel_indices])
            #self.surrogate.update_model(self.state[most_novel_indices, :-1], self.obj_values[most_novel_indices])

        else:
            # Standard evaluation if not using surrogate
            self.obj_values = self.objective_function.evaluate(params=self.state[:, :-1])
            self.num_of_function_evals += self.n_agents

        # Enforce good actions
        if self.enforce_good_actions:
            revert_condition = self.obj_values > self.prev_obj_values if self.optimization_type == "minimize" else self.obj_values < self.prev_obj_values
            self.state[revert_condition, :-1] = self.prev_state[revert_condition, :-1]
        
        self.current_step += 1
        self._update_env_state()
        self._update_pbest()
        self._update_done_flag()

        agents_done = np.array([self.done for _ in range(self.n_agents)])
        reward = self.reward_schemes.compute_reward()
        observation = self.observation_schemes.generate_observation(pbest=self.pbest.copy(), use_gbest=self.use_gbest)
        info = self._get_info()
        self.state_history[:, self.current_step, -2] = observation[1][0].flatten()

        # Store IDs of agents evaluated by the true function for analysis
        self.ids_true_function_eval = most_novel_indices if self.use_surrogate else np.arange(self.n_agents)
        
        self.ids_true_function_eval_history = np.append(self.ids_true_function_eval_history, self.ids_true_function_eval)
        
        novelty = np.zeros_like(self.state[:, -1])
        novelty[self.ids_true_function_eval] = 1
        self.state_history[:, self.current_step, -1] = novelty

        if self.optimization_type == "minimize":
            if np.any(self.obj_values < self.best_obj_value):
                raise ValueError("Objective value is greater than the best objective value")
        elif self.optimization_type == "maximize":
            if np.any(self.obj_values > self.best_obj_value):
                raise ValueError("Objective value is less than the best objective value")
        
        print(f"Current time {self.current_step} - {self.ep_length}, best obj value: {self.best_obj_value}, best agent: {self.best_agent}, best agent value: {self.best_agent_value}, number of function evals: {self.num_of_function_evals} \n")
            
        #print(f"current_step: {self.current_step}, best_obj_value: {self.gbest[-1]}, number of function evals: {self.num_of_function_evals} \n")
        return observation, reward, agents_done, info

    
    def render(self, type: str = "state",fps=1, file_path: Optional[str] = None):
        """ Render the environment
        Args:
            type: type of rendering : "state" or "history"
        """
        if type == "state":
            self.render_helper.render_state()
        elif type == "history":
            self.render_helper.render_state_history(file_path=file_path, fps=fps)
        elif type == "gmm":
            self.gmm.plot_distribution()
        elif type == "surrogate":
            self.surrogate.plot_surrogate(save_dir=file_path)
            # save the state history in the file_path after rendering, in the same folder as the video
            #np.save(file_path[:-4], self.state_history)
        
        
    def _check_boundary_violations(self) -> np.ndarray:
        """ Check if the agents are violating the boundaries
        Returns:
            boundary_violating_agents: list of agents that are violating the boundaries
        """
        boundary_violating_agents = np.any(
            (self.state[:] <= np.zeros_like(self.state)) | (self.state[:] >= np.ones_like(self.state)), axis=1)
        return boundary_violating_agents
        
    def _get_info(self) -> Dict[str, Any]:
        """
        Get the info of the environment
        Returns:
            info: info of the environment
        """
        info = {
            "state": self.state,
            "best_obj_value": self.best_obj_value,
            "worst_obj_value": self.worst_obj_value,
            "best_agent": self.best_agent,
            "actual_state": self._get_actual_state(),
            "current_step": self.current_step,
            "done": self.done,
            "opt_value": self.opt_value,
            "opt_bound": self.opt_bound,
            "pbest": self.pbest,
            "gbest": self.gbest,
            "voilating_agents": self.boundary_violating_agents,
        }
        return info
    
    def _update_done_flag(self):
        # if time step is greater than the maximum episode length
        if self.current_step >= self.ep_length:
            self.done = True
            #print(f"Reached maximum episode length: {self.ep_length}")
        elif self.optimization_type == "minimize" and self.opt_value is not None:
            # if the best objective value is less than the optimal value
            if self.best_obj_value <= self.opt_value + self.opt_bound:
                self.done = True
                #print(f"Best objective value: {self.best_obj_value} is less than the optimal value: {self.opt_value}, within the bound: {self.opt_bound}")
            else:
                self.done = False
            
        elif self.optimization_type == "maximize" and self.opt_value is not None:
            # if the best objective value is greater than the optimal value
            if self.best_obj_value >= self.opt_value - self.opt_bound:
                self.done = True
                #print(f"Best objective value: {self.best_obj_value} is greater than the optimal value: {self.opt_value}")
            else:
                self.done = False
        
        else:
            self.done = False
            
    def _get_optimal_agents(self) -> List[int]:
        """
        Check if agents are optimal
        Returns:
            optimal_agents: list of agents that are optimal
        """
        if self.use_optimal_value:
            if self.optimization_type == "minimize":
                optimal_agents = np.where(
                    self.obj_values <= self.opt_value * (1+self.opt_bound))[0]
            elif self.optimization_type == "maximize":
                optimal_agents = np.where(
                    self.obj_values >= self.opt_value * self.opt_bound)[0]
            else:
                raise ValueError("optimization_type should be either 'minimize' or 'maximize'")
            return optimal_agents
        else:
            return []
        
    def _generate_init_state(self):
        """ Generate a random initial state for all agents
        Returns:
            init_state: initial state of all agents
        """
        if self.init_state is not None:
            init_pos = np.array(self.init_state)
        else:
            init_pos = np.round(np.random.uniform(
                low=self.low[0][:-1], high=self.high[0][:-1], size=(self.n_agents, self.n_dim)), decimals=2)
        # get the objective value of the initial position
        self.obj_values = self.objective_function.evaluate(params=init_pos)
        self.num_of_function_evals += self.n_agents
        # combine the position and objective value
        init_obs = np.append(init_pos, self.obj_values.reshape(-1, 1), axis=1)
        return init_obs
    
    def _get_actual_state(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """ Get the actual state of the agents by rescaling the state to the original bounds
        Args:
            state: state to rescale
        Returns:
            actual_state: actual state of the agents 
        """
        if state is None:
            state = self.state.copy()
            obj_value = self.obj_values.copy()
            actual_state = self.scaler_helper.rescale(
                state[:, :self.n_dim], self.min_pos, self.max_pos)
            actual_state = np.append(
                actual_state, obj_value.reshape(-1, 1), axis=1)
        else:
            actual_state = self.scaler_helper.rescale(
                state[:, :self.n_dim], self.min_pos, self.max_pos)
            actual_state = np.append(
                actual_state, self.objective_function.evaluate(params=actual_state).reshape(-1, 1), axis=1) 
        return actual_state
        
    def _update_env_state(self):
        def _update():
            if self.optimization_type == "minimize":
                self.best_obj_value = min(np.min(self.obj_values), self.best_obj_value)
                self.worst_obj_value = max(np.max(self.obj_values), self.worst_obj_value)
                self.best_agent = np.argmin(self.obj_values)
                self.best_agent_value = np.min(self.obj_values)
            elif self.optimization_type == "maximize":
                self.best_obj_value = max(np.max(self.obj_values), self.best_obj_value)
                self.worst_obj_value = min(np.min(self.obj_values), self.worst_obj_value)
                self.best_agent = np.argmax(self.obj_values)
                self.best_agent_value = np.max(self.obj_values)
            else:
                raise ValueError("optimization_type should be either 'minimize' or 'maximize'")
            
        _update()
        self.state[:, :-1] = self.scaler_helper.scale(
            self.state[:, :-1], self.min_pos, self.max_pos)
        self.state[:, -1] = self.scaler_helper.scale(
            self.obj_values, self.worst_obj_value, self.best_obj_value)
        
        #print(self.state, self.min_pos, self.max_pos, self.worst_obj_value, self.best_obj_value, self.current_step)
        # assert that the normalized state is within the bounds [0, 1]
        assert np.all((self.state >= 0) & (self.state <= 1)), "State is not within the bounds [0, 1]"
        
    def _update_pbest(self):
        # # update the pbest and gbest
        actual_state = self._get_actual_state().copy()
        if self.optimization_type == "minimize":
            condition = (self.obj_values < self.pbest[:, -1])[:, None]  # Reshape to (5, 1)
            condition = np.repeat(condition, self.n_dim+1, axis=1)  # Repeat columns to match shape (5, 3)
            self.pbest = np.where(condition, actual_state, self.pbest)
            self.gbest = self.pbest[np.argmin(self.pbest[:, -1])]
        elif self.optimization_type == "maximize":
            condition = (self.obj_values > self.pbest[:, -1])[:, None]  # Reshape to (5, 1)
            condition = np.repeat(condition, self.n_dim+1, axis=1)  # Repeat columns to match shape (5, 3)
            self.pbest = np.where(condition, actual_state, self.pbest)
            self.gbest = self.pbest[np.argmax(self.pbest[:, -1])]
        else:
            raise ValueError("optimization_type should be either 'minimize' or 'maximize'")
        self.gbest_history[self.current_step, :] = self.gbest
        
        self.state_history[:, self.current_step, :self.n_dim+1] = self._get_actual_state()

    def _get_stuck_agents(self, threshold: int = 2) -> List[int]:
        """
        Check if agents are stuck in a local minimum by comparing the objective values
        in state_history for the past 'threshold' time steps.
        
        Args:
            threshold: number of previous steps to check
        
        Returns:
            stuck_agents: list of agents that are stuck
        """
        # Ensure the threshold is at least 1 to prevent negative indexing
        threshold = max(1, threshold)
        
        # Determine the range of steps to check in the state_history
        start_step = max(0, self.current_step - threshold)
        end_step = self.current_step
        
        # List to hold the indices of stuck agents
        stuck_agents = []
        if self.current_step > 3:
            for agent in range(self.n_agents):
                # Get the objective values from state_history for the specified range of steps
                past_obj_values = self.state_history[agent, start_step:end_step, -3]
                
                # Check if the objective values have not changed over the threshold time steps
                if np.all(past_obj_values == past_obj_values[0]) and agent != self.best_agent:
                    stuck_agents.append(agent)
        
        return stuck_agents


        
if __name__ == "__main__":
    optimizer = OptimizationEnv("environment/config/env_config.json")
    print(optimizer)
    state, obs  = optimizer.reset()
    for episode in range(4):
        print(f"Episode: {episode}")
        actions = np.random.uniform(low=-0.5, high=0.5, size=(optimizer.n_agents, optimizer.n_dim))
        observation, reward, done, info = optimizer.step(actions)
        print(f"Done: {done}")
        print(f"Reward: {reward}")
        if done.all():
            break
    
