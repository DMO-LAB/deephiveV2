import importlib
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple
from deephive.environment.optimization_functions import OptimizationFunctionBase
from deephive.environment.observation_schemes import ObservationScheme
from deephive.environment.reward_schemes import RewardScheme
from deephive.environment.utils import parse_config, ScalingHelper, Render, filter_points
from deephive.environment.utils import initialize_grid
from deephive.exploration.gp_surrogate import GPSurrogateModule
    
class OptimizationEnv(gym.Env):
    def __init__(self, config: Dict[str, Any]):
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
        self.config = config
        self.setup_config()
        
    def initialize_neighborhoods(self, topology_type):
        self.neighborhoods = []
        if topology_type == 'global':
            self.neighborhoods = [list(range(self.n_agents)) for _ in range(self.n_agents)]
            
        elif topology_type == 'ring':
            for i in range(self.n_agents):
                neighborhood_indices = list(range(i - self.neighborhood_size, i)) + \
                                       list(range(i + 1, i + 1 + self.neighborhood_size))
                neighborhood_indices = [index % self.n_agents for index in neighborhood_indices]
                self.neighborhoods.append(neighborhood_indices)
                
        elif topology_type == 'von_neumann':
            # Assuming a square grid for simplicity
            grid_side = int(np.sqrt(self.n_agents))
            assert grid_side**2 == self.n_agents, "n_agents must be a perfect square for von Neumann topology"
            for i in range(self.n_agents):
                x, y = i % grid_side, i // grid_side
                neighbors = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < grid_side and 0 <= ny < grid_side:
                        neighbors.append(ny * grid_side + nx)
                self.neighborhoods.append(neighbors)
                
        elif topology_type == 'star':
            best_index = 0  # This could be dynamic based on your algorithm's state
            for i in range(self.n_agents):
                self.neighborhoods.append([best_index] if i != best_index else list(range(self.n_agents)))
                
        elif topology_type == 'random':
            for i in range(self.n_agents):
                # Ensure we don't include the particle itself in its neighborhood
                neighborhood_indices = np.random.choice([x for x in range(self.n_agents) if x != i], 
                                                        self.neighborhood_size, replace=False).tolist()
                self.neighborhoods.append(neighborhood_indices)
                
        else:
            raise ValueError(f"Unknown topology type: {topology_type}")

        
    # def initialize_neighborhoods(self):
    #     self.neighborhoods = []
    #     for i in range(self.n_agents):
    #         neighborhood_indices = list(range(i - self.neighborhood_size, i)) + list(range(i + 1, i + 1 + self.neighborhood_size))
    #         neighborhood_indices = [index % self.n_agents for index in neighborhood_indices]  # Ensure indices are within bounds
    #         self.neighborhoods.append(neighborhood_indices)
    
    def compute_lbest(self):
        self.lbest_positions = np.zeros((self.n_agents, self.n_dim))
        for i in range(self.n_agents):
            neighborhood_fitnesses = self.state[:, -1][self.neighborhoods[i]]
            best_neighbor_idx = self.neighborhoods[i][np.argmax(neighborhood_fitnesses)]
            self.lbest_positions[i] = self.state[best_neighbor_idx, :-1]
            
    def permute_lbest(self, particle_index):
        # Apply a permutation to the lbest of the selected particle
        # This is a simplification; in reality, you might want to shuffle the entire neighborhood
        neighbor_indices = self.neighborhoods[particle_index]
        np.random.shuffle(neighbor_indices)
        self.lbest_positions[particle_index] = self.state[neighbor_indices[0], :-1]
        
    def setup_config(self):
        # Configuration code from the original __init__ method
        try:
            self.env_name = self.config["env_name"]
            self.function_id = self.config["function_id"]
            self.objective_function:OptimizationFunctionBase = getattr(importlib.import_module(".barrel", "deephive.environment.optimization_functions"), self.config["objective_function"])(function_id=self.config["function_id"], negative=self.config["negative"])
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
            self.observation_schemes: ObservationScheme = getattr(importlib.import_module(".barrel", "deephive.environment.observation_schemes"), self.config["observation_scheme"])(self)
            self.reward_schemes:RewardScheme = getattr(importlib.import_module(".barrel", "deephive.environment.reward_schemes"), self.config["reward_scheme"])(self)
            self.scaler_helper = ScalingHelper()
            self.render_helper = Render(self)
            self.use_gbest = self.config["use_gbest"]
            self.use_lbest = self.config["use_lbest"]
            self.use_optimal_value = self.config["use_optimal_value"]
            self.use_surrogate = self.config["use_surrogate"] if "use_surrogate" in self.config else False
            self.debug = self.config["debug"] if "debug" in self.config else False
            self.grid_resolution = self.config["grid_resolution"] if "grid_resolution" in self.config else 0.1
            self.split_ratio = self.config["split_ratio"] if "split_ratio" in self.config else 0.5  
            self.log_scale = self.config["log_scale"] if "log_scale" in self.config else False
            self.include_gbest = self.config["include_gbest"] if "include_gbest" in self.config else False
            self.neighborhood_size = self.config["neighborhood_size"] if "neighborhood_size" in self.config else 3
            self.topology_type = self.config["topology_type"] if "topology_type" in self.config else "ring"
            
        except KeyError as e:
            raise KeyError(f"Key {e} not found in config file.")

    def _reset_variables(self):
        self.state_history = np.zeros((self.n_agents, self.ep_length+1, self.n_dim+3))
        self.gbest_history = np.zeros((self.ep_length+1, self.n_dim+1))
        self.best_obj_value = np.inf if self.optimization_type == "minimize" else -np.inf
        if self.use_optimal_value:
            self.best_obj_value = self.opt_value
        self.worst_obj_value = -np.inf if self.optimization_type == "minimize" else np.inf

        # Use bounds directly for min_pos and max_pos, no change needed here
        self.min_pos, self.max_pos = self.bounds[0], self.bounds[1]

        # Vectorized bounds for actions
        self.lower_bound_actions = np.full(self.n_dim, -np.inf, dtype=np.float64)
        self.upper_bound_actions = np.full(self.n_dim, np.inf, dtype=np.float64)

        # Efficiently create lower and upper bounds for observations
        self.lower_bound_obs = np.concatenate((self.min_pos, [-np.inf]))
        self.upper_bound_obs = np.concatenate((self.max_pos, [np.inf]))

        # Broadcasting to create agent-specific bounds without explicit loops
        self.low = np.tile(self.lower_bound_obs, (self.n_agents, 1))
        self.high = np.tile(self.upper_bound_obs, (self.n_agents, 1))
        self.action_low = np.tile(self.lower_bound_actions, (self.n_agents, 1))
        self.action_high = np.tile(self.upper_bound_actions, (self.n_agents, 1))

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float64)
        self.observation_space = spaces.Box(low=self.low, high=self.high, dtype=np.float64)
        self.initialize_neighborhoods(self.topology_type)
        
    def __str__(self):
        return f"OptimizationEnv: {self.env_name} with {self.n_agents} agents in {self.n_dim} dimensions"
    
    
    def reset(self):
        self.current_step = 0
        self._reset_variables()
        self._generate_init_state()
        self.objective_function.tracker.reset()
    
        if self.config["use_grid"]:
            self.grid_points = initialize_grid(self.bounds, resolution=self.grid_resolution, n_dim=self.n_dim) 
            # scale the grid points
            self.grid_points = self.scaler_helper.scale(self.grid_points, self.min_pos, self.max_pos)
        actual_samples = self._get_actual_state()
        self.evaluated_points = self.state[:, :-1].copy() # scaled points
        if self.use_surrogate:
            #print("Instantiating the surrogate")
            self.surrogate = GPSurrogateModule(initial_samples=actual_samples[:, :-1], initial_values=actual_samples[:, -1], bounds=self.bounds) 
            _, self.agents_pos_std = self.surrogate.evaluate(actual_samples[:, :-1], scale=True)
            self.prev_agents_pos_std = self.agents_pos_std.copy()
        
        observation = self.observation_schemes.generate_observation(pbest=self.pbest.copy(), use_gbest=self.use_gbest, ratio=self.split_ratio, 
                                                                    include_gbest=self.include_gbest, include_lbest=self.use_lbest)
        self.state_history[:, self.current_step, -2] = observation[1][0].flatten()
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Run one time step of the environment's dynamics.
        """
        assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid"
        self.prev_state = self.state.copy()
        self.prev_agents_pos = self._get_actual_state()[:, :-1] # get the previous agents position
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
        if self.use_surrogate:
            _ , self.agents_pos_std = self.surrogate.evaluate(self.state[:, :-1], scale=True)
            surrogate_state = self.state.copy()
            # filter the points that are close to each other
            surrogate_state = filter_points(surrogate_state, min_distance=0.1)
            self.surrogate.update_model(surrogate_state[:, :-1], surrogate_state[:, -1])

        self.obj_values = self.objective_function.evaluate(params=self.state[:, :-1])
        self.current_step += 1
        self._update_env_state()
        self._update_pbest()
        self._update_done_flag()

        self.evaluated_points = np.vstack((self.evaluated_points, self.state[:, :-1])).copy() # scaled points
        agents_done = np.zeros(self.n_agents, dtype=bool) if not self.done else np.ones(self.n_agents, dtype=bool)
        # self.surrogate_error = self.surrogate.evaluate_accuracy(self.objective_function.evaluate)
        reward = self.reward_schemes.compute_reward()
        observation = self.observation_schemes.generate_observation(pbest=self.pbest.copy(), use_gbest=self.use_gbest, ratio=self.split_ratio, 
                                                                    include_gbest=self.include_gbest, include_lbest=self.use_lbest)
        self.state_history[:, self.current_step, -2] = observation[1][0].flatten()

        info = self._get_info()

        if self.optimization_type == "minimize":
            if np.any(self.obj_values < self.best_obj_value):
                raise ValueError("Objective value is greater than the best objective value")
        elif self.optimization_type == "maximize":
            if np.any(self.obj_values > self.best_obj_value):
                raise ValueError("Objective value is less than the best objective value")
        
        return observation, reward, agents_done, info

    
    def render(self, type: str = "state",fps=1, file_path: Optional[str] = None, **kwargs):
        """ Render the environment
        Args:
            type: type of rendering : "state" or "history"
        """
        optimal_positons = kwargs.get("optimal_positions", None)
        try:
            if type == "state":
                self.render_helper.render_state(file_path, optimal_positions=optimal_positons)
            elif type == "history":
                self.render_helper.render_state_history(file_path=file_path, fps=fps, optimal_positions=optimal_positons)
            elif type == "surrogate":
                self.surrogate.plot_surrogate(save_dir=file_path, **kwargs)

            elif type == "surrogate_variance":
                self.surrogate.plot_variance(save_dir=file_path, **kwargs)
            else:
                raise ValueError("type should be either 'state' or 'history'")
        except Exception as e:
            raise e
            
        
    def _check_boundary_violations(self) -> np.ndarray:
        """ Check if the agents are violating the boundaries
        Returns:
            boundary_violating_agents: list of agents that are violating the boundaries
        """
        boundary_violating_agents = np.any(
            (self.state[:] <= np.zeros_like(self.state)) | (self.state[:] >= np.ones_like(self.state)), axis=1)
        return boundary_violating_agents
    
    def _get_unexplored_area(self) -> np.ndarray:
        """ Get the unexplored area of the agents from the surrounding model
        Returns:
            unexplored_area: unexplored area of the agents
        """
        high_std_points, high_std_points_std = self.surrogate.check_checkpoints(alpha=1, percentile=90)
        # normalize the points to the bounds
        high_std_points = self.scaler_helper.scale(high_std_points, self.min_pos, self.max_pos)
        return high_std_points, high_std_points_std
        
        
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
            "new_evaluated": self._get_actual_state(),
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
        
    def _generate_init_state(self, count=False):
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
        # combine the position and objective value
        self.state = np.append(init_pos, self.obj_values.reshape(-1, 1), axis=1)
        self._update_env_state()
        self.prev_state = self.state.copy()
        self.prev_agents_pos = self._get_actual_state()[:, :-1] # get the previous agents position
        self.pbest = self._get_actual_state()
        self.gbest = self.pbest[np.argmin(self.pbest[:, -1])] if self.optimization_type == "minimize" else self.pbest[np.argmax(self.pbest[:, -1])]
        self._update_pbest()
        if count:
            self.current_step += 1
            
    def _get_scaled_pbest(self):
        pbest_scaled = np.zeros_like(self.pbest)
        pbest_scaled[:, :2] = self.scaler_helper.scale(self.pbest[:, :2], self.min_pos, self.max_pos)
        pbest_scaled[:, -1] = self.scaler_helper.scale(self.pbest[:,-1], self.worst_obj_value, self.best_obj_value, self.log_scale)
        return pbest_scaled
    
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
            self.obj_values, self.worst_obj_value, self.best_obj_value, log_scale=self.log_scale)
        
        # assert that the normalized state is within the bounds [0, 1]
        assert np.all((self.state >= 0) & (self.state <= 1)), f"State is not within the bounds [0, 1]: {self.state}"
        
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


    def _get_stuck_agents(self, threshold: int = 2) -> np.ndarray:
        """
        Check if agents are stuck in a local minimum by comparing the objective values
        in state_history for the past 'threshold' time steps.

        Args:
            threshold: number of previous steps to check

        Returns:
            stuck_agents: numpy array of agents that are stuck
        """
        # Ensure the threshold is at least 1 to prevent negative indexing
        threshold = max(1, threshold)
        
        # Check if there's enough history to compare, otherwise return empty array
        if self.current_step < 3 or self.current_step < threshold:
            return np.array([])
        
        # Determine the range of steps to check in the state_history
        start_step = max(0, self.current_step - threshold)
        end_step = self.current_step

        # Extract the relevant slice of state_history for all agents, focusing on the objective value
        past_obj_values = self.state_history[:, start_step:end_step, -3]

        # Check if the objective values have not changed over the threshold time steps for each agent
        # This is done by checking if the standard deviation along the time axis (axis=1) is 0
        # Meaning the values have not changed
        unchanged_values = np.all(past_obj_values == past_obj_values[:, [0]], axis=1)

        # Exclude the best agent from being considered stuck
        if hasattr(self, 'best_agent'):
            unchanged_values[self.best_agent] = False

        # Find the indices (agent numbers) where values have not changed
        stuck_agents = np.where(unchanged_values)[0]

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
    
