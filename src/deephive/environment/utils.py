import json
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import scipy.stats as stats
import scipy
from typing import Optional, List

# stop warnings from showing up
import warnings
warnings.filterwarnings("ignore")

# set figure size, dpi and fontsize
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 12


class ScalingHelper:
    """
    A class that provides methods for scaling and rescaling values.
    """

    @staticmethod
    def scale(d, d_min, d_max, log_scale=False):
        """
        Scales a value between 0 and 1 based on the given minimum and maximum values.

        Args:
            d (float): The value to be scaled.
            d_min (float): The minimum value of the range.
            d_max (float): The maximum value of the range.
            log_scale (bool): Whether to apply logarithmic scaling.

        Returns:
            float: The scaled value between 0 and 1.
        """
        if log_scale:
            # Ensure d_max is less than 0 for log10 scaling of negative values
            d_max = np.minimum(d_max, -1e-10)
            d = np.where(d == 0, -1e-10, d)  # Replace 0 with -1e-10 to avoid log(0)
            
            d = np.log10(-d)
            d_min_modified = np.log10(-d_max)  # Inverted due to negative log scaling
            d_max_modified = np.log10(-d_min)
            
            scaled_d = (d - d_min_modified) / (d_max_modified - d_min_modified) + 1e-10  # Add small value to avoid division by zero
        else:
            scaled_d = (d - d_min) / (d_max - d_min) + 1e-10  # Add small value to avoid division by zero   
        
        scaled_d = np.round(scaled_d, 4)  # Round to 4 decimal places
        if log_scale:
            scaled_d = 1 - scaled_d  # Invert scaling direction for logarithmic scale
        
        # if any value in scaled_d is greater than 1 or less than 0, or NaN, raise an error
        if np.any(scaled_d > 1) or np.any(scaled_d < 0) or np.any(np.isnan(scaled_d)):
            print(f"Scaling failed for d={d}, d_min={d_min}, d_max={d_max}, log_scale={log_scale}")
            raise ValueError("Scaling failed")
        
        return scaled_d

    @staticmethod
    def rescale(d, d_min, d_max, log_scale=False):
        """
        Rescales a value to its original range.

        Args:
            d (float): The value to be rescaled.
            d_min (float): The minimum value of the range.
            d_max (float): The maximum value of the range.
            log_scale (bool): Whether the original scaling was logarithmic.

        Returns:
            float: The rescaled value.
        """
        if log_scale:
            # Adjust d_min and d_max for logarithmic scale
            d_max = np.minimum(d_max, -1e-10)
            d_min_modifies = np.log10(-d_min)
            d_max_modifies = np.log10(-d_max)

            # Apply inverse of the scaling transformation
            d = d * (d_max_modifies - d_min_modifies) + d_min_modifies
            rescaled_d = -10 ** d  # Inverse of log10 scaling
        else:
            # Linear scaling inverse
            rescaled_d = d_min + (d_max - d_min) * d

        return rescaled_d

    

class StdController:
    def __init__(self, num_agents, n_dim, role_std={'explorer': 1.0, 'exploiter': 0.5}, decay_rate=0.99, min_std=0.01, max_std=0.5):
        self.num_agents = num_agents
        self.n_dim = n_dim
        self.role_std = role_std
        self.decay_rate = decay_rate
        self.min_std = min_std
        self.max_std = max_std
        self.iteration_num = 0
        # Use NumPy array for std initialization for efficiency
        self.std = np.full((n_dim, num_agents), role_std['explorer'])
        self.decayed_role_std = np.array([role_std['explorer'], role_std['exploiter']])
        # Keep track of current roles using a NumPy array
        self.current_roles = np.zeros((n_dim, num_agents), dtype=int)

    def update_roles(self, roles):
        roles = np.array(roles)
        # Efficiently find the indices where roles have changed
        changed = np.where(roles != self.current_roles)
        new_roles = roles[changed]
        self.std[changed] = np.where(new_roles == 1, self.decayed_role_std[1], self.decayed_role_std[0])
        self.current_roles = roles

    def decay_std(self):
        # Efficiently decay std values based on the role
        decay_factor = self.decay_rate ** self.iteration_num
        self.decayed_role_std = np.clip(self.decayed_role_std * decay_factor, self.min_std, self.max_std)
        self.std = np.clip(self.std * decay_factor, self.min_std, self.max_std)
        self.iteration_num += 1

    def get_std(self, agent_id):
        # Efficient retrieval of std values for a specific agent across all dimensions
        return self.std[:, agent_id]

    def get_all_std(self, roles=None, std=None):
        if std is not None:
            # Return a custom std value if specified
            return np.full((self.n_dim, self.num_agents), std)
        if roles is not None:
            self.update_roles(roles)
        return self.std

    def reset_std(self):
        # Efficiently reset std values for all agents
        self.std.fill(self.role_std['explorer'])
        self.decayed_role_std = np.array([self.role_std['explorer'], self.role_std['exploiter']])
        self.current_roles.fill(0)
        self.iteration_num = 0



def parse_config(file_path: str) -> dict:
    """
    Parses a JSON configuration file and returns a dictionary.

    Args:
        file_path (str): The path to the JSON configuration file.

    Returns:
        dict: A dictionary containing the parsed configuration data.
    """
    with open(file_path, "r") as f:
        config = json.load(f)
    return config

def mean_confidence_interval(data, confidence=0.95):
    # check if data is just a single array and reshape it to a 2D array
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis = 0), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def num_function_evaluation(
    fopt: np.ndarray, 
    n_agents: int, 
    save_dir: str, 
    opt_value: Optional[float] = None, 
    label: str = "TEST OPT", 
    num_function_evaluations: Optional[np.ndarray] = None,
    plot_error_bounds: bool = False,
    log_scale: bool = True,
    title=None,
    minimize: bool = False
) -> None:
    """
    Plots and saves the number of function evaluations.

    :param fopt: Array of function optimization results.
    :param n_agents: Number of agents.
    :param save_dir: Directory to save the plot.
    :param opt_value: Optional known optimal value for comparison.
    :param label: Label for the plot.
    :param num_function_evaluations: Array of number of function evaluations.
    """
    fopt = np.array(fopt)
    mf1, ml1, mh1 = mean_confidence_interval(fopt, 0.95)
    x = (np.arange(len(mf1)) + 1) * n_agents if num_function_evaluations is None else np.arange(0, np.mean(num_function_evaluations), np.mean(num_function_evaluations) / len(mf1))
    
    plt.figure(figsize=(6, 4), dpi=200)
    if plot_error_bounds:
        plt.fill_between(x, ml1, mh1, alpha=0.1, edgecolor='#3F7F4C', facecolor='#7EFF99')
    plt.plot(x, mf1, linewidth=2.0, label=label, color='#3F7F4C')

    if opt_value is not None:
        plt.plot(x, np.full(len(mf1), opt_value), linewidth=1.0, label='True OPT', color='#CC4F1B')

    plt.xlabel('Number of Function Evaluations', fontsize=14)
    plt.ylabel('Best Fitness Value', fontsize=14)
    if not minimize:
        plt.legend(fontsize=8, frameon=False, loc="lower right")
    else:
        plt.legend(fontsize=14, frameon=False, loc="upper right")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if log_scale:
        plt.xscale('log')
    if title is not None:
        plt.title(title, fontsize=10)

    with open(save_dir, 'wb') as f:
        print("Saving plot to: ", save_dir)
        plt.savefig(f)
    plt.close()

def plot_individual_function_evaluation(
    fopt: np.ndarray,
    n_agents: int,
    save_dir: str,
    opt_value: Optional[float] = None,
    log_scale: bool = True,
    title=None
) -> None:
    """
    Plots and saves the number of function evaluations for each individual agent.

    :param fopt: Array of function optimization results.
    :param n_agents: Number of agents.
    :param save_dir: Directory to save the plot.
    :param opt_value: Optional known optimal value for comparison.
    """
    plt.figure(figsize=(6, 4), dpi=200)
    fopt = np.array(fopt)
    # plot each individual run in the same plot
    for i in range(fopt.shape[0]):
        plt.plot((np.arange(len(fopt[i])) + 1) * n_agents, fopt[i], linewidth=1.0, alpha=0.5)
    
    # plot the opt value if given
    if opt_value is not None:
        plt.axhline(opt_value, color='r', linestyle='--', label='Optimal Value')
        
    # plot the mean and std
    mean = np.mean(fopt, axis=0)
    std = np.std(fopt, axis=0)
    
    plt.plot((np.arange(len(mean)) + 1) * n_agents, mean, color='k', label='Mean')
    plt.fill_between((np.arange(len(mean)) + 1) * n_agents, mean - std, mean + std, alpha=0.2, color='k', label='Std')
    
        
    plt.xlabel('Number of Function Evaluations', fontsize=14)
    plt.ylabel('Best Fitness Value', fontsize=14)
    plt.legend(fontsize=14, frameon=False)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if log_scale:
        plt.xscale('log')
    if title is not None:
        plt.title(title, fontsize=10)

    with open(save_dir, 'wb') as f:
        plt.savefig(f)
    plt.close()

def plot_num_function_evaluation(
    fopt: List[np.ndarray], 
    n_agents: int, 
    save_dir: str, 
    opt_value: Optional[float] = None, 
    show_std: bool = False, 
    symbol_list: Optional[List[str]] = None, 
    color_list: Optional[List[str]] = None, 
    label_list: Optional[List[str]] = None,
    log_scale: bool = True,
    title=None,
    minimize: bool = False
) -> None:
    """
    Plots the number of function evaluations for different algorithms.

    :param fopt: List of arrays of function optimization results for different algorithms.
    :param n_agents: Number of agents.
    :param save_dir: Directory to save the plot.
    :param opt_value: Optional known optimal value for comparison.
    :param show_std: Flag to show standard deviation.
    :param symbol_list: List of symbols for each plot line.
    :param color_list: List of colors for each plot line.
    :param label_list: List of labels for each plot line.
    :param log_scale: Flag to use log scale for x-axis.
    """
    symbol_list = symbol_list if symbol_list is not None else ['-' for _ in range(len(fopt))]
    color_list = color_list if color_list is not None else ['#3F7F4C', '#CC4F1B', '#FFB852', '#64B5CD']
    label_list = label_list if label_list is not None else ['EXP1', 'EXP2', 'EXP3', 'EXP4']

    # print(f"Number of function evaluations: {len(fopt[0])}")
    # print(f"Number of algorithms: {len(fopt)}")

    plt.figure(figsize=(6, 4), dpi=200)
    for i, single_fopt in enumerate(fopt):
        mf1, ml1, mh1 = mean_confidence_interval(single_fopt, 0.95)
        x = (np.arange(len(mf1)) + 1) * n_agents
        if show_std:
            plt.errorbar(x, mf1, yerr=mh1 - ml1, fmt=symbol_list[i], linewidth=1.0, label=label_list[i], color=color_list[i])
        else:
            plt.plot(x, mf1, symbol_list[i], linewidth=1.0, label=label_list[i], color=color_list[i])

    if opt_value is not None:
        plt.plot(x, np.full(len(mf1), opt_value), linewidth=1.0, label='True OPT', color='#CC4F1B')

    plt.xlabel('Number of Function Evaluations', fontsize=14)
    plt.ylabel('Best Fitness Value', fontsize=14)
    if not minimize:
        plt.legend(fontsize=8, frameon=False, loc="lower right")
    else:
        plt.legend(fontsize=8, frameon=False, loc="upper right")
    if log_scale:
        plt.xscale('log')
    plt.yticks(fontsize=14)
    if title is not None:
        plt.title(title, fontsize=10)


    with open(save_dir, 'wb') as f:
        plt.savefig(f)
    plt.close()


class Render:
    """ Helper class for rendering the environment. 
        The class should be able to plot the particles actual position in the optimization function landscape if it is 
        a 1D or 2D function. Also the class should be able to plot the state history as a gif or video.
    """
    def __init__(self, env):
        self.env = env
    
        
    def render_state(self, file_path: Optional[str] = None, **kwargs):
        self.optimal_position = kwargs.get("optimal_positions", None)
        if self.env.n_dim > 2:
            raise ValueError("Cannot render state for n_dim > 2")
        
        if self.env.n_dim == 1:
            self._render_state_1d(file_path)
        else:
            self._render_state_2d(file_path)
            
    def _render_state_1d(self, file_path: Optional[str] = None):
        fig, ax = plt.subplots()
        x = np.linspace(self.env.bounds[0], self.env.bounds[1], 1000)
        y = self.env.objective_function.evaluate(x)
        ax.plot(x, y)
        # plot the optimal position if given
        if self.optimal_position is not None:
            ax.axvline(self.optimal_position, color="red", linestyle="--", label="Optimal Position")
        ax.set_xlim(self.env.bounds[0], self.env.bounds[1])
        ax.set_ylim(self.env.bounds[0], self.env.bounds[1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Particle positions")
        state = self.env._get_actual_state()
        ax.scatter(state, np.zeros_like(state), c="red", s=10)
        if file_path is not None:
            plt.savefig(file_path)
        else:
            plt.show()

        
    def _render_state_2d(self, file_path: Optional[str] = None):
        fig, ax = plt.subplots()
        x = np.linspace(self.env.bounds[0], self.env.bounds[1], 1000)
        y = np.linspace(self.env.bounds[0], self.env.bounds[1], 1000)
        X, Y = np.meshgrid(x, y)
        if self.env.log_scale: 
            Z = np.log10(-self.env.objective_function.evaluate(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape))
        else:
            Z = self.env.objective_function.evaluate(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)
        ax.contour(X, Y, Z, 50)
        ax.set_xlim(self.env.bounds[0][0], self.env.bounds[1][0])
        ax.set_ylim(self.env.bounds[0][1], self.env.bounds[1][1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Particle positions")
        state = self.env._get_actual_state()
        # use triangles to for the shape of the particles
        # plot scatter plot of the novel points with color red and the rest with color blue
        roles = self.env.state_history[:, self.env.current_step, -2]
        previous_state = self.env.state_history[:, self.env.current_step - 1, :-2]
        exploiter_id = np.where(roles == 1)[0]
        non_exploiter_id = np.isin(np.arange(len(state)), exploiter_id, invert=True)
        ax.scatter(state[exploiter_id, 0], state[exploiter_id, 1], c="red", s=20, marker="*", edgecolors="black", label="Exploiter's points", alpha=1)
        ax.scatter(state[non_exploiter_id, 0], state[non_exploiter_id, 1], c="green", s=20, marker="^", edgecolors="black", label="Explorer's points", alpha=1)
        # add a text of it particle index at the position of the particle
        for i, txt in enumerate(range(len(state))):
            ax.text(state[i, 0], state[i, 1], str(i), fontsize=14)
        
        # plot a line between the previous state and the current state
        for i in range(len(state)):
            ax.plot([previous_state[i, 0], state[i, 0]], [previous_state[i, 1], state[i, 1]], c="black", alpha=0.5)
        
        # plot the optimal position if given
        if self.optimal_position is not None:
            ax.scatter(self.optimal_position[0], self.optimal_position[1], c="blue", s=300, marker="o", edgecolors="black", label="Optimal Position")
        
        ax.legend()
        
        
        if file_path is not None:
            plt.savefig(file_path)
        else:
            plt.show()
        
    
    def render_state_history(self, file_path: str, fps: int = 10, **kwargs):
        self.optimal_position = kwargs.get("optimal_positions", None)
        if self.env.n_dim > 2:
            raise ValueError("Cannot render state for n_dim > 2")
        
        if self.env.n_dim == 1:
            self._render_state_history_1d(file_path, fps)
        else:
            self._render_state_history_2d(file_path, fps)
            
    def _render_state_history_1d(self, file_path: str, fps: int):
        fig, ax = plt.subplots()
        x = np.linspace(self.env.bounds[0], self.env.bounds[1], 1000)
        y = self.env.objective_function.evaluate(x)
        ax.plot(x, y)
        if self.optimal_position is not None:
            ax.axvline(self.optimal_position, color="red", linestyle="--", label="Optimal Position")
        ax.set_xlim(self.env.bounds[0], self.env.bounds[1])
        ax.set_ylim(self.env.bounds[0], self.env.bounds[1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Particle positions")
        scat = ax.scatter([], [], c="red", s=10)
        
        def animate(i):
            scat.set_offsets(self.env.state_history[i]) 
            if self.optimal_position is not None:
                ax.axvline(self.optimal_position, color="red", linestyle="--", label="Optimal Position")
            return scat,
        
        anim = animation.FuncAnimation(fig, animate, frames=len(self.env.state_history), interval=1000/fps, blit=True)
        anim.save(file_path, writer="Pillow")
        
    def _render_state_history_2d(self, file_path: str, fps: int):
        fig, ax = plt.subplots()
        x = np.linspace(self.env.bounds[0], self.env.bounds[1], 1000)
        y = np.linspace(self.env.bounds[0], self.env.bounds[1], 1000)
        X, Y = np.meshgrid(x, y)
        if self.env.log_scale:
            Z = np.log10(-self.env.objective_function.evaluate(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape))
        else:
            Z = self.env.objective_function.evaluate(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)
        ax.contour(X, Y, Z, 50)
        if self.optimal_position is not None:
            ax.scatter(self.optimal_position[0], self.optimal_position[1], c="green", s=300, marker="o", edgecolors="black", label="Optimal Position")
        ax.set_xlim(self.env.bounds[0][0], self.env.bounds[1][0])
        ax.set_ylim(self.env.bounds[0][1], self.env.bounds[1][1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Particle positions")
        ax.legend()
        scat = ax.scatter([], [], c="red", s=100, marker="^", edgecolors="black")
        # add a text box to display the iteration number
        text = ax.text(0.05, 0.95, "", transform=ax.transAxes)
        self.previous_state_history = self.env.state_history[:, 0, :-2]
        
        
        def animate(i):
            scat.set_offsets(self.env.state_history[:, i, :-2])
            text.set_text(f"Iteration: {i}")
            # plot a line between the previous state and the current state
            for j in range(len(self.env.state_history)):
                ax.plot([self.previous_state_history[j, 0], self.env.state_history[j, i, 0]], [self.previous_state_history[j, 1], self.env.state_history[j, i, 1]], c="black", alpha=0.1)
                
            self.previous_state_history = self.env.state_history[:, i, :-2]
            # clear the line between the previous state and the current state
        
            # use different colors for the particles based on their role - red for closer half, blue for farther half
            scat.set_color(["red" if role == 1 else "blue" for role in self.env.state_history[:, i, -2]])
    
            return scat,
        print("Creating animation")
        anim = animation.FuncAnimation(fig, animate, frames=self.env.state_history.shape[1], interval=1000/fps, blit=True)
        print("Saving animation to: ", file_path)
        anim.save(file_path, writer="Pillow")
            
    
from scipy.spatial.distance import cdist

def filter_points(points, min_distance):
    """
    Efficiently filter points along with their function output to ensure minimum distance between them
    using vectorized operations.
    
    :param points_with_output: A numpy array of points with the last dimension being the function output.
    :param min_distance: The minimum allowed distance between any two points, ignoring the output value.
    :return: A numpy array of filtered points with their output.
    """
    # Separate the coordinates and outputs
    coordinates = points[:, :-1]
    _ = points[:, -1]
    
    # Calculate the condensed distance matrix between points
    distance_matrix = cdist(coordinates, coordinates, 'euclidean')
    
    # We only care about the upper triangle of the distance matrix, since it is symmetric.
    # We also fill the diagonal with np.inf to ignore zero distance to itself.
    np.fill_diagonal(distance_matrix, np.inf)
    
    # Filter points that are too close to each other
    filtered_indices = np.full(len(coordinates), True)
    for i in range(len(coordinates)):
        if filtered_indices[i]:
            # Find points that are too close to the current point and mark them as False
            too_close_indices = np.where(distance_matrix[i] < min_distance)[0]
            filtered_indices[too_close_indices] = False
            # Ensure the current point is always kept
            filtered_indices[i] = True
    
    return points[filtered_indices]

def select_candidate_points(grid_points, evaluated_points, n_select):
    next_candidate_points = []
    # Calculate all distances
    for n in range(n_select):
        distances = cdist(grid_points, evaluated_points)

        # Find minimum distance to evaluated points for each grid point
        min_distances = np.min(distances, axis=1)

        # Select new points (farthest points first)
        indices_to_select = np.argmax(min_distances)
        new_evaluated = grid_points[indices_to_select]
        evaluated_points = np.vstack([evaluated_points, new_evaluated])
        next_candidate_points.append(new_evaluated)

    return evaluated_points, np.array(next_candidate_points)


def initialize_grid(bounds, resolution, n_dim):
    # Ensure bounds are provided for each dimension
    if len(bounds[0]) != n_dim:
        raise ValueError("Bounds length must match the number of dimensions")

    # Generate grid points for each dimension
    grid_ranges = [np.arange(bounds[0][dim], bounds[1][dim] + resolution, resolution) for dim in range(n_dim)]

    # Create a meshgrid for the given dimensions
    grids = np.meshgrid(*grid_ranges, indexing='ij')

    # Flatten and combine the grids to create a list of points
    points = np.vstack([grid.flatten() for grid in grids]).T
    return points