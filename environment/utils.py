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
plt.rcParams['figure.figsize'] = [7, 7]
plt.rcParams['figure.dpi'] = 80
plt.rcParams['font.size'] = 12

class ScalingHelper:
    """
    A class that provides methods for scaling and rescaling values.
    """
    @staticmethod
    def scale(d, d_min, d_max):
        """
        Scales a value between 0 and 1 based on the given minimum and maximum values.

        Args:
            d (float): The value to be scaled.
            d_min (float): The minimum value of the range.
            d_max (float): The maximum value of the range.

        Returns:
            float: The scaled value between 0 and 1.
        """
        
        # ensure that the values has the same decimal precision
        scaled_d = (d - d_min) / ((d_max - d_min) + 1e-10)
        # ROUND THE VALUE TO 4 DECIMAL PLACES
        scaled_d = np.round(scaled_d, 4)
        return scaled_d
        
    @staticmethod
    def rescale(d, d_min, d_max):
        """
        Rescales a value between the given minimum and maximum values to its original range.

        Args:
            d (float): The value to be rescaled.
            d_min (float): The minimum value of the range.
            d_max (float): The maximum value of the range.

        Returns:
            float: The rescaled value between d_min and d_max.
        """
        
        rescaled_d = d_min + (d_max - d_min) * d
        return rescaled_d
    

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
    plot_error_bounds: bool = False
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
    plt.legend(fontsize=14, frameon=False)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xscale('log')

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
    label_list: Optional[List[str]] = None
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
    """
    symbol_list = symbol_list if symbol_list is not None else ['-' for _ in range(len(fopt))]
    color_list = color_list if color_list is not None else ['#3F7F4C', '#CC4F1B', '#FFB852', '#64B5CD']
    label_list = label_list if label_list is not None else ['EXP1', 'EXP2', 'EXP3', 'EXP4']

    print(f"Number of function evaluations: {len(fopt[0])}")
    print(f"Number of algorithms: {len(fopt)}")

    plt.figure(figsize=(6, 4), dpi=200)
    for i, single_fopt in enumerate(fopt):
        mf1, ml1, mh1 = mean_confidence_interval(single_fopt, 0.95)
        x = (np.arange(len(mf1)) + 1) * n_agents
        if show_std:
            plt.errorbar(x, mf1, yerr=mh1 - ml1, fmt=symbol_list[i], linewidth=2.0, label=label_list[i], color=color_list[i])
        else:
            plt.plot(x, mf1, symbol_list[i], linewidth=2.0, label=label_list[i], color=color_list[i])

    if opt_value is not None:
        plt.plot(x, np.full(len(mf1), opt_value), linewidth=1.0, label='True OPT', color='#CC4F1B')

    plt.xlabel('Number of Function Evaluations', fontsize=14)
    plt.ylabel('Best Fitness Value', fontsize=14)
    plt.legend(fontsize=8, frameon=False, loc="lower right")
    plt.xscale('log')
    plt.yticks(fontsize=14)

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
        
    def render_state(self, file_path: Optional[str] = None):
        if self.env.n_dim > 2:
            raise ValueError("Cannot render state for n_dim > 2")
        
        if self.env.n_dim == 1:
            self._render_state_1d(file_path )
        else:
            self._render_state_2d(file_path)
            
    def _render_state_1d(self, file_path: Optional[str] = None):
        fig, ax = plt.subplots()
        x = np.linspace(self.env.bounds[0], self.env.bounds[1], 1000)
        y = self.env.objective_function.evaluate(x)
        ax.plot(x, y)
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
        novel_id = self.env.ids_true_function_eval
        ax.scatter(state[novel_id, 0], state[novel_id, 1], c="red", s=100, marker="^", edgecolors="black", label="novel points", alpha=1)
        ax.scatter(state[~novel_id, 0], state[~novel_id, 1], c="red", s=100, marker="^", edgecolors="black", label="non-novel points", alpha=1)
        
        if file_path is not None:
            plt.savefig(file_path)
        else:
            plt.show()
        
    
    def render_state_history(self, file_path: str, fps: int = 10):
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
        ax.set_xlim(self.env.bounds[0], self.env.bounds[1])
        ax.set_ylim(self.env.bounds[0], self.env.bounds[1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Particle positions")
        scat = ax.scatter([], [], c="red", s=10)
        
        def animate(i):
            scat.set_offsets(self.env.state_history[i])
            return scat,
        
        anim = animation.FuncAnimation(fig, animate, frames=len(self.env.state_history), interval=1000/fps, blit=True)
        anim.save(file_path, writer="Pillow")
        
    def _render_state_history_2d(self, file_path: str, fps: int):
        fig, ax = plt.subplots()
        x = np.linspace(self.env.bounds[0], self.env.bounds[1], 1000)
        y = np.linspace(self.env.bounds[0], self.env.bounds[1], 1000)
        X, Y = np.meshgrid(x, y)
        Z = self.env.objective_function.evaluate(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)
        ax.contour(X, Y, Z, 50)
        ax.set_xlim(self.env.bounds[0][0], self.env.bounds[1][0])
        ax.set_ylim(self.env.bounds[0][1], self.env.bounds[1][1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Particle positions")
        ax.legend()
        scat = ax.scatter([], [], c="red", s=100, marker="^", edgecolors="black")
        # add a text box to display the iteration number
        text = ax.text(0.05, 0.95, "", transform=ax.transAxes)
        
        def animate(i):
            scat.set_offsets(self.env.state_history[:, i, :-2])
            text.set_text(f"Iteration: {i}")
            # use different colors for the particles based on their role - red for closer half, blue for farther half
            scat.set_color(["red" if role == 1 else "blue" for role in self.env.state_history[:, i, -2]])
    
            return scat,
        
        anim = animation.FuncAnimation(fig, animate, frames=self.env.state_history.shape[1], interval=1000/fps, blit=True)
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

def initialize_grid(bounds, resolution):
        # Define the bounds of the grid
        x_min, y_min = bounds[0]
        x_max, y_max = bounds[1]
        # Create a meshgrid of x and y values within the bounds
        x_values = np.arange(x_min, x_max + resolution, resolution)
        y_values = np.arange(y_min, y_max + resolution, resolution)
        xx, yy = np.meshgrid(x_values, y_values)

        # Flatten the meshgrid to get the individual x and y coordinates
        x_coordinates = xx.flatten()
        y_coordinates = yy.flatten()

        # Create a list of points as (x, y) tuples
        points = list(zip(x_coordinates, y_coordinates))

        # Convert the list of points to a NumPy array
        points_array = np.array(points)
        
        return points_array