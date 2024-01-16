import numpy as np
from deephive.policies.mappo import MAPPO
from deephive.environment.optimization_environment import OptimizationEnv
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

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


def initialize(config_path, mode="train", **kwargs):
    env = OptimizationEnv(config_path)
    agent_policy = MAPPO(config_path)
    if mode == "test" or mode == "benchmark":
        model_path = kwargs.get("model_path", None)
        if model_path is None:
            raise ValueError("Model path must be provided for testing")
        agent_policy.load(model_path)
    return env, agent_policy

def print_items(**kwargs):
    for key, value in kwargs.items():
        print(key, value)
        
def get_action(observation_info, observation_std, agent_policy, env):
    observation = observation_info
    actions = np.zeros((env.n_agents, env.n_dim))
    for dim in range(env.n_dim):
        observation[dim] = observation[dim].astype(np.float32)
        observation_std[dim] = observation_std[dim].astype(np.float32)
        action = agent_policy.select_action(observation[dim], observation_std[dim])
        actions[:, dim] = action
    return actions


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

def check_proximity(checkpoint, target_array, threshold=0.01):
    # check if the point is very close to any of the points in the target_array
    if len(target_array) == 0:
        return False
    for target in target_array:
        if np.linalg.norm(checkpoint - target) < threshold:
            return True
    return False

def get_informed_action(env, number_of_points=5):
    # let the action be the distance it takes for the agents to get to the a random point in the high std points
    grid_points = env.grid_points
    actions = np.zeros((number_of_points, env.n_dim))
    evaluated_points = env.evaluated_points
    # get the next candidate points
    evaluated_points, next_candidate_points = select_candidate_points(grid_points, evaluated_points, number_of_points)
                                                                      
    # scale the next candidate points to the bounds of the environment
    #next_candidate_points = env.scaler_helper.scale(next_candidate_points, env.min_pos, env.max_pos)
    actions = next_candidate_points - env.state[(10-number_of_points):, :env.n_dim]

    return actions, next_candidate_points
        
def plot_point(grid_points, evaluated_points, new_evaluated,save_dir=None):
    if len(grid_points[0]) == 2:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.scatter(grid_points[:, 0], grid_points[:, 1], s=20, color='black', label="Checkpoints")
        ax.scatter(evaluated_points[:, 0], evaluated_points[:, 1], s=60, color='red', label="Evaluated Points")
        if new_evaluated is not None:
            ax.scatter(new_evaluated[:, 0], new_evaluated[:, 1], s=60, color='green', label="New Evaluated Points")
        plt.legend()
        if save_dir is not None:
            plt.savefig(save_dir)
        else:
            plt.show()
    elif len(grid_points[0]) == 3:
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], s=20, color='black', label="Checkpoints")
        ax.scatter(evaluated_points[:, 0], evaluated_points[:, 1], evaluated_points[:, 2], s=60, color='red', label="Evaluated Points")
        if new_evaluated is not None:
            ax.scatter(new_evaluated[:, 0], new_evaluated[:, 1], new_evaluated[:, 2], s=60, color='green', label="New Evaluated Points")
        plt.legend()
        if save_dir is not None:
            plt.savefig(save_dir)
        else:
            plt.show()
    else:
        raise ValueError("Grid points must be 2 or 3 dimensional")
    plt.close()