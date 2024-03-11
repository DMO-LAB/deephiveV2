

from matplotlib import pyplot as plt
from typing import List, Optional
import numpy as np
from deephive.environment.utils import mean_confidence_interval

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