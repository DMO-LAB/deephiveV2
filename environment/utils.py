import json
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np

# set figure size, dpi and fontsize
plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 14



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


class Render:
    """ Helper class for rendering the environment. 
        The class should be able to plot the particles actual position in the optimization function landscape if it is 
        a 1D or 2D function. Also the class should be able to plot the state history as a gif or video.
    """
    def __init__(self, env):
        self.env = env
        
    def render_state(self):
        if self.env.n_dim > 2:
            raise ValueError("Cannot render state for n_dim > 2")
        
        if self.env.n_dim == 1:
            self._render_state_1d()
        else:
            self._render_state_2d()
            
    def _render_state_1d(self):
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
        plt.show()

        
    def _render_state_2d(self):
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
        ax.scatter(state[:, 0], state[:, 1], c="red", s=500, marker="^", edgecolors="black")
        plt.savefig("test.png")
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
        anim.save(file_path, writer="ffmpeg")
        
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
        scat = ax.scatter([], [], c="red", s=300, marker="^", edgecolors="black")
        # add a text box to display the iteration number
        text = ax.text(0.05, 0.95, "", transform=ax.transAxes)
        
        def animate(i):
            scat.set_offsets(self.env.state_history[:, i, :])
            text.set_text(f"Iteration: {i}")
            return scat,
        
        anim = animation.FuncAnimation(fig, animate, frames=self.env.state_history.shape[1], interval=1000/fps, blit=True)
        anim.save(file_path, writer="ffmpeg")
            
    