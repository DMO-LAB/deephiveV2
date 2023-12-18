import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt

class GPSurrogateModule:
    def __init__(self, initial_samples, initial_values, kernel=None, **kwargs):
        """
        Initialize the Gaussian Process Surrogate Module.

        Parameters:
        - initial_samples: An array of initial sample points.
        - initial_values: Values at the initial sample points.
        - kernel: The kernel to use for the Gaussian Process.
        """
        self.samples = initial_samples
        self.values = initial_values
        self.kernel = kernel if kernel is not None else C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer=9)
        self.gp.fit(self.samples, self.values)
        self.bounds = kwargs.get("bounds", np.array([[-1, -1], [1, 1]]))
        
        self.checkpoints = self._initialize_checkpoints()
        
    def _initialize_checkpoints(self, resolution=0.05):
        # Define the bounds of the grid
        x_min, y_min = self.bounds[0]
        x_max, y_max = self.bounds[1]
    

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

    def update_model(self, new_samples, new_values):
        """
        Update the GP model with new samples and values.

        Parameters:
        - new_samples: An array of new sample points.
        - new_values: Values at the new sample points.
        """
        self.samples = np.vstack([self.samples, new_samples])
        self.values = np.append(self.values, new_values)
        self.gp.fit(self.samples, self.values)

    def evaluate(self, points, scale=False):
        """
        Evaluate new points using the GP model.

        Parameters:
        - points: An array of points to evaluate.
        """
        predictions, std_dev = self.gp.predict(points, return_std=True)
        if scale:
            # ensure the points are more than 1 and then scale the std_dev from 0 to 1
            if len(std_dev) == 1:
                raise ValueError("The points must be more than 1")
            std_dev = (std_dev - np.min(std_dev)) / (np.max(std_dev) - np.min(std_dev))
        return predictions, std_dev

    def plot_surrogate(self, save_dir = "gp_surrogate.png"):
        """
        Plot the learned surrogate function according to the dimensionality of the data.
        """
        dim = self.samples.shape[1]

        if dim == 1:
            # Plot for 1D data
            self.__plot_1d(save_dir)
        elif dim == 2:
            # Plot for 2D data
            self.__plot_2d(save_dir)
        else:
            # For higher dimensions, visualize a 2D slice or projection
            print("Data is higher than 2D. Plotting a 2D slice.")
            self.__plot_higher_dims()

    def __plot_1d(self, save_dir="gp_surrogate_1d.png"):
        x = np.linspace(self.bounds[0][0], self.bounds[1][0], 1000)
        X = np.atleast_2d(x).T
        y_pred, sigma = self.gp.predict(X, return_std=True)

        plt.figure()
        plt.plot(x, y_pred, 'b-', label='GP Prediction')
        plt.fill_between(x, y_pred - sigma, y_pred + sigma, alpha=0.2, color='blue')
        plt.plot(self.samples[-10:, 0], self.values[-10:], 'r.', markersize=10, label='Observations')

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Gaussian Process Surrogate Model (1D) with Variance")
        plt.legend()
        plt.savefig(save_dir)

    def __plot_2d(self, save_dir="gp_surrogate_2d.png"):
        # Create a grid for plotting
        x = np.linspace(self.bounds[0][0], self.bounds[1][0], 100)
        y = np.linspace(self.bounds[0][1], self.bounds[1][1], 100)
        X, Y = np.meshgrid(x, y)
        XY = np.vstack([X.ravel(), Y.ravel()]).T

        # Predictions and variance
        Z, sigma = self.gp.predict(XY, return_std=True)
        Z = Z.reshape(X.shape)
        sigma = sigma.reshape(X.shape)

        # Plot prediction
        fig, ax = plt.subplots()
        contour = ax.contourf(X, Y, Z, cmap='viridis')
        plt.colorbar(contour, label='Prediction')

        ax.scatter(self.samples[:, 0], self.samples[:, 1], c='red', label='Observations')
        ax.set_title("Gaussian Process Surrogate Model (2D) Mean Prediction")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        plt.legend()
        plt.savefig(save_dir)


    def plot_variance(self, save_dir="gp_surrogate_variance.png"):
        """
        Plot the variance of the learned surrogate function according to the dimensionality of the data.
        """
        dim = self.samples.shape[1]

        if dim == 1:
            # Plot for 1D data
            self.__plot_1d_variance(save_dir)
        elif dim == 2:
            # Plot for 2D data
            self.__plot_2d_variance(save_dir)
        else:
            # For higher dimensions, visualize a 2D slice or projection
            print("Data is higher than 2D. Plotting a 2D slice.")
            self.__plot_higher_dims()

    def __plot_1d_variance(self, save_dir="gp_surrogate_variance_1d.png"):
        x = np.linspace(self.bounds[0], self.bounds[1], 1000)
        X = np.atleast_2d(x).T
        y_pred, sigma = self.gp.predict(X, return_std=True)

        plt.figure()
        plt.plot(x, sigma, 'b-', label='GP Variance')
        plt.plot(self.samples[-10:, 0], self.values[-10:], 'r.', markersize=10, label='Observations')

        plt.xlabel("X")
        plt.ylabel("Variance")
        plt.title("Gaussian Process Surrogate Model Variance (1D)")
        plt.legend()
        plt.savefig(save_dir)

    def __plot_2d_variance(self, save_dir="gp_surrogate_variance_2d.png"):
        # Create a grid for plotting
        x = np.linspace(self.bounds[0][0], self.bounds[1][0], 100)
        y = np.linspace(self.bounds[0][1], self.bounds[1][1], 100)
        X, Y = np.meshgrid(x, y)
        XY = np.vstack([X.ravel(), Y.ravel()]).T

        # Predictions and variance
        Z, sigma = self.gp.predict(XY, return_std=True)
        Z = Z.reshape(X.shape)
        sigma = sigma.reshape(X.shape)

        # Plot variance
        fig, ax = plt.subplots()
        contour = ax.contourf(X, Y, sigma, cmap='viridis')
        plt.colorbar(contour, label='Variance')

        ax.scatter(self.samples[-20:, 0], self.samples[-20:, 1], c='red', label='Observations')
        ax.set_title("Gaussian Process Surrogate Model Variance (2D)")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        plt.legend()
        plt.savefig(save_dir)

    def __plot_higher_dims(self):
        raise NotImplementedError("Plotting higher dimensional data is not implemented yet.")

    def evaluate_accuracy(self, real_func, bounds=None, num_points=100):
        """
        Evaluate the accuracy of the GP model against a real function in n-dimensions.

        Parameters:
        - real_func: The real function to compare against. This should be a function that
                    accepts an n-dimensional array of points and returns their corresponding values.
        - bounds: A list of tuples specifying the lower and upper bounds for each dimension.
        - num_points: Number of points to evaluate along each dimension.
        """
        if bounds is None:
            bounds = self.bounds
        if len(bounds) != self.samples.shape[1]:
            raise ValueError("The bounds dimensionality must match the sample dimensionality.")

        # Generate a grid of test points
        grid = np.meshgrid(*[np.linspace(b[0], b[1], num_points) for b in bounds])
        test_points = np.vstack(map(np.ravel, grid)).T

        # Evaluate the real function
        y_real = real_func(test_points)

        # Evaluate the GP model
        y_pred, _ = self.gp.predict(test_points, return_std=True)

        # Compute MSE or RMSE
        mse = np.mean((y_real - y_pred) ** 2)
        rmse = np.sqrt(mse)

        return mse, rmse
        
    def check_checkpoints(self, alpha=1, percentile=85):
        self.checkpoints_mean, self.checkpoints_std = self.evaluate(self.checkpoints, scale=True)
        mean_std = np.array(self.checkpoints_mean) + np.array(self.checkpoints_std) * alpha
        
        high_std = np.percentile(self.checkpoints_std, percentile)
        high_mean_std = np.percentile(mean_std, percentile)
        low_std = np.percentile(self.checkpoints_std, 100 - percentile)
        
        high_std_points = self.checkpoints[np.where(self.checkpoints_std >= high_std)]
        high_std_points_std = self.checkpoints_std[np.where(self.checkpoints_std >= high_std)]
        low_std_points = self.checkpoints[np.where(self.checkpoints_std <= low_std)]
        self.percent_high_std = high_std_points.shape[0] / self.checkpoints.shape[0] * 100
        
        high_mean_std_points = self.checkpoints[np.where(mean_std >= high_mean_std)]

        
        self.high_std_points = high_std_points
        self.high_mean_std_points = high_mean_std_points
        self.low_std_points = low_std_points
        
        return high_std_points, high_std_points_std


    def cal_reward(self):
        _ = self.check_checkpoints()
        reward = 1 - self.percent_high_std / 100
        std_std = np.std(self.checkpoints_std)
        
        return reward * (1 - std_std)
        
        # calculate the 
        
    def plot_checkpoints_state(self):
        
        if not hasattr(self, "high_std_points"):
            self.check_checkpoints()
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.scatter(self.checkpoints[:, 0], self.checkpoints[:, 1], s=10, color='black', label="Checkpoints")
        ax.scatter(self.high_std_points[:, 0], self.high_std_points[:, 1], s=20, color='red', label="High Std")
        # overlay the sampled points on the plot
        ax.scatter(self.samples[:, 0], self.samples[:, 1], s=20, color='yellow', label="Sampled Points")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Grid of Points')
        ax.grid(True)
        plt.show()
