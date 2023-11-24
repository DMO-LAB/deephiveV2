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
        self.bounds = kwargs.get("bounds", [-1, 1])  # Define bounds for plotting

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

    def evaluate(self, points):
        """
        Evaluate new points using the GP model.

        Parameters:
        - points: An array of points to evaluate.
        """
        predictions, std_dev = self.gp.predict(points, return_std=True)
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
        x = np.linspace(self.bounds[0], self.bounds[1], 1000)
        X = np.atleast_2d(x).T
        y_pred, sigma = self.gp.predict(X, return_std=True)

        plt.figure()
        plt.plot(x, y_pred, 'b-', label='GP Prediction')
        plt.fill_between(x, y_pred - sigma, y_pred + sigma, alpha=0.2, color='blue')
        plt.plot(self.samples[-10:, 0], self.values[-10:], 'r.', markersize=10, label='Observations')

        # Plot variance
        plt.fill_between(x, sigma, alpha=0.5, color='orange', label='Variance')
        
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Gaussian Process Surrogate Model (1D) with Variance")
        plt.legend()
        plt.savefig(save_dir)

    def __plot_2d(self, save_dir="gp_surrogate_2d.png"):
        # Create a grid for plotting
        x = np.linspace(self.bounds[0], self.bounds[1], 100)
        y = np.linspace(self.bounds[0], self.bounds[1], 100)
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

        # Plot variance
        variance_contour = ax.contour(X, Y, sigma, colors='orange')
        plt.colorbar(variance_contour, label='Variance')

        ax.scatter(self.samples[-10:, 0], self.samples[-10:, 1], c='red', label='Observations')
        ax.set_title("Gaussian Process Surrogate Model (2D) with Variance")
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        plt.legend()
        plt.savefig(save_dir)

    def __plot_higher_dims(self):
        raise NotImplementedError("Plotting higher dimensional data is not implemented yet.")



