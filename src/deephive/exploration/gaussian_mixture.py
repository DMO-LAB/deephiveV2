import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse

class ExplorationModule:
    def __init__(self, initial_samples, n_components=1, max_samples=None, **kwargs):
        """
        Initialize the exploration module with a Gaussian Mixture Model.

        Parameters:
        - initial_samples: An array of initial sample points.
        - n_components: Number of components (Gaussians) in the initial GMM.
        - max_samples: Maximum number of samples to use for updating the GMM.
        """
        self.n_components = n_components
        self.max_samples = max_samples
        self.samples = initial_samples
        self.gmm = GaussianMixture(n_components=self.n_components)
        self.gmm.fit(self.samples)
        self.upper_bound = kwargs.get("upper_bound", 1)
        self.lower_bound = kwargs.get("lower_bound", -1)
        # Initialize fixed points in the domain
        self.fixed_points = self.initialize_fixed_points(num_points=100)

    def initialize_fixed_points(self, num_points):
        """
        Initialize fixed points uniformly across the domain.

        Parameters:
        - num_points: Number of fixed points to generate.

        Returns:
        - Array of fixed points.
        """
        x = np.linspace(self.lower_bound, self.upper_bound, int(np.sqrt(num_points)))
        y = np.linspace(self.lower_bound, self.upper_bound, int(np.sqrt(num_points)))
        xv, yv = np.meshgrid(x, y)
        return np.column_stack([xv.ravel(), yv.ravel()])
    
    def calculate_reward(self, use_coverage=True, use_uniformity=True):
        """
        Calculate the reward based on exploration progress, with options to use coverage,
        uniformity, or both.

        Parameters:
        - use_coverage: Boolean, whether to use coverage in the reward calculation.
        - use_uniformity: Boolean, whether to use uniformity in the reward calculation.

        Returns:
        - Reward value.
        """
        densities = self.gmm.score_samples(self.fixed_points)
        
        reward = 0

        # Coverage-based reward
        if use_coverage:
            threshold = np.percentile(densities, 50)  # Example threshold
            coverage = np.mean(densities > threshold)
            reward += coverage

        # Uniformity-based reward
        if use_uniformity:
            uniformity = 1 / np.std(densities) if np.std(densities) != 0 else 0
            reward += uniformity

        return reward

    
    def update_distribution(self, new_samples):
        """
        Update the GMM with new samples.

        Parameters:
        - new_samples: An array of new sample points.
        """
        # Optionally limit the number of samples to prevent excessive growth
        if self.max_samples and len(self.samples) >= self.max_samples:
            self.samples = self.samples[-self.max_samples:]
        
        self.samples = np.vstack([self.samples, new_samples])
        self.gmm = GaussianMixture(n_components=self.n_components)
        self.gmm.fit(self.samples)

    def sample_candidate_points(self, n_samples):
        """
        Generate new candidate points based on the current GMM.

        Parameters:
        - n_samples: Number of candidate points to generate.
        """
        return self.gmm.sample(n_samples)[0]

    def assess_novelty(self, points, scale=True):
        """
        Assess the novelty of given points based on the current GMM.

        Parameters:
        - points: An array of points to assess.
        """
        # Evaluate the probability density of each point under each GMM component
        densities = np.array([multivariate_normal(mean=mean, cov=cov, allow_singular=True).pdf(points)
                              for mean, cov in zip(self.gmm.means_, self.gmm.covariances_)])

       # cap densities to prevent very small values - limits the novelty score
        #densities[densities < 1e-6] = 1e-6
        log_density = np.log10(densities)
        # scale the logs to [0, 1] -  the smallest log will be 1 and the largest will be 0
        scaled_log_density = (log_density - np.min(log_density)) / (np.max(log_density) - np.min(log_density))
        # Novelty score could be the inverse of density or a more complex function
        novelty_scores = 1 / np.max(densities, axis=0)
        # Scale the novelty scores to [0, 1]
        if scale:
            novelty_scores = (novelty_scores - np.min(novelty_scores)) / (np.max(novelty_scores) - np.min(novelty_scores))

        return novelty_scores

    def access_novelty_density(self, points):
        """
        Assess the novelty of given points based on the current GMM.

        Parameters:
        - points: An array of points to assess.
        """
        # Evaluate the probability density of each point under each GMM component
        densities = np.array([multivariate_normal(mean=mean, cov=cov, allow_singular=True).pdf(points)
                              for mean, cov in zip(self.gmm.means_, self.gmm.covariances_)])

        # replace all 0 densities with 1/10th of the minimum density
        densities[densities == 0] = np.min(densities[densities != 0]) / 10
        # take the log of the densities
        log_density = np.log10(densities)
        # calculate the novelty score from the log of the densities
        # small log values will have a high novelty score and large log values will have a low novelty score (- and +)
        novelty_scores = np.max(log_density, axis=0) - log_density
        # scale the novelty scores to [0, 1]
        novelty_scores = (novelty_scores - np.min(novelty_scores)) / (np.max(novelty_scores) - np.min(novelty_scores))

        # # print densities, log_density, novelty_scores
        # print("Novelty scores: ", novelty_scores)
        # print("Densities: ", densities)
        # print("Log densities: ", log_density)
        # print("Max log density: ", np.max(log_density, axis=0))
        # print("Min log density: ", np.min(log_density, axis=0))
        # print("Max novelty score: ", np.max(novelty_scores))
        # print("Min novelty score: ", np.min(novelty_scores))

        return novelty_scores, densities

    def get_variance(self, point):
        """
        Estimate the variance of a given point based on the GMM.

        Parameters:
        - point: The point to estimate variance for.
        """
        # Find the nearest GMM component to the point
        nearest_component = np.argmin(np.linalg.norm(self.gmm.means_ - point, axis=1))
        # Return the variance (diagonal of the covariance matrix) of the nearest component
        return np.diag(self.gmm.covariances_[nearest_component])

    def plot_distribution(self):
        """
        Plot the current GMM.
        """
        # Create a mesh grid on which to evaluate the GMM
        x = np.linspace(self.lower_bound, self.upper_bound, 100)
        y = np.linspace(self.lower_bound, self.upper_bound, 100)
        X, Y = np.meshgrid(x, y)
        XY = np.array([X.ravel(), Y.ravel()]).T

        # Evaluate the GMM's probability density function (PDF) on the grid
        Z = self.gmm.score_samples(XY)
    
        Z = Z.reshape(X.shape)
        # Plot the contour
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title('GMM Contour Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

    def plot_distribution_no_exp(self):
        """
        Plot the current GMM.
        """
        # Create a mesh grid on which to evaluate the GMM
        x = np.linspace(self.lower_bound, self.upper_bound, 100)
        y = np.linspace(self.lower_bound, self.upper_bound, 100)
        X, Y = np.meshgrid(x, y)
        XY = np.array([X.ravel(), Y.ravel()]).T

        # Evaluate the GMM's probability density function (PDF) on the grid
        Z = self.gmm.score_samples(XY)
    
        Z = Z.reshape(X.shape)
        # Plot the contour
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title('GMM Contour Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

    def plot_2d_distribution(self, candidate_points, novelty_scores, env, particles=None):
        fig, ax = plt.subplots()
        x = np.linspace(env.bounds[0], env.bounds[1], 1000)
        y = np.linspace(env.bounds[0], env.bounds[1], 1000)
        X, Y = np.meshgrid(x, y)
        Z = env.objective_function.evaluate(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)

        # Evaluate the GMM PDF on the grid
        XY = np.vstack([X.ravel(), Y.ravel()]).T
        Z_gmm = np.exp(self.gmm.score_samples(XY)).reshape(X.shape)
        
        # Plot the objective function contour
        ax.contour(X, Y, Z, 50)
        
        # Plot the GMM contour with some transparency
        ax.contourf(X, Y, Z_gmm, 50, cmap='viridis', alpha=0.5)  # Set alpha for transparency

        ax.set_xlim(env.bounds[0][0], env.bounds[1][0])
        ax.set_ylim(env.bounds[0][1], env.bounds[1][1])

        # Plot particles and candidate points with scaled novelty scores
        if particles is not None:
            ax.scatter(particles[:, 0], particles[:, 1], c='blue', label='Particles')
        novelty_scores = (novelty_scores - np.min(novelty_scores)) / (np.max(novelty_scores) - np.min(novelty_scores)) * 9 + 1
        ax.scatter(candidate_points[:, 0], candidate_points[:, 1], c='red', label='Candidate points', s=novelty_scores*100)

        ax.legend()
        plt.show()
        
    def calculate_novelty(self, point):
        """
        Calculate the novelty of a given point based on the GMM.

        Parameters:
        - point: A 2D point to evaluate.

        Returns:
        - novelty score of the point.
        """
        distance = cdist(self.gmm.means_, [point], metric='euclidean')
        min_distance = np.min(distance)
        return min_distance

    def plot_heatmap(self, resolution=100):
        """
        Plot a heatmap of the explored areas.

        Parameters:
        - resolution: The resolution of the grid for the heatmap.
        """
        x, y = np.meshgrid(np.linspace(self.lower_bound, self.upper_bound, resolution), 
                           np.linspace(self.lower_bound, self.upper_bound, resolution))
        xy_sample = np.column_stack([x.ravel(), y.ravel()])
        z = -self.gmm.score_samples(xy_sample).reshape(x.shape)

        plt.contourf(x, y, z, levels=50, cmap='viridis')
        plt.colorbar(label='Negative Log-Likelihood')
        plt.scatter(self.samples[:, 0], self.samples[:, 1], c='red', s=2)
        plt.title("Heatmap of Explored Areas")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()
        
    def calculate_entropy(self):
        """
        Calculate the entropy of the space based on the GMM.

        Returns:
        - entropy value.
        """
        weights = self.gmm.weights_
        return entropy(weights)
    
    def plot_gmm_components(self):
        """
        Plot the distribution of each GMM component using their means and stds.
        """
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Plotting each component
        for mean, cov in zip(self.gmm.means_, self.gmm.covariances_):
            # Eigenvalues and eigenvectors for the covariance matrix
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sort the eigenvalues and eigenvectors
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

            # Angle between x-axis and the first eigenvector in degrees
            angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

            # Ellipse representing the Gaussian component
            width, height = 2 * np.sqrt(5.991 * eigenvalues)  # 5.991 corresponds to 2 stds
            ellipse = Ellipse(mean, width, height, angle, edgecolor='blue', lw=1, facecolor='none')
            ax.add_patch(ellipse)

        # Plot the sample points
        plt.scatter(self.samples[:, 0], self.samples[:, 1], c='red', s=2)
        plt.title("GMM Component Distributions")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.show()