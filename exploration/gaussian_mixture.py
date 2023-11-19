import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt

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

    def assess_novelty(self, points):
        """
        Assess the novelty of given points based on the current GMM.

        Parameters:
        - points: An array of points to assess.
        """
        # Evaluate the probability density of each point under each GMM component
        densities = np.array([multivariate_normal(mean=mean, cov=cov, allow_singular=True).pdf(points)
                              for mean, cov in zip(self.gmm.means_, self.gmm.covariances_)])

        # Novelty score could be the inverse of density or a more complex function
        novelty_scores = 1 / np.max(densities, axis=0)
        return novelty_scores

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
        Z = np.exp(self.gmm.score_samples(XY))
        Z = Z.reshape(X.shape)
        # Plot the contour
        plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar()
        plt.title('GMM Contour Plot')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
