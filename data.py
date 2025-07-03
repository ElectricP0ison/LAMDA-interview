import numpy as np

class GaussianStream:
    """Generate an endless stream of (x, y) pairs from a logistic model."""
    def __init__(self, dim: int, theta_star: np.ndarray, seed: int = 42):
        self.dim = dim
        self.theta_star = theta_star
        self.rng = np.random.default_rng(seed)

    def sample(self):
        x = self.rng.normal(size=self.dim)
        p = 1.0 / (1.0 + np.exp(-x @ self.theta_star))
        y = self.rng.binomial(1, p)
        return x, y
