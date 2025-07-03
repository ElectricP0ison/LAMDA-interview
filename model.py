import numpy as np

class LinearRewardModel:
    """Simple linear model r(x) = x^T theta."""
    def __init__(self, dim: int, theta_init: np.ndarray = None):
        self.dim = dim
        if theta_init is None:
            self.theta = np.zeros(dim)
        else:
            self.theta = theta_init.astype(float)

    def predict_proba(self, x: np.ndarray):
        z = x @ self.theta
        return 1.0 / (1.0 + np.exp(-z))

    def copy(self):
        return LinearRewardModel(self.dim, self.theta.copy())