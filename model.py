import numpy as np

class LinearRewardModel:
    """Simple linear model r(x) = x^T theta."""
    def __init__(self, dim: int, theta_init: np.ndarray | None = None, seed: int | None = None):
        self.dim = dim
        rng = np.random.default_rng(seed)
        if theta_init is None:
            self.theta = rng.normal(size=dim)
        else:
            self.theta = theta_init.astype(float)

    def predict_proba(self, x: np.ndarray) -> float:
        z = x @ self.theta
        return 1.0 / (1.0 + np.exp(-z))

    def score(self, x: np.ndarray) -> float:
        return x @ self.theta

    def copy(self) -> "LinearRewardModel":
        return LinearRewardModel(self.dim, self.theta.copy())
