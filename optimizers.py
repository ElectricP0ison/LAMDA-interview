import math
import numpy as np
from model import LinearRewardModel


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


class OnlineOGD:
    """Online Gradient Descent baseline for logistic loss."""

    def __init__(self, model: LinearRewardModel, eta: float):
        self.model = model
        self.eta = eta

    def step(self, x: np.ndarray, y: int):
        theta = self.model.theta
        p = _sigmoid(x @ theta)
        g = (p - y) * x
        self.model.theta = theta - self.eta * g
        return g


class OnlineOMD:
    """One-pass Online Mirror Descent with local second-order metric."""

    def __init__(self, model: LinearRewardModel, eta: float, lam: float = 1.0):
        self.model = model
        self.eta = eta
        self.H_cum = lam * np.eye(model.dim)

    def step(self, x: np.ndarray, y: int):
        theta = self.model.theta
        z = x @ theta
        p = _sigmoid(z)
        g = (p - y) * x
        s = p * (1 - p)
        H_sample = s * np.outer(x, x)
        H_tilde = self.H_cum + self.eta * H_sample
        v = np.linalg.solve(H_tilde, g)
        self.model.theta = theta - self.eta * v
        self.H_cum += H_sample
        return g, H_sample


class OnlineMLE:
    """Incremental MLE with full gradient descent over all data."""

    def __init__(self, model: LinearRewardModel, lr: float = 0.1, iter_factor: float = 1.0):
        self.model = model
        self.xs: list[np.ndarray] = []
        self.ys: list[int] = []
        self.lr = lr
        self.iter_factor = iter_factor

    def _batch_grad(self) -> np.ndarray:
        X = np.stack(self.xs)
        y = np.array(self.ys)
        p = _sigmoid(X @ self.model.theta)
        grad = ((p - y)[:, None] * X).mean(axis=0) * len(y)
        return grad

    def step(self, x: np.ndarray, y: int):
        self.xs.append(x)
        self.ys.append(y)
        t = len(self.xs)
        k = max(1, int(self.iter_factor * math.log(t + 1)))
        for _ in range(k):
            g = self._batch_grad()
            self.model.theta -= self.lr * g


class ImplicitOMD:
    """Implicit OMD solved by few Newton steps on subproblem (4)."""

    def __init__(self, model: LinearRewardModel, eta: float, lam: float = 1.0, newton_steps: int = 3):
        self.model = model
        self.eta = eta
        self.H_cum = lam * np.eye(model.dim)
        self.newton_steps = newton_steps

    def step(self, x: np.ndarray, y: int):
        theta_prev = self.model.theta
        theta = theta_prev.copy()
        for _ in range(self.newton_steps):
            u = x @ theta
            p = _sigmoid(u)
            g_sample = (p - y) * x
            s = p * (1 - p)
            H_sample = s * np.outer(x, x)
            grad_Q = g_sample + (1 / self.eta) * (self.H_cum @ (theta - theta_prev))
            Hess_Q = H_sample + (1 / self.eta) * self.H_cum
            delta = np.linalg.solve(Hess_Q, grad_Q)
            theta -= delta
        self.model.theta = theta
        p_final = _sigmoid(x @ theta)
        s_final = p_final * (1 - p_final)
        self.H_cum += s_final * np.outer(x, x)
