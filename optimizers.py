import math
import numpy as np

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

class OnlineOGD:
    """Online Gradient Descent baseline for logistic loss."""
    def __init__(self, dim: int, eta: float):
        self.theta = np.zeros(dim)
        self.eta = eta

    def step(self, x: np.ndarray, y: int):
        # gradient of logistic loss
        p = _sigmoid(x @ self.theta)
        g = (p - y) * x
        self.theta -= self.eta * g
        return g

class OnlineOMD:
    """One-pass Online Mirror Descent with local second-order metric (Eq. 5)."""
    def __init__(self, dim: int, eta: float, lam: float = 1.0):
        self.theta = np.zeros(dim)
        self.eta = eta
        self.lam = lam
        # cumulative Hessian matrix (dim x dim)
        self.H_cum = lam * np.eye(dim)

    def step(self, x: np.ndarray, y: int):
        # compute gradient and Hessian for current sample at current theta
        z = x @ self.theta
        p = _sigmoid(z)
        g = (p - y) * x                                          # gradient (dim,)
        s = p * (1 - p)                                          # scalar
        H_sample = s * np.outer(x, x)                            # Hessian (dim x dim)

        # build local metric
        H_tilde = self.H_cum + self.eta * H_sample               # dim x dim

        # solve v = H_tilde^{-1} g
        v = np.linalg.solve(H_tilde, g)

        # mirror descent update
        self.theta -= self.eta * v

        # update cumulative Hessian
        self.H_cum += H_sample

        return g, H_sample

class OnlineMLE:
    """Incremental MLE with full gradient descent over all data."""
    def __init__(self, dim, lr=0.1, iter_factor=1.0):
        self.theta = np.zeros(dim)
        self.xs = []
        self.ys = []
        self.lr = lr
        self.iter_factor = iter_factor

    def _batch_grad(self):
        X = np.stack(self.xs)  # n x d
        y = np.array(self.ys)
        p = 1/(1+np.exp(-X @ self.theta))
        grad = ((p - y)[:,None] * X).mean(axis=0) * len(y)  # sum gradient
        return grad

    def step(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        t = len(self.xs)
        k = max(1, int(self.iter_factor * math.log(t+1)))
        for _ in range(k):
            g = self._batch_grad()
            self.theta -= self.lr * g

class ImplicitOMD:
    """Implicit OMD solved by few Newton steps on subproblem (4)."""
    def __init__(self, dim, eta, lam=1.0, newton_steps=3):
        self.theta = np.zeros(dim)
        self.eta = eta
        self.H_cum = lam * np.eye(dim)
        self.newton_steps = newton_steps

    def step(self, x, y):
        theta_prev = self.theta.copy()
        # initial guess
        theta = theta_prev.copy()
        for _ in range(self.newton_steps):
            u = x @ theta
            p = _sigmoid(u)
            g_sample = (p - y) * x          # gradient of loss wrt theta
            s = p*(1-p)
            H_sample = s * np.outer(x,x)    # Hessian of loss
            grad_Q = g_sample + (1/self.eta) * (self.H_cum @ (theta - theta_prev))
            Hess_Q = H_sample + (1/self.eta) * self.H_cum
            delta = np.linalg.solve(Hess_Q, grad_Q)
            theta -= delta
        # update stored theta
        self.theta = theta
        # update cumulative Hessian with final s
        p_final = _sigmoid(x @ theta)
        s_final = p_final*(1-p_final)
        self.H_cum += s_final*np.outer(x,x)
