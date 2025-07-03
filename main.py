import numpy as np
import matplotlib.pyplot as plt

from data import GaussianStream
from model import LinearRewardModel
from optimizers import OnlineOGD, OnlineOMD, OnlineMLE, ImplicitOMD


def run_experiment(dim: int = 20, T: int = 2000, seed: int = 0,
                   eta_ogd: float = 0.5, eta_omd: float = 0.5, eta_iomd: float = 0.5,
                   lam: float = 1.0, mle_lr: float = 0.1):
    rng = np.random.default_rng(seed)
    theta_star = rng.normal(size=dim)

    stream = GaussianStream(dim, theta_star, seed=seed + 1)

    model_ogd = LinearRewardModel(dim, seed=seed + 2)
    model_omd = LinearRewardModel(dim, seed=seed + 3)
    model_iomd = LinearRewardModel(dim, seed=seed + 4)
    model_mle = LinearRewardModel(dim, seed=seed + 5)

    ogd = OnlineOGD(model_ogd, eta_ogd)
    omd = OnlineOMD(model_omd, eta_omd, lam)
    iomd = ImplicitOMD(model_iomd, eta_iomd, lam)
    mle = OnlineMLE(model_mle, lr=mle_lr)

    errors = {
        'OGD': [],
        'One-Pass OMD': [],
        'Implicit OMD': [],
        'MLE': []
    }

    for _ in range(T):
        x, y = stream.sample()
        ogd.step(x, y)
        omd.step(x, y)
        iomd.step(x, y)
        mle.step(x, y)

        errors['OGD'].append(np.linalg.norm(model_ogd.theta - theta_star))
        errors['One-Pass OMD'].append(np.linalg.norm(model_omd.theta - theta_star))
        errors['Implicit OMD'].append(np.linalg.norm(model_iomd.theta - theta_star))
        errors['MLE'].append(np.linalg.norm(model_mle.theta - theta_star))

    plt.figure(figsize=(7, 4))
    for name, vals in errors.items():
        plt.plot(vals, label=name)
    plt.xlabel('Iteration')
    plt.ylabel('‖θ - θ*‖₂')
    plt.title('Parameter Error vs. Iterations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('result.png')
    return errors


if __name__ == "__main__":
    run_experiment()
