import numpy as np
import matplotlib.pyplot as plt
import time
import csv

from data import GaussianStream
from model import LinearRewardModel
from optimizers import OnlineOGD, OnlineOMD, OnlineMLE, ImplicitOMD
from timer import Timer


def run_experiment(dim: int = 20, T: int = 2000, seed: int = 0,
                   eta_ogd: float = 0.5, eta_omd: float = 0.5, eta_iomd: float = 0.5,
                   lam: float = 1.0, mle_lr: float = 0.01):
    rng = np.random.default_rng(seed)
    theta_star = rng.normal(size=dim)

    stream = GaussianStream(dim, theta_star, seed=seed + 1)

    model_ogd = LinearRewardModel(dim, seed=seed)
    model_omd = LinearRewardModel(dim, seed=seed)
    model_iomd = LinearRewardModel(dim, seed=seed)
    model_mle = LinearRewardModel(dim, seed=seed)

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
    step_times = {
        'OGD': [],
        'One-Pass OMD': [],
        'Implicit OMD': [],
        'MLE': []
    }

    wall_start = time.perf_counter()

    for _ in range(T):
        x, y = stream.sample()
        with Timer(step_times['OGD']):
            ogd.step(x, y)
        with Timer(step_times['One-Pass OMD']):
            omd.step(x, y)
        with Timer(step_times['Implicit OMD']):
            iomd.step(x, y)
        with Timer(step_times['MLE']):
            mle.step(x, y)

        errors['OGD'].append(np.linalg.norm(model_ogd.theta - theta_star))
        errors['One-Pass OMD'].append(np.linalg.norm(model_omd.theta - theta_star))
        errors['Implicit OMD'].append(np.linalg.norm(model_iomd.theta - theta_star))
        errors['MLE'].append(np.linalg.norm(model_mle.theta - theta_star))

    wall_time = time.perf_counter() - wall_start

    plt.figure(figsize=(7, 4))
    for name, vals in errors.items():
        plt.plot(vals, label=name)
    plt.xlabel('Iteration')
    plt.ylabel('‖θ - θ*‖₂')
    plt.title('Parameter Error vs. Iterations')
    plt.legend()
    plt.tight_layout()
    plt.savefig('result.pdf')

    plt.figure(figsize=(7, 4))
    for name, vals in step_times.items():
        plt.plot(vals, label=name)
    plt.xlabel('Iteration')
    plt.ylabel('Time (s)')
    plt.title('Step Time per Iteration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('timing.pdf')

    with open('timing.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Optimizer', 'TotalTime', 'AverageTime'])
        for name, vals in step_times.items():
            writer.writerow([name, sum(vals), np.mean(vals)])
        writer.writerow(['TotalWallTime', wall_time, ''])

    return errors, step_times, wall_time


if __name__ == "__main__":
    run_experiment()
