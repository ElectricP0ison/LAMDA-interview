import numpy as np
import matplotlib.pyplot as plt
from data import GaussianStream
from optimizers import OnlineOGD, OnlineOMD

def run_experiment(dim=20, T=2000, eta_ogd=0.5, eta_omd=0.5, lam=1.0, seed=0):
    rng = np.random.default_rng(seed)
    theta_star = rng.normal(scale=1.0, size=dim)

    stream = GaussianStream(dim, theta_star, seed=seed)

    ogd = OnlineOGD(dim, eta_ogd)
    omd = OnlineOMD(dim, eta_omd, lam)

    error_ogd = []
    error_omd = []

    for t in range(T):
        x, y = stream.sample()

        ogd.step(x, y)
        omd.step(x, y)

        err_ogd = np.linalg.norm(ogd.theta - theta_star)
        err_omd = np.linalg.norm(omd.theta - theta_star)

        error_ogd.append(err_ogd)
        error_omd.append(err_omd)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(error_ogd, label='OGD')
    plt.plot(error_omd, label='One-Pass OMD')
    plt.xlabel('Iteration')
    plt.ylabel('‖θ - θ*‖₂')
    plt.title('Parameter Error vs. Iterations')
    plt.legend()
    plt.tight_layout()
    return theta_star, ogd.theta, omd.theta, error_ogd, error_omd

if __name__ == "__main__":
    run_experiment()