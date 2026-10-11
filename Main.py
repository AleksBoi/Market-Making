"""Step 1: Geometric Brownian Motion price simulation.

dS = mu * S * dt + sigma * S * dW
Exact discretisation:
S_{t+dt} = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z),  Z ~ N(0, 1)
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_gbm(s0, mu, sigma, T, n_steps, n_paths, seed=None):
    """Simulate GBM paths. Returns (time grid, array of shape (n_paths, n_steps + 1))."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    t = np.linspace(0.0, T, n_steps + 1)
    return t, s0 * np.exp(log_paths)


def plot_gbm(t, paths, s0, mu, sigma):
    fig, ax = plt.subplots(figsize=(10, 6))
    for path in paths:
        ax.plot(t, path, linewidth=0.8, alpha=0.7)
    ax.plot(t, s0 * np.exp(mu * t), color="black", linestyle="--",
            linewidth=1.5, label=r"Expected value $S_0 e^{\mu t}$")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Price")
    ax.set_title(f"Geometric Brownian Motion ({len(paths)} paths, "rf"$\mu$={mu}, $\sigma$={sigma})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    S0, MU, SIGMA = 100.0, 0.05, 0.20
    T, N_STEPS, N_PATHS = 1.0, 252, 10

    t, paths = simulate_gbm(S0, MU, SIGMA, T, N_STEPS, N_PATHS, seed=42)
    fig = plot_gbm(t, paths, S0, MU, SIGMA)
    plt.show()
