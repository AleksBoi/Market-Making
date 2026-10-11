"""Step 1: Geometric Brownian Motion price simulation.

dS = mu * S * dt + sigma * S * dW
Exact discretisation:
S_{t+dt} = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z),  Z ~ N(0, 1)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


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


def black_scholes_call(s, k, tau, r, sigma):
    """Black-Scholes European call price (no dividends).

    s may be an array (e.g. a whole simulated path). tau is the time to maturity
    in years and is held constant, i.e. at every timestep we price a fresh option
    with the same remaining life (a rolling constant-maturity option).
    """
    s = np.asarray(s, dtype=float)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return s * norm.cdf(d1) - k * np.exp(-r * tau) * norm.cdf(d2)


def plot_spot_and_call(t, paths, k, tau, r, sigma):
    """Spot paths (top) and constant-maturity call price (bottom) on one figure."""
    call = black_scholes_call(paths, k, tau, r, sigma)
    fig, (ax_spot, ax_call) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i in range(len(paths)):
        line, = ax_spot.plot(t, paths[i], linewidth=0.8, alpha=0.8)
        ax_call.plot(t, call[i], linewidth=0.8, alpha=0.8, color=line.get_color())
    ax_spot.axhline(k, color="black", linestyle="--", linewidth=1, label=f"Strike K={k}")
    ax_spot.set_ylabel("Spot price")
    ax_spot.legend()
    ax_spot.set_title(rf"Spot and Black-Scholes call, constant maturity $\tau$={tau:.3f}y "
                      rf"(r={r}, $\sigma$={sigma})")
    ax_call.set_ylabel("Call price")
    ax_call.set_xlabel("Time (years)")
    for ax in (ax_spot, ax_call):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    S0, MU, SIGMA = 100.0, 0.05, 0.20
    T, N_STEPS, N_PATHS = 1.0, 252, 10

    t, paths = simulate_gbm(S0, MU, SIGMA, T, N_STEPS, N_PATHS, seed=42)

    # Option parameters (assumed defaults: change as needed)
    K, TAU, R = 100.0, 21 / 252, 0.03  # ATM strike, ~1 month maturity, 3% rate
    fig = plot_spot_and_call(t, paths, K, TAU, R, SIGMA)
    plt.show()
