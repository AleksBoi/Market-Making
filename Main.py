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


def plot_call_quotes(t, paths, k, tau, r, sigma, vol_spread):
    """Call mid at sigma; bid priced at sigma*(1 - vol_spread), ask at sigma*(1 + vol_spread)."""
    bid_vol, ask_vol = sigma * (1 - vol_spread), sigma * (1 + vol_spread)
    mid_px = black_scholes_call(paths, k, tau, r, sigma)
    bid_px = black_scholes_call(paths, k, tau, r, bid_vol)
    ask_px = black_scholes_call(paths, k, tau, r, ask_vol)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(paths)):
        mid, = ax.plot(t, mid_px[i], linewidth=1.0, label=f"Path {i + 1} mid")
        ax.plot(t, bid_px[i], linewidth=0.8, linestyle="--", color=mid.get_color())
        ax.plot(t, ask_px[i], linewidth=0.8, linestyle="--", color=mid.get_color())
        ax.fill_between(t, bid_px[i], ask_px[i], color=mid.get_color(), alpha=0.15)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Call price")
    ax.set_title(rf"Bid at $\sigma$={bid_vol:.2f}, ask at $\sigma$={ask_vol:.2f}, mid at $\sigma$={sigma} "
                 rf"(K={k}, $\tau$={tau:.3f}y, r={r})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    S0, MU, SIGMA = 100.0, 0.05, 0.20
    T, N_STEPS, N_PATHS = 1.0, 252, 3

    t, paths = simulate_gbm(S0, MU, SIGMA, T, N_STEPS, N_PATHS, seed=42)

    # Option parameters (assumed defaults: change as needed)
    K, TAU, R = 100.0, 21 / 252, 0.03  # ATM strike, ~1 month maturity, 3% rate
    VOL_SPREAD = 0.10  # bid vol = SIGMA * 0.9 (18%), ask vol = SIGMA * 1.1 (22%)
    fig = plot_call_quotes(t, paths, K, TAU, R, SIGMA, VOL_SPREAD)
    plt.show()
