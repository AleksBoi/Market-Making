"""Step 1: Geometric Brownian Motion price simulation.

dS = mu * S * dt + sigma * S * dW
Exact discretisation:
S_{t+dt} = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z),  Z ~ N(0, 1)
"""

import os

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


def compute_quotes(paths, k, tau, r, sigma, vol_spread, min_half_spread=0.0):
    """Model mid at sigma; bid priced at sigma*(1 - vol_spread), ask at sigma*(1 + vol_spread).

    The vol-based spread collapses towards 0 for far OTM options, so each side is
    kept at least min_half_spread (price units) away from the mid. When the mid is
    too small to leave that much room above a zero price, there is no bid (NaN),
    so no client can sell to us at a price that earns nothing.
    """
    mid_px = black_scholes_call(paths, k, tau, r, sigma)
    bid_px = black_scholes_call(paths, k, tau, r, sigma * (1 - vol_spread))
    ask_px = black_scholes_call(paths, k, tau, r, sigma * (1 + vol_spread))
    bid_px = np.where(mid_px > min_half_spread,
                      np.minimum(bid_px, mid_px - min_half_spread), np.nan)
    ask_px = np.maximum(ask_px, mid_px + min_half_spread)
    return mid_px, bid_px, ask_px


def simulate_client_flow(mid_px, bid_px, ask_px, p0_sell, k_sell, p0_buy, k_buy, seed=None):
    """Random client order flow whose arrival probability decays with quote distance.

    Per timestep, independently:
      client sells (we buy at our bid) with prob p0_sell * exp(-k_sell * (mid - bid))
      client buys  (we sell at our ask) with prob p0_buy  * exp(-k_buy  * (ask - mid))
    Distances are in option price units, so k has units of 1/price. A tighter quote
    (smaller delta) means a higher fill probability, and p0 is the prob at delta = 0.
    Returns boolean masks (client_sells, client_buys).
    """
    if not (0 <= p0_sell <= 1 and 0 <= p0_buy <= 1):
        raise ValueError("p0_sell and p0_buy must each be in [0, 1]")
    if k_sell < 0 or k_buy < 0:
        raise ValueError("k_sell and k_buy must be non-negative")
    rng = np.random.default_rng(seed)
    prob_sell = p0_sell * np.exp(-k_sell * (mid_px - bid_px))
    prob_buy = p0_buy * np.exp(-k_buy * (ask_px - mid_px))
    client_sells = rng.random(mid_px.shape) < prob_sell
    client_buys = rng.random(mid_px.shape) < prob_buy
    return client_sells, client_buys


def plot_call_quotes(t, mid_px, bid_px, ask_px, client_sells, client_buys, title):
    """Quotes with trades marked: client sells hit our bid, client buys lift our ask."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(mid_px)):
        mid, = ax.plot(t, mid_px[i], linewidth=1.0, label=f"Path {i + 1} mid")
        ax.plot(t, bid_px[i], linewidth=0.8, linestyle="--", color=mid.get_color())
        ax.plot(t, ask_px[i], linewidth=0.8, linestyle="--", color=mid.get_color())
        ax.fill_between(t, bid_px[i], ask_px[i], color=mid.get_color(), alpha=0.15)
        ax.scatter(t[client_sells[i]], bid_px[i][client_sells[i]], marker="v", s=22,
                   color="red", zorder=3, label="Client sells (we buy at bid)" if i == 0 else None)
        ax.scatter(t[client_buys[i]], ask_px[i][client_buys[i]], marker="^", s=22,
                   color="green", zorder=3, label="Client buys (we sell at ask)" if i == 0 else None)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Call price")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def compute_pnl(mid_px, bid_px, ask_px, client_sells, client_buys):
    """Cash, signed inventory and mark-to-market equity at every timestep.

    Client sells: we buy one contract at the bid (cash down, inventory up).
    Client buys:  we sell one contract at the ask (cash up, inventory down).
        C_t = C_{t-1} - b_t s_t + a_t u_t
        q_t = q_{t-1} + s_t - u_t
        V_t = C_t + q_t m_t
    Starts from C = 0, q = 0 before the first step. Bid is NaN where we do not
    quote, but no sell can fill there, so those entries are masked out.
    """
    cash = np.cumsum(np.where(client_sells, -bid_px, 0.0)
                     + np.where(client_buys, ask_px, 0.0), axis=1)
    inventory = np.cumsum(client_sells.astype(int) - client_buys.astype(int), axis=1)
    equity = cash + inventory * mid_px
    return cash, inventory, equity


def plot_pnl(t, equity, title):
    """Cumulative mark-to-market PnL against time, one line per path."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(equity)):
        ax.plot(t, equity[i], linewidth=1.0, label=f"Path {i + 1}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(r"Equity $V_t = C_t + q_t m_t$")
    ax.set_title(title)
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
    MIN_HALF_SPREAD = 0.20  # minimum distance of bid/ask from mid, in option price units
    mid_px, bid_px, ask_px = compute_quotes(paths, K, TAU, R, SIGMA, VOL_SPREAD, MIN_HALF_SPREAD)

    # Client flow: p(delta) = p0 * exp(-k * delta), delta = distance from mid in price units
    P0_SELL, K_SELL = 0.30, 3.0
    P0_BUY, K_BUY = 0.30, 3.0
    client_sells, client_buys = simulate_client_flow(
        mid_px, bid_px, ask_px, P0_SELL, K_SELL, P0_BUY, K_BUY, seed=7)

    title = (rf"Bid $\sigma$={SIGMA * (1 - VOL_SPREAD):.2f}, ask $\sigma$={SIGMA * (1 + VOL_SPREAD):.2f}, "
             rf"fill prob $p_0 e^{{-k\delta}}$ (K={K}, $\tau$={TAU:.3f}y, r={R})")
    fig = plot_call_quotes(t, mid_px, bid_px, ask_px, client_sells, client_buys, title)

    cash, inventory, equity = compute_pnl(mid_px, bid_px, ask_px, client_sells, client_buys)
    fig_pnl = plot_pnl(t, equity, "Cumulative mark-to-market PnL")

    plt.show()
