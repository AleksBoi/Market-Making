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
    too small to leave that much room above a zero price we quote neither side
    (both NaN), so we never end up one-sided and no client can trade at a price
    that earns nothing.
    """
    mid_px = black_scholes_call(paths, k, tau, r, sigma)
    bid_px = black_scholes_call(paths, k, tau, r, sigma * (1 - vol_spread))
    ask_px = black_scholes_call(paths, k, tau, r, sigma * (1 + vol_spread))
    quoting = mid_px > min_half_spread
    bid_px = np.where(quoting, np.minimum(bid_px, mid_px - min_half_spread), np.nan)
    ask_px = np.where(quoting, np.maximum(ask_px, mid_px + min_half_spread), np.nan)
    return mid_px, bid_px, ask_px


def simulate_client_flow(mid_px, bid_px, ask_px, p0_sell, k_sell, p0_buy, k_buy,
                         gamma=0.0, seed=None):
    """Client order flow with inventory-skewed quotes, simulated forward in time.

    Distances are RELATIVE: a fraction of the mid (0.10 = quote 10% away from mid), so
    k is dimensionless and the same percentage spread gives the same fill probability
    whatever the option's price level.

    Both quotes are shifted by gamma * x_t, in PRICE units, where x_t is the inventory
    carried into step t (contracts, positive = long): down when long, up when short.
    With h the unskewed relative distance from the mid:
        bid distance = h_bid + gamma * x / mid   ((mid - bid_eff) / mid)
        ask distance = h_ask - gamma * x / mid   ((ask_eff - mid) / mid)
    Per timestep, independently:
        client sells (we buy at bid_eff) w.p. p0_sell * exp(-k_sell * bid distance)
        client buys  (we sell at ask_eff) w.p. p0_buy  * exp(-k_buy  * ask distance)
    gamma may be a scalar or an array of the same shape as mid_px (a per-step,
    per-path skew per contract, e.g. one that scales with the option's price volatility).
    The skew is kept in price units (not scaled by the mid): a skew proportional to the
    mid is huge for expensive options relative to their spread, so the quotes cross the
    mid and trade at a loss.
    Probabilities are capped at 1 and the effective bid is floored at 0. Inventory
    feeds back into the next step's quotes, hence the loop over time.
    Returns (client_sells, client_buys, bid_eff, ask_eff); gamma = 0 gives no skew.
    """
    if not (0 <= p0_sell <= 1 and 0 <= p0_buy <= 1):
        raise ValueError("p0_sell and p0_buy must each be in [0, 1]")
    if k_sell < 0 or k_buy < 0:
        raise ValueError("k_sell and k_buy must be non-negative")
    gamma = np.broadcast_to(np.asarray(gamma, dtype=float), mid_px.shape)
    rng = np.random.default_rng(seed)
    u_sell = rng.random(mid_px.shape)
    u_buy = rng.random(mid_px.shape)
    client_sells = np.zeros(mid_px.shape, dtype=bool)
    client_buys = np.zeros(mid_px.shape, dtype=bool)
    bid_eff = np.full(mid_px.shape, np.nan)
    ask_eff = np.full(mid_px.shape, np.nan)
    inventory = np.zeros(mid_px.shape[0])
    for j in range(mid_px.shape[1]):
        mid = mid_px[:, j]
        shift = gamma[:, j] * inventory
        bid_eff[:, j] = np.maximum(bid_px[:, j] - shift, 0.0)  # NaN stays NaN
        ask_eff[:, j] = ask_px[:, j] - shift
        prob_sell = np.minimum(p0_sell * np.exp(-k_sell * (mid - bid_eff[:, j]) / mid), 1.0)
        prob_buy = np.minimum(p0_buy * np.exp(-k_buy * (ask_eff[:, j] - mid) / mid), 1.0)
        client_sells[:, j] = u_sell[:, j] < prob_sell  # False where not quoting (NaN)
        client_buys[:, j] = u_buy[:, j] < prob_buy
        inventory = inventory + client_sells[:, j] - client_buys[:, j]
    return client_sells, client_buys, bid_eff, ask_eff


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
        ax.plot(t, equity[i], linewidth=1.0 if len(equity) <= 10 else 0.5,
                alpha=1.0 if len(equity) <= 10 else 0.5, label=f"Path {i + 1}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(r"Equity $V_t = C_t + q_t m_t$")
    ax.set_title(title)
    if len(equity) <= 10:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_position(t, inventory, title):
    """Signed inventory (contracts, positive = long) against time, one line per path."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(len(inventory)):
        ax.step(t, inventory[i], where="post", linewidth=1.0, label=f"Path {i + 1}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel(r"Inventory $q_t$ (contracts)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pnl_distribution(final_equity, title):
    """Histogram of final mark-to-market PnL across paths, with mean and median."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(final_equity, bins=min(40, max(10, len(final_equity) // 3)),
            color="tab:blue", alpha=0.7, edgecolor="white")
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.axvline(final_equity.mean(), color="red", linestyle="--",
               label=f"Mean {final_equity.mean():.2f}")
    ax.axvline(np.median(final_equity), color="green", linestyle=":",
               label=f"Median {np.median(final_equity):.2f}")
    ax.set_xlabel(r"Final equity $V_T$")
    ax.set_ylabel("Number of paths")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    S0, MU, SIGMA = 100.0, 0.05, 0.20
    T, N_STEPS, N_PATHS = 1.0, 252, 1000

    t, paths = simulate_gbm(S0, MU, SIGMA, T, N_STEPS, N_PATHS, seed=42)

    # Option parameters (assumed defaults: change as needed)
    K, TAU, R = 100.0, 21 / 252, 0.03  # ATM strike, ~1 month maturity, 3% rate
    VOL_SPREAD = 0.10  # bid vol = SIGMA * 0.9 (18%), ask vol = SIGMA * 1.1 (22%)
    MIN_HALF_SPREAD = 0.20  # minimum distance of bid/ask from mid, in option price units
    mid_px, bid_px, ask_px = compute_quotes(paths, K, TAU, R, SIGMA, VOL_SPREAD, MIN_HALF_SPREAD)

    # Client flow: p(delta) = p0 * exp(-k * delta), delta = distance from mid as a fraction of mid
    P0_SELL, K_SELL = 0.30, 7.0
    P0_BUY, K_BUY = 0.30, 7.0

    # Inventory skew: both quotes shift by GAMMA price units per contract of inventory.
    # 0.1 chosen from gamma_sweep.py (1000 paths): mean PnL within 1 s.e. of the best, with
    # mean |position| about a third of the unskewed run.
    USE_SKEW = True  # set False for GAMMA = 0 (no inventory skew)
    GAMMA = 0.1 if USE_SKEW else 0.0
    client_sells, client_buys, bid_px, ask_px = simulate_client_flow(
        mid_px, bid_px, ask_px, P0_SELL, K_SELL, P0_BUY, K_BUY, gamma=GAMMA, seed=7)

    title = (rf"Bid $\sigma$={SIGMA * (1 - VOL_SPREAD):.2f}, ask $\sigma$={SIGMA * (1 + VOL_SPREAD):.2f}, "
             rf"fill prob $p_0 e^{{-k\delta}}$ (K={K}, $\tau$={TAU:.3f}y, r={R})")
    #fig = plot_call_quotes(t, mid_px, bid_px, ask_px, client_sells, client_buys, title)

    cash, inventory, equity = compute_pnl(mid_px, bid_px, ask_px, client_sells, client_buys)
    #fig_pnl = plot_pnl(t, equity, "Cumulative mark-to-market PnL")
    #fig_pos = plot_position(t, inventory, "Position (signed inventory) over time")

    final = equity[:, -1]
    print(f"Final PnL over {N_PATHS} paths: mean {final.mean():.2f}, median {np.median(final):.2f}, "
          f"std {final.std(ddof=1):.2f}, min {final.min():.2f}, max {final.max():.2f}, "
          f"P(PnL>0) {(final > 0).mean():.0%}, mean/std {final.mean() / final.std(ddof=1):.2f}")
    print(f"Max |inventory| across paths: {np.abs(inventory).max()}")
    fig_dist = plot_pnl_distribution(final, f"Final PnL distribution ({N_PATHS} paths)")

    plt.show()
