"""Sweep the inventory-skew strength for two skew models.

  constant:  shift per contract = gamma                       (price units)
  vol:       shift per contract = gamma * (Delta_t S_t sigma)^2 * TAU
             i.e. proportional to the variance of the option's price over the holding
             horizon TAU (Avellaneda-Stoikov reservation-price form). Delta_t S_t sigma is
             the option price's volatility per sqrt(year), so the skew is large for
             options that carry a lot of risk (ATM / ITM) and ~0 for far OTM ones.

Every GAMMA is run on the same GBM paths and the same client-flow random numbers
(common random numbers), so differences between rows come from GAMMA alone.
Objectives: high mean final PnL and small |position|.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from Main import simulate_gbm, compute_quotes, simulate_client_flow, compute_pnl

# Same market and strategy settings as Main.py
S0, MU, SIGMA = 100.0, 0.05, 0.20
T, N_STEPS, N_PATHS = 1.0, 252, 1000
K, TAU, R = 100.0, 21 / 252, 0.03
VOL_SPREAD, MIN_HALF_SPREAD = 0.10, 0.20
P0, K_FILL = 0.30, 7.0
PATH_SEED, FLOW_SEED = 42, 7

GAMMAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.167, 0.25, 0.4, 0.6, 1.0, 1.5, 2.0]
GAMMAS_VOL = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def option_variance_scale(paths):
    """(Delta_t * S_t * sigma)^2 * TAU: variance of the option price over the horizon TAU."""
    d1 = (np.log(paths / K) + (R + 0.5 * SIGMA**2) * TAU) / (SIGMA * np.sqrt(TAU))
    return (norm.cdf(d1) * paths * SIGMA) ** 2 * TAU


def run_gamma(gamma, mid_px, bid_px, ask_px, model="constant", var_scale=None):
    skew = gamma if model == "constant" else gamma * var_scale
    sells, buys, bid_eff, ask_eff = simulate_client_flow(
        mid_px, bid_px, ask_px, P0, K_FILL, P0, K_FILL, gamma=skew, seed=FLOW_SEED)
    _, inventory, equity = compute_pnl(mid_px, bid_eff, ask_eff, sells, buys)
    final = equity[:, -1]
    abs_pos = np.abs(inventory)
    return {
        "model": model,
        "gamma": gamma,
        "mean_pnl": final.mean(),
        "se_pnl": final.std(ddof=1) / np.sqrt(len(final)),
        "std_pnl": final.std(ddof=1),
        "mean_abs_pos": abs_pos.mean(),          # time-and-path average |q_t|
        "mean_abs_final_pos": abs_pos[:, -1].mean(),
        "max_abs_pos": abs_pos.max(),
        "p_positive": (final > 0).mean(),
        "trades": int(sells.sum() + buys.sum()),
    }


def pareto_front(rows):
    """Rows not dominated on (higher mean_pnl, lower mean_abs_pos)."""
    front = []
    for a in rows:
        dominated = any(
            b["mean_pnl"] >= a["mean_pnl"] and b["mean_abs_pos"] <= a["mean_abs_pos"]
            and (b["mean_pnl"] > a["mean_pnl"] or b["mean_abs_pos"] < a["mean_abs_pos"])
            for b in rows)
        if not dominated:
            front.append(a)
    return front


def plot_sweep(rows, front):
    """Left/middle: mean PnL and mean |position| vs gamma per model. Right: PnL vs |position|."""
    colors = {"constant": "tab:blue", "vol": "tab:orange"}
    labels = {"constant": r"constant: $\gamma$", "vol": r"vol: $\gamma\,(\Delta S\sigma)^2\tau$"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for model in ("constant", "vol"):
        sub = sorted((r for r in rows if r["model"] == model), key=lambda r: r["gamma"])
        g = np.array([max(r["gamma"], 1e-4) for r in sub])  # plot gamma = 0 at the left edge
        pnl = np.array([r["mean_pnl"] for r in sub])
        se = np.array([r["se_pnl"] for r in sub])
        pos = np.array([r["mean_abs_pos"] for r in sub])
        axes[0].errorbar(g, pnl, yerr=se, marker="o", capsize=3, color=colors[model], label=labels[model])
        axes[1].plot(g, pos, marker="o", color=colors[model], label=labels[model])
        axes[2].plot(pos, pnl, marker="o", color=colors[model], alpha=0.5, label=labels[model])
    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_xlabel(r"$\gamma$ (gamma = 0 drawn at $10^{-4}$)")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Mean final PnL (± 1 s.e.)")
    axes[1].set_ylabel("Mean |position| (contracts)")
    fp = sorted(front, key=lambda r: r["mean_abs_pos"])
    axes[2].plot([r["mean_abs_pos"] for r in fp], [r["mean_pnl"] for r in fp],
                 color="tab:green", linestyle="--", linewidth=2, label="Pareto front (both models)")
    axes[2].set_xlabel("Mean |position| (contracts)")
    axes[2].set_ylabel("Mean final PnL")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    fig.suptitle(f"Inventory skew sweep ({N_PATHS} paths)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    t, paths = simulate_gbm(S0, MU, SIGMA, T, N_STEPS, N_PATHS, seed=PATH_SEED)
    mid_px, bid_px, ask_px = compute_quotes(paths, K, TAU, R, SIGMA, VOL_SPREAD, MIN_HALF_SPREAD)
    var_scale = option_variance_scale(paths)
    print(f"Option variance scale (Delta S sigma)^2 tau: mean {var_scale.mean():.2f}, "
          f"median {np.median(var_scale):.2f}, 5-95% {np.percentile(var_scale, 5):.2f}"
          f"-{np.percentile(var_scale, 95):.2f}")

    rows = [run_gamma(g, mid_px, bid_px, ask_px) for g in GAMMAS]
    rows += [run_gamma(g, mid_px, bid_px, ask_px, model="vol", var_scale=var_scale) for g in GAMMAS_VOL]
    front = pareto_front(rows)

    print(f"{'model':>9} {'gamma':>7} {'mean PnL':>9} {'s.e.':>6} {'std':>7} {'mean|q|':>8} "
          f"{'mean|q_T|':>9} {'max|q|':>7} {'P(>0)':>6} {'trades':>7}  pareto")
    for r in rows:
        print(f"{r['model']:>9} {r['gamma']:7g} {r['mean_pnl']:9.2f} {r['se_pnl']:6.2f} {r['std_pnl']:7.2f} "
              f"{r['mean_abs_pos']:8.2f} {r['mean_abs_final_pos']:9.2f} {r['max_abs_pos']:7d} "
              f"{r['p_positive']:6.0%} {r['trades']:7d}  {'*' if r in front else ''}")

    fig = plot_sweep(rows, front)
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/gamma_sweep.pdf")
    plt.show()
