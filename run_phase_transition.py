#!/usr/bin/env python3
"""
run_phase_transition.py
=======================
Experiment C: Phase transition sweep.

Fix epsilon and sweep gamma_min to vary alpha across the transition at alpha=1.
For each alpha, run GD and measure the empirical error exponent of the minority.
Plot measured exponent vs theoretical alpha.

Usage:
  python run_phase_transition.py                # full run
  python run_phase_transition.py --quick        # quick sanity check
  python run_phase_transition.py --plot_only    # just re-plot from existing runs
"""

import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth_utils import (
    SynthConfig, build_isotropic_operators, generate_dataset,
    train_population_gd, kappa_theory, save_logs, ensure_dir,
)


def measure_exponent(logs, h, t_min_frac=0.5):
    """
    Estimate the empirical decay exponent of the minority error.
    Fit log(err_min) = -beta * log(z_t) + C  on the tail (z_t >= t_min_frac * z_max).
    Returns beta (the empirical exponent).
    """
    t = np.array(logs["t"], dtype=float)
    err = np.array(logs["err_min"], dtype=float)
    z = h * t

    # Filter tail
    z_max = z[-1]
    mask = (z >= t_min_frac * z_max) & (err > 1e-14)
    if mask.sum() < 10:
        return float('nan')

    log_z = np.log(z[mask])
    log_err = np.log(err[mask])

    # Linear regression:  log_err = -beta * log_z + C
    # We also account for the (ln z)^{alpha-1} correction by fitting
    # log_err = -beta * log_z + (alpha-1) * log(log_z) + C
    # But for simplicity, just fit the slope (the log-log correction is small)
    coeffs = np.polyfit(log_z, log_err, 1)
    beta = -coeffs[0]
    return beta


def run_sweep(gamma_mins, cfg_base, out_root, seeds):
    """Run one sweep over gamma_min values, repeating each over multiple seeds."""
    results = []

    for gm in gamma_mins:
        betas_seeds = []
        for seed in seeds:
            cfg = SynthConfig()
            cfg.d_r = cfg_base.d_r
            cfg.d_s = cfg_base.d_s
            cfg.mu_A = cfg_base.mu_A
            cfg.mu_B = cfg_base.mu_B
            cfg.mu = cfg_base.mu
            cfg.epsilon = cfg_base.epsilon
            cfg.N = cfg_base.N
            cfg.lr = cfg_base.lr
            cfg.steps = cfg_base.steps
            cfg.log_every = cfg_base.log_every
            cfg.print_every = cfg_base.print_every
            cfg.seed = seed
            cfg.gamma_min = gm
            cfg.gamma_maj = 1.0
            cfg.__post_init__()
            cfg.out_dir = os.path.join(out_root, f"gm_{gm:.3f}", f"seed_{seed}")

            print(f"\n  gamma_min={gm:.3f}  =>  alpha={cfg.alpha:.3f}  seed={seed}")

            A, B, v = build_isotropic_operators(cfg)
            X, groups, meta = generate_dataset(cfg, A, B, v)
            meta["theory"] = kappa_theory(cfg)
            meta["seed"] = seed

            logs, w = train_population_gd(cfg, X, groups)
            save_logs(logs, meta, cfg)

            beta = measure_exponent(logs, cfg.lr)
            betas_seeds.append(beta)
            print(f"    Empirical exponent: {beta:.3f}  (theory alpha={cfg.alpha:.3f})")

        beta_mean = float(np.nanmean(betas_seeds))
        beta_std = float(np.nanstd(betas_seeds))
        print(f"  => gamma_min={gm:.3f}  beta={beta_mean:.3f} +/- {beta_std:.3f}")

        results.append({
            "gamma_min": gm,
            "alpha_theory": cfg.alpha,
            "beta_mean": beta_mean,
            "beta_std": beta_std,
            "beta_per_seed": betas_seeds,
            "seeds": seeds,
            "mu_A": cfg.mu_A, "mu_B": cfg.mu_B, "mu": cfg.mu,
            "epsilon": cfg.epsilon,
        })

    return results


def plot_phase_transition(results, out_dir):
    """Plot empirical exponent vs theoretical alpha with error bars."""
    ensure_dir(out_dir)

    alphas = np.array([r["alpha_theory"] for r in results])
    # Support both old (single beta) and new (mean/std) formats
    if "beta_mean" in results[0]:
        betas = np.array([r["beta_mean"] for r in results])
        beta_stds = np.array([r["beta_std"] for r in results])
    else:
        betas = np.array([r["beta_empirical"] for r in results])
        beta_stds = np.zeros_like(betas)

    n_seeds = len(results[0].get("seeds", [1]))

    fig, ax = plt.subplots(figsize=(7, 5))

    # Theory: exponent = min(alpha, 1) for the minority
    # When alpha < 1: error ~ 1/z_t, so exponent = 1
    # When alpha >= 1: error ~ z_t^{-alpha}, so exponent = alpha
    alpha_grid = np.linspace(alphas.min() - 0.1, alphas.max() + 0.1, 200)
    theory_exponent = np.where(alpha_grid < 1, 1.0, alpha_grid)

    ax.plot(alpha_grid, theory_exponent, "k-", lw=2.5, label="Theory: $\\min(\\alpha, 1)$ / $\\alpha$",
            zorder=5)
    ax.errorbar(alphas, betas, yerr=beta_stds, fmt="o", ms=7, color="tab:blue",
                ecolor="tab:blue", elinewidth=1.5, capsize=3, capthick=1.2,
                markeredgecolor="k", markeredgewidth=0.8,
                label=rf"Empirical exponent (mean $\pm$ 1 std, {n_seeds} seeds)", zorder=10)

    # Mark the transition
    ax.axvline(1.0, color="red", ls=":", lw=1.5, alpha=0.7, label=r"$\alpha = 1$ (transition)")

    ax.set_xlabel(r"Theoretical $\alpha = \tilde{\gamma}_{\min}(1+\mu)/(1+\mu_A)$", fontsize=13)
    ax.set_ylabel(r"Empirical decay exponent $\beta$", fontsize=13)
    ax.set_title("Phase transition in minority error decay rate", fontsize=14)
    ax.legend(fontsize=11, frameon=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = "phase_transition.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)
    print(f"\n  Saved {fname}")

    # Also save results as JSON
    with open(os.path.join(out_dir, "phase_transition_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Phase transition sweep")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip training, just re-plot from existing results")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="List of random seeds for error bars (default: [1, 2, 3])")
    parser.add_argument("--out_root", type=str, default="./runs_synth/phase_transition")
    parser.add_argument("--fig_dir", type=str, default="./figures")
    args = parser.parse_args()

    if args.plot_only:
        results_path = os.path.join(args.fig_dir, "phase_transition_results.json")
        if not os.path.exists(results_path):
            print(f"No results file at {results_path}. Run without --plot_only first.")
            return
        with open(results_path) as f:
            results = json.load(f)
        plot_phase_transition(results, args.fig_dir)
        return

    # Configure base
    cfg_base = SynthConfig()
    cfg_base.epsilon = args.epsilon
    cfg_base.mu_A = 1.0
    cfg_base.mu_B = 1.0
    cfg_base.mu = 0.0  # alpha = gamma_min * 1.0 / 2.0 = gamma_min / 2

    if args.quick:
        cfg_base.steps = args.steps or 20_000
        cfg_base.N = args.N or 10_000
        cfg_base.log_every = 200
        cfg_base.print_every = 5_000
        gamma_mins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        seeds = args.seeds or [1, 2, 3]
    else:
        cfg_base.steps = args.steps or 200_000
        cfg_base.N = args.N or 50_000
        cfg_base.log_every = 500
        cfg_base.print_every = 50_000
        gamma_mins = np.arange(0.5, 4.1, 0.4).tolist()  # 10 values
        seeds = args.seeds or [1, 2, 3]

    print(f"Phase transition sweep: gamma_min in {gamma_mins}")
    print(f"  => alpha in {[gm * (1+cfg_base.mu)/(1+cfg_base.mu_A) for gm in gamma_mins]}")
    print(f"  epsilon={cfg_base.epsilon}, steps={cfg_base.steps}, N={cfg_base.N}, seeds={seeds}")

    results = run_sweep(gamma_mins, cfg_base, args.out_root, seeds)
    plot_phase_transition(results, args.fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
