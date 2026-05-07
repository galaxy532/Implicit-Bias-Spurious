#!/usr/bin/env python3
"""
plot_isotropic.py
=================
Plotting for Experiment A (isotropic regime).

Reads the runs produced by run_isotropic.py and generates publication-quality figures.
Supports multi-seed runs: loads all seeds per (panel, epsilon), plots mean curves
with ±1 standard deviation shaded bands.

Usage:
  python plot_isotropic.py                           # default runs_synth/
  python plot_isotropic.py --run_root ./runs_synth   # custom root
  python plot_isotropic.py --t_min 50000             # only plot tail
"""

import argparse
import os
import glob
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize


def load_panel_runs(run_root, panel):
    """
    Load all epsilon runs for a given panel.

    Supports two directory layouts:
      - Multi-seed (new):  panel/eps_X/seed_Y/logs.json
      - Single-seed (old): panel/eps_X/logs.json

    Returns: dict[eps] -> {"seeds": [data1, data2, ...], "mean_logs": {...}}
    where mean_logs has pointwise mean and std of the error trajectories.
    """
    panel_dir = os.path.join(run_root, panel)
    if not os.path.isdir(panel_dir):
        print(f"  Warning: {panel_dir} not found, skipping.")
        return {}

    runs = {}
    for d in sorted(glob.glob(os.path.join(panel_dir, "eps_*"))):
        m = re.search(r"eps_([0-9.]+)$", os.path.basename(d))
        if not m:
            continue
        eps = float(m.group(1))

        # Try multi-seed layout first
        seed_dirs = sorted(glob.glob(os.path.join(d, "seed_*")))
        seed_data = []
        for sd in seed_dirs:
            log_path = os.path.join(sd, "logs.json")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    seed_data.append(json.load(f))

        # Fallback: single-seed layout
        if not seed_data:
            log_path = os.path.join(d, "logs.json")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    seed_data.append(json.load(f))

        if not seed_data:
            continue

        # Aggregate across seeds: pointwise mean and std
        ref = seed_data[0]
        t = np.array(ref["logs"]["t"], dtype=float)
        n_seeds = len(seed_data)

        agg = {"t": t}
        for key in ["err_min", "err_maj", "err_min_rescaled", "err_maj_rescaled", "loss"]:
            arr = np.array([np.array(sd["logs"][key], dtype=float) for sd in seed_data])
            agg[f"{key}_mean"] = arr.mean(axis=0)
            agg[f"{key}_std"] = arr.std(axis=0)

        runs[eps] = {
            "seeds": seed_data,
            "agg": agg,
            "meta": ref["meta"],
            "config": ref["config"],
            "n_seeds": n_seeds,
        }

    return dict(sorted(runs.items()))


def plot_error_decay(runs, panel, out_dir, t_min=0, group="min"):
    """
    Plot E_g[1 - p_y] vs z_t on log-log scale, colored by epsilon.
    Mean line with ±1-std shaded band across seeds.  Overlay theory curve.
    """
    if not runs:
        return

    first = next(iter(runs.values()))
    alpha = first["meta"]["theory"]["alpha"]
    h = first["config"]["lr"]
    n_seeds = first["n_seeds"]

    eps_vals = np.array(sorted(runs.keys()))
    norm = Normalize(vmin=eps_vals.min(), vmax=eps_vals.max())
    cmap = get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7.5, 5))

    key = "err_min" if group == "min" else "err_maj"

    for eps in eps_vals:
        data = runs[eps]
        t = data["agg"]["t"]
        z = h * t
        mean = data["agg"][f"{key}_mean"]
        std = data["agg"][f"{key}_std"]

        mask = t >= t_min
        if not np.any(mask):
            continue
        c = cmap(norm(eps))
        ax.loglog(z[mask], np.maximum(mean[mask], 1e-15), lw=1.4, alpha=0.85, color=c)
        if n_seeds > 1:
            ax.fill_between(z[mask],
                            np.maximum(mean[mask] - std[mask], 1e-15),
                            mean[mask] + std[mask],
                            color=c, alpha=0.15)

    # Theory reference line (pick median epsilon)
    eps_ref = float(np.median(eps_vals))
    data_ref = runs[eps_ref]
    t_ref = data_ref["agg"]["t"]
    z_ref = h * t_ref
    mask = t_ref >= t_min
    z_tail = z_ref[mask]

    theory_info = data_ref["meta"]["theory"]

    if group == "min":
        if alpha < 1:
            kappa = theory_info["kappa_min"]
            theory_curve = kappa / (eps_ref * z_tail)
            label = (rf"Theory: $\kappa_{{\min}}\,/\,(\varepsilon\,z_t)$"
                     rf"  ($\kappa_{{\min}}={kappa:.3f}$, $\varepsilon={eps_ref}$)")
        else:
            theory_curve = z_tail**(-alpha) * np.log(z_tail)**(alpha - 1)
            err_ref = data_ref["agg"][f"{key}_mean"]
            C = err_ref[mask][-1] / theory_curve[-1]
            theory_curve *= C
            label = rf"Theory: $C\,z_t^{{-\alpha}}(\ln z_t)^{{\alpha-1}}$  ($\alpha={alpha:.2f}$)"
    else:
        kappa = theory_info["kappa_maj"]
        eps_g = 1 - eps_ref
        theory_curve = kappa / (eps_g * z_tail)
        label = (rf"Theory: $\kappa_{{\mathrm{{maj}}}}\,/\,((1-\varepsilon)\,z_t)$"
                 rf"  ($\kappa_{{\mathrm{{maj}}}}={kappa:.3f}$, $\varepsilon={eps_ref}$)")

    ax.loglog(z_tail, theory_curve, "k--", lw=2.2, label=label)

    group_name = "minority" if group == "min" else "majority"
    ax.set_xlabel(r"$z_t = h\,t$", fontsize=13)
    ax.set_ylabel(rf"$\mathbb{{E}}_{{\mathcal{{G}}_{{\mathrm{{{group_name}}}}}}}[1 - p_y]$", fontsize=13)
    title = rf"{group_name.capitalize()} error decay  ($\alpha = {alpha:.2f}$)"
    if n_seeds > 1:
        title += rf"  [{n_seeds} seeds, $\pm 1\sigma$ bands]"
    ax.set_title(title, fontsize=14)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\varepsilon$", fontsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fname = f"{panel}_{group}_error_decay.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_rescaled_collapse(runs, panel, out_dir, t_min=0, group="min"):
    """
    Plot the rescaled error  err_g * eps_g * z_t  vs z_t.
    If theory holds, this should converge to kappa_g (horizontal line).
    Mean line with ±1-std shaded band across seeds.
    """
    if not runs:
        return

    first = next(iter(runs.values()))
    alpha = first["meta"]["theory"]["alpha"]
    h = first["config"]["lr"]
    n_seeds = first["n_seeds"]

    if alpha >= 1 and group == "min":
        return

    eps_vals = np.array(sorted(runs.keys()))
    norm = Normalize(vmin=eps_vals.min(), vmax=eps_vals.max())
    cmap = get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(7.5, 5))

    key = "err_min" if group == "min" else "err_maj"

    for eps in eps_vals:
        data = runs[eps]
        t = data["agg"]["t"]
        z = h * t
        mean = data["agg"][f"{key}_mean"]
        std = data["agg"][f"{key}_std"]
        eps_g = eps if group == "min" else (1 - eps)
        rescaled_mean = mean * eps_g * z
        rescaled_std = std * eps_g * z

        mask = t >= t_min
        if not np.any(mask):
            continue
        c = cmap(norm(eps))
        ax.plot(z[mask], rescaled_mean[mask], lw=1.3, alpha=0.85, color=c)
        if n_seeds > 1:
            ax.fill_between(z[mask],
                            rescaled_mean[mask] - rescaled_std[mask],
                            rescaled_mean[mask] + rescaled_std[mask],
                            color=c, alpha=0.15)

    # Theory horizontal line
    theory_info = first["meta"]["theory"]
    kappa_key = "kappa_min" if group == "min" else "kappa_maj"
    kappa = theory_info.get(kappa_key)
    if kappa is not None:
        ax.axhline(kappa, color="k", ls="--", lw=2,
                   label=rf"$\kappa_{{\mathrm{{{group}}}}} = {kappa:.4f}$ (theory)")

    group_name = "minority" if group == "min" else "majority"
    ax.set_xlabel(r"$z_t = h\,t$", fontsize=13)
    ax.set_ylabel(rf"$\mathbb{{E}}_{{\mathcal{{G}}_{{\mathrm{{{group_name}}}}}}}[1 - p_y] \times \varepsilon_{{{group_name[:3]}}} \times z_t$",
                  fontsize=12)
    title = rf"Rescaled {group_name} error  ($\alpha = {alpha:.2f}$)"
    if n_seeds > 1:
        title += rf"  [{n_seeds} seeds, $\pm 1\sigma$ bands]"
    ax.set_title(title, fontsize=14)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$\varepsilon$", fontsize=12)
    ax.legend(loc="best", frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f"{panel}_{group}_rescaled.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_eps_independence(runs, panel, out_dir, t_min=0):
    """
    For alpha >= 1: overlay all minority error curves (mean ± std).
    They should collapse (eps-independent). Contrast with majority (eps-dependent).
    """
    if not runs:
        return

    first = next(iter(runs.values()))
    alpha = first["meta"]["theory"]["alpha"]
    if alpha < 1:
        return

    h = first["config"]["lr"]
    n_seeds = first["n_seeds"]
    eps_vals = np.array(sorted(runs.keys()))
    norm = Normalize(vmin=eps_vals.min(), vmax=eps_vals.max())
    cmap = get_cmap("viridis")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for eps in eps_vals:
        data = runs[eps]
        t = data["agg"]["t"]
        z = h * t
        err_min_mean = data["agg"]["err_min_mean"]
        err_min_std = data["agg"]["err_min_std"]
        err_maj_mean = data["agg"]["err_maj_mean"]
        err_maj_std = data["agg"]["err_maj_std"]
        mask = t >= t_min
        if not np.any(mask):
            continue
        c = cmap(norm(eps))
        axes[0].loglog(z[mask], np.maximum(err_min_mean[mask], 1e-15),
                       lw=1.4, alpha=0.85, color=c)
        axes[1].loglog(z[mask], np.maximum(err_maj_mean[mask], 1e-15),
                       lw=1.4, alpha=0.85, color=c)
        if n_seeds > 1:
            axes[0].fill_between(z[mask],
                                 np.maximum(err_min_mean[mask] - err_min_std[mask], 1e-15),
                                 err_min_mean[mask] + err_min_std[mask],
                                 color=c, alpha=0.12)
            axes[1].fill_between(z[mask],
                                 np.maximum(err_maj_mean[mask] - err_maj_std[mask], 1e-15),
                                 err_maj_mean[mask] + err_maj_std[mask],
                                 color=c, alpha=0.12)

    # Theory references
    eps_ref = float(np.median(eps_vals))
    data_ref = runs[eps_ref]
    t_ref = data_ref["agg"]["t"]
    z_ref = h * t_ref
    mask = t_ref >= t_min
    z_tail = z_ref[mask]

    # Minority: z_t^{-alpha} (ln z_t)^{alpha-1}
    theory_min = z_tail**(-alpha) * np.log(z_tail)**(alpha - 1)
    err_min_ref = data_ref["agg"]["err_min_mean"]
    C = err_min_ref[mask][-1] / theory_min[-1]
    axes[0].loglog(z_tail, C * theory_min, "k--", lw=2.2,
                   label=rf"$\Theta(z_t^{{-{alpha:.1f}}} (\ln z_t)^{{{alpha-1:.1f}}})$")

    # Majority: kappa_maj / ((1-eps) z_t)
    kappa_maj = data_ref["meta"]["theory"]["kappa_maj"]
    theory_maj = kappa_maj / ((1 - eps_ref) * z_tail)
    axes[1].loglog(z_tail, theory_maj, "k--", lw=2.2,
                   label=rf"$\kappa_{{\mathrm{{maj}}}}/((1-\varepsilon)\,z_t)$  ($\varepsilon={eps_ref}$)")

    seed_note = rf"  [{n_seeds} seeds]" if n_seeds > 1 else ""
    axes[0].set_title(rf"Minority error ($\alpha={alpha:.2f}$): $\varepsilon$-independent{seed_note}",
                      fontsize=13)
    axes[1].set_title(rf"Majority error ($\alpha={alpha:.2f}$): $\varepsilon$-dependent{seed_note}",
                      fontsize=13)
    for ax in axes:
        ax.set_xlabel(r"$z_t = h\,t$", fontsize=12)
        ax.set_ylabel(r"$\mathbb{E}_g[1 - p_y]$", fontsize=12)
        ax.legend(loc="upper right", frameon=True, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.02, shrink=0.85)
    cbar.set_label(r"$\varepsilon$", fontsize=12)

    fig.tight_layout()
    fname = f"{panel}_eps_independence.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)
    print(f"  Saved {fname}")


def main():
    parser = argparse.ArgumentParser(description="Plot isotropic regime experiments")
    parser.add_argument("--run_root", type=str, default="./runs_synth")
    parser.add_argument("--out_dir", type=str, default="./figures")
    parser.add_argument("--t_min", type=float, default=0,
                        help="Only plot t >= t_min (step number, not z_t)")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    for panel in ["alpha_lt_1", "alpha_ge_1"]:
        print(f"\n--- Panel: {panel} ---")
        runs = load_panel_runs(args.run_root, panel)
        if not runs:
            continue
        n_seeds = next(iter(runs.values()))["n_seeds"]
        print(f"  Loaded {len(runs)} eps values: {list(runs.keys())}  ({n_seeds} seed(s) each)")

        # Error decay plots
        for group in ["min", "maj"]:
            plot_error_decay(runs, panel, args.out_dir, t_min=args.t_min, group=group)
            plot_rescaled_collapse(runs, panel, args.out_dir, t_min=args.t_min, group=group)

        # eps-independence (only for alpha >= 1)
        plot_eps_independence(runs, panel, args.out_dir, t_min=args.t_min)

    print(f"\nAll figures saved to {args.out_dir}/")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


if __name__ == "__main__":
    main()
