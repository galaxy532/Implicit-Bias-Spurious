#!/usr/bin/env python3
"""
replot_tabular_vendor.py
========================
Re-generate all figures from saved logs.json files, cropped to z_t <= T_max.
No experiment is re-run; this script only reads and plots.

Usage
-----
  # Crop all plots at z_t = 20000:
  python replot_tabular_vendor.py --T_max 20000

  # Custom input/output dirs:
  python replot_tabular_vendor.py --T_max 15000 \\
      --in_dir ./tabular_vendor_results \\
      --out_dir ./tabular_vendor_results_crop

  # Select specific alphas:
  python replot_tabular_vendor.py --T_max 20000 --alphas 0.5 1.0 2.0
"""

import argparse
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from synth_utils import SynthConfig, kappa_theory, load_logs, ensure_dir


# ============================================================
#  Plotting helpers
# ============================================================

def smooth(arr, window=51):
    """Simple moving average for noisy curves."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def prepare_data(logs, cfg, z_max):
    """Load, filter, and crop trajectory data."""
    t = np.array(logs["t"], dtype=np.float64)
    err_maj = np.array(logs["err_maj"])
    err_min = np.array(logs["err_min"])
    loss = np.array(logs["loss"]) if "loss" in logs else None
    z_t = cfg.lr * t

    ERR_FLOOR = 1e-9  # lowered: N=5M gives floors ~1e-6, so 1e-9 is safe
    mask = (err_maj > ERR_FLOOR) & (err_min > ERR_FLOOR) & (z_t > 1.0)
    if z_max is not None:
        mask &= (z_t <= z_max)

    idx = np.where(mask)[0]
    return {
        "t": t[idx], "z_t": z_t[idx],
        "err_maj": err_maj[idx], "err_min": err_min[idx],
        "loss": loss[idx] if loss is not None else None,
    }


# ============================================================
#  Per-alpha plots
# ============================================================

def plot_error_ratio(d, cfg, theory, out_dir):
    """Plot 1: err_min / err_maj  (converges for alpha<1, diverges for alpha>=1)."""
    z_t, err_maj, err_min = d["z_t"], d["err_maj"], d["err_min"]
    alpha = cfg.alpha

    fig, ax = plt.subplots(figsize=(8, 5))
    ratio = err_min / err_maj
    ax.plot(z_t, ratio, "k-", lw=1.5)
    ax.set_xlabel(r"$z_t = h \cdot t$")
    ax.set_ylabel(r"$\mathrm{err}_{\min} \,/\, \mathrm{err}_{\mathrm{maj}}$")
    ax.set_xscale("log")

    if alpha < 1.0:
        if theory.get("kappa_min") is not None:
            expected = (theory["kappa_min"] * (1 - cfg.epsilon)) / \
                       (theory["kappa_maj"] * cfg.epsilon)
            ax.axhline(expected, color="green", ls="--", lw=1.5,
                       label=f"Theory: {expected:.2f}")
            ax.legend()
        # y-axis: from 0 to 30% above the theory line (or data max)
        y_top = max(ratio.max(), expected if theory.get("kappa_min") else ratio.max()) * 1.3
        ax.set_ylim([0, y_top])
        ax.set_title(f"Error ratio ($\\alpha = {alpha:.2f} < 1$: should converge)")
    else:
        ax.set_yscale("log")
        ax.set_ylabel(
            r"$\mathrm{err}_{\min} \,/\, \mathrm{err}_{\mathrm{maj}}$ (log)")
        ax.set_title(
            f"Error ratio ($\\alpha = {alpha:.2f} \\geq 1$: should diverge)")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_ratio.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "error_ratio.pdf"))
    plt.close(fig)


def plot_compensated(d, cfg, theory, out_dir):
    """Plot 2: compensated errors  err * z_t^beta."""
    z_t, err_maj, err_min = d["z_t"], d["err_maj"], d["err_min"]
    alpha = cfg.alpha

    if alpha > 1.0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: err * z_t
        axes[0].plot(z_t, err_maj * z_t, "b-", lw=1.5, label=r"Majority $\cdot\, z_t$")
        axes[0].plot(z_t, err_min * z_t, "r-", lw=1.5, label=r"Minority $\cdot\, z_t$")
        if theory.get("kappa_maj") is not None:
            axes[0].axhline(theory["kappa_maj"] / (1 - cfg.epsilon), color="blue",
                            ls="--", alpha=0.6,
                            label=f"Theory: $\\kappa_{{\\mathrm{{maj}}}}/(1-\\varepsilon)"
                                  f" = {theory['kappa_maj']/(1-cfg.epsilon):.3f}$")
        axes[0].set_xlabel(r"$z_t$")
        axes[0].set_ylabel(r"$\mathrm{err} \cdot z_t$")
        axes[0].set_title("Compensated by $z_t$\n(plateau $\\Leftrightarrow$ $1/z_t$ decay)")
        axes[0].set_xscale("log")
        # Zoom out: y from 0 to 2x the theory line
        y_top_left = (theory["kappa_maj"] / (1 - cfg.epsilon)) * 2.0 if theory.get("kappa_maj") else None
        if y_top_left:
            axes[0].set_ylim([0, y_top_left])
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Right: err_min * z_t^alpha only
        comp_min_alpha = err_min * z_t**alpha
        axes[1].plot(z_t, comp_min_alpha, "r-", lw=1.5,
                     label=f"Minority $\\cdot\\, z_t^{{{alpha:.2f}}}$")
        axes[1].set_xlabel(r"$z_t$")
        axes[1].set_ylabel(
            f"$\\mathrm{{err}}_{{\\min}} \\cdot z_t^{{{alpha:.2f}}}$")
        axes[1].set_title(
            f"Minority compensated by $z_t^{{\\alpha}}$\n"
            f"(plateau $\\Leftrightarrow$ $z_t^{{-\\alpha}}$ decay)")
        axes[1].set_xscale("log")
        # Zoom out: 0 to 2x the data max
        axes[1].set_ylim([0, comp_min_alpha.max() * 2.0])
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Compensated errors  ($\\alpha = {alpha:.2f}$)", fontsize=13)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(z_t, err_maj * z_t, "b-", lw=1.5, label=r"Majority $\cdot\, z_t$")
        ax.plot(z_t, err_min * z_t, "r-", lw=1.5, label=r"Minority $\cdot\, z_t$")
        if theory.get("kappa_maj") is not None:
            ax.axhline(theory["kappa_maj"] / (1 - cfg.epsilon), color="blue",
                       ls="--", alpha=0.6,
                       label=f"Theory: $\\kappa_{{\\mathrm{{maj}}}}/(1-\\varepsilon)"
                             f" = {theory['kappa_maj']/(1-cfg.epsilon):.3f}$")
        if theory.get("kappa_min") is not None:
            ax.axhline(theory["kappa_min"] / cfg.epsilon, color="red",
                       ls="--", alpha=0.6,
                       label=f"Theory: $\\kappa_{{\\min}}/\\varepsilon"
                             f" = {theory['kappa_min']/cfg.epsilon:.3f}$")
        ax.set_xlabel(r"$z_t$")
        ax.set_ylabel(r"$\mathrm{err} \cdot z_t$")
        ax.set_xscale("log")
        # Zoom out: y from 0 to 2x the higher theory line
        theory_lines = []
        if theory.get("kappa_maj") is not None:
            theory_lines.append(theory["kappa_maj"] / (1 - cfg.epsilon))
        if theory.get("kappa_min") is not None:
            theory_lines.append(theory["kappa_min"] / cfg.epsilon)
        if theory_lines:
            ax.set_ylim([0, max(theory_lines) * 2.0])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if alpha < 1.0:
            ax.set_title(
                f"Compensated errors ($\\alpha = {alpha:.2f} < 1$: both groups plateau)")
        else:
            ax.set_title(
                f"Compensated errors ($\\alpha = 1.0$: majority plateaus, "
                f"minority $\\to 0$ with log correction)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compensated.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "compensated.pdf"))
    plt.close(fig)


def plot_rescaled(d, cfg, theory, out_dir):
    """Plot 3: rescaled errors  eps_g * err_g * z_t  (shows kappa_g prefactors)."""
    z_t, err_maj, err_min = d["z_t"], d["err_maj"], d["err_min"]
    alpha = cfg.alpha

    fig, ax = plt.subplots(figsize=(8, 5))
    rescaled_maj = (1 - cfg.epsilon) * err_maj * z_t
    rescaled_min = cfg.epsilon * err_min * z_t

    ax.plot(z_t, rescaled_maj, "b-", lw=1.5,
            label=r"$(1-\varepsilon)\cdot\mathrm{err}_{\mathrm{maj}}\cdot z_t$")
    ax.plot(z_t, rescaled_min, "r-", lw=1.5,
            label=r"$\varepsilon\cdot\mathrm{err}_{\min}\cdot z_t$")

    if theory.get("kappa_maj") is not None:
        ax.axhline(theory["kappa_maj"], color="blue", ls="--", alpha=0.6,
                   label=f"$\\kappa_{{\\mathrm{{maj}}}} = {theory['kappa_maj']:.4f}$")
    if theory.get("kappa_min") is not None:
        ax.axhline(theory["kappa_min"], color="red", ls="--", alpha=0.6,
                   label=f"$\\kappa_{{\\min}} = {theory['kappa_min']:.4f}$")

    ax.set_xlabel(r"$z_t = h \cdot t$")
    ax.set_ylabel("Rescaled error")
    ax.set_xscale("log")

    # Fixed y-axis across all alphas so plateau flatness is visually obvious
    # and cross-alpha comparison is immediate.
    # Max theory line is kappa_maj = 0.5 (for alpha >= 1); curves peak ~0.4.
    ax.set_ylim([0, 0.6])

    ax.legend()
    ax.grid(True, alpha=0.3)

    if alpha < 1.0:
        ax.set_title(
            f"Rescaled errors ($\\alpha = {alpha:.2f} < 1$: "
            f"both plateau at $\\kappa_g$)")
    elif alpha == 1.0:
        ax.set_title(
            f"Rescaled errors ($\\alpha = 1.0$: "
            f"majority plateaus, minority $\\to 0$)")
    else:
        ax.set_title(
            f"Rescaled errors ($\\alpha = {alpha:.2f} > 1$: "
            f"majority plateaus, minority $\\to 0$)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rescaled_errors.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "rescaled_errors.pdf"))
    plt.close(fig)


# ============================================================
#  Summary plot
# ============================================================

def plot_summary(all_results, out_dir, z_max):
    """Measured decay exponent vs. theoretical alpha."""
    ensure_dir(out_dir)
    alphas_theory = []
    exponents_measured_min = []
    exponents_measured_maj = []

    for res in all_results:
        cfg = res["cfg"]
        d = res["data"]
        z_t, err_maj, err_min = d["z_t"], d["err_maj"], d["err_min"]

        if len(z_t) < 50:
            continue

        log_z = np.log(z_t)
        dlog_z = np.diff(log_z)

        slope_maj = np.diff(np.log(err_maj)) / dlog_z
        n_tail = max(10, len(slope_maj) // 5)
        exp_maj = np.median(slope_maj[-n_tail:])

        slope_min = np.diff(np.log(err_min)) / dlog_z
        exp_min = np.median(slope_min[-n_tail:])

        alphas_theory.append(cfg.alpha)
        exponents_measured_maj.append(-exp_maj)
        exponents_measured_min.append(-exp_min)

    alphas_theory = np.array(alphas_theory)
    exponents_measured_min = np.array(exponents_measured_min)
    exponents_measured_maj = np.array(exponents_measured_maj)

    fig, ax = plt.subplots(figsize=(8, 6))
    a_range = np.linspace(0.3, 2.2, 100)
    ax.plot(a_range, np.ones_like(a_range), "b--", lw=1.5, alpha=0.6,
            label="Theory: majority exponent = 1")
    theory_min_exp = np.where(a_range < 1, 1.0, a_range)
    ax.plot(a_range, theory_min_exp, "r--", lw=1.5, alpha=0.6,
            label=r"Theory: minority exponent = $\max(\alpha, 1)$")

    ax.scatter(alphas_theory, exponents_measured_maj, c="blue", s=80, zorder=5,
               marker="o", label="Measured: majority")
    ax.scatter(alphas_theory, exponents_measured_min, c="red", s=80, zorder=5,
               marker="^", label="Measured: minority")

    ax.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.5)
    ax.text(1.02, 0.5, r"$\alpha = 1$" + "\n(phase transition)", fontsize=9,
            color="gray", transform=ax.get_xaxis_transform())
    ax.set_xlabel(r"Theoretical $\alpha$")
    ax.set_ylabel(r"Measured decay exponent ($-$slope)")
    ax.set_title("Phase transition verification: measured exponent vs. "
                 r"$\alpha$")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.3, 2.2])
    ax.set_ylim([0.0, 2.5])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_transition_summary.png"), dpi=150)
    fig.savefig(os.path.join(out_dir, "phase_transition_summary.pdf"))
    plt.close(fig)
    print(f"  Summary plot saved to {out_dir}/phase_transition_summary.png")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Re-plot tabular vendor results with x-axis cropping")
    parser.add_argument("--T_max", type=float, required=True,
                        help="Crop all plots at z_t <= T_max")
    parser.add_argument("--in_dir", default="./tabular_vendor_results",
                        help="Input directory containing alpha_*/logs.json")
    parser.add_argument("--out_dir", default="./tabular_vendor_results_crop",
                        help="Output directory for cropped plots")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.5, 0.7, 1.0, 1.3, 1.5, 2.0],
                        help="Alpha values to replot")
    args = parser.parse_args()

    print("=" * 60)
    print("  Replot: Tabular Vendor Results (cropped)")
    print("=" * 60)
    print(f"  Input:  {args.in_dir}")
    print(f"  Output: {args.out_dir}")
    print(f"  T_max:  {args.T_max:.0f}")
    print(f"  Alphas: {args.alphas}")

    all_results = []

    for a in args.alphas:
        run_dir = os.path.join(args.in_dir, f"alpha_{a:.2f}")
        logs_path = os.path.join(run_dir, "logs.json")
        if not os.path.exists(logs_path):
            print(f"  [SKIP] No logs.json for alpha={a:.2f} in {run_dir}")
            continue

        # Load saved data
        payload = load_logs(run_dir)
        logs = payload["logs"]
        cfg_dict = payload["config"]

        # Reconstruct SynthConfig
        cfg = SynthConfig(
            mu_A=cfg_dict["mu_A"], mu_B=cfg_dict["mu_B"], mu=cfg_dict["mu"],
            gamma_min=cfg_dict["gamma_min"], gamma_maj=cfg_dict["gamma_maj"],
            epsilon=cfg_dict["epsilon"],
            noise_R=cfg_dict.get("noise_R", 0.1),
            N=cfg_dict["N"], steps=cfg_dict["steps"], lr=cfg_dict["lr"],
            log_every=cfg_dict.get("log_every", 1),
            print_every=cfg_dict.get("print_every", 1),
            out_dir=run_dir,
        )
        theory = kappa_theory(cfg)

        # Prepare cropped data
        d = prepare_data(logs, cfg, z_max=args.T_max)
        if len(d["z_t"]) < 50:
            print(f"  [SKIP] Too few points after crop for alpha={a:.2f}")
            continue

        # Output directory for this alpha
        alpha_out = os.path.join(args.out_dir, f"alpha_{a:.2f}")
        ensure_dir(alpha_out)

        # Generate all 3 plots
        plot_error_ratio(d, cfg, theory, alpha_out)
        plot_compensated(d, cfg, theory, alpha_out)
        plot_rescaled(d, cfg, theory, alpha_out)
        print(f"  alpha={a:.2f}: 3 plots saved to {alpha_out}/")

        all_results.append({"cfg": cfg, "data": d, "theory": theory})

    # Summary plot
    if all_results:
        plot_summary(all_results, args.out_dir, args.T_max)

    print("\nDone.")


if __name__ == "__main__":
    main()
