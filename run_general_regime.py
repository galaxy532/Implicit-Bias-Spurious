#!/usr/bin/env python3
"""
run_general_regime.py
=====================
Experiment B: General (non-isotropic) regime validation on MNIST.

Uses MNIST digit images as core features r (full 28x28 = 784 dims),
generates spatially-rich spurious features via group-specific convolutions:
  s = Conv(r, kernel_A) + noise   (majority)
  s = Conv(r, kernel_B) + noise   (minority)
Convolution is a linear operation (Toeplitz matrix), so s = A_g * r + xi
holds exactly, with A dense and naturally in the general regime.

After generating x = [r; s], we:
  1. Regress s on r per group to estimate A_hat, B_hat and check R^2.
  2. Verify the general regime (A^T A and A^T B don't share eigenvectors).
  3. Train a linear classifier via full-batch logistic GD (label-absorbed).
  4. Plot per-group error decay with error bands (multi-seed).

Usage:
  python run_general_regime.py --quick        # sanity check (~2 min)
  python run_general_regime.py                # full run
  python run_general_regime.py --plot_only    # re-plot from saved results
"""

import argparse
import os
import json
import glob
import re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from scipy.ndimage import convolve

from synth_utils import SynthConfig, train_population_gd, save_logs, ensure_dir


# ============================================================
#  Data loading
# ============================================================

def load_mnist_images():
    """
    Load MNIST images (full 28x28) and binary labels (0-4 vs 5-9).
    Returns images (N, 28, 28) float32 in [0,1], y (N,) in {-1, +1}.
    """
    try:
        from torchvision import datasets
        train_ds = datasets.MNIST('./data', train=True, download=True)
        test_ds = datasets.MNIST('./data', train=False, download=True)
        images = np.concatenate([
            train_ds.data.numpy(), test_ds.data.numpy()
        ]).astype(np.float32)
        labels = np.concatenate([
            train_ds.targets.numpy(), test_ds.targets.numpy()
        ])
    except Exception as e:
        print(f"  torchvision failed ({e}), trying sklearn...")
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        images = mnist.data.astype(np.float32).reshape(-1, 28, 28)
        labels = mnist.target.astype(int)

    images = images / 255.0
    y = np.where(labels < 5, -1.0, 1.0)

    print(f"  Loaded {len(images)} images (28x28).  "
          f"Class split: {(y==-1).sum()} / {(y==1).sum()}")
    return images, y


# ============================================================
#  Convolution-based spurious feature construction
# ============================================================

def make_gaussian_kernel(size, sigma, angle_deg=0.0):
    """
    Create a 2D Gaussian blur kernel, optionally elongated along an angle.

    For isotropic: sigma_x = sigma_y = sigma.
    For directional: sigma along the angle is 2*sigma, perpendicular is 0.5*sigma.
    """
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)

    if angle_deg is not None:
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Rotate coordinates
        u = cos_t * xx + sin_t * yy
        v = -sin_t * xx + cos_t * yy
        # Elongated: wide along angle, narrow perpendicular
        sigma_u = 2.0 * sigma
        sigma_v = 0.5 * sigma
        kernel = np.exp(-0.5 * (u**2 / sigma_u**2 + v**2 / sigma_v**2))
    else:
        kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)

    kernel = kernel / kernel.sum()
    return kernel.astype(np.float32)


def apply_conv_spurious(images, kernel, noise_std=0.05, seed=42):
    """
    Apply a 2D convolution kernel to each image, add noise, and flatten.

    images: (N, 28, 28) float32
    Returns: s (N, 784) float32 = Conv(image, kernel) + noise
    """
    rng = np.random.RandomState(seed)
    N = images.shape[0]
    s = np.zeros((N, 28, 28), dtype=np.float32)

    for i in range(N):
        s[i] = convolve(images[i], kernel, mode='constant', cval=0.0)

    s_flat = s.reshape(N, -1)

    if noise_std > 0:
        s_flat += rng.randn(N, 784).astype(np.float32) * noise_std

    return s_flat


# ============================================================
#  Regime analysis
# ============================================================

def estimate_operators(r, s, groups):
    """
    Estimate A, B via least-squares regression: s = A * r (majority), s = B * r (minority).
    Returns A_hat, B_hat, R2_A, R2_B.
    """
    maj = groups == 0
    mino = groups == 1

    # A_hat: least squares  s_maj = r_maj @ A_hat^T
    r_maj, s_maj = r[maj], s[maj]
    A_hat, res_A, _, _ = np.linalg.lstsq(r_maj, s_maj, rcond=None)
    # A_hat is (d_r, d_s), we want (d_s, d_r)
    A_hat = A_hat.T

    # R^2 for majority
    s_pred_A = r_maj @ A_hat.T
    ss_res_A = np.sum((s_maj - s_pred_A)**2)
    ss_tot_A = np.sum((s_maj - s_maj.mean(axis=0))**2)
    R2_A = 1.0 - ss_res_A / (ss_tot_A + 1e-12)

    # B_hat: least squares  s_min = r_min @ B_hat^T
    r_min, s_min = r[mino], s[mino]
    B_hat, res_B, _, _ = np.linalg.lstsq(r_min, s_min, rcond=None)
    B_hat = B_hat.T

    s_pred_B = r_min @ B_hat.T
    ss_res_B = np.sum((s_min - s_pred_B)**2)
    ss_tot_B = np.sum((s_min - s_min.mean(axis=0))**2)
    R2_B = 1.0 - ss_res_B / (ss_tot_B + 1e-12)

    return A_hat, B_hat, float(R2_A), float(R2_B)


def analyze_regime(A, B):
    """
    Check whether (A, B) are in the isotropic or general regime.

    Isotropic: A^T A, A^T B share a common eigenvector.
    General:   they don't.

    Returns dict with alignment metrics.
    """
    AtA = A.T @ A
    AtB = A.T @ B
    BtB = B.T @ B

    _, evecs_AA = np.linalg.eigh(AtA)
    _, evecs_BB = np.linalg.eigh(BtB)

    v_top = evecs_AA[:, -1]  # top eigenvector of A^T A

    # Check if v_top is also an eigenvector of A^T B
    AtB_v = AtB @ v_top
    nrm = np.linalg.norm(AtB_v)
    cos_angle = abs(v_top @ AtB_v) / (nrm + 1e-12) if nrm > 1e-10 else 0.0

    # Alignment between top eigenvectors of A^T A and B^T B
    eigvec_align = abs(v_top @ evecs_BB[:, -1])

    # Effective spectral values along v_top
    mu_A_eff = float(v_top @ AtA @ v_top)
    mu_B_eff = float(v_top @ BtB @ v_top)
    mu_eff   = float(v_top @ AtB @ v_top)

    return {
        "cos_angle_v_AtA_vs_AtB_v": round(float(cos_angle), 4),
        "eigvec_alignment_AtA_vs_BtB": round(float(eigvec_align), 4),
        "is_isotropic": bool(cos_angle > 0.99),
        "mu_A_eff": round(mu_A_eff, 4),
        "mu_B_eff": round(mu_B_eff, 4),
        "mu_eff": round(mu_eff, 4),
    }


# ============================================================
#  Dataset assembly
# ============================================================

def generate_general_dataset(images, y, kernel_A, kernel_B,
                              epsilon, noise_std=0.05, seed=42):
    """
    Construct x = [r; s] with label absorption.

    r = flattened image (784 dims)
    s = Conv(image, kernel_A) + noise  (majority, group=0)
      = Conv(image, kernel_B) + noise  (minority, group=1)
    x = y * [r; s]   (label absorption)

    Returns: X (N, 1568), groups (N,), r_flat (N,784), s_flat (N,784)
    """
    rng = np.random.RandomState(seed)
    N = images.shape[0]

    groups = (rng.rand(N) < epsilon).astype(np.int64)
    maj = groups == 0
    mino = groups == 1

    r_flat = images.reshape(N, -1).astype(np.float32)  # (N, 784)

    # Apply group-specific convolutions
    s_flat = np.zeros((N, 784), dtype=np.float32)
    # Use different noise seeds per group to avoid correlation artifacts
    s_flat[maj] = apply_conv_spurious(
        images[maj], kernel_A, noise_std=noise_std, seed=seed)[:]
    s_flat[mino] = apply_conv_spurious(
        images[mino], kernel_B, noise_std=noise_std, seed=seed + 1000)[:]

    # Concatenate: x = [r; s]
    x = np.concatenate([r_flat, s_flat], axis=1)  # (N, 1568)

    # Label absorption: x <- y * x
    x = x * y[:, None].astype(np.float32)

    return x, groups, r_flat, s_flat


# ============================================================
#  Post-hoc analysis
# ============================================================

def compute_margins(w, X, groups):
    """Compute per-group normalized margins from converged w."""
    w_np = w.cpu().numpy() if isinstance(w, torch.Tensor) else w
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    g_np = groups.cpu().numpy() if isinstance(groups, torch.Tensor) else groups

    margins = X_np @ w_np
    w_norm = np.linalg.norm(w_np) + 1e-12
    m = margins / w_norm

    return {
        "gamma_min": round(float(m[g_np == 1].min()), 4),
        "gamma_maj": round(float(m[g_np == 0].min()), 4),
        "gamma_min_mean": round(float(m[g_np == 1].mean()), 4),
        "gamma_maj_mean": round(float(m[g_np == 0].mean()), 4),
        "w_norm": round(float(w_norm), 4),
    }


def measure_exponent(logs, h, t_min_frac=0.5):
    """Fit log(err) = -beta * log(z_t) + C on the tail."""
    t = np.array(logs["t"], dtype=float)
    err = np.array(logs["err_min"], dtype=float)
    z = h * t
    mask = (z >= t_min_frac * z[-1]) & (err > 1e-14)
    if mask.sum() < 10:
        return float('nan')
    coeffs = np.polyfit(np.log(z[mask]), np.log(err[mask]), 1)
    return -coeffs[0]


# ============================================================
#  Plotting
# ============================================================

def load_general_runs(run_root):
    """Load all (eps, seed) runs from runs_synth/general_regime/."""
    runs = {}
    for d in sorted(glob.glob(os.path.join(run_root, "eps_*"))):
        m = re.search(r"eps_([0-9.]+)$", os.path.basename(d))
        if not m:
            continue
        eps = float(m.group(1))

        seed_dirs = sorted(glob.glob(os.path.join(d, "seed_*")))
        seed_data = []
        for sd in seed_dirs:
            lp = os.path.join(sd, "logs.json")
            if os.path.exists(lp):
                with open(lp) as f:
                    seed_data.append(json.load(f))

        if not seed_data:
            lp = os.path.join(d, "logs.json")
            if os.path.exists(lp):
                with open(lp) as f:
                    seed_data.append(json.load(f))

        if not seed_data:
            continue

        ref = seed_data[0]
        t = np.array(ref["logs"]["t"], dtype=float)
        agg = {"t": t}
        for key in ["err_min", "err_maj", "loss"]:
            arr = np.array([np.array(sd["logs"][key], dtype=float) for sd in seed_data])
            agg[f"{key}_mean"] = arr.mean(axis=0)
            agg[f"{key}_std"]  = arr.std(axis=0)

        runs[eps] = {
            "seeds": seed_data, "agg": agg,
            "meta": ref["meta"], "config": ref["config"],
            "n_seeds": len(seed_data),
        }

    return dict(sorted(runs.items()))


def plot_general_regime(runs, out_dir, t_min=0):
    """Plot error decay for general regime experiment."""
    ensure_dir(out_dir)
    if not runs:
        print("  No runs to plot.")
        return

    first = next(iter(runs.values()))
    h = first["config"]["lr"]
    n_seeds = first["n_seeds"]

    eps_vals = np.array(sorted(runs.keys()))
    norm = Normalize(vmin=eps_vals.min(), vmax=eps_vals.max())
    cmap = get_cmap("viridis")

    for key, group_name in [("err_min", "minority"), ("err_maj", "majority")]:
        fig, ax = plt.subplots(figsize=(7.5, 5))

        for eps in eps_vals:
            data = runs[eps]
            t = data["agg"]["t"]
            z = h * t
            mean = data["agg"][f"{key}_mean"]
            std  = data["agg"][f"{key}_std"]
            mask = t >= t_min

            c = cmap(norm(eps))
            ax.loglog(z[mask], np.maximum(mean[mask], 1e-15),
                      lw=1.4, alpha=0.85, color=c,
                      label=rf"$\varepsilon={eps}$")
            if n_seeds > 1:
                ax.fill_between(z[mask],
                                np.maximum(mean[mask] - std[mask], 1e-15),
                                mean[mask] + std[mask],
                                color=c, alpha=0.15)

        # Theory reference: z_t^{-gamma} slope
        meta = first["meta"]
        margins = meta.get("margins", {})
        if group_name == "minority" and "gamma_min" in margins:
            gamma = abs(margins["gamma_min"])
            if gamma > 0:
                z_ref = h * np.array(first["agg"]["t"])
                mask_ref = z_ref > z_ref[-1] * 0.3
                z_r = z_ref[mask_ref]
                ref_line = z_r**(-gamma) * (z_r[0]**gamma * mean[mask][0] if len(mean[mask]) > 0 else 1.0)
                ax.loglog(z_r, ref_line, 'k--', lw=1.5, alpha=0.5,
                          label=rf"$z_t^{{-{gamma:.2f}}}$ (measured $\gamma_{{\min}}$)")

        ax.set_xlabel(r"$z_t = h\,t$", fontsize=13)
        ax.set_ylabel(
            rf"$\mathbb{{E}}_{{\mathcal{{G}}_{{\mathrm{{{group_name}}}}}}}[1 - p_y]$",
            fontsize=13)
        title = rf"{group_name.capitalize()} error decay (general regime, MNIST conv.)"
        if n_seeds > 1:
            title += rf"  [{n_seeds} seeds, $\pm 1\sigma$]"
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=9, frameon=True, loc="lower left")
        ax.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        fname = f"general_regime_{group_name}_error_decay.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close(fig)
        print(f"  Saved {fname}")

    # ---- Regime analysis summary figure ----
    regime = first["meta"].get("regime_info", {})
    margins = first["meta"].get("margins", {})
    R2 = first["meta"].get("R2", {})

    fig, ax = plt.subplots(figsize=(6, 5))
    lines = [
        "General Regime Verification (MNIST + Conv)",
        "=" * 45,
        f"Kernel A: horizontal Gaussian blur (sigma=2)",
        f"Kernel B: vertical Gaussian blur (sigma=2)",
        "",
        f"R^2 (s ~ A*r, majority)     = {R2.get('R2_A', '?')}",
        f"R^2 (s ~ B*r, minority)     = {R2.get('R2_B', '?')}",
        "",
        f"cos(v_top, A^T B v_top)     = {regime.get('cos_angle_v_AtA_vs_AtB_v', '?')}",
        f"align(top evec A^TA, B^TB)  = {regime.get('eigvec_alignment_AtA_vs_BtB', '?')}",
        f"Is isotropic?                 {regime.get('is_isotropic', '?')}",
        "",
        f"Effective mu_A  = {regime.get('mu_A_eff', '?')}",
        f"Effective mu_B  = {regime.get('mu_B_eff', '?')}",
        f"Effective mu    = {regime.get('mu_eff', '?')}",
    ]
    if margins:
        lines += [
            "",
            f"gamma_min (converged) = {margins.get('gamma_min', '?')}",
            f"gamma_maj (converged) = {margins.get('gamma_maj', '?')}",
            f"||w||                 = {margins.get('w_norm', '?')}",
        ]
    beta = first["meta"].get("beta_empirical", None)
    if beta is not None:
        lines += [f"Empirical exponent   = {beta}"]

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "general_regime_analysis.png"), dpi=200)
    plt.close(fig)
    print("  Saved general_regime_analysis.png")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment B: general regime on MNIST with convolution-based spurious features")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot_only", action="store_true",
                        help="Re-plot from existing runs without re-training")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--noise_std", type=float, default=0.05,
                        help="Noise std added to convolved spurious features")
    parser.add_argument("--kernel_size", type=int, default=7,
                        help="Size of the Gaussian convolution kernel")
    parser.add_argument("--kernel_sigma", type=float, default=2.0,
                        help="Sigma for Gaussian kernel")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epsilons", nargs="+", type=float, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--out_root", type=str,
                        default="./runs_synth/general_regime")
    parser.add_argument("--fig_dir", type=str, default="./figures")
    parser.add_argument("--t_min", type=float, default=0)
    args = parser.parse_args()

    # ---- Plot-only mode ----
    if args.plot_only:
        runs = load_general_runs(args.out_root)
        if not runs:
            print("No runs found. Run without --plot_only first.")
            return
        plot_general_regime(runs, args.fig_dir, t_min=args.t_min)
        return

    # ---- Defaults ----
    if args.quick:
        steps    = args.steps or 10_000
        epsilons = args.epsilons or [0.05, 0.2, 0.5]
        seeds    = args.seeds or [1, 2, 3]
        log_every, print_every = 100, 2_000
    else:
        steps    = args.steps or 200_000
        epsilons = args.epsilons or [0.01, 0.05, 0.1, 0.2, 0.5]
        seeds    = args.seeds or [1, 2, 3]
        log_every, print_every = 500, 50_000

    # ---- Build convolution kernels ----
    print("Building convolution kernels...")
    # Majority: horizontal Gaussian blur (angle=0 degrees)
    kernel_A = make_gaussian_kernel(args.kernel_size, args.kernel_sigma, angle_deg=0.0)
    # Minority: vertical Gaussian blur (angle=90 degrees)
    kernel_B = make_gaussian_kernel(args.kernel_size, args.kernel_sigma, angle_deg=90.0)
    print(f"  Kernel A (horizontal): {kernel_A.shape}, sum={kernel_A.sum():.4f}")
    print(f"  Kernel B (vertical):   {kernel_B.shape}, sum={kernel_B.sum():.4f}")

    # ---- Load MNIST images ----
    print("\nLoading MNIST images...")
    images, y = load_mnist_images()
    N = len(images)
    d_r = 784
    d_s = 784  # convolution preserves spatial dims (same padding)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning: steps={steps}, eps={epsilons}, seeds={seeds}, device={device}")

    # ---- Regime analysis (do once with a representative split) ----
    print("\nAnalyzing regime with eps=0.1...")
    X_check, g_check, r_check, s_check = generate_general_dataset(
        images, y, kernel_A, kernel_B, epsilon=0.1, noise_std=args.noise_std, seed=0)

    A_hat, B_hat, R2_A, R2_B = estimate_operators(r_check, s_check, g_check)
    print(f"  R^2 (majority, s ~ A*r): {R2_A:.4f}")
    print(f"  R^2 (minority, s ~ B*r): {R2_B:.4f}")

    regime_info = analyze_regime(A_hat, B_hat)
    print(f"  cos(v, A^T B v) = {regime_info['cos_angle_v_AtA_vs_AtB_v']}")
    print(f"  eigvec alignment = {regime_info['eigvec_alignment_AtA_vs_BtB']}")
    print(f"  isotropic? {regime_info['is_isotropic']}")

    R2_info = {"R2_A": round(R2_A, 4), "R2_B": round(R2_B, 4)}

    # ---- Sweep ----
    for eps in epsilons:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"  eps={eps}  seed={seed}  steps={steps}  N={N}")
            print(f"{'='*60}")

            X_np, groups_np, _, _ = generate_general_dataset(
                images, y, kernel_A, kernel_B,
                epsilon=eps, noise_std=args.noise_std, seed=seed)

            X_t = torch.tensor(X_np, device=device)
            g_t = torch.tensor(groups_np, device=device)

            # Config (spectral values are effective/approximate from regression)
            cfg = SynthConfig()
            cfg.d_r = d_r
            cfg.d_s = d_s
            cfg.epsilon = eps
            cfg.N = N
            cfg.lr = args.lr
            cfg.steps = steps
            cfg.log_every = log_every
            cfg.print_every = print_every
            cfg.seed = seed
            cfg.noise_R = args.noise_std
            cfg.mu_A = max(regime_info["mu_A_eff"], 0.01)
            cfg.mu_B = max(regime_info["mu_B_eff"], 0.01)
            cfg.mu = regime_info["mu_eff"]
            cfg.gamma_min = 1.0  # placeholder, will measure from converged w
            cfg.__post_init__()
            cfg.out_dir = os.path.join(args.out_root, f"eps_{eps}", f"seed_{seed}")

            logs, w_final = train_population_gd(cfg, X_t, g_t)

            margins = compute_margins(w_final, X_t, g_t)
            beta = measure_exponent(logs, cfg.lr)
            print(f"  margins: min={margins['gamma_min']}, maj={margins['gamma_maj']}")
            print(f"  empirical exponent (minority): {beta:.3f}")

            meta = {
                "N": N, "d_r": d_r, "d_s": d_s,
                "epsilon": eps, "seed": seed,
                "noise_std": args.noise_std,
                "kernel_size": args.kernel_size,
                "kernel_sigma": args.kernel_sigma,
                "kernel_A_type": "horizontal_gaussian",
                "kernel_B_type": "vertical_gaussian",
                "regime_info": regime_info,
                "R2": R2_info,
                "margins": margins,
                "beta_empirical": round(beta, 4),
                "n_maj": int((groups_np == 0).sum()),
                "n_min": int((groups_np == 1).sum()),
            }
            save_logs(logs, meta, cfg)

    # ---- Plot ----
    runs = load_general_runs(args.out_root)
    plot_general_regime(runs, args.fig_dir, t_min=args.t_min)

    print("\nDone. Figures in", args.fig_dir)


if __name__ == "__main__":
    main()
