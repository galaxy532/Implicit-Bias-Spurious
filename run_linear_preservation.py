"""
run_linear_preservation.py
--------------------------
Experiment: does a nonlinear model preserve the linear spurious relationship
             s = A_g r + xi  in its learned representation phi(x)?

Protocol:
  1. Generate synthetic data from the isotropic regime (same setup as the paper).
  2. Train a 2-hidden-layer MLP to classify y from x = [r, s].
  3. Extract the penultimate-layer representation phi(x).
  4. Fit linear probes:  P_r such that P_r phi(x) ~ r,
                         P_s such that P_s phi(x) ~ s.
  5. Project onto v and scatter-plot  v^T P_r phi(x)  vs  v^T P_s phi(x),
     coloured by group, to verify the linear relationship is preserved.

Usage (from the Implicit-Bias-Spurious directory):
    python run_linear_preservation.py [--out_dir ./results_linear_preservation]

Requires: torch, numpy, matplotlib, sklearn.
          Run from the same directory as synth_utils.py.
"""

import os
import argparse
import math
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synth_utils import SynthConfig, build_isotropic_operators


# ============================================================
#  Data generation (un-absorbed, returns r, s, y separately)
# ============================================================

def generate_data_with_labels(cfg: SynthConfig, A: torch.Tensor, B: torch.Tensor,
                              v: torch.Tensor):
    """
    Generate data with explicit labels y in {-1, +1} and un-absorbed features.

    Unlike synth_utils.generate_dataset (which absorbs y into x), this function
    returns the raw (r, s, y, groups) so that we can train a standard classifier
    and later probe the representation for r and s separately.

    Data model (matches the paper):
      - y ~ Uniform({-1, +1})
      - Z = v^T r  drawn from Uniform[gamma_g, K]  (always positive in absorbed
        space, so in un-absorbed space  y * v^T r  is in [gamma_g, K])
      - r_perp ~ uniform on ball
      - r = y * (Z * v + r_perp)       (un-absorb)
      - s = A r + xi  (majority)  or  s = B r + xi  (minority)
      - x = [r; s]

    The spurious relationship  s = A_g r + xi  holds in the un-absorbed space
    because  s = A_g (y * r_absorbed) + xi = A_g r_original + xi  (xi is
    symmetric so y*xi ~ xi).

    Returns:
        r       (N, d_r)
        s       (N, d_s)
        y       (N,)        in {-1, +1}
        groups  (N,)        0 = majority, 1 = minority
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    N = cfg.N
    d_r, d_s = cfg.d_r, cfg.d_s
    K = max(cfg.gamma_maj, cfg.gamma_min) + 2.0
    R_perp = 0.5

    # Groups
    groups = (torch.rand(N) < cfg.epsilon).long()

    # Labels y ~ Uniform({-1, +1})
    y = 2 * torch.randint(0, 2, (N,)).float() - 1.0  # {-1, +1}

    # Draw Z in absorbed space (always positive)
    gamma_g = torch.where(groups == 0, cfg.gamma_maj, cfg.gamma_min).float()
    Z_absorbed = gamma_g + torch.rand(N) * (K - gamma_g)

    # r_perp in absorbed space
    r_perp_raw = torch.randn(N, d_r - 1)
    r_perp_raw = r_perp_raw / (r_perp_raw.norm(dim=1, keepdim=True) + 1e-12)
    r_perp_scale = R_perp * torch.rand(N, 1).pow(1.0 / max(d_r - 1, 1))
    r_perp_raw = r_perp_raw * r_perp_scale

    # r_absorbed = Z_absorbed * v + r_perp  (v = e_1)
    r_absorbed = torch.zeros(N, d_r)
    r_absorbed[:, 0] = Z_absorbed
    r_absorbed[:, 1:] = r_perp_raw

    # Un-absorb: r = y * r_absorbed
    r = y.unsqueeze(1) * r_absorbed

    # s = A_g r + xi  (the linear relationship holds in un-absorbed space)
    A_cpu, B_cpu = A.cpu(), B.cpu()
    s = torch.zeros(N, d_s)
    maj_mask = (groups == 0)
    min_mask = (groups == 1)
    s[maj_mask] = r[maj_mask] @ A_cpu.T
    s[min_mask] = r[min_mask] @ B_cpu.T

    # Add noise xi if requested
    if cfg.noise_R > 0:
        xi = torch.randn(N, d_s)
        xi = xi / (xi.norm(dim=1, keepdim=True) + 1e-12)
        xi = xi * cfg.noise_R * torch.rand(N, 1).pow(1.0 / d_s)
        s = s + xi

    return r, s, y, groups


# ============================================================
#  MLP model
# ============================================================

class TwoLayerMLP(nn.Module):
    """
    input(d_r + d_s) -> hidden1(64) -> hidden2(32) -> output(2)

    The penultimate representation phi(x) is the 32-dim output of hidden2
    (after ReLU).
    """
    def __init__(self, d_in, h1=64, h2=32):
        super().__init__()
        self.layer1 = nn.Linear(d_in, h1)
        self.layer2 = nn.Linear(h1, h2)
        self.head   = nn.Linear(h2, 2)
        self.relu   = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.head(x)

    def representation(self, x):
        """Return the penultimate-layer activations phi(x) in R^{h2}."""
        with torch.no_grad():
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
        return x


# ============================================================
#  Training
# ============================================================

def train_mlp(model, X_train, y_train, epochs=200, lr=1e-3, batch_size=2048,
              verbose=True):
    """Train the MLP with cross-entropy loss and Adam."""
    device = next(model.parameters()).device
    # Convert labels from {-1, +1} to {0, 1} for cross-entropy
    labels = ((y_train + 1) / 2).long().to(device)
    X = X_train.to(device)
    N = X.shape[0]

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            logits = model(X[idx])
            loss = criterion(logits, labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if verbose and (epoch % 50 == 0 or epoch == 1):
            avg_loss = epoch_loss / n_batches
            # Accuracy
            with torch.no_grad():
                preds = model(X).argmax(dim=1)
                acc = (preds == labels).float().mean().item()
            print(f"  Epoch {epoch:>4d}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    # Final accuracy per group
    with torch.no_grad():
        preds = model(X).argmax(dim=1)
        acc = (preds == labels).float().mean().item()
    return acc


# ============================================================
#  Linear probes and evaluation
# ============================================================

def fit_linear_probes(phi, r, s, v):
    """
    Fit linear probes P_r, P_s via ridge regression:
        P_r: phi(x) -> r       (R^{h2} -> R^{d_r})
        P_s: phi(x) -> s       (R^{h2} -> R^{d_s})

    Returns dict with fitted probes, predicted r_hat/s_hat, and R^2 scores.
    """
    phi_np = phi.cpu().numpy()
    r_np = r.cpu().numpy()
    s_np = s.cpu().numpy()
    v_np = v.cpu().numpy()

    # Fit P_r: phi -> r
    probe_r = Ridge(alpha=1.0)
    probe_r.fit(phi_np, r_np)
    r_hat = probe_r.predict(phi_np)
    r2_r = r2_score(r_np, r_hat, multioutput="variance_weighted")

    # Fit P_s: phi -> s
    probe_s = Ridge(alpha=1.0)
    probe_s.fit(phi_np, s_np)
    s_hat = probe_s.predict(phi_np)
    r2_s = r2_score(s_np, s_hat, multioutput="variance_weighted")

    # Projections are handled in make_figure using the operator A
    # (project onto Av/||Av|| direction in R^{d_s}).

    return {
        "probe_r": probe_r, "probe_s": probe_s,
        "r_hat": r_hat, "s_hat": s_hat,
        "r2_r": r2_r, "r2_s": r2_s,
    }


# ============================================================
#  Plotting
# ============================================================

def make_figure(r, s, r_hat, s_hat, groups, v, A, B, cfg, probe_results,
                out_path):
    """
    Two-panel scatter plot:
      Left:   v^T r  vs  (Av)^T s   in the RAW input space  (ground truth)
      Right:  v^T r_hat  vs  (Av)^T s_hat  in the PROBED representation space

    Both should show two linear clusters (majority slope = mu_A, minority
    slope ~ mu) if the linear spurious relationship is preserved.
    """
    v_np = v.cpu().numpy()
    A_np = A.cpu().numpy()
    B_np = B.cpu().numpy()
    r_np = r.cpu().numpy()
    s_np = s.cpu().numpy()
    groups_np = groups.cpu().numpy()

    # Projection directions
    Av = A_np @ v_np       # direction of s that correlates with v^T r (majority)
    Av_norm = Av / (np.linalg.norm(Av) + 1e-12)

    # Raw-space projections
    vr_raw = r_np @ v_np                # v^T r, shape (N,)
    Avs_raw = s_np @ Av_norm            # (Av/||Av||)^T s, shape (N,)

    # Probed-space projections
    vr_probe = r_hat @ v_np
    Avs_probe = s_hat @ Av_norm

    # Masks
    maj = (groups_np == 0)
    mino = (groups_np == 1)

    # Subsample for plot readability
    max_pts = 3000
    rng = np.random.RandomState(0)
    if maj.sum() > max_pts:
        idx_maj = rng.choice(np.where(maj)[0], max_pts, replace=False)
    else:
        idx_maj = np.where(maj)[0]
    if mino.sum() > max_pts:
        idx_min = rng.choice(np.where(mino)[0], max_pts, replace=False)
    else:
        idx_min = np.where(mino)[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: raw space ---
    ax = axes[0]
    ax.scatter(vr_raw[idx_maj], Avs_raw[idx_maj], s=4, alpha=0.3,
               c="tab:blue", label=f"Majority (1-ε={1-cfg.epsilon:.2f})")
    ax.scatter(vr_raw[idx_min], Avs_raw[idx_min], s=4, alpha=0.5,
               c="tab:red", label=f"Minority (ε={cfg.epsilon:.2f})")

    # Theoretical slopes
    # For majority:  (Av/||Av||)^T (A r) = (Av/||Av||)^T A (v * (v^T r) + r_perp)
    #              = ||Av|| * (v^T r)  + (Av/||Av||)^T A r_perp
    # So slope ~ ||Av|| = sqrt(mu_A)
    slope_maj = np.linalg.norm(Av)
    # For minority:  (Av/||Av||)^T (B r) = (Av/||Av||)^T B v * (v^T r) + ...
    #              = (Av . Bv) / ||Av|| * (v^T r) + ...
    Bv = B_np @ v_np
    slope_min = np.dot(Av_norm, Bv)

    xlim = ax.get_xlim()
    xx = np.linspace(xlim[0], xlim[1], 100)
    ax.plot(xx, slope_maj * xx, "b--", linewidth=1.5,
            label=f"slope = √μ_A = {slope_maj:.2f}")
    ax.plot(xx, slope_min * xx, "r--", linewidth=1.5,
            label=f"slope = μ/√μ_A = {slope_min:.2f}")

    ax.set_xlabel(r"$v^\top r$", fontsize=12)
    ax.set_ylabel(r"$\hat{a}^\top s$  ($\hat{a} = Av/\|Av\|$)", fontsize=12)
    ax.set_title("Raw input space", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")

    # --- Right panel: probed representation space ---
    ax = axes[1]
    ax.scatter(vr_probe[idx_maj], Avs_probe[idx_maj], s=4, alpha=0.3,
               c="tab:blue", label="Majority")
    ax.scatter(vr_probe[idx_min], Avs_probe[idx_min], s=4, alpha=0.5,
               c="tab:red", label="Minority")

    # Fit empirical slopes via least-squares
    from numpy.polynomial.polynomial import polyfit
    if len(idx_maj) > 10:
        c_maj = polyfit(vr_probe[idx_maj], Avs_probe[idx_maj], 1)
        emp_slope_maj = c_maj[1]
    else:
        emp_slope_maj = float("nan")
    if len(idx_min) > 10:
        c_min = polyfit(vr_probe[idx_min], Avs_probe[idx_min], 1)
        emp_slope_min = c_min[1]
    else:
        emp_slope_min = float("nan")

    xlim2 = ax.get_xlim()
    xx2 = np.linspace(xlim2[0], xlim2[1], 100)
    if not np.isnan(emp_slope_maj):
        ax.plot(xx2, c_maj[0] + emp_slope_maj * xx2, "b--", linewidth=1.5,
                label=f"fitted slope = {emp_slope_maj:.2f}")
    if not np.isnan(emp_slope_min):
        ax.plot(xx2, c_min[0] + emp_slope_min * xx2, "r--", linewidth=1.5,
                label=f"fitted slope = {emp_slope_min:.2f}")

    ax.set_xlabel(r"$v^\top P_r \varphi(x)$", fontsize=12)
    ax.set_ylabel(r"$\hat{a}^\top P_s \varphi(x)$", fontsize=12)
    ax.set_title("Learned representation (probed)", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")

    # Annotation with R^2 scores
    r2_r = probe_results["r2_r"]
    r2_s = probe_results["r2_s"]
    fig.suptitle(
        f"Linear spurious preservation in MLP representation\n"
        f"(α={cfg.alpha:.2f}, ε={cfg.epsilon}, μ_A={cfg.mu_A}, μ={cfg.mu})   "
        f"Probe R²: r→{r2_r:.3f}, s→{r2_s:.3f}   "
        f"Slopes — raw: ({slope_maj:.2f}, {slope_min:.2f}), "
        f"probed: ({emp_slope_maj:.2f}, {emp_slope_min:.2f})",
        fontsize=10, y=1.02
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    plt.close()

    return {
        "slope_maj_raw": float(slope_maj),
        "slope_min_raw": float(slope_min),
        "slope_maj_probed": float(emp_slope_maj),
        "slope_min_probed": float(emp_slope_min),
        "r2_r": float(r2_r),
        "r2_s": float(r2_s),
    }


# ============================================================
#  Main
# ============================================================

def run_experiment(cfg: SynthConfig, out_dir: str, mlp_epochs: int = 200,
                   mlp_lr: float = 1e-3):
    """Full pipeline: generate data, train MLP, probe, plot."""
    os.makedirs(out_dir, exist_ok=True)
    device = cfg.device
    print(f"Device: {device}")
    print(f"Config: d_r={cfg.d_r}, d_s={cfg.d_s}, epsilon={cfg.epsilon}, "
          f"mu_A={cfg.mu_A}, mu_B={cfg.mu_B}, mu={cfg.mu}, "
          f"gamma_min={cfg.gamma_min}, alpha={cfg.alpha:.3f}, "
          f"noise_R={cfg.noise_R}")

    # Step 1: build operators and generate data
    print("\n[1/4] Generating data...")
    A, B, v = build_isotropic_operators(cfg)
    r, s, y, groups = generate_data_with_labels(cfg, A, B, v)
    X = torch.cat([r, s], dim=1)  # (N, d_r + d_s)
    print(f"  N={cfg.N}, majority={int((groups==0).sum())}, "
          f"minority={int((groups==1).sum())}")

    # Step 2: train MLP
    print(f"\n[2/4] Training MLP ({mlp_epochs} epochs, lr={mlp_lr})...")
    d_in = cfg.d_r + cfg.d_s
    model = TwoLayerMLP(d_in=d_in, h1=64, h2=32).to(device)
    X_dev = X.to(device)
    acc = train_mlp(model, X_dev, y, epochs=mlp_epochs, lr=mlp_lr)
    print(f"  Final accuracy: {acc:.4f}")

    # Step 3: extract representations and fit probes
    print("\n[3/4] Extracting representations and fitting linear probes...")
    model.eval()
    phi = model.representation(X_dev)  # (N, 32)
    print(f"  Representation shape: {phi.shape}")

    probe_results = fit_linear_probes(phi, r, s, v)
    print(f"  Linear probe R² for r: {probe_results['r2_r']:.4f}")
    print(f"  Linear probe R² for s: {probe_results['r2_s']:.4f}")

    # Step 4: make figure
    print("\n[4/4] Plotting...")
    fig_path = os.path.join(out_dir, "linear_preservation.png")
    plot_stats = make_figure(
        r, s,
        probe_results["r_hat"], probe_results["s_hat"],
        groups, v, A, B, cfg, probe_results,
        fig_path
    )

    # Save summary
    summary = {
        "config": {
            "d_r": cfg.d_r, "d_s": cfg.d_s, "epsilon": cfg.epsilon,
            "mu_A": cfg.mu_A, "mu_B": cfg.mu_B, "mu": cfg.mu,
            "gamma_min": cfg.gamma_min, "alpha": cfg.alpha,
            "noise_R": cfg.noise_R, "N": cfg.N, "seed": cfg.seed,
        },
        "mlp": {
            "architecture": "16->64->32->2",
            "epochs": mlp_epochs, "lr": mlp_lr, "final_acc": acc,
        },
        "probes": plot_stats,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print(f"Done.  Key result: slope_maj raw={plot_stats['slope_maj_raw']:.3f} "
          f"-> probed={plot_stats['slope_maj_probed']:.3f},  "
          f"slope_min raw={plot_stats['slope_min_raw']:.3f} "
          f"-> probed={plot_stats['slope_min_probed']:.3f}")

    return summary


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test whether an MLP preserves linear spurious structure "
                    "in its learned representation.")
    parser.add_argument("--out_dir", type=str,
                        default="./results_linear_preservation",
                        help="Output directory for figures and logs")
    # Data
    parser.add_argument("--d_r", type=int, default=8)
    parser.add_argument("--d_s", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--gamma_min", type=float, default=1.5,
                        help="Minority r-margin (gamma_maj fixed at 1.0)")
    parser.add_argument("--mu_A", type=float, default=1.0)
    parser.add_argument("--mu_B", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.0,
                        help="Cross-coupling eigenvalue (0 = decoupled)")
    parser.add_argument("--noise_R", type=float, default=0.3,
                        help="Noise radius for xi (0 = noiseless)")
    parser.add_argument("--N", type=int, default=50000,
                        help="Dataset size")
    parser.add_argument("--seed", type=int, default=42)
    # MLP
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)

    args = parser.parse_args()

    cfg = SynthConfig(
        d_r=args.d_r,
        d_s=args.d_s,
        gamma_min=args.gamma_min,
        gamma_maj=1.0,
        mu_A=args.mu_A,
        mu_B=args.mu_B,
        mu=args.mu,
        noise_R=args.noise_R,
        epsilon=args.epsilon,
        N=args.N,
        seed=args.seed,
        # Training params below are for the linear-model experiments;
        # the MLP uses its own optimizer, so these don't matter here.
        lr=0.01, steps=1,
        out_dir=args.out_dir,
    )

    run_experiment(cfg, out_dir=args.out_dir, mlp_epochs=args.epochs,
                   mlp_lr=args.mlp_lr)


if __name__ == "__main__":
    main()
