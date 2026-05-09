"""
scenario1_semisynthetic.py — Semi-synthetic sanity check for the Δ_i partition pipeline

Validates the methodology for discovering the isotropic regime in learned
representations.  We generate synthetic data with KNOWN feature-mediated
spurious structure (isotropic regime), train an MLP, then apply the full
analysis pipeline on the learned representations.

Pipeline:
  1. Generate synthetic data with known (A, B, v, μ_A, μ_B, μ)
  2. Train a small MLP on x = [r; s]
  3. Extract penultimate-layer representations Φ(x)
  4. Compute group-sensitivity Δ_i for each coordinate of Φ
  5. Partition Φ into r̃ (low Δ) and s̃ (high Δ)
  6. Fit Â, B̂ by group-wise linear regression
  7. Compute v̂ (max-margin direction on r̃)
  8. Check isotropic condition on (Â, B̂, v̂)
  9. Project onto isotropic constraint, measure residual
  10. Compare recovered α with the ground truth

Usage:
    python scenario1_semisynthetic.py
    python scenario1_semisynthetic.py --d_r 16 --d_s 16 --hidden_dim 128
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===========================================================
#  Configuration
# ===========================================================

def get_config():
    p = argparse.ArgumentParser(description="Scenario 1: semi-synthetic sanity check")

    # ground-truth geometry
    p.add_argument('--d_r', type=int, default=8)
    p.add_argument('--d_s', type=int, default=8)
    p.add_argument('--mu_A', type=float, default=1.0)
    p.add_argument('--mu_B', type=float, default=1.0)
    p.add_argument('--mu',   type=float, default=0.0)
    p.add_argument('--gamma_min', type=float, default=2.0,
                   help='r-margin of the minority group (γ̃_min)')
    p.add_argument('--noise_R', type=float, default=0.1,
                   help='Bound on ||ξ||')

    # data
    p.add_argument('--epsilon', type=float, default=0.1)
    p.add_argument('--N', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)

    # MLP
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--n_layers', type=int, default=3)
    p.add_argument('--lr_mlp', type=float, default=1e-3)
    p.add_argument('--steps_mlp', type=int, default=10000)
    p.add_argument('--batch_size', type=int, default=512)

    # Δ partition
    p.add_argument('--delta_quantile', type=float, default=0.5,
                   help='Quantile threshold for the Δ_i partition')

    # output
    p.add_argument('--out_dir', type=str, default='./scenario1_results')

    return p.parse_args()


# ===========================================================
#  Isotropic-regime operator construction (mirrors synth_utils.py)
# ===========================================================

def build_isotropic_operators(cfg):
    d_r, d_s = cfg.d_r, cfg.d_s
    assert d_s >= d_r

    v = np.zeros(d_r); v[0] = 1.0

    A = np.zeros((d_s, d_r))
    A[0, 0] = math.sqrt(cfg.mu_A)
    for j in range(1, d_r):
        A[j, j] = 1.0

    B = np.zeros((d_s, d_r))
    c1 = cfg.mu / math.sqrt(cfg.mu_A)
    c2 = math.sqrt(max(cfg.mu_B - cfg.mu ** 2 / cfg.mu_A, 0.0))
    B[0, 0] = c1
    B[1, 0] = c2
    for j in range(1, d_r):
        B[j, j] = 1.0

    # sanity
    assert abs(v @ A.T @ A @ v - cfg.mu_A) < 1e-10
    assert abs(v @ A.T @ B @ v - cfg.mu)   < 1e-10
    assert abs(v @ B.T @ B @ v - cfg.mu_B) < 1e-10
    return A, B, v


def generate_data(cfg, A, B, v):
    """Label-absorbed synthetic dataset (y = +1 always)."""
    np.random.seed(cfg.seed)
    N, d_r, d_s = cfg.N, cfg.d_r, cfg.d_s
    K = max(1.0, cfg.gamma_min) + 2.0
    R_perp = 0.5

    groups = (np.random.rand(N) < cfg.epsilon).astype(int)
    gamma_g = np.where(groups == 0, 1.0, cfg.gamma_min)
    Z = gamma_g + np.random.rand(N) * (K - gamma_g)

    rp = np.random.randn(N, d_r - 1)
    rp /= (np.linalg.norm(rp, axis=1, keepdims=True) + 1e-12)
    rp *= R_perp * np.random.rand(N, 1) ** (1.0 / max(d_r - 1, 1))

    r = np.zeros((N, d_r))
    r[:, 0] = Z
    r[:, 1:] = rp

    s = np.zeros((N, d_s))
    s[groups == 0] = r[groups == 0] @ A.T
    s[groups == 1] = r[groups == 1] @ B.T

    if cfg.noise_R > 0:
        xi = np.random.randn(N, d_s)
        xi /= (np.linalg.norm(xi, axis=1, keepdims=True) + 1e-12)
        xi *= cfg.noise_R * np.random.rand(N, 1) ** (1.0 / d_s)
        s += xi

    x = np.concatenate([r, s], axis=1)
    return x, r, s, groups


# ===========================================================
#  MLP
# ===========================================================

class MLP(nn.Module):
    def __init__(self, d_in, d_hid, n_layers):
        super().__init__()
        layers = [nn.Linear(d_in, d_hid), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(d_hid, d_hid), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(d_hid, 1)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)

    def get_features(self, x):
        return self.features(x)


def train_mlp(x_np, cfg):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.from_numpy(x_np).float().to(dev)
    N = len(X)

    model = MLP(X.shape[1], cfg.hidden_dim, cfg.n_layers).to(dev)
    opt = optim.Adam(model.parameters(), lr=cfg.lr_mlp)

    for step in range(1, cfg.steps_mlp + 1):
        idx = torch.randint(N, (cfg.batch_size,))
        logits = model(X[idx])
        loss = torch.log1p(torch.exp(-logits)).mean()   # absorbed: y=+1
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0 or step == 1:
            print(f"    [MLP step {step:>6d}]  loss = {loss.item():.6f}")

    model.eval()
    return model


# ===========================================================
#  Δ_i partition
# ===========================================================

def delta_partition(phi, groups, quantile):
    """
    Δ_i = |E[Φ_i | g=0] − E[Φ_i | g=1]|.
    Coordinates with Δ_i > quantile-threshold → spurious.
    """
    mean_0 = phi[groups == 0].mean(axis=0)
    mean_1 = phi[groups == 1].mean(axis=0)
    delta = np.abs(mean_0 - mean_1)
    thr = np.quantile(delta, quantile)
    r_idx = np.where(delta <= thr)[0]
    s_idx = np.where(delta >  thr)[0]
    return delta, r_idx, s_idx, thr


# ===========================================================
#  Isotropic check + projection  (general, non-circulant)
# ===========================================================

def _eig_residual(M, v):
    Mv = M @ v
    mu = float(v @ Mv)
    res = np.linalg.norm(Mv - mu * v) / (np.linalg.norm(Mv) + 1e-15)
    return mu, res


def check_isotropic_general(A, B, v):
    mu_A,  rA  = _eig_residual(A.T @ A, v)
    mu_AB, rAB = _eig_residual(A.T @ B, v)
    mu_B,  rB  = _eig_residual(B.T @ B, v)
    mu_BA, rBA = _eig_residual(B.T @ A, v)
    return dict(
        mu_A=mu_A, mu_B=mu_B, mu_AtB=mu_AB, mu_BtA=mu_BA,
        res_AtA=rA, res_BtB=rB, res_AtB=rAB, res_BtA=rBA,
        max_residual=max(rA, rB, rAB, rBA),
    )


def isotropic_projection(A, B, v):
    """
    Project A, B onto the isotropic constraint wrt v.

    A'v = (I − P_W) A v,   A'u = Au  for u ⊥ v
    where W = span{ A(v⊥) ∪ B(v⊥) }.
    """
    d_s, d_r = A.shape

    # orthonormal basis for v⊥
    Q = np.eye(d_r) - np.outer(v, v)
    _, S, Vt = np.linalg.svd(Q)
    vp = Vt[S > 1e-10].T                         # (d_r, d_r-1)

    # W = col span of [A vp, B vp]
    W = np.hstack([A @ vp, B @ vp])               # (d_s, 2*(d_r-1))
    Uw, Sw, _ = np.linalg.svd(W, full_matrices=False)
    rank = (Sw > 1e-10).sum()
    Uw = Uw[:, :rank]
    PW = Uw @ Uw.T

    Av = A @ v;  Bv = B @ v
    Av_p = (np.eye(d_s) - PW) @ Av
    Bv_p = (np.eye(d_s) - PW) @ Bv

    A_proj = A + np.outer(Av_p - Av, v)
    B_proj = B + np.outer(Bv_p - Bv, v)

    err_A = np.linalg.norm(Av_p - Av) / (np.linalg.norm(Av) + 1e-15)
    err_B = np.linalg.norm(Bv_p - Bv) / (np.linalg.norm(Bv) + 1e-15)
    return A_proj, B_proj, err_A, err_B


# ===========================================================
#  Max-margin v on label-absorbed r̃  (QP: min ‖v‖² s.t. Φ_r v ≥ 1)
# ===========================================================

def compute_v_qp(phi_r):
    from scipy.optimize import minimize

    d = phi_r.shape[1]
    res = minimize(
        fun=lambda v: 0.5 * np.sum(v ** 2),
        x0=np.ones(d) * 0.01,
        jac=lambda v: v,
        constraints={'type': 'ineq', 'fun': lambda v: phi_r @ v - 1.0},
        method='SLSQP',
        options=dict(maxiter=10000, ftol=1e-14),
    )
    v = res.x
    v_norm = v / np.linalg.norm(v)
    margins = phi_r @ v
    return v_norm, margins, res.success


# ===========================================================
#  Plots
# ===========================================================

def make_plots(delta, iso_before, iso_after, cfg):
    fig_dir = Path(cfg.out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Δ_i bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    order = np.argsort(delta)[::-1]
    ax.bar(range(len(delta)), delta[order], color='steelblue', alpha=.7)
    thr = np.quantile(delta, cfg.delta_quantile)
    ax.axhline(thr, color='red', ls='--', label=f'Threshold (q={cfg.delta_quantile})')
    ax.set(xlabel='Coordinate (sorted)', ylabel='Δ_i',
           title='Group-sensitivity scores of Φ(x) coordinates')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / 'delta_scores.png', dpi=200)
    fig.savefig(fig_dir / 'delta_scores.pdf', bbox_inches='tight')
    plt.close(fig)

    # residuals before / after projection
    labels = ['$A^\\top\\!A$', '$B^\\top\\!B$', '$A^\\top\\!B$', '$B^\\top\\!A$']
    bef = [iso_before['res_AtA'], iso_before['res_BtB'],
           iso_before['res_AtB'], iso_before['res_BtA']]
    aft = [iso_after['res_AtA'], iso_after['res_BtB'],
           iso_after['res_AtB'], iso_after['res_BtA']]

    x_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x_pos - .17, bef, .34, label='Before projection', color='salmon')
    ax.bar(x_pos + .17, aft, .34, label='After projection', color='steelblue')
    ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Relative eigenvector residual')
    ax.set_title('Isotropic condition: before vs. after projection')
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / 'isotropic_residuals.png', dpi=200)
    fig.savefig(fig_dir / 'isotropic_residuals.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Figures saved to {fig_dir}")


# ===========================================================
#  Main
# ===========================================================

def main():
    cfg = get_config()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    alpha_true = cfg.gamma_min * (1 + cfg.mu) / (1 + cfg.mu_A)

    print("=" * 65)
    print("  Scenario 1 — Semi-synthetic sanity check")
    print("=" * 65)

    # ---- 1. synthetic data ----
    print("\n[1/8] Generating synthetic isotropic-regime data ...")
    A_true, B_true, v_true = build_isotropic_operators(cfg)
    x, r_true, s_true, groups = generate_data(cfg, A_true, B_true, v_true)
    print(f"  d_r={cfg.d_r}, d_s={cfg.d_s}, N={cfg.N}, ε={cfg.epsilon}")
    print(f"  True α = {alpha_true:.4f}  "
          f"(μ_A={cfg.mu_A}, μ_B={cfg.mu_B}, μ={cfg.mu}, γ̃_min={cfg.gamma_min})")

    # ---- 2. train MLP ----
    print(f"\n[2/8] Training MLP ({cfg.n_layers} layers, "
          f"{cfg.hidden_dim} hidden) ...")
    model = train_mlp(x, cfg)

    # ---- 3. extract representations ----
    print("\n[3/8] Extracting penultimate-layer representations Φ(x) ...")
    X_t = torch.from_numpy(x).float().to(dev)
    with torch.no_grad():
        phi = model.get_features(X_t).cpu().numpy()
    d_phi = phi.shape[1]
    print(f"  Φ(x) ∈ R^{d_phi}")

    # ---- 4. Δ_i partition ----
    print("\n[4/8] Computing Δ_i partition ...")
    delta, r_idx, s_idx, thr = delta_partition(phi, groups, cfg.delta_quantile)
    print(f"  Threshold = {thr:.6f}  (quantile {cfg.delta_quantile})")
    print(f"  r̃ dim = {len(r_idx)},  s̃ dim = {len(s_idx)}")

    phi_r = phi[:, r_idx]
    phi_s = phi[:, s_idx]

    # ---- 5. fit Â, B̂ ----
    print("\n[5/8] Fitting Â, B̂ by group-wise OLS ...")
    maj, minn = (groups == 0), (groups == 1)

    A_hat = np.linalg.lstsq(phi_r[maj], phi_s[maj], rcond=None)[0].T
    B_hat = np.linalg.lstsq(phi_r[minn], phi_s[minn], rcond=None)[0].T

    def _r2(y, yhat):
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean(0)) ** 2) + 1e-15
        return 1 - ss_res / ss_tot

    r2_maj = _r2(phi_s[maj],  phi_r[maj]  @ A_hat.T)
    r2_min = _r2(phi_s[minn], phi_r[minn] @ B_hat.T)
    print(f"  R² majority: {r2_maj:.6f}")
    print(f"  R² minority: {r2_min:.6f}")

    # ---- 6. v̂ via QP ----
    print("\n[6/8] Computing v̂ (max-margin on r̃) ...")
    v_hat, margins, qp_ok = compute_v_qp(phi_r)
    print(f"  QP converged: {qp_ok}")

    gm_maj = margins[maj].min()
    gm_min = margins[minn].min()
    sc = min(gm_maj, gm_min)
    gm_maj /= sc; gm_min /= sc
    print(f"  γ̃_maj = {gm_maj:.4f},  γ̃_min = {gm_min:.4f}")

    # ---- 7. isotropic check (before projection) ----
    print("\n[7/8] Isotropic condition — before projection ...")
    iso_bef = check_isotropic_general(A_hat, B_hat, v_hat)
    print(f"  μ_A = {iso_bef['mu_A']:.6f},  μ_B = {iso_bef['mu_B']:.6f}")
    print(f"  μ(A^TB) = {iso_bef['mu_AtB']:.6f},  μ(B^TA) = {iso_bef['mu_BtA']:.6f}")
    print(f"  Residuals:  AtA={iso_bef['res_AtA']:.4f}  BtB={iso_bef['res_BtB']:.4f}  "
          f"AtB={iso_bef['res_AtB']:.4f}  BtA={iso_bef['res_BtA']:.4f}")
    print(f"  Max residual: {iso_bef['max_residual']:.6f}")

    # ---- 8. projection + post-check ----
    print("\n[8/8] Projecting onto isotropic constraint ...")
    A_proj, B_proj, pe_A, pe_B = isotropic_projection(A_hat, B_hat, v_hat)
    print(f"  Projection error  A: {pe_A:.6f},  B: {pe_B:.6f}")

    iso_aft = check_isotropic_general(A_proj, B_proj, v_hat)
    print(f"  After projection:")
    print(f"    μ_A = {iso_aft['mu_A']:.6f},  μ_B = {iso_aft['mu_B']:.6f}")
    print(f"    μ(A^TB) = {iso_aft['mu_AtB']:.6f}")
    print(f"    Residuals:  AtA={iso_aft['res_AtA']:.4f}  BtB={iso_aft['res_BtB']:.4f}  "
          f"AtB={iso_aft['res_AtB']:.4f}")
    print(f"    Max residual: {iso_aft['max_residual']:.6f}")

    # recovered alpha
    mu_A_r = iso_aft['mu_A']
    mu_r   = iso_aft['mu_AtB']
    tgm    = max(gm_maj, gm_min)
    alpha_rec = tgm * (1 + mu_r) / (1 + mu_A_r)
    print(f"\n  Recovered α = {alpha_rec:.4f}   (true α = {alpha_true:.4f})")

    # residual noise norm
    xi_maj = phi_s[maj]  - phi_r[maj]  @ A_proj.T
    xi_min = phi_s[minn] - phi_r[minn] @ B_proj.T
    xi_nm = np.linalg.norm(xi_maj, axis=1).mean()
    xi_nn = np.linalg.norm(xi_min, axis=1).mean()
    print(f"  Mean ‖ξ'‖  maj: {xi_nm:.6f},  min: {xi_nn:.6f}")

    # ---- figures ----
    make_plots(delta, iso_bef, iso_aft, cfg)

    # ---- save ----
    results = dict(
        config=vars(cfg),
        ground_truth=dict(mu_A=cfg.mu_A, mu_B=cfg.mu_B, mu=cfg.mu,
                          gamma_min=cfg.gamma_min, alpha=alpha_true),
        recovered=dict(mu_A=float(mu_A_r), mu=float(mu_r),
                        gamma_min=float(tgm), alpha=float(alpha_rec)),
        partition=dict(n_r=len(r_idx), n_s=len(s_idx),
                       threshold=float(thr)),
        regression=dict(r2_maj=float(r2_maj), r2_min=float(r2_min)),
        isotropic_before=iso_bef,
        isotropic_after=iso_aft,
        projection_error=dict(A=float(pe_A), B=float(pe_B)),
        residual_norm=dict(maj=float(xi_nm), min=float(xi_nn)),
    )
    with open(out / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"  Done — all outputs in  {cfg.out_dir}")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
