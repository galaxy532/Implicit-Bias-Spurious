"""
train_sae_analyze.py
--------------------
Train a Sparse Autoencoder (SAE) on the MLP representations phi(x) from the
colored-MNIST experiment, then analyze whether the linear spurious correlation
s = A r + xi is preserved in the learned representation.

Pipeline:
  1. Load representations phi(x) and metadata.
  2. Train a tied-weight SAE:  c = ReLU(M phi + b),  phi_hat = M^T c
     with loss = ||phi - phi_hat||^2 + alpha * ||c||_1
  3. Classify each SAE feature as r-related or s-related by correlating
     its activation c_k with (a) digit class and (b) background intensity.
  4. Decompose phi into phi_r and phi_s components:
       phi_r = [c_k for k in r-features]
       phi_s = [c_k for k in s-features]
  5. Fit a linear map phi_r -> phi_s per group and measure R^2.
  6. Produce figures.

Reference: Cunningham et al. (2023) "Sparse Autoencoders Find Highly
Interpretable Features in Language Models" (arXiv:2309.08600).

Usage (from Implicit-Bias-Spurious/):
    python train_sae_analyze.py --data_dir ./data_colored_mnist
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
#  Sparse Autoencoder (tied weights, following the SAE paper)
# ============================================================

class SparseAutoencoder(nn.Module):
    """
    Tied-weight sparse autoencoder.

        c     = ReLU(M x + b)          (encoding)
        x_hat = M^T c                  (decoding with tied weights)

    The dictionary consists of the rows of M (each row f_k is a feature
    direction in the activation space).  The hidden dimension d_hid is
    the dictionary size and should be > d_in for an overcomplete basis.

    Loss = ||x - x_hat||^2  +  alpha * ||c||_1
    """

    def __init__(self, d_in, d_hid):
        super().__init__()
        self.d_in = d_in
        self.d_hid = d_hid

        # M: (d_hid, d_in) — rows are dictionary features
        self.M = nn.Parameter(torch.randn(d_hid, d_in) * 0.01)
        self.b = nn.Parameter(torch.zeros(d_hid))
        self.relu = nn.ReLU()

        # Normalize rows at init
        with torch.no_grad():
            self.M.div_(self.M.norm(dim=1, keepdim=True) + 1e-12)

    def encode(self, x):
        """Compute sparse coefficients c = ReLU(M x + b)."""
        return self.relu(x @ self.M.T + self.b)   # (batch, d_hid)

    def decode(self, c):
        """Reconstruct x_hat = M^T c."""
        return c @ self.M                          # (batch, d_in)

    def forward(self, x):
        c = self.encode(x)
        x_hat = self.decode(c)
        return x_hat, c

    def normalize_features(self):
        """Normalize dictionary rows to unit norm (prevents trivial sparsity
        reduction by scaling up M)."""
        with torch.no_grad():
            norms = self.M.norm(dim=1, keepdim=True)
            self.M.div_(norms + 1e-12)


# ============================================================
#  SAE training
# ============================================================

def train_sae(model, data_loader, epochs, lr, alpha, device, verbose=True):
    """
    Train SAE with reconstruction + L1 sparsity loss.

    Parameters
    ----------
    alpha : float
        Weight on the L1 sparsity penalty.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1, epochs + 1):
        total_recon, total_l1, total_n = 0.0, 0.0, 0
        for (X_batch,) in data_loader:
            X_batch = X_batch.to(device)
            x_hat, c = model(X_batch)

            recon_loss = ((X_batch - x_hat) ** 2).sum(dim=1).mean()
            l1_loss = c.abs().sum(dim=1).mean()
            loss = recon_loss + alpha * l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Re-normalize dictionary rows after each step
            model.normalize_features()

            total_recon += recon_loss.item() * X_batch.size(0)
            total_l1 += l1_loss.item() * X_batch.size(0)
            total_n += X_batch.size(0)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            avg_r = total_recon / total_n
            avg_l1 = total_l1 / total_n
            # Fraction of variance unexplained
            print(f"  Epoch {epoch:>3d}/{epochs}  "
                  f"recon={avg_r:.4f}  L1={avg_l1:.4f}")

    return model


# ============================================================
#  Feature classification: r-related vs s-related
# ============================================================

def classify_features(c_all, digit_classes, intensity_used, threshold=0.1):
    """
    Classify each SAE feature as r-related, s-related, or mixed, based on
    Pearson correlation with digit class (proxy for r) and background
    intensity (proxy for s).

    Parameters
    ----------
    c_all : ndarray (N, d_hid), SAE activations for all samples
    digit_classes : ndarray (N,), 0-9
    intensity_used : ndarray (N,), actual background intensity
    threshold : float, minimum |correlation| to assign a feature

    Returns
    -------
    feature_type : list of str, one per feature ('r', 's', 'mixed', 'dead')
    corr_digit : ndarray (d_hid,), correlation with digit class
    corr_intensity : ndarray (d_hid,), correlation with intensity
    """
    d_hid = c_all.shape[1]
    corr_digit = np.zeros(d_hid)
    corr_intensity = np.zeros(d_hid)
    feature_type = []

    for k in range(d_hid):
        ck = c_all[:, k]
        # Skip dead features (never activate)
        if ck.std() < 1e-8:
            corr_digit[k] = 0.0
            corr_intensity[k] = 0.0
            feature_type.append("dead")
            continue

        corr_digit[k] = np.corrcoef(ck, digit_classes)[0, 1]
        corr_intensity[k] = np.corrcoef(ck, intensity_used)[0, 1]

        abs_d = abs(corr_digit[k])
        abs_i = abs(corr_intensity[k])

        if abs_d >= threshold and abs_i < threshold:
            feature_type.append("r")
        elif abs_i >= threshold and abs_d < threshold:
            feature_type.append("s")
        elif abs_d >= threshold and abs_i >= threshold:
            feature_type.append("mixed")
        else:
            feature_type.append("weak")

    return feature_type, corr_digit, corr_intensity


# ============================================================
#  Decompose into phi_r and phi_s, measure linearity
# ============================================================

def decompose_and_measure(c_all, feature_type, groups):
    """
    Decompose phi(x) into r-activations and s-activations using the SAE
    feature classification, then fit a linear map phi_r -> phi_s per group.

    phi_r = [c_k for k where feature_type[k] == 'r']
    phi_s = [c_k for k where feature_type[k] == 's']

    Returns
    -------
    results : dict with R^2 scores, number of features, etc.
    """
    r_idx = [k for k, t in enumerate(feature_type) if t == "r"]
    s_idx = [k for k, t in enumerate(feature_type) if t == "s"]
    mixed_idx = [k for k, t in enumerate(feature_type) if t == "mixed"]
    dead_idx = [k for k, t in enumerate(feature_type) if t == "dead"]
    weak_idx = [k for k, t in enumerate(feature_type) if t == "weak"]

    print(f"\n  Feature classification:")
    print(f"    r-features:     {len(r_idx)}")
    print(f"    s-features:     {len(s_idx)}")
    print(f"    mixed features: {len(mixed_idx)}")
    print(f"    weak features:  {len(weak_idx)}")
    print(f"    dead features:  {len(dead_idx)}")

    if len(r_idx) == 0 or len(s_idx) == 0:
        print("  WARNING: not enough r or s features to measure linearity.")
        return {"r2_maj": None, "r2_min": None,
                "n_r": len(r_idx), "n_s": len(s_idx)}

    phi_r = c_all[:, r_idx]   # (N, n_r)
    phi_s = c_all[:, s_idx]   # (N, n_s)

    results = {"n_r": len(r_idx), "n_s": len(s_idx),
               "n_mixed": len(mixed_idx), "n_dead": len(dead_idx),
               "n_weak": len(weak_idx),
               "r_idx": r_idx, "s_idx": s_idx}

    # Fit linear map phi_r -> phi_s per group
    for g, name in [(0, "majority"), (1, "minority")]:
        mask = (groups == g)
        if mask.sum() < 10:
            results[f"r2_{name}"] = None
            continue

        probe = Ridge(alpha=1.0)
        probe.fit(phi_r[mask], phi_s[mask])
        phi_s_hat = probe.predict(phi_r[mask])
        r2 = r2_score(phi_s[mask], phi_s_hat,
                       multioutput="variance_weighted")
        results[f"r2_{name}"] = float(r2)
        print(f"    Linear map phi_r -> phi_s ({name}):  R^2 = {r2:.4f}")

    # Also fit on pooled data (ignoring group)
    probe_all = Ridge(alpha=1.0)
    probe_all.fit(phi_r, phi_s)
    phi_s_hat_all = probe_all.predict(phi_r)
    r2_all = r2_score(phi_s, phi_s_hat_all, multioutput="variance_weighted")
    results["r2_pooled"] = float(r2_all)
    print(f"    Linear map phi_r -> phi_s (pooled):  R^2 = {r2_all:.4f}")

    return results


# ============================================================
#  Figures
# ============================================================

def plot_feature_correlations(corr_digit, corr_intensity, feature_type,
                              out_path):
    """Scatter plot of per-feature correlations: digit vs intensity."""
    colors = {"r": "tab:blue", "s": "tab:red", "mixed": "tab:purple",
              "weak": "lightgray", "dead": "black"}

    fig, ax = plt.subplots(figsize=(7, 6))
    for ftype in ["weak", "dead", "mixed", "s", "r"]:
        idx = [k for k, t in enumerate(feature_type) if t == ftype]
        if idx:
            ax.scatter(corr_digit[idx], corr_intensity[idx],
                       s=12, alpha=0.6, c=colors[ftype], label=ftype)

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("Correlation with digit class (r-proxy)", fontsize=11)
    ax.set_ylabel("Correlation with background intensity (s-proxy)", fontsize=11)
    ax.set_title("SAE feature classification", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Feature correlation plot saved to {out_path}")


def plot_phi_r_vs_phi_s(c_all, r_idx, s_idx, groups, out_path):
    """
    Scatter plot: first principal direction of phi_r vs first principal
    direction of phi_s, colored by group.
    """
    phi_r = c_all[:, r_idx]
    phi_s = c_all[:, s_idx]

    # Use the first SVD component as a 1D summary
    def first_pc(X):
        X_centered = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return X_centered @ Vt[0]

    pr = first_pc(phi_r)   # (N,)
    ps = first_pc(phi_s)   # (N,)

    fig, ax = plt.subplots(figsize=(7, 6))
    maj = (groups == 0)
    mino = (groups == 1)

    # Subsample for readability
    rng = np.random.RandomState(0)
    max_pts = 3000
    for mask, color, label in [
        (maj, "tab:blue", "Majority (maroon)"),
        (mino, "tab:red", "Minority (teal)"),
    ]:
        idx = np.where(mask)[0]
        if len(idx) > max_pts:
            idx = rng.choice(idx, max_pts, replace=False)
        ax.scatter(pr[idx], ps[idx], s=6, alpha=0.3, c=color, label=label)

    ax.set_xlabel(r"First PC of $\varphi_r$ (r-feature activations)", fontsize=11)
    ax.set_ylabel(r"First PC of $\varphi_s$ (s-feature activations)", fontsize=11)
    ax.set_title(r"$\varphi_r$ vs $\varphi_s$ in SAE feature space", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  phi_r vs phi_s scatter saved to {out_path}")


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train SAE on MLP representations and analyze "
                    "spurious correlation preservation.")
    parser.add_argument("--data_dir", type=str, default="./data_colored_mnist")
    parser.add_argument("--out_dir", type=str, default=None)
    # SAE hyperparameters
    parser.add_argument("--dict_ratio", type=int, default=4,
                        help="Ratio of dictionary size to input dim (R)")
    parser.add_argument("--alpha", type=float, default=1e-3,
                        help="L1 sparsity coefficient")
    parser.add_argument("--sae_epochs", type=int, default=30)
    parser.add_argument("--sae_lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=512)
    # Feature classification
    parser.add_argument("--corr_threshold", type=float, default=0.1,
                        help="Min |correlation| to classify a feature as r or s")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.data_dir
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    #  Load representations
    # ------------------------------------------------------------------
    print("[1/5] Loading representations...")
    rep = np.load(os.path.join(args.data_dir, "representations.npz"))
    phi = rep["phi"]                       # (N, 128)
    y = rep["y"]
    groups = rep["groups"]
    digit_classes = rep["digit_classes"]
    intensity_used = rep["intensity_used"]

    N, d_in = phi.shape
    d_hid = d_in * args.dict_ratio
    print(f"  N={N}, d_in={d_in}, dictionary size={d_hid} (R={args.dict_ratio})")

    phi_tensor = torch.from_numpy(phi).float()
    ds = TensorDataset(phi_tensor)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # ------------------------------------------------------------------
    #  Train SAE
    # ------------------------------------------------------------------
    print(f"\n[2/5] Training SAE (alpha={args.alpha}, "
          f"{args.sae_epochs} epochs)...")
    sae = SparseAutoencoder(d_in=d_in, d_hid=d_hid).to(device)
    train_sae(sae, loader, args.sae_epochs, args.sae_lr, args.alpha, device)

    # Save SAE
    sae_path = os.path.join(args.out_dir, "sae_model.pt")
    torch.save(sae.state_dict(), sae_path)
    print(f"  SAE saved to {sae_path}")

    # ------------------------------------------------------------------
    #  Encode all samples
    # ------------------------------------------------------------------
    print("\n[3/5] Encoding all samples through SAE...")
    sae.eval()
    c_list = []
    with torch.no_grad():
        for i in range(0, N, 2048):
            batch = phi_tensor[i:i+2048].to(device)
            c_batch = sae.encode(batch)
            c_list.append(c_batch.cpu().numpy())
    c_all = np.concatenate(c_list, axis=0)   # (N, d_hid)

    active = (c_all > 0).any(axis=0).sum()
    avg_active = (c_all > 0).sum(axis=1).mean()
    print(f"  Active features: {active}/{d_hid}")
    print(f"  Avg features active per sample: {avg_active:.1f}")

    # Reconstruction quality
    with torch.no_grad():
        phi_hat = sae.decode(torch.from_numpy(c_all).float().to(device))
        phi_hat = phi_hat.cpu().numpy()
    total_var = ((phi - phi.mean(axis=0)) ** 2).sum()
    resid_var = ((phi - phi_hat) ** 2).sum()
    frac_explained = 1 - resid_var / total_var
    print(f"  Variance explained: {frac_explained:.4f}")

    # ------------------------------------------------------------------
    #  Classify features
    # ------------------------------------------------------------------
    print("\n[4/5] Classifying SAE features...")
    feature_type, corr_digit, corr_intensity = classify_features(
        c_all, digit_classes, intensity_used,
        threshold=args.corr_threshold)

    plot_feature_correlations(
        corr_digit, corr_intensity, feature_type,
        os.path.join(args.out_dir, "sae_feature_correlations.png"))

    # ------------------------------------------------------------------
    #  Decompose and measure linearity
    # ------------------------------------------------------------------
    print("\n[5/5] Decomposing phi into phi_r and phi_s...")
    results = decompose_and_measure(c_all, feature_type, groups)

    if results.get("r_idx") and results.get("s_idx"):
        plot_phi_r_vs_phi_s(
            c_all, results["r_idx"], results["s_idx"], groups,
            os.path.join(args.out_dir, "phi_r_vs_phi_s.png"))

    # Save summary
    summary = {
        "sae": {
            "d_in": d_in, "d_hid": d_hid, "dict_ratio": args.dict_ratio,
            "alpha": args.alpha, "epochs": args.sae_epochs,
            "active_features": int(active),
            "avg_active_per_sample": float(avg_active),
            "variance_explained": float(frac_explained),
        },
        "features": {
            "n_r": results["n_r"], "n_s": results["n_s"],
            "n_mixed": results.get("n_mixed", 0),
            "n_dead": results.get("n_dead", 0),
            "n_weak": results.get("n_weak", 0),
        },
        "linearity": {
            "r2_majority": results.get("r2_majority"),
            "r2_minority": results.get("r2_minority"),
            "r2_pooled": results.get("r2_pooled"),
        },
        "corr_threshold": args.corr_threshold,
    }
    summary_path = os.path.join(args.out_dir, "sae_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
