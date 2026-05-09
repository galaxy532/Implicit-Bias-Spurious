"""
scenario3_convolution.py — Feature-mediated spurious correlation via group-specific spatial filtering

Validates the isotropic regime (Definition 2, Theorem 2) on real images from Fashion-MNIST.
  - Causal features r:  clothing item images (28×28, vectorised to R^784)
  - Spurious features s: circular convolution of r with a group-specific Gaussian kernel
      Majority (1-ε):  s = conv(r, Gaussian(σ_A)) + ξ
      Minority (ε):    s = conv(r, Gaussian(σ_B)) + ξ
  - Full input: x = [r; s] ∈ R^1568, label-absorbed (x ← y·x)

The circular convolution makes A, B block-circulant ⇒ diagonalised by the 2D DFT.
Every Fourier mode is a common eigenvector, so the isotropic condition holds
exactly if v is a Fourier mode, and approximately otherwise.

Pipeline:
  1. Load Fashion-MNIST, select binary classes
  2. Assign groups, apply group-specific circular blur
  3. Save sample images for the paper
  4. Compute v (max-margin direction on r-space via SVM)
  5. Check the isotropic condition via FFT
  6. Train a linear model with full-batch GD + logistic loss
  7. Compare empirical error decay with theoretical predictions
  8. Save all figures and results

Usage:
    python scenario3_convolution.py --sigma_A 1.0 --sigma_B 3.0 --epsilon 0.1
    python scenario3_convolution.py --dataset MNIST --class_pos 3 --class_neg 8
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===========================================================
#  Configuration
# ===========================================================

def get_config():
    p = argparse.ArgumentParser(description="Scenario 3: convolution experiment")

    # Dataset
    p.add_argument('--dataset', type=str, default='FashionMNIST',
                   choices=['FashionMNIST', 'MNIST'])
    p.add_argument('--class_pos', type=int, default=9,
                   help='Positive class (9=Ankle boot for FashionMNIST)')
    p.add_argument('--class_neg', type=int, default=0,
                   help='Negative class (0=T-shirt for FashionMNIST)')
    p.add_argument('--N', type=int, default=10000,
                   help='Number of samples (subsampled from train set)')

    # Blur
    p.add_argument('--sigma_A', type=float, default=1.0,
                   help='Gaussian blur σ for majority group')
    p.add_argument('--sigma_B', type=float, default=3.0,
                   help='Gaussian blur σ for minority group')
    p.add_argument('--noise_std', type=float, default=0.02,
                   help='Std of small bounded additive noise on s')

    # Training
    p.add_argument('--epsilon', type=float, default=0.1,
                   help='Minority fraction')
    p.add_argument('--lr', type=float, default=0.01,
                   help='Learning rate (constant)')
    p.add_argument('--steps', type=int, default=200000,
                   help='Number of full-batch GD steps')

    # Logging
    p.add_argument('--log_every', type=int, default=500)
    p.add_argument('--print_every', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', type=str, default='./scenario3_results')

    return p.parse_args()


# ===========================================================
#  FFT-based circular convolution
# ===========================================================

def make_gaussian_kernel_2d(size, sigma):
    """
    Discrete 2D Gaussian kernel on a (size × size) grid,
    shifted so that the peak sits at index [0, 0] (ready for FFT).
    """
    ax = np.arange(size, dtype=np.float64)
    center = size / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    # wrap so that peak is at (0, 0)
    kernel = np.roll(kernel, -int(center), axis=0)
    kernel = np.roll(kernel, -int(center), axis=1)
    return kernel


def fft_blur_batch(images_np, kernel_fft):
    """
    Circular convolution via FFT.
    images_np : (N, H, W)  real
    kernel_fft: (H, W)     complex  (pre-computed FFT of the kernel)
    """
    F = np.fft.fft2(images_np, axes=(-2, -1))
    return np.real(np.fft.ifft2(F * kernel_fft[None, :, :], axes=(-2, -1)))


# ===========================================================
#  Data loading
# ===========================================================

def load_data(cfg):
    """
    Load dataset → select two classes → assign groups →
    apply circular blur → absorb labels → return everything.
    """
    transform = transforms.ToTensor()
    DS = torchvision.datasets.FashionMNIST if cfg.dataset == 'FashionMNIST' \
         else torchvision.datasets.MNIST
    ds = DS(root='./data', train=True, download=True, transform=transform)

    # binary mask
    mask = (ds.targets == cfg.class_pos) | (ds.targets == cfg.class_neg)
    images = ds.data[mask].float() / 255.0          # (N_all, 28, 28)
    labels = ((ds.targets[mask] == cfg.class_pos).float() * 2 - 1)  # +1 / -1

    # sub-sample
    torch.manual_seed(cfg.seed)
    N_use = min(cfg.N, len(images))
    perm  = torch.randperm(len(images))[:N_use]
    images, labels = images[perm], labels[perm]
    N = len(images)

    # assign groups
    torch.manual_seed(cfg.seed + 1)
    groups = (torch.rand(N) < cfg.epsilon).long()

    # build blur kernels
    H = W = 28
    kern_A     = make_gaussian_kernel_2d(H, cfg.sigma_A)
    kern_B     = make_gaussian_kernel_2d(H, cfg.sigma_B)
    kern_A_fft = np.fft.fft2(kern_A)
    kern_B_fft = np.fft.fft2(kern_B)

    # apply blur per group
    img_np = images.numpy()
    s_np   = np.zeros_like(img_np)
    maj, minn = (groups == 0).numpy(), (groups == 1).numpy()
    if maj.sum() > 0:
        s_np[maj] = fft_blur_batch(img_np[maj], kern_A_fft)
    if minn.sum() > 0:
        s_np[minn] = fft_blur_batch(img_np[minn], kern_B_fft)
    s_images = torch.from_numpy(s_np).float()

    # bounded additive noise
    if cfg.noise_std > 0:
        xi = torch.randn(N, H, W) * cfg.noise_std
        xi = xi.clamp(-3 * cfg.noise_std, 3 * cfg.noise_std)
        s_images = s_images + xi

    # vectorise
    r_vec = images.reshape(N, -1)           # (N, 784)
    s_vec = s_images.reshape(N, -1)         # (N, 784)

    # keep pre-absorption copies for SVM and plotting
    r_orig, s_orig, labels_orig = r_vec.clone(), s_vec.clone(), labels.clone()

    # label absorption: x ← y·x
    r_abs = r_vec * labels.unsqueeze(1)
    s_abs = s_vec * labels.unsqueeze(1)
    x     = torch.cat([r_abs, s_abs], dim=1)   # (N, 1568)

    return dict(
        x=x, r_abs=r_abs, s_abs=s_abs,
        r_orig=r_orig, s_orig=s_orig, labels=labels_orig,
        images=images, s_images=s_images, groups=groups,
        kern_A_fft=kern_A_fft, kern_B_fft=kern_B_fft,
        N=N, n_maj=int(maj.sum()), n_min=int(minn.sum()),
    )


# ===========================================================
#  Sample images for the paper
# ===========================================================

CLASS_NAMES_FMNIST = {
    0: 'T-shirt', 1: 'Trouser', 2: 'Pullover', 3: 'Dress',
    4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker',
    8: 'Bag', 9: 'Ankle boot',
}

def save_sample_images(data, cfg):
    fig_dir = Path(cfg.out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    images, s_images = data['images'], data['s_images']
    groups, labels   = data['groups'], data['labels']
    n_ex = 5

    name = CLASS_NAMES_FMNIST if cfg.dataset == 'FashionMNIST' else {}
    pos_name = name.get(cfg.class_pos, str(cfg.class_pos))
    neg_name = name.get(cfg.class_neg, str(cfg.class_neg))

    fig, axes = plt.subplots(4, n_ex, figsize=(2.8 * n_ex, 11))

    row_labels = [
        f'Majority (σ={cfg.sigma_A}) — {pos_name}',
        f'Majority (σ={cfg.sigma_A}) — {neg_name}',
        f'Minority (σ={cfg.sigma_B}) — {pos_name}',
        f'Minority (σ={cfg.sigma_B}) — {neg_name}',
    ]
    conditions = [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0)]

    for row, (g, c) in enumerate(conditions):
        idxs = ((groups == g) & (labels == c)).nonzero(as_tuple=True)[0][:n_ex]
        for j, idx in enumerate(idxs):
            r_img = images[idx].numpy()
            s_img = s_images[idx].numpy()
            sep   = np.full((28, 2), 0.5)
            composite = np.concatenate([r_img, sep, s_img], axis=1)

            ax = axes[row, j]
            ax.imshow(composite, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
        axes[row, 0].set_ylabel(row_labels[row], fontsize=9, rotation=0,
                                 labelpad=130, va='center')

    fig.suptitle('Original  r  |  Filtered  s', fontsize=13, y=0.93)
    fig.tight_layout(rect=[0.18, 0, 1, 0.92])
    fig.savefig(fig_dir / 'sample_images.png', dpi=200, bbox_inches='tight')
    fig.savefig(fig_dir / 'sample_images.pdf', bbox_inches='tight')
    plt.close(fig)

    # individual pairs (for fine paper-figure control)
    ind = fig_dir / "individual"
    ind.mkdir(exist_ok=True)
    for g, c in conditions:
        idxs = ((groups == g) & (labels == c)).nonzero(as_tuple=True)[0][:3]
        for j, idx in enumerate(idxs):
            for tag, arr in [('r', images[idx].numpy()),
                             ('s', s_images[idx].numpy())]:
                np.save(ind / f'{tag}_g{g}_c{int(c)}_ex{j}.npy', arr)
                fig_s, ax_s = plt.subplots(figsize=(2, 2))
                ax_s.imshow(arr, cmap='gray', vmin=0, vmax=1)
                ax_s.axis('off')
                fig_s.savefig(ind / f'{tag}_g{g}_c{int(c)}_ex{j}.png',
                              dpi=150, bbox_inches='tight', pad_inches=0.02)
                plt.close(fig_s)
    print(f"  Sample images saved to {fig_dir}")


# ===========================================================
#  SVM direction v on r-space
# ===========================================================

def compute_v_svm(data, cfg):
    """
    Hard-margin SVM on the *pre-absorption* data (r_orig, labels).
    Returns v (unit-norm), per-group margins γ̃_g (rescaled so min=1).
    """
    from sklearn.svm import LinearSVC

    r = data['r_orig'].numpy()
    y = data['labels'].numpy()

    svm = LinearSVC(C=1e5, loss='hinge', max_iter=50000,
                    fit_intercept=False, dual='auto')
    svm.fit(r, y)
    v = svm.coef_[0].astype(np.float64)
    v /= np.linalg.norm(v)

    # functional margins y_i (v^T r_i)
    margins = y * (r @ v)
    # scale v so that the smallest margin equals 1
    m_min = margins.min()
    v     = v / m_min
    margins = margins / m_min

    g = data['groups'].numpy()
    gm_maj = margins[g == 0].min()
    gm_min = margins[g == 1].min()

    # normalise so min(γ_maj, γ_min) = 1
    sc     = min(gm_maj, gm_min)
    v      = v / sc
    gm_maj /= sc
    gm_min /= sc

    acc = (svm.predict(r) == y).mean()
    print(f"  SVM accuracy on r: {acc:.4f}")
    print(f"  γ̃_maj = {gm_maj:.4f},  γ̃_min = {gm_min:.4f}")
    return v, gm_maj, gm_min, acc


# ===========================================================
#  Isotropic-condition check via FFT
# ===========================================================

def check_isotropic(v, kern_A_fft, kern_B_fft, img_size=28):
    """
    For circulant A, B with DFT coefficients H_A, H_B:
      (A^T A v)_k = |H_A(k)|^2 V_k          ⇒ eigenvalue |H_A(k)|^2
      (A^T B v)_k = H_A(k) H_B(k) V_k       ⇒ eigenvalue H_A(k)H_B(k)

    Isotropic ⟺ these "eigenvalues" are constant over the support of V.
    We measure the power-weighted variance of each spectrum.
    """
    V = np.fft.fft2(v.reshape(img_size, img_size))
    Vp = np.abs(V) ** 2                 # power spectrum of v
    tot = Vp.sum()

    # for real symmetric Gaussian kernels H is real
    H_A = np.real(kern_A_fft)
    H_B = np.real(kern_B_fft)

    spec = {
        'AtA': H_A ** 2,
        'BtB': H_B ** 2,
        'AtB': H_A * H_B,
    }

    def weighted_stats(s):
        mu  = np.sum(s * Vp) / tot
        var = np.sum((s - mu) ** 2 * Vp) / tot
        mv2 = np.sum(s ** 2 * Vp) / tot
        rel = np.sqrt(var / (mv2 + 1e-30))
        return mu, rel

    mu_A,  res_AtA = weighted_stats(spec['AtA'])
    mu_B,  res_BtB = weighted_stats(spec['BtB'])
    mu,    res_AtB = weighted_stats(spec['AtB'])

    # frequency concentration of v
    sorted_p = np.sort(Vp.ravel())[::-1]
    cum = np.cumsum(sorted_p) / tot
    n90 = int(np.searchsorted(cum, 0.9)) + 1
    n95 = int(np.searchsorted(cum, 0.95)) + 1

    return dict(
        mu_A=float(mu_A), mu_B=float(mu_B), mu=float(mu),
        res_AtA=float(res_AtA), res_BtB=float(res_BtB),
        res_AtB=float(res_AtB), res_BtA=float(res_AtB),
        max_residual=float(max(res_AtA, res_BtB, res_AtB)),
        freq_modes_90pct=n90, freq_modes_95pct=n95,
        total_modes=img_size ** 2,
    )


# ===========================================================
#  Training: full-batch GD (logistic loss, label-absorbed)
# ===========================================================

def train_gd(data, cfg):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = data['x'].to(device)
    groups = data['groups'].to(device)
    N, d = X.shape
    w = torch.zeros(d, device=device)

    maj = (groups == 0)
    minn = (groups == 1)

    logs = dict(t=[], err_maj=[], err_min=[], loss=[],
                err_maj_resc=[], err_min_resc=[])
    t0 = time.time()

    for t in range(1, cfg.steps + 1):
        logit = X @ w
        q = 1.0 - torch.sigmoid(logit)          # 1 - p_y
        w = w + cfg.lr * (X.T @ q) / N

        if t % cfg.log_every == 0 or t == 1:
            with torch.no_grad():
                em = q[maj].mean().item()
                en = q[minn].mean().item()
                lo = torch.log1p(torch.exp(-logit)).mean().item()
                z  = cfg.lr * t
                logs['t'].append(t)
                logs['err_maj'].append(em)
                logs['err_min'].append(en)
                logs['loss'].append(lo)
                logs['err_maj_resc'].append(em * (1 - cfg.epsilon) * z)
                logs['err_min_resc'].append(en * cfg.epsilon * z)
            if t % cfg.print_every == 0 or t == 1:
                print(f"    [t={t:>7d}] loss={lo:.6f}  "
                      f"err_maj={em:.6f}  err_min={en:.6f}  "
                      f"({time.time()-t0:.0f}s)")
    return logs, w.cpu()


# ===========================================================
#  Plotting
# ===========================================================

def plot_results(logs, theory, cfg):
    fig_dir = Path(cfg.out_dir) / "figures"
    fig_dir.mkdir(exist_ok=True)
    alpha = theory['alpha']

    t   = np.array(logs['t'])
    z   = cfg.lr * t
    em  = np.array(logs['err_maj'])
    en  = np.array(logs['err_min'])

    # ---- error decay (log-log) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ok = en > 1e-14
    ax1.loglog(z[ok], en[ok], 'b-', lw=1.2, label='Minority (empirical)')
    k = max(1, ok.sum() // 4)
    if alpha < 1:
        ref = en[ok][k] * z[ok][k] / z[ok]
        ax1.loglog(z[ok], ref, 'b--', alpha=.5,
                   label=f'$1/z_t$  (α={alpha:.2f}<1)')
    else:
        ref = en[ok][k] * (z[ok][k] / z[ok]) ** alpha
        ax1.loglog(z[ok], ref, 'b--', alpha=.5,
                   label=f'$z_t^{{-{alpha:.2f}}}$')
    ax1.set(xlabel='$z_t = h\\,t$', ylabel='Classification error',
            title='Minority error')
    ax1.legend(); ax1.grid(True, alpha=.3)

    ok = em > 1e-14
    ax2.loglog(z[ok], em[ok], 'r-', lw=1.2, label='Majority (empirical)')
    k = max(1, ok.sum() // 4)
    ref = em[ok][k] * z[ok][k] / z[ok]
    ax2.loglog(z[ok], ref, 'r--', alpha=.5, label='$1/z_t$')
    ax2.set(xlabel='$z_t$', ylabel='Classification error',
            title='Majority error')
    ax2.legend(); ax2.grid(True, alpha=.3)

    fig.suptitle(f'Error decay — α={alpha:.3f}, ε={cfg.epsilon}, '
                 f'σ_A={cfg.sigma_A}, σ_B={cfg.sigma_B}', fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / 'error_decay.png', dpi=200, bbox_inches='tight')
    fig.savefig(fig_dir / 'error_decay.pdf', bbox_inches='tight')
    plt.close(fig)

    # ---- rescaled errors ----
    fig2, ax = plt.subplots(figsize=(8, 5))
    emr = np.array(logs['err_maj_resc'])
    enr = np.array(logs['err_min_resc'])
    if alpha < 1:
        ax.plot(z, emr, 'r-', alpha=.8, label='Maj rescaled: err·(1-ε)·z')
        ax.plot(z, enr, 'b-', alpha=.8, label='Min rescaled: err·ε·z')
        if theory.get('kappa_maj') is not None:
            ax.axhline(theory['kappa_maj'], color='r', ls='--', alpha=.5,
                       label=f"κ_maj = {theory['kappa_maj']:.4f}")
        if theory.get('kappa_min') is not None:
            ax.axhline(theory['kappa_min'], color='b', ls='--', alpha=.5,
                       label=f"κ_min = {theory['kappa_min']:.4f}")
        ax.set_title(f'Rescaled errors (α={alpha:.3f} < 1)')
    else:
        ax.plot(z, emr, 'r-', alpha=.8, label='Maj rescaled')
        ax.set_title(f'Rescaled errors (α={alpha:.3f} ≥ 1)')
    ax.set(xlabel='$z_t$', ylabel='Rescaled error')
    ax.legend(); ax.grid(True, alpha=.3)
    fig2.tight_layout()
    fig2.savefig(fig_dir / 'rescaled_errors.png', dpi=200, bbox_inches='tight')
    plt.close(fig2)

    # ---- log-log slope estimation ----
    fig3, ax = plt.subplots(figsize=(7, 5))
    ok = en > 1e-14
    log_z = np.log(z[ok])
    log_e = np.log(en[ok])
    # fit slope on last 50 % of trajectory
    half = len(log_z) // 2
    slope, intercept = np.polyfit(log_z[half:], log_e[half:], 1)
    ax.plot(log_z, log_e, 'b-', lw=1, label='Minority log-error')
    ax.plot(log_z[half:],
            slope * log_z[half:] + intercept,
            'k--', lw=2, label=f'Fitted slope = {slope:.3f}')
    ax.axhline(0, color='gray', lw=.5)
    ax.set(xlabel='log $z_t$', ylabel='log(error)',
           title=f'Minority slope check  (theory: {-min(alpha,1) if alpha<1 else -alpha:.3f})')
    ax.legend(); ax.grid(True, alpha=.3)
    fig3.tight_layout()
    fig3.savefig(fig_dir / 'slope_check.png', dpi=200, bbox_inches='tight')
    plt.close(fig3)

    print(f"  Empirical minority slope: {slope:.4f}  "
          f"(theory: {-min(alpha, 1) if alpha < 1 else -alpha:.4f})")
    print(f"  Plots saved to {fig_dir}")
    return slope


# ===========================================================
#  Theoretical predictions
# ===========================================================

def compute_theory(iso, tgm_maj, tgm_min, cfg):
    """Compute α and κ from the eigenvalues and margins."""
    ma, mb, m = iso['mu_A'], iso['mu_B'], iso['mu']

    if tgm_maj <= tgm_min:
        alpha = tgm_min * (1 + m) / (1 + ma)
    else:
        alpha = tgm_maj * (1 + m) / (1 + mb)

    Sigma = (1 + ma) * (1 + mb) - (1 + m) ** 2
    th = dict(alpha=alpha, Sigma=Sigma,
              tgamma_maj=tgm_maj, tgamma_min=tgm_min)
    th.update(iso)

    gm = max(tgm_maj, tgm_min)
    if alpha < 1 and Sigma > 1e-12:
        th['kappa_maj'] = (gm * (1 + mb) - (1 + m)) / (gm * Sigma)
        th['kappa_min'] = ((1 + ma) - gm * (1 + m)) / (gm ** 2 * Sigma)
    else:
        th['kappa_maj'] = 1.0 / (1 + ma)
        th['kappa_min'] = None
    return th


# ===========================================================
#  Main
# ===========================================================

def main():
    cfg = get_config()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Scenario 3 — Convolution experiment (Fashion-MNIST)")
    print("=" * 65)

    # 1. data
    print("\n[1/6] Loading data + group-specific circular blur ...")
    data = load_data(cfg)
    print(f"  N = {data['N']}  (maj {data['n_maj']}, min {data['n_min']})")

    # 2. sample images
    print("\n[2/6] Saving sample images ...")
    save_sample_images(data, cfg)

    # 3. SVM direction
    print("\n[3/6] Computing max-margin direction v on r ...")
    v, tgm_maj, tgm_min, svm_acc = compute_v_svm(data, cfg)

    # 4. isotropic check
    print("\n[4/6] Isotropic condition (FFT) ...")
    iso = check_isotropic(v, data['kern_A_fft'], data['kern_B_fft'])
    theory = compute_theory(iso, tgm_maj, tgm_min, cfg)
    alpha = theory['alpha']

    print(f"  μ_A = {iso['mu_A']:.6f}")
    print(f"  μ_B = {iso['mu_B']:.6f}")
    print(f"  μ   = {iso['mu']:.6f}")
    print(f"  Residuals  AtA: {iso['res_AtA']:.6f}   "
          f"BtB: {iso['res_BtB']:.6f}   AtB: {iso['res_AtB']:.6f}")
    print(f"  Max residual: {iso['max_residual']:.6f}")
    print(f"  α = {alpha:.4f}")
    print(f"  v freq concentration: {iso['freq_modes_90pct']} modes for 90 % power "
          f"({iso['total_modes']} total)")
    if theory.get('kappa_min') is not None:
        print(f"  κ_maj = {theory['kappa_maj']:.6f},  κ_min = {theory['kappa_min']:.6f}")
    else:
        print(f"  κ_maj = {theory['kappa_maj']:.6f}  (α ≥ 1 regime)")

    # 5. train
    print(f"\n[5/6] Training full-batch GD for {cfg.steps} steps ...")
    logs, w = train_gd(data, cfg)

    # 6. plot
    print("\n[6/6] Generating figures ...")
    slope = plot_results(logs, theory, cfg)

    # persist
    results = dict(
        config={k: v for k, v in vars(cfg).items()},
        theory={k: v for k, v in theory.items()
                if isinstance(v, (int, float, type(None)))},
        svm_accuracy=svm_acc,
        empirical_slope=float(slope),
        final_err_maj=logs['err_maj'][-1],
        final_err_min=logs['err_min'][-1],
    )
    with open(Path(cfg.out_dir) / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    with open(Path(cfg.out_dir) / 'logs.json', 'w') as f:
        json.dump(logs, f)

    print(f"\n{'=' * 65}")
    print(f"  Done — all outputs in  {cfg.out_dir}")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
