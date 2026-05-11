# Experiments: Implicit Bias of GD under Spurious Correlations

Synthetic experiments verifying Theorems 1 & 2 of the paper.

## Setup

```bash
pip install torch numpy matplotlib scipy scikit-learn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Experiment B additionally requires `torchvision` (for MNIST download) and `scipy` (for convolution).

GPU is optional but speeds up large runs. The code auto-detects CUDA.

## Quick sanity check (< 3 min, CPU is fine)

```bash
# Experiment A: isotropic regime (alpha < 1 and alpha >= 1)
python run_isotropic.py --quick

# Experiment B: general regime on MNIST (convolution-based)
python run_general_regime.py --quick

# Experiment C: phase transition sweep
python run_phase_transition.py --quick

# Plot results
python plot_isotropic.py --run_root ./runs_synth --out_dir ./figures
```

Check `./figures/` for output PNGs. The `--quick` flag uses fewer steps (10-20k) and a small dataset (10k samples) to verify the code runs and the curves have the right shape. The asymptotics won't be fully converged.

## Full runs (for the paper)

```bash
# Experiment A: ~2-3 hours on GPU (3 seeds x 7 epsilons x 2 panels)
python run_isotropic.py --steps 500000 --N 100000

# Experiment B: ~1-2 hours on GPU (3 seeds x 5 epsilons, MNIST 70k images)
python run_general_regime.py --steps 200000

# Experiment C: ~2-3 hours on GPU (3 seeds x 15 gamma_min values)
python run_phase_transition.py --steps 500000 --N 100000

# Plot with tail-only view (cleaner asymptotics)
python plot_isotropic.py --t_min 50000
```

For longer runs (sharper convergence):
```bash
python run_isotropic.py --steps 2000000 --N 200000
```

Custom number of seeds:
```bash
python run_isotropic.py --seeds 1 2 3 4 5
python run_phase_transition.py --seeds 1 2 3 4 5
```

## What each script does

### `run_isotropic.py`

Runs Experiment A with two panels:

| Panel | Parameters | alpha | What it tests |
|-------|-----------|-------|---------------|
| `alpha_lt_1` | mu_A=1, mu_B=1, mu=0.4, gamma_min=1 | 0.70 | Both groups ~ kappa_g / (eps_g z_t) |
| `alpha_ge_1` | mu_A=1, mu_B=1, mu=0, gamma_min=3 | 1.50 | Minority ~ z_t^{-1.5}, eps-independent |

For each panel, sweeps epsilon in {0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5}.
Each (panel, epsilon) is repeated over multiple random seeds (default: 3).

**CLI options:**
- `--quick`: fast sanity check
- `--steps N`: override training steps
- `--N N`: override dataset size
- `--epsilons 0.01 0.05 0.1 0.5`: custom epsilon list
- `--seeds 1 2 3 4 5`: random seeds for error bars (default: [1, 2, 3])
- `--panels alpha_lt_1 alpha_ge_1`: which panels to run
- `--out_root DIR`: output directory

### `plot_isotropic.py`

Reads runs from `--run_root` and produces figures in `--out_dir`.
When multiple seeds are present, plots the mean curve with shaded +/-1 standard deviation bands.

- `{panel}_{group}_error_decay.png`: log-log error vs z_t with theory overlay
- `{panel}_{group}_rescaled.png`: rescaled error (should converge to kappa_g)
- `{panel}_eps_independence.png`: side-by-side minority (collapsed) vs majority (spread)

### `run_phase_transition.py`

Experiment C: fixes epsilon=0.1, sweeps gamma_min from 0.5 to 4.0.
For each gamma_min, runs over multiple seeds, measures the empirical decay exponent beta, and reports mean +/- std.

Produces `figures/phase_transition.png`: empirical exponent (with error bars) vs theory, showing the kink at alpha=1.

**CLI options:**
- `--quick`: fast version (7 values of gamma_min)
- `--plot_only`: re-plot from saved results without re-training
- `--epsilon E`: minority fraction (default 0.1)
- `--seeds 1 2 3 4 5`: random seeds for error bars (default: [1, 2, 3])

### `run_general_regime.py`

Experiment B: general regime validation on MNIST with convolution-based spurious features.
Uses raw 28×28 MNIST digit images as core features r, and generates spurious features s by applying group-specific directional Gaussian blur convolutions (horizontal for majority, vertical for minority) plus noise.

Steps: (1) build convolution kernels, (2) load MNIST, (3) generate s = Conv(r, kernel) + noise per group, (4) regress s on r to estimate A, B and verify R^2, (5) analyze regime (confirm non-isotropic), (6) train linear logistic GD on x = [r; s], (7) plot per-group error decay.

Produces `figures/general_regime_{group}_error_decay.png` and `figures/general_regime_analysis.png`.

**CLI options:**
- `--quick`: fast version (3 epsilons, 10k steps)
- `--plot_only`: re-plot from saved results without re-training
- `--noise_std F`: noise added to convolved features (default 0.05)
- `--kernel_size N`: convolution kernel size (default 7)
- `--kernel_sigma F`: Gaussian sigma (default 2.0)
- `--seeds 1 2 3 4 5`: random seeds for error bars (default: [1, 2, 3])

### `synth_utils.py`

Core library:
- `SynthConfig`: all experiment parameters
- `build_isotropic_operators(cfg)`: constructs A, B matrices satisfying the isotropic eigenvalue conditions
- `generate_dataset(cfg, A, B, v)`: samples (x, group) from the paper's data model (label-absorbed)
- `train_population_gd(cfg, X, groups)`: full-batch GD training loop with logging
- `kappa_theory(cfg)`: closed-form kappa_maj, kappa_min from Theorem 2

## File structure

```
.
├── synth_utils.py            # Core utilities
├── run_isotropic.py          # Experiment A
├── run_general_regime.py     # Experiment B
├── run_phase_transition.py   # Experiment C
├── plot_isotropic.py         # Plotting (with error bands)
├── runs_synth/               # Training logs (created by scripts)
│   ├── alpha_lt_1/eps_*/seed_*/
│   ├── alpha_ge_1/eps_*/seed_*/
│   ├── general_regime/eps_*/seed_*/
│   └── phase_transition/gm_*/seed_*/
└── figures/                  # Output plots (created by scripts)
```

## Statistical significance

All experiments are repeated over 3 random seeds (configurable via `--seeds`).
The source of variability is the random draw of the synthetic dataset (the training algorithm is deterministic given the data).
Error bars / shaded bands show +/-1 standard deviation across seeds.
For full-batch GD on large datasets (N=100k), the variance across seeds is expected to be very small, confirming that results are not artifacts of a particular draw.

## Compute resources

All experiments were run on a single NVIDIA A6000 GPU. Each full run (Experiment A or C with 3 seeds) completed in approximately 20 hours. Total compute: 1 GPU-hours.

## Expected results

**Panel 1 (alpha=0.70):** All epsilon curves decay as 1/z_t with different prefactors. Rescaled curves collapse to horizontal lines matching kappa_maj, kappa_min from the formulas.

**Panel 2 (alpha=1.50):** Minority curves for all epsilon values collapse onto a single z_t^{-1.5} curve (epsilon-independent). Majority curves remain spread by epsilon.

**Phase transition:** Empirical exponent = 1 for alpha < 1 (all rates are 1/z_t), then grows linearly as alpha for alpha >= 1.
