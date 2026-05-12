# Experiments: Implicit Bias of GD under Spurious Correlations

Synthetic experiments verifying Theorems 1 & 2 of the paper, using the
**Tabular + Vendor Score** setup (isotropic regime by construction).

## Setup

```bash
pip install torch numpy matplotlib scipy scikit-learn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

GPU is optional but recommended for large N. The code auto-detects CUDA.

## Scripts

| Script | Purpose |
|--------|---------|
| `synth_utils.py` | Core library: `SynthConfig`, data generation, GD training, theory predictions |
| `run_tabular_vendor.py` | **α-sweep**: fix ε=0.1, sweep α ∈ {0.5, 0.7, 1.0, 1.3, 1.5, 2.0} |
| `run_tabular_vendor_eps_sweep.py` | **ε-sweep**: fix α ∈ {0.7, 1.5}, sweep ε ∈ {0.01, 0.05, 0.1, 0.2, 0.5} |
| `replot_tabular_vendor.py` | Re-generate α-sweep plots with x-axis cropping (no GD re-run) |

## Quick sanity check (< 5 min, CPU is fine)

```bash
python run_tabular_vendor.py --quick
python run_tabular_vendor_eps_sweep.py --quick
```

## Full runs (for the paper)

### α-sweep (verifies asymptotic rates + phase transition)

```bash
# Default: 10M steps, N=5M, h=0.01, 6 alpha values (~several hours on GPU)
python run_tabular_vendor.py

# Re-plot with x-axis crop at z_t = 20000 (instant, no GD):
python replot_tabular_vendor.py --T_max 20000
```

Produces per-α plots (rescaled errors, compensated, error ratio) in
`tabular_vendor_results_crop/`, plus an α-sweep overlay and phase transition
summary.

### ε-sweep (verifies ε-dependence and ε-independence)

```bash
# Default: 10M steps, N=1M, 2 alphas × 5 epsilons
python run_tabular_vendor_eps_sweep.py

# Resume if interrupted (only run missing configs):
python run_tabular_vendor_eps_sweep.py --alphas 1.5 --epsilons 0.1 0.2 0.5

# Re-plot from saved logs:
python run_tabular_vendor_eps_sweep.py --plot_only
```

Produces minority/majority raw and rescaled error overlays in
`tabular_vendor_eps_sweep_results/alpha_*/`.

## Data model

All experiments use the **Tabular + Vendor Score** instantiation of the
isotropic regime (Definition 2 in the paper):

- Raw causal features r ∈ ℝ^{d_r} with label-absorbed margins
- Vendor risk scores s = A·r + ξ (majority) or s = B·r + ξ (minority)
- A, B share the same singular-vector basis but differ in singular values
- Full feature vector x = [r; s] ∈ ℝ^{d_r + d_s}

Parameters: μ_A = μ_B = 1, μ = 0, γ_min = 2·α. This isolates the margin
advantage as the sole driver of the phase transition.

## File structure

```
.
├── synth_utils.py                          # Core library
├── run_tabular_vendor.py                   # α-sweep experiment
├── run_tabular_vendor_eps_sweep.py         # ε-sweep experiment
├── replot_tabular_vendor.py                # Replotting with crop/overlay
├── tabular_vendor_results/                 # α-sweep raw logs
│   └── alpha_*/logs.json
├── tabular_vendor_results_crop/            # α-sweep cropped figures
│   ├── alpha_*/rescaled_errors.{png,pdf}
│   ├── alpha_sweep_overlay.{png,pdf}
│   └── phase_transition_summary.{png,pdf}
└── tabular_vendor_eps_sweep_results/       # ε-sweep results
    ├── alpha_0.70/eps_*/logs.json
    ├── alpha_0.70/{min,maj}_{error_raw,rescaled}.{png,pdf}
    ├── alpha_1.50/eps_*/logs.json
    └── alpha_1.50/{min,maj}_{error_raw,rescaled}.{png,pdf}
```

## Expected results

**α-sweep (ε = 0.1):** For α < 1, both rescaled errors plateau at κ_g. For
α ≥ 1, majority plateaus while minority → 0. The α-sweep overlay shows all
curves compensated by z_t^{max(1,α)} approximately plateau. The phase
transition summary shows the measured decay exponent tracking the theoretical
prediction with a kink at α = 1.

**ε-sweep (α = 0.7):** Minority raw errors fan out ∝ 1/ε; rescaled errors
collapse onto κ_min regardless of ε.

**ε-sweep (α = 1.5):** Minority raw errors collapse (ε-independent rate
z_t^{-1.5}); majority errors fan out ∝ 1/(1−ε).

## Compute resources

All experiments were run on a single NVIDIA RTX A6000 GPU (49 GB).
The α-sweep (N=5M, 10M steps, 6 configs) takes approximately 2–4 hours.
The ε-sweep (N=1M, 10M steps, 10 configs) takes approximately 4–6 hours.
