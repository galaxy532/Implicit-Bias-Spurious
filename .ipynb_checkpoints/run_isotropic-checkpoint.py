#!/usr/bin/env python3
"""
run_isotropic.py
================
Experiment A: Synthetic isotropic-regime verification of Theorems 1 & 2.

Produces two sets of runs:
  Panel 1 (alpha < 1):  Both groups decay as kappa_g / (eps_g * z_t).
  Panel 2 (alpha >= 1): Minority escapes eps-dependence, decays as z_t^{-alpha}.

Usage:
  # Full run (may take a few hours on GPU):
  python run_isotropic.py

  # Quick sanity check (~2 min):
  python run_isotropic.py --quick

  # Custom settings:
  python run_isotropic.py --steps 1000000 --N 100000 --epsilons 0.01 0.05 0.1 0.2 0.5

  # Multiple seeds for error bars (default: 3 seeds):
  python run_isotropic.py --seeds 1 2 3 4 5
"""

import argparse
import os
import sys
import json
import numpy as np
from synth_utils import (
    SynthConfig, build_isotropic_operators, generate_dataset,
    train_population_gd, kappa_theory, save_logs, ensure_dir,
)


def make_panel_configs(panel: str, epsilons: list, base_steps: int, base_N: int,
                       log_every: int, print_every: int, out_root: str):
    """
    Return a list of SynthConfig for one panel.

    Panel "alpha_lt_1":  alpha ~ 0.7
      mu_A=1.0, mu_B=1.0, mu=-0.3, gamma_min=1.0  =>  alpha = 1.0*(1-0.3)/(1+1.0) = 0.35
      Actually let's pick values that give a cleaner alpha.
      mu_A=1.0, mu_B=1.0, mu=0.4, gamma_min=1.0  =>  alpha = 1.0*1.4/2.0 = 0.7

    Panel "alpha_ge_1":  alpha ~ 1.5
      mu_A=1.0, mu_B=1.0, mu=0.0, gamma_min=3.0   =>  alpha = 3.0*1.0/2.0 = 1.5
    """
    configs = []
    for eps in epsilons:
        cfg = SynthConfig()
        cfg.epsilon = eps
        cfg.steps = base_steps
        cfg.N = base_N
        cfg.log_every = log_every
        cfg.print_every = print_every

        if panel == "alpha_lt_1":
            cfg.mu_A = 1.0
            cfg.mu_B = 1.0
            cfg.mu = 0.4
            cfg.gamma_min = 1.0
            cfg.gamma_maj = 1.0
            # alpha = 1.0 * 1.4 / 2.0 = 0.7
        elif panel == "alpha_ge_1":
            cfg.mu_A = 1.0
            cfg.mu_B = 1.0
            cfg.mu = 0.0
            cfg.gamma_min = 3.0
            cfg.gamma_maj = 1.0
            # alpha = 3.0 * 1.0 / 2.0 = 1.5
        else:
            raise ValueError(f"Unknown panel: {panel}")

        cfg.__post_init__()  # recompute alpha
        cfg.out_dir = os.path.join(out_root, panel, f"eps_{eps}")
        configs.append(cfg)

    return configs


def run_single(cfg: SynthConfig):
    """Run one (panel, epsilon, seed) experiment."""
    print(f"\n{'='*60}")
    print(f"  eps={cfg.epsilon}  alpha={cfg.alpha:.3f}  seed={cfg.seed}  "
          f"steps={cfg.steps}  N={cfg.N}")
    print(f"  mu_A={cfg.mu_A}  mu_B={cfg.mu_B}  mu={cfg.mu}  gamma_min={cfg.gamma_min}")
    print(f"  out: {cfg.out_dir}")
    print(f"{'='*60}")

    # Theory
    theory = kappa_theory(cfg)
    print(f"  Theory: {theory}")

    # Build operators & data
    A, B, v = build_isotropic_operators(cfg)
    X, groups, meta = generate_dataset(cfg, A, B, v)
    meta["theory"] = theory
    meta["seed"] = cfg.seed

    # Train
    logs, w_final = train_population_gd(cfg, X, groups)

    # Save
    path = save_logs(logs, meta, cfg)
    print(f"  Saved to {path}")
    return logs, meta


def main():
    parser = argparse.ArgumentParser(description="Isotropic regime experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick sanity check (fewer steps, smaller dataset)")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--N", type=int, default=None, help="Override dataset size")
    parser.add_argument("--epsilons", nargs="+", type=float, default=None,
                        help="List of epsilon values to sweep")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="List of random seeds for error bars (default: [1, 2, 3])")
    parser.add_argument("--panels", nargs="+", default=["alpha_lt_1", "alpha_ge_1"],
                        choices=["alpha_lt_1", "alpha_ge_1"],
                        help="Which panels to run")
    parser.add_argument("--out_root", type=str, default="./runs_synth",
                        help="Root output directory")
    args = parser.parse_args()

    # Defaults
    if args.quick:
        steps = args.steps or 10_000
        N = args.N or 10_000
        epsilons = args.epsilons or [0.05, 0.2, 0.5]
        seeds = args.seeds or [1, 2, 3]
        log_every = 100
        print_every = 2_000
    else:
        steps = args.steps or 200_000
        N = args.N or 50_000
        epsilons = args.epsilons or [0.01, 0.05, 0.1, 0.2, 0.5]
        seeds = args.seeds or [1, 2, 3]
        log_every = 500
        print_every = 50_000

    print(f"Running with: steps={steps}, N={N}, epsilons={epsilons}, seeds={seeds}")
    print(f"Panels: {args.panels}")
    print(f"Device: {'cuda' if __import__('torch').cuda.is_available() else 'cpu'}")

    for panel in args.panels:
        print(f"\n{'#'*60}")
        print(f"  PANEL: {panel}")
        print(f"{'#'*60}")
        configs = make_panel_configs(
            panel, epsilons, steps, N, log_every, print_every, args.out_root
        )
        for cfg in configs:
            base_out_dir = cfg.out_dir
            for seed in seeds:
                cfg.seed = seed
                cfg.out_dir = os.path.join(base_out_dir, f"seed_{seed}")
                run_single(cfg)

    print("\n\nAll runs complete. Now run:  python plot_isotropic.py")


if __name__ == "__main__":
    main()
