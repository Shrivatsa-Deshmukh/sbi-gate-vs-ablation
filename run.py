#!/usr/bin/env python
"""
CLI entrypoint for the Jansen-Rit SBI pipeline.

For interactive exploration, use sbi_pipeline.ipynb instead — this script is
for headless/scripted runs.

Usage:
    python run.py                      # Default run (4 params)
    python run.py --use_gating         # Enable feature gating
    python run.py --no_noise           # Disable observation noise
"""

import argparse
import time
import warnings

import matplotlib
# Headless/scripted runner: force a non-interactive backend so
# plot_training_curves' plt.show() never pops up a blocking GUI window.
matplotlib.use('Agg')

import torch

# Suppress benign warnings (script-level concern, not a library import side effect)
warnings.filterwarnings('ignore', message='.*triangular_solve.*')
warnings.filterwarnings('ignore', message='.*Torchinductor does not support.*')
warnings.filterwarnings('ignore', message='.*TensorFloat32.*')
warnings.filterwarnings('ignore', message='.*torch._dynamo.*')
warnings.filterwarnings('ignore', message='.*Moving.*to the data_device.*')

# Enable TensorFloat32 for better GPU performance
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

from config import Config
from training import train_and_evaluate, generate_fixed_eval_set, run_retrain_ablation_analysis
from simulator import simulate_jansen_rit, generate_sobol_parameters

try:
    from forward_model import LeadFieldModel, clear_cache
except ImportError:
    LeadFieldModel = None
    clear_cache = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run stats-only model for Jansen-Rit SBI pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Gating argument (enable with flag since default is off)
    parser.add_argument(
        '--use_gating',
        action='store_true',
        help='Enable learned feature gating (disabled by default)'
    )

    parser.add_argument(
        '--gate_activation',
        type=str,
        choices=['sigmoid', 'softmax'],
        default=None,
        help='Gating activation (default: sigmoid). Only matters with --use_gating.'
    )

    parser.add_argument(
        '--run_ablation',
        action='store_true',
        help='Gold-standard ablation: after the baseline run, retrain once per '
             'base feature (excluding it) and log the R² drop to results/ablation.csv'
    )

    parser.add_argument(
        '--exclude_features',
        type=str,
        nargs='+',
        default=[],
        help='Features to exclude (e.g., --exclude_features skewness kurtosis)'
    )

    parser.add_argument(
        '--snr_min',
        type=float,
        default=None,
        help='Minimum SNR in dB (default: 5.0, more noise)'
    )

    parser.add_argument(
        '--snr_max',
        type=float,
        default=None,
        help='Maximum SNR in dB (default: 25.0, less noise)'
    )

    parser.add_argument(
        '--no_noise',
        action='store_true',
        help='Disable observation noise entirely'
    )

    parser.add_argument(
        '--noise_exponent',
        type=float,
        default=None,
        help='Noise spectral exponent (default: 1.0 = pink noise, 0 = white, 2 = brown)'
    )

    parser.add_argument(
        '--eval_snr',
        type=float,
        default=None,
        help='Fixed SNR for evaluation only (default: variable SNR). Use to test at specific noise levels.'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Training seed (default: 0, from Config.SEED)'
    )

    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Label for this run (e.g. "n_channels_5"), used in the results log and run directory name'
    )

    parser.add_argument(
        '--notes',
        type=str,
        default='',
        help='Free-text note about this run, stored in the results log (e.g. what is being swept)'
    )

    parser.add_argument(
        '--no_log',
        action='store_true',
        help='Skip logging this run to results/experiments.csv'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    config = Config()

    print("=" * 70)
    print("   STATS-ONLY MODEL FOR JANSEN-RIT SBI")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  • Simulations:    {config.NUM_SIMULATIONS}")
    print(f"  • Eval samples:   {config.N_EVAL_SAMPLES}")
    print(f"  • Max epochs:     {config.MAX_EPOCHS}")
    print(f"  • Early stopping: {config.STOP_AFTER_EPOCHS}")
    print(f"  • Device:         {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  • Gating:         {'ENABLED' if args.use_gating else 'DISABLED'}")
    print(f"  • Ablation:       {'ENABLED' if args.run_ablation else 'DISABLED'}")
    print(f"  • Forward model:  ENABLED (source: {config.SOURCE_LOCATION})")

    config.USE_GATING = args.use_gating
    config.EXCLUDED_FEATURES = args.exclude_features

    if args.gate_activation is not None:
        config.GATE_ACTIVATION = args.gate_activation

    if args.seed is not None:
        config.SEED = args.seed

    if args.no_noise:
        config.ADD_NOISE = False
    if args.snr_min is not None:
        config.NOISE_SNR_MIN = args.snr_min
    if args.snr_max is not None:
        config.NOISE_SNR_MAX = args.snr_max
    if args.noise_exponent is not None:
        config.NOISE_EXPONENT = args.noise_exponent
    if args.eval_snr is not None:
        config.EVAL_SNR_DB = args.eval_snr

    if config.ADD_NOISE:
        noise_type = {0.0: 'white', 1.0: 'pink', 2.0: 'brown'}.get(
            config.NOISE_EXPONENT, f'1/f^{config.NOISE_EXPONENT}'
        )
        print(f"  • Train noise:    variable SNR={config.NOISE_SNR_MIN}-{config.NOISE_SNR_MAX} dB, type={noise_type}")
        if config.EVAL_SNR_DB is not None:
            print(f"  • Eval noise:     fixed SNR={config.EVAL_SNR_DB} dB")
        else:
            print(f"  • Eval noise:     variable (same as training)")
    else:
        print(f"  • Obs. noise:     DISABLED")

    print(f"  • Sigma:          FIXED at {config.SIGMA_VALUE} (not inferred)")
    print()

    print("  Generating fixed evaluation set (independent of training seed)...")
    theta_test, x_test = generate_fixed_eval_set(config)

    # Ablation needs the exact baseline theta_train/x_train reused for every
    # retrain condition (see run_retrain_ablation_analysis docstring), so
    # pre-generate them here instead of letting train_and_evaluate generate
    # (and discard) its own internally.
    theta_train = x_train = lead_field_model = channel_names = None
    if args.run_ablation:
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)
        prior_min, prior_max, _ = config.get_effective_prior_bounds()

        if config.USE_FORWARD_MODEL:
            if LeadFieldModel is None:
                raise ImportError("Lead field model requested but forward_model.py not found")
            lead_field_model = LeadFieldModel(config.SOURCE_LOCATION, verbose=True)
            if config.N_CHANNELS > 1:
                channel_names = lead_field_model.resolve_channels(config)

        theta_train = generate_sobol_parameters(
            config.NUM_SIMULATIONS, prior_min, prior_max, seed=config.SEED
        )
        x_train = simulate_jansen_rit(
            theta_train, config, use_gpu=torch.cuda.is_available(), lead_field_model=lead_field_model
        )

    start_time = time.time()
    results = train_and_evaluate(config=config, seed=config.SEED, verbose=True,
                                  theta_test=theta_test, x_test=x_test,
                                  theta_train=theta_train, x_train=x_train)
    total_time = time.time() - start_time

    _, _, param_names = config.get_effective_prior_bounds()

    print("\n" + "=" * 70)
    print("   STATS-ONLY MODEL COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {total_time / 60:.1f} minutes")

    print("\n" + "-" * 70)
    print("RESULTS SUMMARY")
    print("-" * 70)
    print(f"\n  Mean R²:      {results.mean_r2:.3f}")
    print(f"  Mean NRMSE:   {results.mean_nrmse:.3f}")
    print(f"  Mean Pearson: {results.mean_pearson:.3f}")
    print(f"\n  Per-parameter performance:")
    for i, name in enumerate(param_names):
        print(f"    {name:12s} R²={results.r2_per_param[i]:.3f}  "
              f"NRMSE={results.nrmse_per_param[i]:.3f}  "
              f"r={results.pearson_per_param[i]:.3f}  "
              f"Cov90={results.coverage_90[i]:.2f}")
    if not args.no_log:
        from experiment_log import log_run
        run_label = log_run(config, results, run_name=args.run_name, notes=args.notes)
        print(f"\n  Logged run as: {run_label}  (see results/experiments.csv)")

        if config.USE_GATING and results.gate_importance is not None:
            from experiment_log import log_gating_importance
            log_gating_importance(config, results.gate_importance, run_label, notes=args.notes)
            print(f"  Logged gating importance  (see results/gating_importance.csv)")
    else:
        run_label = args.run_name or "unlogged_run"

    if args.run_ablation:
        print("\n" + "=" * 70)
        print("   GOLD-STANDARD ABLATION (retrain per feature)")
        print("=" * 70)
        ablation_start = time.time()
        ablation_results = run_retrain_ablation_analysis(
            theta_train=theta_train, x_train=x_train, config=config,
            baseline_r2=results.r2_per_param, seed=config.SEED, verbose=True,
            lead_field_model=lead_field_model, channel_names=channel_names,
            theta_test=theta_test, x_test=x_test,
        )
        ablation_time = time.time() - ablation_start
        print(f"\nAblation complete in {ablation_time / 60:.1f} minutes")

        print("\nR² drop when feature removed (sorted by importance):")
        for feat, res in sorted(ablation_results.items(), key=lambda kv: -kv[1]['mean_r2_drop']):
            print(f"    {feat:20s} ΔR²={res['mean_r2_drop']:+.4f}  (R² without = {res['mean_r2']:.3f})")

        if not args.no_log:
            from experiment_log import log_ablation
            log_ablation(config, ablation_results, baseline_run_label=run_label, notes=args.notes)
            print(f"\n  Logged ablation  (see results/ablation.csv)")

    if clear_cache is not None:
        clear_cache()

    return results


if __name__ == "__main__":
    main()
