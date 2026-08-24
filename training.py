"""
Training, evaluation, and ablation analysis for the stats-only SBI model.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from sbi.inference import SNPE
from sbi.utils import BoxUniform
from sbi.neural_nets import posterior_nn

from config import Config
from simulator import simulate_jansen_rit, generate_sobol_parameters
from metrics import (compute_r2, compute_nrmse, evaluate_posterior,
                      _sampling_timeout, PosteriorSamplingTimeout, POSTERIOR_SAMPLE_TIMEOUT_S)
from embedding import StatsOnlyEmbeddingNet

# Conditional import for lead field model
try:
    from forward_model import LeadFieldModel
    LEAD_FIELD_AVAILABLE = True
except ImportError:
    LEAD_FIELD_AVAILABLE = False
    LeadFieldModel = None


@dataclass
class StatsOnlyResults:
    """Container for results from stats_only model."""
    r2_per_param: np.ndarray
    nrmse_per_param: np.ndarray
    pearson_per_param: np.ndarray
    coverage_50: np.ndarray
    coverage_90: np.ndarray
    coverage_95: np.ndarray
    mean_r2: float
    mean_nrmse: float
    mean_pearson: float
    training_epochs: int
    training_time: float
    best_val_loss: float
    theta_true: np.ndarray = field(default_factory=lambda: np.array([]))
    theta_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    posterior: Optional[Any] = None
    param_names: List[str] = field(default_factory=list)
    gate_importance: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        result = {
            'model': 'stats_only',
            'mean_r2': self.mean_r2,
            'mean_nrmse': self.mean_nrmse,
            'mean_pearson': self.mean_pearson,
            'training_epochs': self.training_epochs,
            'training_time': self.training_time,
            'best_val_loss': self.best_val_loss,
        }
        param_names = self.param_names
        for i, r2 in enumerate(self.r2_per_param):
            result[f'r2_{param_names[i]}'] = r2
        for i, nrmse in enumerate(self.nrmse_per_param):
            result[f'nrmse_{param_names[i]}'] = nrmse
        for i, pearson in enumerate(self.pearson_per_param):
            result[f'pearson_{param_names[i]}'] = pearson
        for i, cov in enumerate(self.coverage_50):
            result[f'coverage_50_{param_names[i]}'] = cov
        for i, cov in enumerate(self.coverage_90):
            result[f'coverage_90_{param_names[i]}'] = cov
        for i, cov in enumerate(self.coverage_95):
            result[f'coverage_95_{param_names[i]}'] = cov
        return result


def plot_training_curves(summary: Dict[str, Any], title: str = "Training Progress"):
    """
    Plot training and validation loss curves from SBI summary.

    Args:
        summary: The inference._summary dictionary from SBI training
        title: Optional title for the plot
    """
    train_loss = summary.get('training_loss', [])
    val_loss = summary.get('validation_loss', [])
    best_val_loss = summary.get('best_validation_loss', None)
    epochs_trained = summary.get('epochs_trained', 0)

    if not train_loss and not val_loss:
        print("  No training history available for plotting.")
        return

    epochs = list(range(1, len(train_loss) + 1)) if train_loss else list(range(1, len(val_loss) + 1))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    if train_loss:
        ax.plot(epochs, train_loss, label='Training Loss', color='blue', alpha=0.7, linewidth=1.5)
    if val_loss:
        ax.plot(epochs, val_loss, label='Validation Loss', color='orange', alpha=0.9, linewidth=2)
        if best_val_loss is not None:
            # SBI's summary wraps this in a list/array in some versions.
            if isinstance(best_val_loss, (list, np.ndarray)):
                best_val_loss = float(best_val_loss[0]) if len(best_val_loss) > 0 else None

            if best_val_loss is not None and len(val_loss) > 0:
                val_loss_array = np.array(val_loss)
                best_epoch = int(np.argmin(val_loss_array)) + 1
                min_val_loss = float(val_loss_array.min())

                ax.axhline(y=min_val_loss, color='red', linestyle='--', alpha=0.5,
                           label=f'Best Val Loss: {min_val_loss:.4f} (epoch {best_epoch})')
                ax.scatter([best_epoch], [min_val_loss], color='red', s=100, zorder=5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (Negative Log-Likelihood)', fontsize=12)
    ax.set_title(f'{title} ({epochs_trained} epochs)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    if train_loss:
        final_train = train_loss[-1]
        ax.annotate(f'Final Train: {final_train:.4f}',
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    if val_loss:
        final_val = val_loss[-1]
        ax.annotate(f'Final Val: {final_val:.4f}',
                    xy=(0.02, 0.88), xycoords='axes fraction',
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.show()


def generate_fixed_eval_set(config: Config, n_samples: Optional[int] = None,
                             eval_seed: int = 12345, verbose: bool = False,
                             lead_field_model=None):
    """
    Generate a fixed evaluation set (theta_test, x_test), independent of any
    particular training seed, for passing into train_and_evaluate /
    run_retrain_ablation_analysis as theta_test/x_test.

    Args:
        config: Determines the prior bounds, N_EVAL_SAMPLES (default sample
            count), forward model / noise settings used to simulate x_test.
        n_samples: Number of eval samples. Defaults to config.N_EVAL_SAMPLES.
        eval_seed: Seed for the eval set draw only — unrelated to, and not
            affected by, any training seed used later.
        verbose: Passed through to LeadFieldModel construction.
        lead_field_model: Reuse an existing LeadFieldModel instead of
            constructing a new one (they're deterministic per source
            location via disk cache, so either is equivalent) — saves a
            redundant load when the caller already has one.

    Returns:
        (theta_test, x_test) tensors, ready to pass as train_and_evaluate's
        or run_retrain_ablation_analysis's theta_test/x_test arguments.
    """
    if n_samples is None:
        n_samples = config.N_EVAL_SAMPLES

    torch.manual_seed(eval_seed)
    torch.cuda.manual_seed_all(eval_seed)
    np.random.seed(eval_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prior_min, prior_max, _ = config.get_effective_prior_bounds()
    prior = BoxUniform(
        low=torch.tensor(prior_min, device=device),
        high=torch.tensor(prior_max, device=device)
    )
    theta_test = prior.sample((n_samples,))

    if config.USE_FORWARD_MODEL and lead_field_model is None:
        if not LEAD_FIELD_AVAILABLE or LeadFieldModel is None:
            raise ImportError("Lead field model requested but forward_model.py not found")
        lead_field_model = LeadFieldModel(config.SOURCE_LOCATION, verbose=verbose)

    x_test = simulate_jansen_rit(
        theta_test.cpu(), config, use_gpu=torch.cuda.is_available(),
        lead_field_model=lead_field_model, fixed_snr_db=config.EVAL_SNR_DB
    )

    return theta_test, x_test.to(device)


def run_retrain_ablation_analysis(theta_train, x_train, config, baseline_r2,
                                  seed=42, verbose=True,
                                  lead_field_model=None, channel_names=None,
                                  feature_groups=None,
                                  theta_test=None, x_test=None):
    """

    Args:
        theta_train: Training parameters
        x_train: Training simulations
        config: Configuration
        baseline_r2: Baseline R² scores (full model)
        seed: Random seed base
        verbose: Print progress
        lead_field_model: The same LeadFieldModel used to generate x_train (if
            any) — must be passed through so ablation's own test-time
            simulations match x_train's shape/scale (forward-projected and,
            for multi-channel, the same number of electrodes). Ignored if
            theta_test/x_test are given.
        channel_names: Electrode names for multi-channel embedding nets
            (n_channels=config.N_CHANNELS), for interpretable feature naming.
        feature_groups: Optional list of feature-name lists to exclude
            jointly. Each group is keyed in the result dict by '+'.join(group).
        theta_test, x_test: Optional fixed evaluation set, shared across every
            condition in this call (and ideally across seeds/other run types
            too — see evaluate_posterior) so ablation drops are comparable
            rather than each condition scored on its own random test draw.

    Returns:
        Dictionary with ablation results per feature (or per feature group)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prior_min, prior_max, param_names = config.get_effective_prior_bounds()

    # feature_names() doesn't depend on gating; only used here to enumerate feature names.
    all_features = StatsOnlyEmbeddingNet(config.SIGNAL_LENGTH, fs=config.FS_OUT).feature_names()

    if feature_groups is not None:
        conditions = [(list(group), "+".join(group)) for group in feature_groups]
    else:
        conditions = [([feat_name], feat_name) for feat_name in all_features]

    ablation_results = {}

    if verbose:
        print(f"\n  Running retrain ablation analysis (retraining for each condition)...")
        print(f"  This may take a while ({len(conditions)} models to train)...")

    for feat_idx, (excluded, feat_name) in enumerate(conditions):
        if verbose:
            print(f"\n  [{feat_idx+1}/{len(conditions)}] Training model WITHOUT '{feat_name}'...")

        embedding_net = StatsOnlyEmbeddingNet(
            input_len=config.SIGNAL_LENGTH,
            fs=config.FS_OUT,
            use_gating=config.USE_GATING,
            excluded_features=excluded,
            n_channels=config.N_CHANNELS,
            channel_names=channel_names,
            gate_hidden_width=config.GATE_HIDDEN_WIDTH,
            gate_activation=config.GATE_ACTIVATION
        )
        embedding_net.compute_normalization_stats(x_train)

        prior = BoxUniform(
            low=torch.tensor(prior_min, device=device),
            high=torch.tensor(prior_max, device=device)
        )

        inference = SNPE(
            prior=prior,
            density_estimator=posterior_nn(
                model="nsf",
                embedding_net=embedding_net,
                hidden_features=config.HIDDEN_FEATURES,
                num_transforms=config.NUM_TRANSFORMS
            ),
            device=str(device)
        )

        density_estimator = inference.append_simulations(theta_train, x_train).train(
            learning_rate=config.LEARNING_RATE,
            max_num_epochs=config.MAX_EPOCHS,
            stop_after_epochs=config.STOP_AFTER_EPOCHS,
            training_batch_size=config.BATCH_SIZE,
            validation_fraction=config.VALIDATION_FRACTION,
            show_train_summary=True
        )

        posterior = inference.build_posterior(density_estimator)

        if theta_test is not None:
            cond_theta_test = theta_test
            cond_x_test = x_test.to(device)
        else:
            cond_theta_test = prior.sample((config.ABLATION_EVAL_SAMPLES,))
            cond_x_test = simulate_jansen_rit(cond_theta_test.cpu(), config, use_gpu=torch.cuda.is_available(),
                                               lead_field_model=lead_field_model)
            cond_x_test = cond_x_test.to(device)

        n_eval = len(cond_theta_test)
        if verbose:
            print(f"    Evaluating on {n_eval} test samples...")

        theta_true = cond_theta_test.cpu().numpy()

        # Sequential posterior sampling (reliable, slightly slower)
        all_theta_pred = []
        all_coverage = {level: [] for level in config.CREDIBLE_LEVELS}

        kept_theta_true = []
        for i in range(n_eval):
            try:
                with _sampling_timeout(POSTERIOR_SAMPLE_TIMEOUT_S):
                    samples = posterior.sample((config.N_POSTERIOR_SAMPLES,), x=cond_x_test[i:i+1], show_progress_bars=False)
            except PosteriorSamplingTimeout:
                # See metrics._sampling_timeout: sbi's rejection sampler has no
                # cap and can spin forever at ~0% acceptance for a bad observation.
                if verbose:
                    print(f"    WARNING: skipped eval point {i}: posterior sampling timed out after {POSTERIOR_SAMPLE_TIMEOUT_S}s")
                continue
            samples_np = samples.cpu().numpy()
            all_theta_pred.append(samples_np.mean(axis=0))
            kept_theta_true.append(theta_true[i])

            for level in config.CREDIBLE_LEVELS:
                lower = np.percentile(samples_np, (1 - level) / 2 * 100, axis=0)
                upper = np.percentile(samples_np, (1 + level) / 2 * 100, axis=0)
                in_ci = (theta_true[i] >= lower) & (theta_true[i] <= upper)
                all_coverage[level].append(in_ci)

        theta_true = np.array(kept_theta_true)
        theta_pred = np.array(all_theta_pred)

        r2_ablated = compute_r2(theta_true, theta_pred)
        nrmse_ablated = compute_nrmse(theta_true, theta_pred, prior_min, prior_max)
        coverage_90 = np.mean(all_coverage[0.9], axis=0)

        mean_r2_ablated = np.mean(r2_ablated)
        mean_r2_drop = (np.mean(baseline_r2) - mean_r2_ablated) if baseline_r2 is not None else float('nan')

        ablation_results[feat_name] = {
            'mean_r2': mean_r2_ablated,
            'mean_r2_drop': mean_r2_drop,
            'r2_per_param': r2_ablated,
            'nrmse_per_param': nrmse_ablated,
            'coverage_90_per_param': coverage_90
        }

        if verbose:
            print(f"    Per-parameter results (without {feat_name}):")
            for j, pname in enumerate(param_names):
                r2_drop = baseline_r2[j] - r2_ablated[j] if baseline_r2 is not None else 0
                print(f"      {pname:12s} R²={r2_ablated[j]:.3f} (Δ={r2_drop:+.3f}) NRMSE={nrmse_ablated[j]:.3f} Cov90={coverage_90[j]:.2f}")

        # Explicit memory cleanup to prevent RAM accumulation. Note: cond_theta_test/
        # cond_x_test are deleted here, NOT the theta_test/x_test params — those may be
        # the caller's fixed evaluation set, reused across every remaining condition.
        del embedding_net, inference, density_estimator, posterior
        del cond_theta_test, cond_x_test, theta_pred, theta_true
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return ablation_results


def train_and_evaluate(config: Optional[Config] = None,
                   theta_train: Optional[torch.Tensor] = None,
                   x_train: Optional[torch.Tensor] = None,
                   seed: int = 42,
                   verbose: bool = True,
                   theta_test: Optional[torch.Tensor] = None,
                   x_test: Optional[torch.Tensor] = None) -> StatsOnlyResults:
    """
    Run the stats_only model experiment.

    Args:
        config: Experiment configuration
        theta_train: Pre-generated training parameters
        x_train: Pre-generated training simulations
        seed: Random seed
        verbose: Whether to print progress
        theta_test, x_test: Optional fixed evaluation set (see
            evaluate_posterior) — pass the same fixed set across
            runs/conditions.

    Returns:
        StatsOnlyResults object with all metrics
    """
    if config is None:
        config = Config()

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f"\nRunning inference...")

    # Multi-channel projection requires the forward model
    if config.N_CHANNELS > 1 and not config.USE_FORWARD_MODEL:
        raise ValueError("N_CHANNELS > 1 requires USE_FORWARD_MODEL=True")

    lead_field_model = None
    channel_names = None
    if config.USE_FORWARD_MODEL:
        if not LEAD_FIELD_AVAILABLE or LeadFieldModel is None:
            raise ImportError("Lead field model requested but forward_model.py not found")
        if verbose:
            print(f"  Initializing forward model for source: {config.SOURCE_LOCATION}")
        lead_field_model = LeadFieldModel(config.SOURCE_LOCATION, verbose=verbose)
        if config.N_CHANNELS > 1:
            channel_names = lead_field_model.resolve_channels(config)
            if verbose:
                print(f"  Multi-channel mode: {config.N_CHANNELS} electrodes selected: {channel_names}")
        elif verbose:
            print(f"  Forward model ready: electrode={lead_field_model.selected_electrode}, "
                  f"gain={lead_field_model.lead_field_value:.6f}")

    prior_min, prior_max, param_names = config.get_effective_prior_bounds()
    expected_n_params = len(param_names)

    if theta_train is not None and x_train is not None:
        provided_n_params = theta_train.shape[1]
        if provided_n_params != expected_n_params:
            if verbose:
                print(f"  ⚠ Dimension mismatch: theta_train has {provided_n_params} params, "
                      f"expected {expected_n_params}")
                print(f"    Regenerating training data with correct dimensionality...")
            theta_train = None
            x_train = None

    if theta_train is None or x_train is None:
        start_time = time.time()
        theta_train = generate_sobol_parameters(
            config.NUM_SIMULATIONS, prior_min, prior_max, seed=seed
        )
        simulator_fn = lambda theta, lf=lead_field_model: simulate_jansen_rit(
            theta, config, use_gpu=torch.cuda.is_available(), lead_field_model=lf
        )
        x_train = simulator_fn(theta_train)
        if verbose:
            print(f"  Generated {config.NUM_SIMULATIONS} simulations in {time.time() - start_time:.1f}s")
            print(f"  Parameters: {param_names}")
    else:
        simulator_fn = lambda theta, lf=lead_field_model: simulate_jansen_rit(
            theta, config, use_gpu=torch.cuda.is_available(), lead_field_model=lf
        )

    prior = BoxUniform(
        low=torch.tensor(prior_min, device=device),
        high=torch.tensor(prior_max, device=device)
    )

    embedding_net = StatsOnlyEmbeddingNet(
        input_len=config.SIGNAL_LENGTH,
        fs=config.FS_OUT,
        use_gating=config.USE_GATING,
        excluded_features=config.EXCLUDED_FEATURES,
        n_channels=config.N_CHANNELS,
        channel_names=channel_names,
        gate_hidden_width=config.GATE_HIDDEN_WIDTH,
        gate_activation=config.GATE_ACTIVATION
    )

    embedding_net.compute_normalization_stats(x_train)

    inference = SNPE(
        prior=prior,
        density_estimator=posterior_nn(
            model="nsf",
            embedding_net=embedding_net,
            hidden_features=config.HIDDEN_FEATURES,
            num_transforms=config.NUM_TRANSFORMS
        ),
        device=str(device)
    )

    if verbose:
        print(f"  Training density estimator...")
    start_time = time.time()

    density_estimator = inference.append_simulations(theta_train, x_train).train(
        learning_rate=config.LEARNING_RATE,
        max_num_epochs=config.MAX_EPOCHS,
        stop_after_epochs=config.STOP_AFTER_EPOCHS,
        training_batch_size=config.BATCH_SIZE,
        validation_fraction=config.VALIDATION_FRACTION,
        show_train_summary=False
    )

    training_time = time.time() - start_time
    posterior = inference.build_posterior(density_estimator)

    summary = inference._summary
    training_epochs = summary.get('epochs_trained', 0)
    best_val_loss = summary.get('best_validation_loss', float('inf'))
    # SBI's summary wraps these in lists in some versions.
    if isinstance(training_epochs, (list, np.ndarray)):
        training_epochs = int(training_epochs[0]) if len(training_epochs) > 0 else 0
    if isinstance(best_val_loss, (list, np.ndarray)):
        best_val_loss = float(best_val_loss[0]) if len(best_val_loss) > 0 else float('inf')

    if verbose:
        print(f"  Training complete in {training_time:.1f}s ({training_epochs} epochs)")
        plot_training_curves(summary, title="Stats-Only Model Training")
        if config.EVAL_SNR_DB is not None:
            print(f"  Evaluating on {config.N_EVAL_SAMPLES} test cases at fixed SNR={config.EVAL_SNR_DB} dB...")
        else:
            print(f"  Evaluating on {config.N_EVAL_SAMPLES} test cases with variable SNR...")

    eval_simulator_fn = lambda theta, lf=lead_field_model: simulate_jansen_rit(
        theta, config, use_gpu=torch.cuda.is_available(), lead_field_model=lf,
        fixed_snr_db=config.EVAL_SNR_DB
    )

    theta_true, theta_pred, r2, nrmse, pearson, coverage, x_obs_used = evaluate_posterior(
        posterior, eval_simulator_fn, prior, config,
        theta_test=theta_test, x_test=x_test
    )

    gate_importance = None
    if config.USE_GATING:
        gate_importance = embedding_net.get_feature_importance(x_obs_used, detailed=True)
        if verbose:
            print(f"\n  Feature Importance:")
            for item in gate_importance['summary']:
                bar = "█" * int(item['contribution'] * 10)
                print(f"    {item['feature']:20s} {item['contribution']:.3f} {bar}")

    return StatsOnlyResults(
        r2_per_param=r2,
        nrmse_per_param=nrmse,
        pearson_per_param=pearson,
        coverage_50=coverage[0.5],
        coverage_90=coverage[0.9],
        coverage_95=coverage[0.95],
        mean_r2=float(np.mean(r2)),
        mean_nrmse=float(np.mean(nrmse)),
        mean_pearson=float(np.mean(pearson)),
        training_epochs=training_epochs,
        training_time=training_time,
        best_val_loss=best_val_loss,
        theta_true=theta_true,
        theta_pred=theta_pred,
        posterior=posterior,
        param_names=param_names,
        gate_importance=gate_importance
    )
