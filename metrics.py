"""
Evaluation metrics for posterior parameter recovery.
"""

import logging
import signal
from contextlib import contextmanager
from typing import Callable, Dict, List, Tuple

import numpy as np

from config import Config


class PosteriorSamplingTimeout(Exception):
    """Raised when posterior.sample() exceeds POSTERIOR_SAMPLE_TIMEOUT_S."""


POSTERIOR_SAMPLE_TIMEOUT_S = 60


@contextmanager
def _sampling_timeout(seconds: int):
    """
    Abort a single posterior.sample() call that's stuck in sbi's rejection
    sampler (see samplers/rejection/rejection.py: `while num_remaining > 0`
    has no cap and can spin forever if the trained flow's acceptance rate
    against the prior support is ~0% for a given observation). SIGALRM fires
    between the loop's proposal-batch calls, so this reliably interrupts it
    without corrupting torch/CUDA state. Main-thread/Unix only.
    """
    def _on_alarm(signum, frame):
        raise PosteriorSamplingTimeout(f"posterior.sample() exceeded {seconds}s")

    previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute R² score per parameter."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    return 1 - (ss_res / (ss_tot + 1e-8))


def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray,
                  prior_min: List[float], prior_max: List[float]) -> np.ndarray:
    """Compute Normalized RMSE per parameter."""
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    prior_range = np.array(prior_max) - np.array(prior_min)
    return rmse / prior_range


def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation coefficient per parameter."""
    n_params = y_true.shape[1]
    pearson = np.zeros(n_params)
    for i in range(n_params):
        pearson[i] = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
    return pearson


def evaluate_posterior(posterior, simulator: Callable, prior,
                       config: Config, theta_test=None, x_test=None) -> Tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray, np.ndarray,
                                                          Dict[float, np.ndarray], "torch.Tensor"]:
    """
    Evaluate posterior quality with R², NRMSE, Pearson correlation, and coverage.

    Args:
        theta_test, x_test: Optional fixed evaluation set. If given, these are
            used directly instead of drawing a fresh sample from `prior` —
            pass the same fixed set across runs/conditions so results are
            comparable (not confounded by which random test points got drawn).

    Returns:
        theta_true_arr, theta_pred_arr, r2, nrmse, pearson, coverage_dict, x_obs_batch
            (x_obs_batch is the eval signal batch actually used — either the
            caller-supplied x_test or the freshly-simulated one — so callers
            that need the eval inputs themselves, e.g. for gate feature
            importance, don't have to re-simulate or silently fall back to None)
    """
    device = prior.base_dist.low.device

    if theta_test is not None:
        theta_true_batch = theta_test
        x_obs_batch = x_test.to(device)
    else:
        # Generate all test parameters at once and batch simulate
        theta_true_batch = prior.sample((config.N_EVAL_SAMPLES,))
        x_obs_batch = simulator(theta_true_batch.cpu())
        x_obs_batch = x_obs_batch.to(device)

    all_theta_true, all_theta_pred = [], []
    all_coverage = {level: [] for level in config.CREDIBLE_LEVELS}
    n_skipped = 0

    for i in range(len(theta_true_batch)):
        theta_true_np = theta_true_batch[i].cpu().numpy()
        x_obs = x_obs_batch[i:i+1]

        try:
            with _sampling_timeout(POSTERIOR_SAMPLE_TIMEOUT_S):
                samples = posterior.sample((config.N_POSTERIOR_SAMPLES,), x=x_obs, show_progress_bars=False)
        except PosteriorSamplingTimeout:
            # Trained flow's acceptance rate against the prior support collapsed
            # to ~0% for this observation (sbi's rejection sampler has no cap —
            # see _sampling_timeout docstring). Drop this eval point rather than
            # stalling the whole run; a handful of skips out of N_EVAL_SAMPLES
            # doesn't meaningfully bias R²/NRMSE/Pearson.
            n_skipped += 1
            logging.warning(f"  Skipped eval point {i}: posterior sampling timed out after {POSTERIOR_SAMPLE_TIMEOUT_S}s (near-zero acceptance rate)")
            continue

        samples_np = samples.cpu().numpy()

        all_theta_true.append(theta_true_np)
        all_theta_pred.append(np.mean(samples_np, axis=0))

        for level in config.CREDIBLE_LEVELS:
            alpha = (1 - level) / 2
            lower = np.percentile(samples_np, alpha * 100, axis=0)
            upper = np.percentile(samples_np, (1 - alpha) * 100, axis=0)
            in_ci = (theta_true_np >= lower) & (theta_true_np <= upper)
            all_coverage[level].append(in_ci)

    if n_skipped > 0:
        print(f"  WARNING: {n_skipped}/{len(theta_true_batch)} eval points skipped due to posterior sampling timeout")

    theta_true_arr = np.array(all_theta_true)
    theta_pred_arr = np.array(all_theta_pred)

    prior_min, prior_max, _ = config.get_effective_prior_bounds()

    r2 = compute_r2(theta_true_arr, theta_pred_arr)
    nrmse = compute_nrmse(theta_true_arr, theta_pred_arr, prior_min, prior_max)
    pearson = compute_pearson(theta_true_arr, theta_pred_arr)

    coverage_dict = {level: np.mean(all_coverage[level], axis=0) for level in config.CREDIBLE_LEVELS}

    return theta_true_arr, theta_pred_arr, r2, nrmse, pearson, coverage_dict, x_obs_batch
