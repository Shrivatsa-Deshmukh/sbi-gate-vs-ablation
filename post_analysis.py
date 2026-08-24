"""
Post-hoc posterior diagnostics for trained SBI models.

Posterior pairwise correlation: average posterior correlation matrix across
a test set, to surface parameter degeneracies that per-parameter R²/NRMSE can't reveal.

Computed directly against a posterior already in memory (e.g.
StatsOnlyResults.posterior from training.train_and_evaluate) — no retraining
needed, and nothing here writes a posterior to disk.
"""

from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt

# Diverging pair (blue <-> red, neutral midpoint) for the correlation heatmap.
DIVERGING_CMAP = "RdBu_r"


def compute_posterior_correlation(posterior, x_test: torch.Tensor, param_names: List[str],
                                   num_posterior_samples: int = 1000,
                                   verbose: bool = True) -> np.ndarray:
    """
    Average posterior pairwise correlation matrix across a test set.

    High off-diagonal |correlation| for a parameter pair indicates a
    degeneracy the model can't resolve from the data alone (e.g. two
    parameters trading off along a ridge in the likelihood) — informative
    even when both parameters individually recover with reasonable R².

    Returns:
        mean_corr: [n_params, n_params] average correlation matrix.
    """
    n_test = x_test.shape[0]
    n_params = len(param_names)
    corr_sum = np.zeros((n_params, n_params))

    device = getattr(posterior, "device", None) or "cpu"
    x_test = x_test.to(device)

    for i in range(n_test):
        samples = posterior.sample((num_posterior_samples,), x=x_test[i:i + 1],
                                    show_progress_bars=False)
        samples_np = samples.cpu().numpy()
        corr_sum += np.corrcoef(samples_np, rowvar=False)

    mean_corr = corr_sum / n_test

    if verbose:
        print("\n  Mean posterior correlation matrix:")
        header = "              " + "".join(f"{p:>12s}" for p in param_names)
        print(header)
        for i, p in enumerate(param_names):
            print(f"    {p:10s}" + "".join(f"{mean_corr[i, j]:12.3f}" for j in range(n_params)))

    return mean_corr


def plot_posterior_correlation(mean_corr: np.ndarray, param_names: List[str],
                                title: str = 'Mean posterior pairwise correlation'):
    """Heatmap of the average posterior pairwise correlation matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mean_corr, vmin=-1, vmax=1, cmap=DIVERGING_CMAP)
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.set_yticklabels(param_names)
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            ax.text(j, i, f"{mean_corr[i, j]:.2f}", ha='center', va='center',
                    color='white' if abs(mean_corr[i, j]) > 0.5 else 'black')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig, ax
