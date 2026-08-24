# SBI - Gate vs Ablation

This repository contains the code accompanying the paper *"Gate-Based Feature
Importance Is Not a Reliable Diagnostic for Simulation-Based Inference: A
Jansen-Rit EEG Case Study."*

We train sequential neural posterior estimation (SNPE) models to infer four
physiological parameters of a Jansen-Rit neural mass model from simulated EEG,
and compare a jointly-trained feature-gating mechanism against gold-standard,
retrain-based ablation as measures of feature importance. Across nine
independent training seeds, the two measures are negatively correlated
(Spearman's ρ ≈ −0.61 for sigmoid gating, −0.56 for softmax): total signal
power, the feature whose exclusion causes the largest drop in ablation
accuracy, is among the features the gate down-weights most.

## Repository structure

| File | Contents |
|---|---|
| `config.py` | Experiment hyperparameters (`Config` dataclass): simulation, training, prior bounds, gating, and noise settings |
| `simulator.py` | Jansen-Rit ODE simulator, pink-noise generation, and Sobol parameter sampling |
| `forward_model.py` | MNE `fsaverage` lead-field model projecting the simulated source to five-electrode scalp EEG |
| `embedding.py` | Seven-feature statistical embedding per channel, and the optional learned gate (sigmoid or softmax) |
| `training.py` | SNPE training loop, posterior evaluation, and the retrain-based ablation procedure |
| `metrics.py` | R², NRMSE, and Pearson correlation, and a timeout guard around `posterior.sample()` |
| `post_analysis.py` | Posterior pairwise-correlation diagnostics for parameter degeneracy |
| `experiment_log.py` | Append-only CSV logging of each run's configuration and metrics to `results/` |
| `run.py` | Command-line entry point that trains and evaluates a single configuration |
| `sbi_pipeline.ipynb` | Interactive walkthrough of the full pipeline for a single seed |
| `requirements.txt` | Pinned dependencies |

## Data

No external dataset is required. All training and evaluation data are generated synthetically at runtime by the Jansen-Rit simulator, optionally projected to scalp EEG via the MNE lead-field model.

## Installation

The pipeline requires Python 3.11 and a POSIX operating system (Linux, macOS,
or WSL), since `experiment_log.py` uses `fcntl` for file locking, which is not
available on native Windows.

```bash
git clone <this-repo-url>
cd <repo-name>
python -m venv .venv && source .venv/bin/activate

# torch is pinned to a CUDA 12.4 build. On a different CUDA version or a CPU-only machine,
# install torch separately first, following https://pytorch.org/get-started/locally/, then run:
pip install -r requirements.txt --no-deps torch
# Alternatively, remove the torch line from requirements.txt and allow torch's own installer
# to select the appropriate build before installing the remaining dependencies.

pip install -r requirements.txt
```

`forward_model.py` downloads the MNE `fsaverage` template head model on first
use and caches the computed lead fields under `~/.mne/lead_field_cache/`. The
first invocation of any script therefore requires network access and incurs a
one-time delay.

## Quickstart

To train the ungated baseline for a single seed:

```bash
python run.py --seed 0
```

To enable learned feature gating (sigmoid weights each feature
independently; softmax forces the weights to compete):

```bash
python run.py --use_gating --gate_activation sigmoid --seed 0 --run_name sigmoid_seed0
python run.py --use_gating --gate_activation softmax --seed 0 --run_name softmax_seed0
```

To run the gold-standard ablation, which retrains the model once per excluded
feature:

```bash
python run.py --run_ablation --seed 0 --run_name baseline_seed0
```

Each run appends one row to `results/experiments.csv`; `--run_ablation`
additionally logs one row per excluded feature to `results/ablation.csv` (see
`experiment_log.py` for the column schema).

| Flag | Purpose |
|---|---|
| `--use_gating` | Enables the learned feature gate (disabled by default) |
| `--gate_activation {sigmoid,softmax}` | Selects independent versus competing gating |
| `--run_ablation` | Retrains once per feature, excluding it, and logs the resulting ΔR² |
| `--exclude_features` | Manually excludes specific features |
| `--seed` | Training seed (the paper reports results across nine independent seeds) |
| `--no_noise` / `--snr_min` / `--snr_max` | Observation-noise controls |
| `--run_name`, `--notes` | Labels recorded in `results/experiments.csv` |

## Reproducing the paper's results

1. Table 1 (ablation columns) and the "No gating" row of Table 3: run python run.py --run_ablation --seed SEED for each of the nine seeds.
2. Table 1 (gate columns) and the "Gated" rows of Table 3: run with --use_gating --gate_activation sigmoid and --gate_activation softmax for the same nine seeds.
3. Table 2 (Spearman ρ and the sign-flip test): compute the per-seed Spearman correlation between each seed's ablation ΔR² and gate weight from the logged CSVs, then apply the exact sign-flip test across the nine per-seed values.
4. Posterior degeneracy diagnostics (Methods, Discussion): post_analysis.compute_posterior_correlation() operates directly on a trained posterior object held in memory and does not require retraining.
5. Interactive walkthrough: sbi_pipeline.ipynb walks through steps 1–4 for a single seed and is a useful first check before running the full nine-seed sweep.
