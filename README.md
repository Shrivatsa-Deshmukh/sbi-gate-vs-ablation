# Gate-Based Feature Importance Is Not a Reliable Diagnostic for Simulation-Based Inference

Code for the paper *"Gate-Based Feature Importance Is Not a Reliable Diagnostic for
Simulation-Based Inference: A Jansen-Rit EEG Case Study"* (NeurIPS 2026 Workshop
submission).

We train sequential neural posterior estimation (SNPE) models to infer four
physiological parameters of a Jansen-Rit neural mass model from simulated EEG,
and compare a jointly-trained feature-gating mechanism against gold-standard,
retrain-based ablation as measures of feature importance. Across nine seeds, the
two measures are **negatively correlated** (Spearman's ρ ≈ −0.61 for sigmoid
gating, −0.56 for softmax) — the feature ablation drops the most when removed,
total signal power, is the one the gate down-weights most.

> **Note (anonymized review copy):** author identity has been scrubbed from this
> repository for double-blind review. See [Anonymity](#anonymity-for-review) below.

## Repository structure

| File | Contents |
|---|---|
| `config.py` | All experiment hyperparameters (`Config` dataclass): simulation, training, prior bounds, gating, noise |
| `simulator.py` | Jansen-Rit ODE simulator, pink-noise generation, Sobol parameter sampling |
| `forward_model.py` | MNE `fsaverage` lead-field model — projects the simulated source to 5-electrode scalp EEG |
| `embedding.py` | Seven-feature statistical embedding per channel, plus the optional learned gate (sigmoid/softmax) |
| `training.py` | SNPE training loop, posterior evaluation, and the retrain-based ablation procedure |
| `metrics.py` | R², NRMSE, Pearson correlation, and a timeout guard around `posterior.sample()` |
| `post_analysis.py` | Posterior pairwise-correlation diagnostics (parameter degeneracy checks) |
| `experiment_log.py` | Append-only CSV logging of each run's config + metrics to `results/` |
| `run.py` | CLI entry point — trains one configuration end-to-end |
| `notebooks/sbi_pipeline.ipynb` | Interactive, single-seed walkthrough of the full pipeline |
| `requirements.txt` | Pinned dependencies |

All modules import flatly (`from config import Config`, `from training import ...`)
— run everything from the repo root, not from inside a subdirectory. `run.py`
creates a `results/` folder on first run to hold the logged CSVs; it isn't
checked in.

## Installation

Requires **Python 3.11** and a POSIX OS (Linux/macOS/WSL) — `experiment_log.py`
uses `fcntl` for file locking, unavailable on native Windows.

```bash
git clone <this-repo-url>
cd <repo-name>
python -m venv .venv && source .venv/bin/activate

# torch is pinned to a CUDA 12.4 build. On a different CUDA version or a
# CPU-only machine, install torch first per
# https://pytorch.org/get-started/locally/, then:
pip install -r requirements.txt --no-deps torch
# ...or drop the torch line from requirements.txt and let torch's own
# installer pick the right build before installing the rest.

pip install -r requirements.txt
```

`forward_model.py` downloads the MNE `fsaverage` template head model on first
use and caches the computed lead fields under `~/.mne/lead_field_cache/` —
expect a one-time delay (and network access) the first time any script runs.

## Quickstart

Ungated baseline, single seed:

```bash
python run.py --seed 0
```

Learned feature gating (sigmoid = independent weighting, softmax = competing):

```bash
python run.py --use_gating --gate_activation sigmoid --seed 0 --run_name sigmoid_seed0
python run.py --use_gating --gate_activation softmax --seed 0 --run_name softmax_seed0
```

Gold-standard ablation (retrains once per excluded feature, on top of a baseline run):

```bash
python run.py --run_ablation --seed 0 --run_name baseline_seed0
```

Each run appends a row to `results/experiments.csv`; `--run_ablation` also logs
per-feature rows to `results/ablation.csv` (schema in `experiment_log.py`).

| Flag | Purpose |
|---|---|
| `--use_gating` | Enable the learned feature gate (off by default) |
| `--gate_activation {sigmoid,softmax}` | Independent vs. competing gating |
| `--run_ablation` | Retrain once per feature, excluding it, log ΔR² |
| `--exclude_features F1 F2 ...` | Manually exclude specific features |
| `--seed` | Training seed (paper: 9 independent seeds) |
| `--no_noise` / `--snr_min` / `--snr_max` | Observation-noise controls |
| `--run_name`, `--notes` | Labels for `results/experiments.csv` |

A single run (32,768 simulations, up to 250 epochs) takes on the order of
tens of minutes on CPU; a CUDA GPU is strongly recommended, especially for
`--run_ablation`, which retrains 7 additional models.

## Reproducing the paper's numbers

1. **Table 1 (ablation columns) + Table 3 "No gating" row:** run
   `python run.py --run_ablation --seed <s>` for each of 9 seeds.
2. **Table 1 (gate columns) + Table 3 gated rows:** run with
   `--use_gating --gate_activation sigmoid` and `--gate_activation softmax`
   for each of the same 9 seeds.
3. **Table 2 (Spearman ρ, sign-flip test):** compute per-seed Spearman
   correlation between each seed's ablation ΔR² and gate weight from the
   logged CSVs, then run the exact sign-flip test across the 9 per-seed
   values (512 sign assignments; see Methods 3.8 in the paper).
4. **Posterior degeneracy diagnostics (Methods, Discussion):**
   `post_analysis.compute_posterior_correlation()` runs directly on a trained
   `posterior` object in memory — no retraining needed.
5. `notebooks/sbi_pipeline.ipynb` walks through steps 1–4 end-to-end for a
   single seed, useful for a first sanity check before launching the full
   9-seed sweep.

Expect mean R² ≈ 0.79 for the ungated baseline (Table 3) as a rough check
that your environment reproduces the paper's setup correctly.

## Anonymity for review

This copy has had author names, affiliations, and repo-identifying history
removed for double-blind review, per the workshop's `dblblindworkshop` LaTeX
option. If hosting this during the review period, consider wrapping it with
[anonymous.4open.science](https://anonymous.4open.science) rather than
pushing to a personally-identifiable GitHub account.

## Citation

```bibtex
@inproceedings{anonymous2026gating,
  title     = {Gate-Based Feature Importance Is Not a Reliable Diagnostic for
               Simulation-Based Inference: A Jansen-Rit EEG Case Study},
  author    = {Anonymous},
  booktitle = {NeurIPS 2026 Workshop},
  year      = {2026}
}
```
(Update with full author list and workshop name after de-anonymization.)
