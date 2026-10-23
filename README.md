# Gate-Based Feature Importance in Simulation-Based Inference for EEG: A Cautionary Tale

[![arXiv](https://img.shields.io/badge/arXiv-2610.30364-b31b1b.svg)](https://arxiv.org/abs/2610.30364)


📄 **Accepted at NeurIPS 2026 workshops:**
- **Interpretability for Discovery: Understanding and Discovering Novel Knowledge in AI Models**
- **Sim2Science: ML with Imperfect Scientific Models**

---

## Summary

We train neural posterior estimation (NPE) models to infer four physiological parameters of a Jansen–Rit neural mass model from simulated EEG. We then compare two measures of feature importance: a feature-gating mechanism trained jointly with the model, and gold-standard retrain-based ablation.

Across nine independent training seeds, the two measures are **negatively correlated** (Spearman's ρ ≈ −0.61 for sigmoid gating, −0.56 for softmax gating). Total signal power is the feature whose exclusion causes the largest drop in accuracy under ablation, yet it is among the features the gate down-weights most. Gating costs almost no predictive accuracy. **Learned gate weights should not be read as feature importance without validating them against a causal measure.**

## Repository structure

| File | Contents |
|---|---|
| `config.py` | Experiment hyperparameters (`Config` dataclass): simulation, training, prior bounds, gating, and noise settings |
| `simulator.py` | Jansen–Rit ODE simulator, pink-noise generation, and Sobol parameter sampling |
| `forward_model.py` | MNE fsaverage lead-field model projecting the simulated source to five-electrode scalp EEG |
| `embedding.py` | Seven-feature statistical embedding per channel, and the optional learned gate (sigmoid or softmax) |
| `training.py` | NPE training loop, posterior evaluation, and the retrain-based ablation procedure |
| `metrics.py` | R², NRMSE, and Pearson correlation, and a timeout guard around `posterior.sample()` |
| `post_analysis.py` | Posterior pairwise-correlation diagnostics for parameter degeneracy |
| `experiment_log.py` | Append-only CSV logging of each run's configuration and metrics to `results/` |
| `run.py` | Command-line entry point that trains and evaluates a single configuration |
| `sbi_pipeline.ipynb` | Interactive walkthrough of the full pipeline for a single seed |
| `requirements.txt` | Pinned dependencies |

## Data

No external dataset is required. The Jansen–Rit simulator generates all training and evaluation data at runtime, and the MNE lead-field model can optionally project it to scalp EEG.

## Installation

The pipeline requires **Python 3.11** and a POSIX operating system (Linux, macOS, or WSL). `experiment_log.py` uses `fcntl` for file locking, which is not available on native Windows.

```bash
git clone https://github.com/Shrivatsa-Deshmukh/sbi-gate-vs-ablation.git
cd sbi-gate-vs-ablation
python -m venv .venv && source .venv/bin/activate
```

`requirements.txt` pins PyTorch 2.5.1 built for **CUDA 12.4**, which is the build used for the paper.

On first use, `forward_model.py` downloads the MNE fsaverage template head model and caches the computed lead fields under `~/.mne/lead_field_cache/`. The first run of any script therefore needs network access and takes longer.

## Quickstart

Train the ungated baseline for a single seed:

```bash
python run.py --seed 0
```

Enable learned feature gating. Sigmoid weights each feature independently; softmax makes the weights compete:

```bash
python run.py --use_gating --gate_activation sigmoid --seed 0 --run_name sigmoid_seed0
python run.py --use_gating --gate_activation softmax --seed 0 --run_name softmax_seed0
```

Run the gold-standard ablation, which retrains the model once per excluded feature:

```bash
python run.py --run_ablation --seed 0 --run_name baseline_seed0
```

Each run appends one row to `results/experiments.csv`. `--run_ablation` also logs one row per excluded feature to `results/ablation.csv`. See `experiment_log.py` for the column schema.

| Flag | Purpose |
|---|---|
| `--use_gating` | Enables the learned feature gate (disabled by default) |
| `--gate_activation {sigmoid,softmax}` | Selects independent versus competing gating |
| `--run_ablation` | Retrains once per feature, excluding it, and logs the resulting ΔR² |
| `--exclude_features` | Manually excludes specific features |
| `--seed` | Training seed (the paper reports results across nine independent seeds) |
| `--no_noise` / `--snr_min` / `--snr_max` | Observation-noise controls |
| `--run_name`, `--notes` | Labels recorded in `results/experiments.csv` |

## Citation

If you use this code, please cite:

```bibtex
@misc{deshmukh2026gate,
  title         = {Gate-Based Feature Importance in Simulation-Based Inference for {EEG}: A Cautionary Tale},
  author        = {Deshmukh, Shrivatsa and Brenner, Sasha and Kn{\"o}sche, Thomas R. and Scherf, Nico},
  year          = {2026},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {NeurIPS 2026 Workshops: Sim2Science and Interp4Discovery}
}
```

## License

See [`LICENSE`](LICENSE).
