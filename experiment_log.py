"""
Persistent logging of final results + config for each run, for later
cross-run comparison.

Each call to log_run() appends one row to results/experiments.csv: the
run's config fields (prefixed "cfg_") plus its final summary metrics
(mean/per-parameter R², NRMSE, Pearson, coverage). No raw arrays or
trained models are stored.
"""

import contextlib
import dataclasses
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from config import Config
from training import StatsOnlyResults

RESULTS_DIR = Path("results")
LOG_PATH = RESULTS_DIR / "experiments.csv"
ABLATION_LOG_PATH = RESULTS_DIR / "ablation.csv"
GATING_LOG_PATH = RESULTS_DIR / "gating_importance.csv"
CORRELATION_LOG_PATH = RESULTS_DIR / "posterior_correlation.csv"

# Passed to pd.read_csv when re-loading a log for appending, so that free-text
# cells (e.g. "notes" containing "NA", "NULL", "None", ...) round-trip as the
# literal string they were written as, instead of pandas silently reading them
# back as NaN on the next append.
_READ_BACK_KWARGS = dict(keep_default_na=False, na_values=[""])


def _flatten(value: Any) -> Any:
    """Make a config/metric value CSV-cell-safe (scalars pass through, everything else becomes a JSON string)."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


@contextlib.contextmanager
def _locked(lock_path: Path):
    """Exclusive, blocking file lock (POSIX advisory lock) guarding a critical section."""
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _append_rows(path: Path, rows: Union[list, "callable"]) -> None:
    """
    Append rows to a results CSV, tolerating column-set changes across runs
    (e.g. Config gaining/losing fields over time).

    `rows` is either a list of dicts, or a callable `(existing_df) -> list[dict]`
    for callers that need to see already-logged rows before building their own
    (e.g. log_run disambiguating run_name) — that lookup-then-write happens
    inside the same lock, so it can't race with a concurrent writer.

    Safe under concurrent writers (e.g. parallel sweep processes) via an
    exclusive file lock, and crash-safe via write-to-temp-then-rename, so a
    process killed mid-write can never truncate/corrupt the existing log.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")

    with _locked(lock_path):
        existing = pd.read_csv(path, **_READ_BACK_KWARGS) if path.exists() else pd.DataFrame()
        new_rows = rows(existing) if callable(rows) else rows
        updated = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        updated.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)  # atomic on POSIX: readers never see a partial file


def _dedupe_run_name(existing: pd.DataFrame, run_name: str) -> str:
    """Disambiguate run_name if a fast run collides with an existing one at second resolution."""
    if "run_name" not in existing.columns or run_name not in existing["run_name"].values:
        return run_name
    existing_names = existing["run_name"].values
    suffix = 2
    while f"{run_name}_{suffix}" in existing_names:
        suffix += 1
    return f"{run_name}_{suffix}"


def log_run(config: Config, results: StatsOnlyResults, run_name: Optional[str] = None,
            notes: str = "") -> str:
    """
    Append one row (config + final metrics) for this run to results/experiments.csv.

    Args:
        config: The Config used for this run.
        results: The StatsOnlyResults returned by train_and_evaluate().
        run_name: Optional label (e.g. "n_channels_5") identifying this run
            in a parameter sweep.
        notes: Free-text note about this run (e.g. what's being swept).

    Returns:
        The run_name used for this row (a timestamp if run_name wasn't given).
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_run_name = run_name or timestamp

    config_dict = dataclasses.asdict(config)
    resolved_run_name = {}

    def build_rows(existing: pd.DataFrame) -> list:
        resolved_name = _dedupe_run_name(existing, base_run_name)
        resolved_run_name["value"] = resolved_name
        return [{
            "run_name": resolved_name,
            "notes": notes,
            **{f"cfg_{k}": _flatten(v) for k, v in config_dict.items()},
            **{k: _flatten(v) for k, v in results.to_dict().items()},
        }]

    _append_rows(LOG_PATH, build_rows)

    return resolved_run_name["value"]


def log_ablation(config: Config, ablation_results: dict, baseline_run_label: str,
                  notes: str = "") -> None:
    """
    Append one row per excluded feature to results/ablation.csv, from the
    dict returned by training.run_retrain_ablation_analysis().

    Each row records that feature's R² when excluded, its R² drop relative
    to the full-feature baseline, and its per-parameter R²/NRMSE/coverage —
    so features can be ranked by importance, overall or per parameter.

    Args:
        config: The Config used for the baseline (and ablation) runs. Not
            duplicated into this table's columns — it's already recorded
            once in experiments.csv under baseline_run_label; join on that
            instead of re-reading it per ablation row.
        ablation_results: Output of run_retrain_ablation_analysis() —
            {feature_name: {'mean_r2', 'mean_r2_drop', 'r2_per_param',
                             'nrmse_per_param', 'coverage_90_per_param'}}.
        baseline_run_label: The run_name (from log_run) of the full-feature
            model these ablations are relative to, for cross-referencing.
        notes: Free-text note about this ablation batch.
    """
    _, _, param_names = config.get_effective_prior_bounds()

    rows = []
    for feature_name, res in ablation_results.items():
        row = {
            "baseline_run_label": baseline_run_label,
            "feature": feature_name,
            "notes": notes,
            "mean_r2": _flatten(res["mean_r2"]),
            "mean_r2_drop": _flatten(res["mean_r2_drop"]),
        }
        for i, pname in enumerate(param_names):
            row[f"r2_{pname}"] = _flatten(res["r2_per_param"][i])
            row[f"nrmse_{pname}"] = _flatten(res["nrmse_per_param"][i])
            row[f"coverage_90_{pname}"] = _flatten(res["coverage_90_per_param"][i])
        rows.append(row)

    _append_rows(ABLATION_LOG_PATH, rows)


def log_gating_importance(config: Config, gate_importance: dict, run_label: str,
                           notes: str = "") -> None:
    """
    Append one row per feature to results/gating_importance.csv, from
    StatsOnlyResults.gate_importance (requires config.USE_GATING=True).

    Each row records that feature's learned gate weight (mean/std across
    the held-out eval batch) and its contribution score.

    Args:
        config: The Config used for this run (must have USE_GATING=True).
            Not duplicated into this table's columns — already recorded once
            in experiments.csv under run_name; join on that instead.
        gate_importance: results.gate_importance — the detailed dict from
            StatsOnlyEmbeddingNet.get_feature_importance(x_obs_used, detailed=True),
            where x_obs_used is the eval signal batch evaluate_posterior actually
            used (caller-supplied or freshly simulated — see its docstring).
        run_label: The run_name (from log_run) of this gating model, for
            cross-referencing.
        notes: Free-text note about this run.
    """
    rows = []
    for item in gate_importance['summary']:
        rows.append({
            "run_label": run_label,
            "feature": item['feature'],
            "notes": notes,
            "mean_gate": _flatten(item['mean_gate']),
            "std_gate": _flatten(item['std_gate']),
            "contribution": _flatten(item['contribution']),
            "context_dependency": item['context_dependency'],
        })

    _append_rows(GATING_LOG_PATH, rows)


def log_posterior_correlation(config: Config, seed: Union[int, str], param_names: list,
                               mean_corr: np.ndarray, run_label: Optional[str] = None,
                               notes: str = "") -> None:
    """
    Append one row per parameter pair (long format: seed, param_i, param_j,
    correlation) to results/posterior_correlation.csv, from a single seed's
    average posterior correlation matrix (post_analysis.compute_posterior_correlation).

    seed may also be a label such as "pooled" or "ensemble" for cross-comparable-rows reason.

    config isn't duplicated into this table's columns — join on run_name against experiments.csv for that.
    """
    n_params = len(param_names)

    rows = []
    for i in range(n_params):
        for j in range(n_params):
            rows.append({
                "run_label": run_label or "",
                "seed": seed,
                "param_i": param_names[i],
                "param_j": param_names[j],
                "correlation": _flatten(float(mean_corr[i, j])),
                "notes": notes,
            })

    _append_rows(CORRELATION_LOG_PATH, rows)
