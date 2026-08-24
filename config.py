"""
Configuration for the Jansen-Rit SBI pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Config:
    """Configuration for stats-only model experiments."""

    # ─── Reproducibility ─────────────────────────────────────────────────────
    SEED: int = 0

    # ─── Training data / hyperparameters ─────────────────────────────────────
    NUM_SIMULATIONS: int = 32768
    BATCH_SIZE: int = 512
    LEARNING_RATE: float = 5e-4
    MAX_EPOCHS: int = 250
    STOP_AFTER_EPOCHS: int = 15
    VALIDATION_FRACTION: float = 0.15

    # ─── Neural density estimator ─────────────────────────────────────────────
    HIDDEN_FEATURES: int = 125
    NUM_TRANSFORMS: int = 5

    # ─── Evaluation ────────────────────────────────────────────────────────────
    N_EVAL_SAMPLES: int = 250
    N_POSTERIOR_SAMPLES: int = 1000
    CREDIBLE_LEVELS: List[float] = field(default_factory=lambda: [0.5, 0.9, 0.95])

    # ─── Forward model (EEG sensor-space projection) ────────────────────────────
    USE_FORWARD_MODEL: bool = True
    SOURCE_LOCATION: str = 'visual'

    # ─── Multi-channel projection ────────────────────────────────────────────
    # N_CHANNELS>1 projects the source signal to multiple electrodes with
    # signed gains (see LeadFieldModel.apply_multi), preserving real
    # topographic information; N_CHANNELS=1 reproduces the original
    # single-electrode scalar-gain behavior. If ELECTRODE_NAMES is None, the
    # top N_CHANNELS electrodes by sensitivity are auto-selected; requires
    # USE_FORWARD_MODEL=True when N_CHANNELS>1.
    N_CHANNELS: int = 5
    ELECTRODE_NAMES: Optional[List[str]] = None

    # ─── Observation noise (variable SNR for robustness) ─────────────────────
    ADD_NOISE: bool = True
    NOISE_SNR_MIN: float = 5.0       # Minimum SNR in dB (more noise)
    NOISE_SNR_MAX: float = 25.0      # Maximum SNR in dB (less noise)
    NOISE_EXPONENT: float = 1.0      # Spectral exponent: 0=white, 1=pink, 2=brown
    EVAL_SNR_DB: Optional[float] = None  # If set (otherwise None), use fixed SNR for evaluation only

    # ─── Feature gating / ablation ────────────────────────────────────────────
    USE_GATING: bool = False
    # "sigmoid": independent gating, no competition between features.
    # "softmax": weights forced to sum to 1 across the features — true
    #     competitive attention.
    # Only matters when USE_GATING=True.
    GATE_ACTIVATION: str = "sigmoid"
    EXCLUDED_FEATURES: List[str] = field(default_factory=list)
    # Hidden layer width of the gating MLP. If None, defaults to a fixed 32.
    GATE_HIDDEN_WIDTH: Optional[int] = None
    ABLATION_EVAL_SAMPLES: int = 250  # Test samples per feature in the gold-standard retrain ablation

    # ─── Advanced / rarely changed (Jansen-Rit physics & priors) ─────────────
    DT: float = 0.001  # Simulation timestep (1000 Hz internal)
    FS_SIM: int = 1000  # Simulation sampling frequency
    FS_OUT: int = 250   # Output sampling frequency (downsampled)
    T_OUTPUT: float = 3.0
    T_DISCARD: float = 2.0

    # Jansen-Rit input noise std, fixed 
    SIGMA_VALUE: float = 4.5   

    PRIOR_MIN: List[float] = field(default_factory=lambda: [np.log(135.0), np.log(120.0), np.log(0.75), np.log(0.5)])
    PRIOR_MAX: List[float] = field(default_factory=lambda: [np.log(270.0), np.log(350.0), np.log(1.25), np.log(2.0)])
    PARAM_NAMES: List[str] = field(default_factory=lambda: ["log_C", "log_mu", "log_kappa", "log_g"])

    @property
    def T_TOTAL(self) -> float:
        return self.T_OUTPUT + self.T_DISCARD

    @property
    def SIGNAL_LENGTH(self) -> int:
        """Length of output signal at FS_OUT sampling rate."""
        return int(self.T_OUTPUT * self.FS_OUT)

    @property
    def DOWNSAMPLE_FACTOR(self) -> int:
        """Factor to downsample from FS_SIM to FS_OUT."""
        return self.FS_SIM // self.FS_OUT

    def get_effective_prior_bounds(self) -> Tuple[List[float], List[float], List[str]]:
        """Prior bounds for the 4 inferred parameters (log_C, log_mu, log_kappa, log_g)."""
        return self.PRIOR_MIN.copy(), self.PRIOR_MAX.copy(), self.PARAM_NAMES.copy()
