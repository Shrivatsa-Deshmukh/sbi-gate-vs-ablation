"""
Stats-Only Embedding Network for Jansen-Rit SBI Pipeline

This module provides a embedding network that uses only statistical
features for inference. 

Statistical features (7 total when all included):
- skewness, kurtosis
- spectral_slope
- total_log_power, dominant_freq
- Hjorth mobility, complexity

Optional gating mechanism for feature importance learning.
Supports feature exclusion for proper ablation by retraining.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np


class StatsOnlyEmbeddingNet(nn.Module):
    """
    Args:
        input_len: Length of input time series
        fs: Sampling frequency (default: 250 Hz)
        n_fft: FFT size for Welch PSD (default: 256, gives ~1 Hz resolution)
        hop_length: Hop length for Welch PSD (default: 128)
        use_gating: Whether to use learned gating for feature importance (default: True)
        excluded_features: List of feature names to exclude from computation (default: [])
        n_channels: Number of input channels/electrodes (default: 1). Each
            channel gets its own copy of the base features, computed
            independently and concatenated (channel-major layout: channel 0's
            features, then channel 1's, etc.)
        channel_names: Optional electrode names for interpretable feature_names()
            output when n_channels>1 (defaults to "ch0", "ch1", ...).
        gate_activation: "sigmoid" (default, independent gating — no
            competition between features) or "softmax" (weights forced to
            sum to 1 across the 7 base features — true competitive
            attention). 
    """

    # All available feature names in order (7 total)
    ALL_FEATURE_NAMES = [
        'skewness', 'kurtosis',
        'spectral_slope',
        'total_log_power', 'dominant_freq',
        'hjorth_mobility', 'hjorth_complexity'
    ]

    GATE_ACTIVATIONS = ("sigmoid", "softmax")

    def __init__(self, input_len, fs=250, n_fft=256, hop_length=128,
                 use_gating=True, excluded_features=None,
                 n_channels: int = 1, channel_names: Optional[List[str]] = None,
                 gate_hidden_width: Optional[int] = None,
                 gate_activation: str = "sigmoid"):
        super().__init__()

        if gate_activation not in self.GATE_ACTIVATIONS:
            raise ValueError(f"Unknown gate_activation: '{gate_activation}'. "
                            f"Valid options: {self.GATE_ACTIVATIONS}")

        self.input_len = input_len
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_fft // 2 + 1
        self.use_gating = use_gating
        self.n_channels = n_channels
        self.channel_names = channel_names
        self.gate_activation = gate_activation

        self.excluded_features = set(excluded_features) if excluded_features else set()

        for feat in self.excluded_features:
            if feat not in self.ALL_FEATURE_NAMES:
                raise ValueError(f"Unknown feature to exclude: '{feat}'. "
                               f"Valid features: {self.ALL_FEATURE_NAMES}")

        self._included_features = [f for f in self.ALL_FEATURE_NAMES
                                   if f not in self.excluded_features]
        self.n_base_features = len(self._included_features)
        # Total output width: each channel gets its own copy of the base features
        self.n_features = self.n_base_features * self.n_channels
        
        # Pre-compute log frequencies for slope calculation (only if needed)
        if 'spectral_slope' not in self.excluded_features:
            freqs = torch.linspace(0, self.fs / 2, self.n_bins)
            idx_1hz = int(1.0 * (self.n_fft / self.fs))
            idx_50hz = int(50.0 * (self.n_fft / self.fs))

            self.slope_idx_start = max(1, idx_1hz)
            self.slope_idx_end = idx_50hz
            
            slope_freqs = freqs[self.slope_idx_start:self.slope_idx_end]
            self.register_buffer('log_freqs', torch.log(slope_freqs))
            self.register_buffer('log_freqs_mean', torch.mean(torch.log(slope_freqs)))
            self.register_buffer('log_freqs_var', torch.var(torch.log(slope_freqs)))
        
        # Dummy parameter to satisfy SBI's device check
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        
        # Normalization buffers (sized for actual n_features)
        self.register_buffer('_normalization_set', torch.tensor(False))
        self.register_buffer('feature_means', torch.zeros(self.n_features))
        self.register_buffer('feature_stds', torch.ones(self.n_features))
        self.epsilon = 1e-8
        
        # Gating mechanism, always channel-tied: the gate net's input/output
        # width is n_base_features regardless of n_channels (it operates on
        # the channel-pooled features and its output gets tiled back out to
        # all channels). Hidden width defaults to a fixed 32, 
        # independent of n_base_features, so capacity stays
        # consistent across EXCLUDED_FEATURES / N_CHANNELS configs.
        if self.use_gating:
            if gate_hidden_width is None:
                gate_hidden_width = 32
            self.gate_hidden_width = gate_hidden_width
            final_activation = nn.Sigmoid() if self.gate_activation == "sigmoid" else nn.Softmax(dim=-1)
            self.gate_net = nn.Sequential(
                nn.Linear(self.n_base_features, gate_hidden_width),
                nn.ReLU(),
                nn.Linear(gate_hidden_width, self.n_base_features),
                final_activation
            )
            # Running average of gate weights (EMA with momentum 0.99). Sized for
            # n_features — every channel's slice is identical by construction
            # (see _apply_gate_net), so this just holds n_channels duplicate copies.
            self.register_buffer('_running_gate_weights', torch.ones(self.n_features) * 0.5)
            self._gate_weight_count: torch.Tensor
            self.register_buffer('_gate_weight_count', torch.tensor(0))
            self.gate_momentum = 0.99
        else:
            self.gate_net = None

    def feature_names(self):
        """
        Return list of feature names in order (channel-major when n_channels>1:
        channel 0's base features, then channel 1's, etc.)
        """
        if self.n_channels == 1:
            return self._included_features.copy()
        names = []
        for c in range(self.n_channels):
            label = self.channel_names[c] if self.channel_names else f"ch{c}"
            names.extend(f"{label}_{f}" for f in self._included_features)
        return names

    def _is_included(self, feature_name):
        """Check if a feature is included (not excluded)."""
        return feature_name not in self.excluded_features

    def compute_normalization_stats(self, signals):
        """
        Compute normalization statistics from training signals.
        
        Args:
            signals: All training signals [n_samples, input_len] as tensor or numpy array
            
        This computes all features for each signal and calculates the mean and
        standard deviation for each feature across the full dataset.
        """
        if isinstance(signals, np.ndarray):
            signals = torch.from_numpy(signals).float()
        signals = signals.to(self.dummy_param.device)

        # Batched to bound peak memory over large training sets.
        batch_size = 1024
        all_features = []

        # forward()'s normalization-set check would otherwise block the raw
        # feature computation below, which is how these stats get set.
        self._normalization_set.fill_(True)

        with torch.no_grad():
            for i in range(0, len(signals), batch_size):
                batch = signals[i:i + batch_size]
                features = self._compute_raw_features_multi(batch)
                all_features.append(features)

        all_features = torch.cat(all_features, dim=0)
        means = all_features.mean(dim=0)
        stds = all_features.std(dim=0)

        self.feature_means.copy_(means)
        self.feature_stds.copy_(stds)
        self._normalization_set.fill_(True)
    
    def set_normalization_stats(self, means, stds):
        """
        Manually set precomputed normalization statistics.
        
        Args:
            means: Mean for each feature [n_features] as tensor or numpy array
            stds: Standard deviation for each feature [n_features] as tensor/numpy
        """
        if isinstance(means, np.ndarray):
            means = torch.from_numpy(means).float()
        if isinstance(stds, np.ndarray):
            stds = torch.from_numpy(stds).float()
        
        if means.shape[0] != self.n_features:
            raise ValueError(f"Expected {self.n_features} means, got {means.shape[0]}")
        if stds.shape[0] != self.n_features:
            raise ValueError(f"Expected {self.n_features} stds, got {stds.shape[0]}")
        
        self.feature_means.copy_(means.to(self.feature_means.device))
        self.feature_stds.copy_(stds.to(self.feature_stds.device))
        self._normalization_set.fill_(True)
    
    def get_feature_importance(self, signals=None, detailed=False):
        """
        Get feature importance from gating weights.
        
        Args:
            signals: Optional signals to compute gate weights for [n_samples, input_len].
                     If None, returns the stored running average from training.
            detailed: If True, returns detailed analysis including weight distribution
                     and feature contributions. If False, returns simple mean weights.
        
        Returns:
            If detailed=False: Dictionary mapping feature names to mean importance weights (0-1)
            If detailed=True: Dictionary with detailed analysis
        
        Raises:
            RuntimeError: If gating is disabled
        """
        if not self.use_gating or self.gate_net is None:
            raise RuntimeError("Gating is disabled. Enable use_gating=True to get feature importance.")

        std_weights = None
        contributions = None

        if signals is not None:
            if isinstance(signals, np.ndarray):
                signals = torch.from_numpy(signals).float()
            signals = signals.to(self.dummy_param.device)

            with torch.no_grad():
                features = self._compute_raw_features_multi(signals)
                normalized = (features - self.feature_means) / (self.feature_stds + self.epsilon)
                gate_weights = self._apply_gate_net(normalized)

                if detailed:
                    mean_weights = gate_weights.mean(dim=0)
                    std_weights = gate_weights.std(dim=0)
                    contributions = (normalized.abs() * gate_weights).mean(dim=0)
                else:
                    mean_weights = gate_weights.mean(dim=0)
        else:
            mean_weights = self._running_gate_weights
            if detailed:
                std_weights = torch.zeros_like(mean_weights)
                contributions = torch.zeros_like(mean_weights)
        
        if not detailed:
            importance = {}
            for i, name in enumerate(self.feature_names()):
                importance[name] = mean_weights[i].item()
            return importance
        
        assert std_weights is not None and contributions is not None  # always set when detailed=True

        result = {
            'mean_weights': {},
            'std_weights': {},
            'contributions': {},
            'summary': []
        }

        for i, name in enumerate(self.feature_names()):
            mw = mean_weights[i].item()
            sw = std_weights[i].item()
            ct = contributions[i].item()
            
            result['mean_weights'][name] = mw
            result['std_weights'][name] = sw
            result['contributions'][name] = ct
            
            context_dep = "high" if sw > 0.1 else "low"
            
            result['summary'].append({
                'feature': name,
                'mean_gate': mw,
                'std_gate': sw,
                'contribution': ct,
                'context_dependency': context_dep
            })
        
        result['summary'] = sorted(result['summary'], key=lambda x: x['contribution'], reverse=True)
        
        return result
    
    def compute_welch_psd(self, x):
        """Compute Welch PSD of input signal."""
        x_unfolded = x.unfold(-1, self.n_fft, self.hop_length)
        window = torch.hann_window(self.n_fft, device=x.device).view(1, 1, -1)
        x_windowed = x_unfolded * window
        fft_windows = torch.fft.rfft(x_windowed, dim=-1)
        psd_windows = torch.abs(fft_windows) ** 2
        return psd_windows.mean(dim=1)
    
    
    def _compute_raw_features(self, x):
        """
        Compute raw (unnormalized) statistical features from input signal.
        Only computes included features.
        
        Args:
            x: Input time series [batch_size, input_len]
            
        Returns:
            Raw statistical features [batch_size, n_features]
        """
        features = []
        
        # Compute PSD once for efficiency
        psd = self.compute_welch_psd(x)
        
        # Compute mean and std for skewness calculation
        mu = x.mean(1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        
        # 1. Skewness
        z = (x - mu) / (std + 1e-6)
        if self._is_included('skewness'):
            skew = (z ** 3).mean(1, True)
            features.append(skew)
        
        # 2. Kurtosis (excess kurtosis, captures distribution "peakedness"; normal = 0)
        if self._is_included('kurtosis'):
            kurt = (z ** 4).mean(1, True) - 3.0
            features.append(kurt)
        
        # 3. Spectral slope (1-50Hz)
        if self._is_included('spectral_slope'):
            log_psd = torch.log(psd + 1e-8).unsqueeze(1)
            psd_band = log_psd[:, 0, self.slope_idx_start:self.slope_idx_end]
            y_mean = torch.mean(psd_band, dim=1, keepdim=True)
            xy_cov = torch.mean((self.log_freqs - self.log_freqs_mean) * (psd_band - y_mean), dim=1, keepdim=True)
            slope = xy_cov / (self.log_freqs_var + 1e-8)
            features.append(slope)
        
        # 4. Total log power
        if self._is_included('total_log_power'):
            total_power = psd.sum(dim=1, keepdim=True)
            total_log_power = torch.log(total_power + 1e-8)
            features.append(total_log_power)
        
        # 5. Dominant frequency (frequency with maximum power in 1-50 Hz range)
        # We restrict to 1-50 Hz to avoid DC artifacts and focus on physiologically
        # relevant frequencies for Jansen-Rit models (which produce alpha-band activity),
        # while staying clear of the anti-alias filter's rolloff near 100 Hz
        if self._is_included('dominant_freq'):
            freqs = torch.linspace(0, self.fs / 2, psd.shape[1], device=psd.device)

            freq_min_hz = 1.0
            freq_max_hz = 50.0
            idx_min = max(1, int(freq_min_hz * (self.n_fft / self.fs)))
            idx_max = min(psd.shape[1], int(freq_max_hz * (self.n_fft / self.fs)))

            psd_band = psd[:, idx_min:idx_max]
            band_freqs = freqs[idx_min:idx_max]
            max_indices_in_band = psd_band.argmax(dim=1)
            dominant_freq = band_freqs[max_indices_in_band].unsqueeze(1)
            
            # Normalize to [0, 1] range (relative to Nyquist)
            dominant_freq = dominant_freq / (self.fs / 2)
            features.append(dominant_freq)
        
        # 6-7. Hjorth parameters (mobility and complexity)
        if self._is_included('hjorth_mobility') or self._is_included('hjorth_complexity'):
            dx = torch.diff(x, dim=1)
            ddx = torch.diff(dx, dim=1)
            
            var_0 = torch.var(x, dim=1, keepdim=True)
            var_1 = torch.var(dx, dim=1, keepdim=True)
            var_2 = torch.var(ddx, dim=1, keepdim=True)
            
            mobility = torch.sqrt(var_1 / (var_0 + 1e-12))
            
            if self._is_included('hjorth_mobility'):
                features.append(mobility)
            
            if self._is_included('hjorth_complexity'):
                mobility_dx = torch.sqrt(var_2 / (var_1 + 1e-12))
                complexity = mobility_dx / (mobility + 1e-12)
                features.append(complexity)

        return torch.cat(features, dim=1)

    def _compute_raw_features_multi(self, x):
        """
        Compute raw features for single- or multi-channel input.

        For 2D input [batch, input_len] (n_channels==1), delegates directly
        to _compute_raw_features unchanged. For 3D input
        [batch, n_channels, input_len], flattens batch and channel dims
        together so _compute_raw_features's per-signal math (which assumes
        strictly-2D input, e.g. compute_welch_psd's dim=1 window-axis mean)
        never has to handle an extra axis, then reshapes the result back to
        [batch, n_channels * n_base_features] in channel-major order.

        Returns:
            Raw statistical features [batch_size, n_features]
        """
        if x.dim() == 2:
            return self._compute_raw_features(x)
        batch, n_ch, length = x.shape
        flat = x.reshape(batch * n_ch, length)
        feats = self._compute_raw_features(flat)  # [batch*n_ch, n_base_features]
        return feats.reshape(batch, n_ch * feats.shape[1])

    def _apply_gate_net(self, normalized_features):
        """
        Run the gate net on normalized features, returning gate weights of
        shape [batch, n_features].

        The flat channel-major [batch, n_features] vector is reshaped to
        [batch, n_channels, n_base_features], mean-pooled across the channel
        dimension, run through the small fixed-size gate net, and tiled back
        out to [batch, n_features] so every channel gets an identical weight
        for a given base feature.
        """
        assert self.gate_net is not None  # callers only invoke this when gating is enabled

        batch_size = normalized_features.shape[0]
        reshaped = normalized_features.view(batch_size, self.n_channels, self.n_base_features)
        pooled = reshaped.mean(dim=1)                     # [batch, n_base_features]
        base_weights = self.gate_net(pooled)              # [batch, n_base_features]
        return base_weights.unsqueeze(1).expand(
            -1, self.n_channels, -1
        ).reshape(batch_size, self.n_features)

    def forward(self, x):
        """
        Compute normalized (and optionally gated) statistical features from input signal.

        Args:
            x: Input time series [batch_size, input_len] or, for multi-channel
                input, [batch_size, n_channels, input_len]

        Returns:
            Normalized statistical features [batch_size, n_features]

        Raises:
            RuntimeError: If normalization stats have not been set via
                         compute_normalization_stats() or set_normalization_stats()
        """
        if not self._normalization_set:
            raise RuntimeError(
                "Normalization statistics have not been set. "
                "Call compute_normalization_stats(signals) with your training data "
                "or set_normalization_stats(means, stds) with precomputed statistics "
                "before running forward pass."
            )

        features = self._compute_raw_features_multi(x)
        normalized_features = (features - self.feature_means) / (self.feature_stds + self.epsilon)

        if self.use_gating and self.gate_net is not None:
            gate_weights = self._apply_gate_net(normalized_features)

            # Only accumulate the EMA during training, not eval-time inference calls.
            if self.training:
                batch_mean_weights = gate_weights.mean(dim=0).detach()
                if self._gate_weight_count == 0:
                    self._running_gate_weights.copy_(batch_mean_weights)
                else:
                    self._running_gate_weights.mul_(self.gate_momentum).add_(
                        batch_mean_weights * (1 - self.gate_momentum)
                    )
                self._gate_weight_count.add_(1)

            gated_features = normalized_features * gate_weights
            return gated_features
        
        return normalized_features
