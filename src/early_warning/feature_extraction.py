"""Advanced feature extraction for NDVI time series classification."""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.stats import entropy
from typing import Tuple


def extract_seasonal_features(curve: np.ndarray) -> np.ndarray:
    """Extract seasonal decomposition features."""
    n = len(curve)
    if n < 4:
        return np.zeros(5, dtype=np.float32)

    # Estimate period (assume ~12 for annual cycle in monthly data)
    period = min(12, n // 2)
    if period < 2:
        return np.zeros(5, dtype=np.float32)

    # Moving average for trend
    trend = np.convolve(curve, np.ones(period)/period, mode='same')

    # Detrended = seasonal + residual
    detrended = curve - trend

    # Seasonal strength: var(detrended) / var(curve)
    curve_var = np.var(curve)
    detrended_var = np.var(detrended)
    seasonal_strength = 1 - (detrended_var / (curve_var + 1e-8))

    # Trend strength
    residual = curve - trend - detrended
    residual_var = np.var(residual)
    trend_strength = 1 - (residual_var / (detrended_var + 1e-8))

    # Peak timing (when does max occur relative to start)
    peak_idx = np.argmax(curve)
    peak_timing = peak_idx / n

    # Trough timing
    trough_idx = np.argmin(curve)
    trough_timing = trough_idx / n

    # Number of peaks (local maxima)
    peaks, _ = signal.find_peaks(curve, distance=max(3, n//6))
    n_peaks = len(peaks) / (n / 12)  # Normalize by expected seasons

    return np.array([
        seasonal_strength,
        trend_strength,
        peak_timing,
        trough_timing,
        n_peaks
    ], dtype=np.float32)


def extract_frequency_features(curve: np.ndarray) -> np.ndarray:
    """Extract frequency domain features using FFT."""
    n = len(curve)
    if n < 4:
        return np.zeros(6, dtype=np.float32)

    # FFT
    fft_vals = np.fft.fft(curve - np.mean(curve))
    fft_mag = np.abs(fft_vals[:n//2])

    # Dominant frequency
    if len(fft_mag) > 1:
        dominant_freq_idx = np.argmax(fft_mag[1:]) + 1
        dominant_freq = dominant_freq_idx / n
        dominant_power = fft_mag[dominant_freq_idx]
    else:
        dominant_freq = 0
        dominant_power = 0

    # Spectral entropy (measure of periodicity)
    total_power = np.sum(fft_mag[1:]) + 1e-8
    normalized_power = fft_mag[1:] / total_power
    spectral_entropy = entropy(normalized_power + 1e-10)

    # Power in different bands
    low_band = int(len(fft_mag) * 0.1)
    mid_band = int(len(fft_mag) * 0.3)

    low_power = np.sum(fft_mag[1:low_band+1]) if low_band > 0 else 0
    mid_power = np.sum(fft_mag[low_band+1:mid_band+1]) if mid_band > low_band else 0
    high_power = np.sum(fft_mag[mid_band+1:]) if len(fft_mag) > mid_band + 1 else 0

    total_band_power = low_power + mid_power + high_power + 1e-8

    return np.array([
        dominant_freq,
        dominant_power / (np.max(fft_mag) + 1e-8),
        spectral_entropy / np.log(len(fft_mag) + 1),
        low_power / total_band_power,
        mid_power / total_band_power,
        high_power / total_band_power
    ], dtype=np.float32)


def extract_rolling_features(curve: np.ndarray, windows: list = None) -> np.ndarray:
    """Extract rolling statistics at multiple window sizes."""
    if windows is None:
        windows = [3, 6, 12]

    n = len(curve)
    features = []

    for w in windows:
        if n <= w:
            features.extend([0, 0, 0, 0])
            continue

        # Rolling mean
        rolling_mean = np.convolve(curve, np.ones(w)/w, mode='valid')

        # Rolling std
        rolling_var = np.convolve(curve**2, np.ones(w)/w, mode='valid') - rolling_mean**2
        rolling_std = np.sqrt(np.maximum(rolling_var, 0))

        # Features: mean of rolling stats, max variability
        features.extend([
            np.mean(rolling_std),           # Average volatility
            np.max(rolling_std),            # Max volatility
            np.mean(np.abs(np.diff(rolling_mean))),  # Mean change rate
            np.max(rolling_mean) - np.min(rolling_mean)  # Range of smoothed
        ])

    return np.array(features, dtype=np.float32)


def extract_change_point_features(curve: np.ndarray) -> np.ndarray:
    """Extract features related to abrupt changes."""
    n = len(curve)
    if n < 4:
        return np.zeros(5, dtype=np.float32)

    # First differences (rate of change)
    diff = np.diff(curve)

    # Second differences (acceleration)
    diff2 = np.diff(diff)

    # Max absolute change
    max_change = np.max(np.abs(diff))

    # Mean absolute change
    mean_change = np.mean(np.abs(diff))

    # Number of significant changes (> 2 std)
    std_change = np.std(diff)
    if std_change > 1e-8:
        significant_changes = np.sum(np.abs(diff) > 2 * std_change)
    else:
        significant_changes = 0

    # Max acceleration
    max_accel = np.max(np.abs(diff2)) if len(diff2) > 0 else 0

    # Sudden drop detection: large negative change followed by sustained low values
    drops = np.where(diff < -2 * std_change)[0] if std_change > 1e-8 else np.array([])
    n_drops = len(drops)

    return np.array([
        max_change,
        mean_change,
        significant_changes / n,  # Normalized
        max_accel,
        n_drops
    ], dtype=np.float32)


def extract_shape_features(curve: np.ndarray) -> np.ndarray:
    """Extract shape-based features."""
    n = len(curve)
    if n < 3:
        return np.zeros(6, dtype=np.float32)

    # Normalized curve
    curve_range = np.max(curve) - np.min(curve) + 1e-8
    curve_norm = (curve - np.min(curve)) / curve_range

    # Area under curve (normalized)
    auc = np.mean(curve_norm)

    # Skewness
    skew = np.mean(((curve - np.mean(curve)) / (np.std(curve) + 1e-8)) ** 3)

    # Kurtosis
    kurt = np.mean(((curve - np.mean(curve)) / (np.std(curve) + 1e-8)) ** 4) - 3

    # Number of zero crossings in detrended signal
    t = np.arange(n)
    slope, intercept = np.polyfit(t, curve, 1)
    detrended = curve - (slope * t + intercept)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(detrended))) > 0)

    # Peak-to-mean ratio
    peak_to_mean = np.max(curve) / (np.mean(curve) + 1e-8)

    # Recovery index: how much does it recover after a drop
    min_idx = np.argmin(curve)
    if min_idx < n - 1:
        recovery = (curve[-1] - curve[min_idx]) / curve_range
    else:
        recovery = 0

    return np.array([
        auc,
        skew,
        kurt,
        zero_crossings / n,
        peak_to_mean,
        recovery
    ], dtype=np.float32)


def extract_all_features(curve: np.ndarray) -> np.ndarray:
    """Extract all features from a curve."""
    seasonal = extract_seasonal_features(curve)
    freq = extract_frequency_features(curve)
    rolling = extract_rolling_features(curve)
    change = extract_change_point_features(curve)
    shape = extract_shape_features(curve)

    # Original basic features
    t = np.arange(len(curve), dtype=np.float32)
    slope = np.polyfit(t, curve, 1)[0] if len(curve) > 1 else 0
    amp = float(np.max(curve) - np.min(curve))
    mean_val = float(np.mean(curve))
    std_val = float(np.std(curve))
    p10 = float(np.percentile(curve, 10))
    p90 = float(np.percentile(curve, 90))
    basic = np.array([mean_val, std_val, amp, slope, p90 - p10], dtype=np.float32)

    return np.concatenate([basic, seasonal, freq, rolling, change, shape])


def get_feature_names() -> list:
    """Get names of all features."""
    return (
        ['mean', 'std', 'amplitude', 'slope', 'seasonal_range'] +
        ['seasonal_strength', 'trend_strength', 'peak_timing', 'trough_timing', 'n_peaks'] +
        ['dominant_freq', 'dominant_power_ratio', 'spectral_entropy',
         'low_band_power', 'mid_band_power', 'high_band_power'] +
        [f'roll_std_{w}' for w in [3, 6, 12] for _ in range(4)] +
        ['max_change', 'mean_change', 'significant_changes', 'max_accel', 'n_drops'] +
        ['auc', 'skew', 'kurt', 'zero_crossings', 'peak_to_mean', 'recovery']
    )
