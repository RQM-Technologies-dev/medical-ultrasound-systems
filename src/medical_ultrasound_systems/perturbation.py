"""Synthetic RF perturbations for robustness experiments."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .simulation import RFChannelData


def copy_rf_with_samples(
    rf: RFChannelData,
    samples: np.ndarray,
    metadata_update: dict | None = None,
) -> RFChannelData:
    """Return a new RFChannelData with replaced sample values."""
    samples_arr = np.asarray(samples, dtype=float)
    if samples_arr.shape != rf.samples.shape:
        raise ValueError("samples must match rf.samples shape.")

    metadata = deepcopy(rf.metadata) if rf.metadata is not None else {}
    if metadata_update:
        metadata.update(metadata_update)
    return RFChannelData(
        samples=samples_arr,
        sample_rate_hz=rf.sample_rate_hz,
        geometry=rf.geometry,
        sound_speed_m_s=rf.sound_speed_m_s,
        metadata=metadata,
    )


def add_awgn(
    rf: RFChannelData,
    snr_db: float,
    seed: int | None = None,
) -> RFChannelData:
    """Add white Gaussian noise to synthetic RF samples."""
    snr_db = float(snr_db)
    rng = np.random.default_rng(seed)

    signal_power = float(np.mean(rf.samples**2))
    baseline = max(signal_power, np.finfo(float).eps)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = baseline / max(snr_linear, np.finfo(float).eps)
    noise_std = np.sqrt(noise_power)
    noisy = rf.samples + rng.normal(0.0, noise_std, size=rf.samples.shape)
    return copy_rf_with_samples(
        rf=rf,
        samples=noisy,
        metadata_update={
            "perturbation": "awgn",
            "snr_db": snr_db,
            "seed": seed,
        },
    )


def drop_channels(
    rf: RFChannelData,
    drop_fraction: float,
    seed: int | None = None,
) -> RFChannelData:
    """Randomly zero a fraction of channels while preserving RF dimensions."""
    drop_fraction = float(drop_fraction)
    if not 0.0 <= drop_fraction < 1.0:
        raise ValueError("drop_fraction must satisfy 0 <= drop_fraction < 1.")

    rng = np.random.default_rng(seed)
    n_channels = rf.n_channels
    if drop_fraction == 0.0 or n_channels == 0:
        dropped_idx = np.array([], dtype=int)
    else:
        n_drop = int(np.floor(drop_fraction * n_channels))
        n_drop = max(1, n_drop)
        n_drop = min(n_drop, n_channels - 1)
        dropped_idx = np.sort(rng.choice(n_channels, size=n_drop, replace=False))

    dropped = rf.samples.copy()
    if dropped_idx.size > 0:
        dropped[dropped_idx, :] = 0.0

    return copy_rf_with_samples(
        rf=rf,
        samples=dropped,
        metadata_update={
            "perturbation": "drop_channels",
            "drop_fraction": drop_fraction,
            "dropped_channel_indices": dropped_idx.tolist(),
            "seed": seed,
        },
    )


def apply_gain_jitter(
    rf: RFChannelData,
    gain_std: float = 0.05,
    seed: int | None = None,
) -> RFChannelData:
    """Apply random per-channel multiplicative gain jitter around unity."""
    gain_std = float(gain_std)
    if gain_std < 0.0:
        raise ValueError("gain_std must be non-negative.")

    rng = np.random.default_rng(seed)
    gains = 1.0 + rng.normal(loc=0.0, scale=gain_std, size=rf.n_channels)
    jittered = rf.samples * gains[:, np.newaxis]
    return copy_rf_with_samples(
        rf=rf,
        samples=jittered,
        metadata_update={
            "perturbation": "gain_jitter",
            "gain_std": gain_std,
            "seed": seed,
        },
    )
