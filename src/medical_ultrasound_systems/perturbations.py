"""Reproducible synthetic RF perturbations for robustness experiments."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .simulation import RFChannelData


def _validate_channel_samples(samples: np.ndarray) -> np.ndarray:
    samples_arr = np.asarray(samples, dtype=float)
    if samples_arr.ndim != 2:
        raise ValueError("samples must have shape (n_channels, n_samples).")
    return samples_arr


def _derive_child_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int32).max, endpoint=True))


def add_awgn(
    samples: np.ndarray,
    snr_db: float,
    seed: int | None = None,
) -> np.ndarray:
    """Add white Gaussian noise to real RF channel samples."""
    samples_arr = _validate_channel_samples(samples)
    snr_db = float(snr_db)
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite.")

    signal_power = float(np.mean(samples_arr**2))
    reference_power = max(signal_power, np.finfo(float).eps)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = reference_power / max(snr_linear, np.finfo(float).eps)
    noise_std = float(np.sqrt(noise_power))

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=noise_std, size=samples_arr.shape)
    return samples_arr + noise


def dropout_channels(
    samples: np.ndarray,
    dropout_fraction: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly zero out a fraction of channels."""
    samples_arr = _validate_channel_samples(samples)
    dropout_fraction = float(dropout_fraction)
    if not 0.0 <= dropout_fraction <= 1.0:
        raise ValueError("dropout_fraction must satisfy 0 <= dropout_fraction <= 1.")

    n_channels = samples_arr.shape[0]
    n_drop = int(np.floor(dropout_fraction * n_channels))
    dropped_mask = np.zeros(n_channels, dtype=bool)
    if n_drop > 0:
        rng = np.random.default_rng(seed)
        dropped_indices = rng.choice(n_channels, size=n_drop, replace=False)
        dropped_mask[dropped_indices] = True

    dropped = samples_arr.copy()
    dropped[dropped_mask, :] = 0.0
    return dropped, dropped_mask


def apply_channel_gain_variation(
    samples: np.ndarray,
    gain_std: float = 0.1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-channel multiplicative gain variation with nonnegative clipping."""
    samples_arr = _validate_channel_samples(samples)
    gain_std = float(gain_std)
    if gain_std < 0.0:
        raise ValueError("gain_std must be non-negative.")

    rng = np.random.default_rng(seed)
    gains = rng.normal(loc=1.0, scale=gain_std, size=samples_arr.shape[0])
    gains = np.clip(gains, 0.0, None)
    return samples_arr * gains[:, np.newaxis], gains


def apply_timing_jitter_nearest(
    samples: np.ndarray,
    max_jitter_samples: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-channel integer timing shifts with zero-fill."""
    samples_arr = _validate_channel_samples(samples)
    max_jitter_samples = int(max_jitter_samples)
    if max_jitter_samples < 0:
        raise ValueError("max_jitter_samples must be non-negative.")

    n_channels, n_samples = samples_arr.shape
    if max_jitter_samples == 0:
        shifts = np.zeros(n_channels, dtype=int)
        return samples_arr.copy(), shifts

    rng = np.random.default_rng(seed)
    shifts = rng.integers(-max_jitter_samples, max_jitter_samples + 1, size=n_channels, dtype=int)
    jittered = np.zeros_like(samples_arr)

    for ch_idx, shift in enumerate(shifts):
        if shift > 0:
            jittered[ch_idx, shift:] = samples_arr[ch_idx, : n_samples - shift]
        elif shift < 0:
            offset = -shift
            jittered[ch_idx, : n_samples - offset] = samples_arr[ch_idx, offset:]
        else:
            jittered[ch_idx, :] = samples_arr[ch_idx, :]
    return jittered, shifts


def perturb_rf_channel_data(
    rf: RFChannelData,
    snr_db: float | None = None,
    dropout_fraction: float = 0.0,
    gain_std: float = 0.0,
    max_jitter_samples: int = 0,
    seed: int | None = None,
) -> RFChannelData:
    """Apply synthetic perturbations to RF data in a reproducible order."""
    parent_rng = np.random.default_rng(seed)
    stage_seeds = {
        "gain_variation": _derive_child_seed(parent_rng),
        "timing_jitter": _derive_child_seed(parent_rng),
        "channel_dropout": _derive_child_seed(parent_rng),
        "awgn": _derive_child_seed(parent_rng),
    }

    samples = np.asarray(rf.samples, dtype=float).copy()
    details: dict[str, object] = {
        "order": ["gain_variation", "timing_jitter", "channel_dropout", "awgn"],
        "requested": {
            "snr_db": snr_db,
            "dropout_fraction": float(dropout_fraction),
            "gain_std": float(gain_std),
            "max_jitter_samples": int(max_jitter_samples),
        },
        "seed": seed,
        "stage_seeds": stage_seeds,
    }

    samples, gains = apply_channel_gain_variation(
        samples=samples,
        gain_std=gain_std,
        seed=stage_seeds["gain_variation"],
    )
    details["gains"] = gains.tolist()

    samples, shifts = apply_timing_jitter_nearest(
        samples=samples,
        max_jitter_samples=max_jitter_samples,
        seed=stage_seeds["timing_jitter"],
    )
    details["timing_shifts_samples"] = shifts.tolist()

    samples, dropped_mask = dropout_channels(
        samples=samples,
        dropout_fraction=dropout_fraction,
        seed=stage_seeds["channel_dropout"],
    )
    details["dropped_channel_mask"] = dropped_mask.tolist()

    if snr_db is not None:
        samples = add_awgn(
            samples=samples,
            snr_db=float(snr_db),
            seed=stage_seeds["awgn"],
        )
        details["applied_snr_db"] = float(snr_db)
    else:
        details["applied_snr_db"] = None

    metadata = deepcopy(rf.metadata) if rf.metadata is not None else {}
    metadata["perturbations"] = details
    return RFChannelData(
        samples=samples,
        sample_rate_hz=rf.sample_rate_hz,
        geometry=rf.geometry,
        sound_speed_m_s=rf.sound_speed_m_s,
        metadata=metadata,
    )
