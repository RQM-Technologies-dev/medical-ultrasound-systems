"""Lightweight analytic-signal helpers for ultrasound research pipelines."""

from __future__ import annotations

import numpy as np


def analytic_signal_fft(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a simple FFT-based analytic signal approximation.

    This is a lightweight research implementation of a Hilbert-transform style
    analytic-signal lift and is intended for synthetic benchmarking workflows,
    not a calibrated scanner signal-processing chain.
    """
    signal = np.asarray(signal)
    if np.iscomplexobj(signal):
        return signal.astype(np.complex128, copy=False)

    signal = signal.astype(float, copy=False)
    axis = int(axis)
    n = signal.shape[axis]
    if n == 0:
        return signal.astype(np.complex128)

    spectrum = np.fft.fft(signal, axis=axis)
    h = np.zeros(n, dtype=float)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0

    shape = [1] * signal.ndim
    shape[axis] = n
    return np.fft.ifft(spectrum * h.reshape(shape), axis=axis)


def instantaneous_phase(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return instantaneous phase from real or analytic-complex input."""
    signal_arr = np.asarray(signal)
    analytic = signal_arr if np.iscomplexobj(signal_arr) else analytic_signal_fft(signal_arr, axis=axis)
    return np.angle(analytic)


def instantaneous_amplitude(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return instantaneous amplitude from real or analytic-complex input."""
    signal_arr = np.asarray(signal)
    analytic = signal_arr if np.iscomplexobj(signal_arr) else analytic_signal_fft(signal_arr, axis=axis)
    return np.abs(analytic)
