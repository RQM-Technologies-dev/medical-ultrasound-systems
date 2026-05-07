"""Quaternion utilities using [w, x, y, z] convention."""

from __future__ import annotations

import numpy as np


def quaternion_norm(q: np.ndarray) -> np.ndarray:
    """Return Euclidean norm of quaternion array with last dimension 4."""
    q = np.asarray(q, dtype=float)
    if q.shape[-1] != 4:
        raise ValueError("Quaternion arrays must have last dimension 4.")
    return np.linalg.norm(q, axis=-1)


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion array and safely handle zero-norm entries."""
    q = np.asarray(q, dtype=float)
    norms = quaternion_norm(q)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return q / safe_norms[..., np.newaxis]


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Return quaternion conjugate [w, -x, -y, -z]."""
    q = np.asarray(q, dtype=float)
    if q.shape[-1] != 4:
        raise ValueError("Quaternion arrays must have last dimension 4.")
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two quaternion arrays using Hamilton product."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError("Quaternion arrays must have last dimension 4.")

    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)

    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return np.stack((w, x, y, z), axis=-1)
