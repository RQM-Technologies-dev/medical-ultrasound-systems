import numpy as np

from medical_ultrasound_systems.quaternion import (
    quaternion_multiply,
    quaternion_norm,
    quaternion_normalize,
)


def test_quaternion_multiply_identity():
    q = np.array([0.9, 0.1, 0.2, 0.3])
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quaternion_multiply(q, identity), q)
    assert np.allclose(quaternion_multiply(identity, q), q)


def test_quaternion_normalization_unit_norm():
    q = np.array([[2.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]])
    qn = quaternion_normalize(q)
    assert np.allclose(quaternion_norm(qn), 1.0)
