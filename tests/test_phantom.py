import numpy as np
import pytest

from medical_ultrasound_systems.phantom import (
    PointScatterer,
    PointScattererPhantom,
    random_point_phantom,
)


def test_point_scatterer_phantom_as_array_and_count():
    phantom = PointScattererPhantom(
        scatterers=[
            PointScatterer(x_m=-0.001, z_m=0.02, amplitude=0.5),
            PointScatterer(x_m=0.001, z_m=0.03, amplitude=1.0),
        ]
    )
    arr = phantom.as_array()
    assert phantom.n_scatterers == 2
    assert arr.shape == (2, 3)
    assert np.allclose(arr[:, 1] > 0.0, True)


def test_point_scatterer_rejects_nonpositive_depth():
    with pytest.raises(ValueError, match="z_m must be positive"):
        PointScatterer(x_m=0.0, z_m=0.0, amplitude=1.0)


def test_random_point_phantom_requires_positive_n_scatterers():
    with pytest.raises(ValueError, match="n_scatterers must be positive"):
        random_point_phantom(0)
