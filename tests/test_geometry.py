import numpy as np

from medical_ultrasound_systems.geometry import LinearArrayGeometry


def test_linear_array_geometry_centering_and_shape():
    geometry = LinearArrayGeometry(n_elements=4, pitch_m=0.001)
    positions = geometry.element_positions_m

    assert positions.shape == (4, 2)
    assert np.allclose(positions[:, 0], [-0.0015, -0.0005, 0.0005, 0.0015])
    assert np.allclose(positions[:, 1], 0.0)
    assert np.isclose(geometry.aperture_m, 0.003)
    assert np.isclose(geometry.center_index, 1.5)
