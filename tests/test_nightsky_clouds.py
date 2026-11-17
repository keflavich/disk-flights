"""Checks for nightsky.clouds (config-driven extra VoxelFields)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nightsky import clouds, gas, spectral  # noqa: E402

OBS = np.array([100.0, -20.0, 5.0])


def _column(f, direction, length, n=20000):
    s = (np.arange(n) + 0.5) * length / n
    return f.sample(OBS + s[:, None] * np.asarray(direction, float))[:, 0].sum() * length / n


@pytest.mark.parametrize("profile", ["gauss", "uniform", "shell"])
def test_extinction_av_center(profile):
    grid = spectral.default_grid()
    f = clouds.cloud_field({"kind": "extinction", "profile": profile, "radius_pc": 1.5, "av_center": 4.0, "n_vox": 81},
                           grid, OBS)
    assert f.kind == "extinction"
    for d in ([1, 0, 0], [0, 0, -1], np.array([1, 1, 1]) / np.sqrt(3)):
        assert _column(f, d, 10.0) == pytest.approx(4.0, rel=0.03)
    i550 = int(np.argmin(np.abs(grid.wl - 550.0)))
    assert f.spectra[0, i550] == pytest.approx(1 / 1.0857, rel=0.05)


def test_emission_measure_uniform():
    grid = spectral.default_grid()
    f = clouds.cloud_field({"kind": "emission", "profile": "uniform", "radius_pc": 2.0, "n_e": 300.0, "t_e": 8000.0,
                            "n_vox": 64}, grid, OBS)
    # H-alpha radiance from the centre to the edge = halpha_coeff * EM (EM = n_e^2 * R)
    ha = _column(f, [0, 1, 0], 4.0)
    assert ha == pytest.approx(gas.halpha_coeff(8000.0) * 300.0 ** 2 * 2.0, rel=0.05)


def test_bad_kind():
    with pytest.raises(ValueError):
        clouds.cloud_field({"kind": "smoke"}, spectral.default_grid(), OBS)
