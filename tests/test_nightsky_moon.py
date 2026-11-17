"""Checks for nightsky.moon (position, phase law, spectrum, shaded disk)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from nightsky import frames, moon, spectral  # noqa: E402


def test_position_and_phase():
    assert np.allclose(moon.ecliptic_celestial(37.0), frames.sun_celestial(37.0))
    assert moon.phase_angle_deg(180.0) == 0.0
    assert moon.phase_angle_deg(90.0) == 90.0 and moon.phase_angle_deg(270.0) == 90.0
    assert moon.vmag(0.0) == pytest.approx(-12.73)
    assert moon.vmag(90.0) == pytest.approx(-12.73 + 2.34 + 4e-9 * 90 ** 4)  # ~ -10.1, first quarter


def test_spectrum_vmag_and_colour():
    grid = spectral.default_grid()
    s = moon.spectrum(-12.73, grid)
    assert spectral.vmag_of(s, grid) == pytest.approx(-12.73, abs=1e-6)
    sun = spectral.blackbody_irradiance(1.0, 5772.0, grid)
    i450, i650 = (int(np.argmin(np.abs(grid.wl - w))) for w in (450.0, 650.0))
    assert s[i650] / s[i450] > sun[i650] / sun[i450]  # redder than sunlight


def test_disk_weights_full_and_quarter():
    d = np.array([1.0, 0.0, 0.0])
    # full Moon: the Sun is opposite the Moon as seen from Earth
    full =moon._disk_integral(d, -d, moon.RADIUS_DEG)
    r = np.radians(moon.RADIUS_DEG)
    # Lommel-Seeliger at full phase: mu0 = mu -> weight 1/2 over the whole disk
    assert full == pytest.approx(0.5 * np.pi * r * r, rel=0.01)
    quarter = moon._disk_integral(d, np.array([0.0, 1.0, 0.0]), moon.RADIUS_DEG)
    assert 0.2 * full < quarter < 0.6 * full
    # the lit half of a quarter Moon faces the Sun (+y)
    pts = np.array([[1.0, 0.5 * r, 0.0], [1.0, -0.5 * r, 0.0]])
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    w = moon.disk_weights(pts, d, [0.0, 1.0, 0.0])
    assert w[0] > 0 and w[1] == 0


def test_inset_never_darker_than_veil():
    import views  # flat module in src/
    grid = spectral.default_grid()
    view = views.View("perspective", 120, 120, azimuth_deg=0.0, pitch_deg=0.0, fov_deg=3.0)
    d = np.array([1.0, 0.0, 0.0])
    sun = np.array([0.0, 1.0, 0.0])  # quarter Moon, lit side toward +y
    irr = moon.spectrum(-10.1, grid)
    near = view.dirs @ d > np.cos(np.radians(moon.RADIUS_DEG))
    # saturated glare halo: the disk stays white (no dark spot), phase hidden
    white = np.full((120, 120, 3), 255, np.uint8)
    out = moon.inset(white, view, d, sun, irr, grid)
    assert (out >= white).all()
    # dark surroundings: the phase shows, lit half brighter than the night half
    dark = np.full((120, 120, 3), 20, np.uint8)
    out = moon.inset(dark, view, d, sun, irr, grid).astype(float).mean(-1)
    side = view.dirs[..., 1]
    assert out[near & (side > 0)].mean() > out[near & (side < 0)].mean() + 50
    assert (out >= 20).all()
