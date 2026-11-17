"""Naked-eye detection threshold for point sources (camera.deposit_points_eye).

    cd /orange/adamginsburg/gc/visualization/src && PYTHONPATH=$PWD python3 -m pytest -q ../tests/test_nightsky_visibility.py
"""
import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from nightsky import camera, config, frames, spectral  # noqa: E402


def garstang_nl(mu_v):
    """Sky brightness (nL) of mu_v V mag/arcsec^2 (Garstang 1986; Schaefer 1990)."""
    return 34.08 * np.exp(20.7233 - 0.92104 * mu_v)


def test_threshold_sanity():
    assert np.isclose(camera.NL_CD_M2, 3.1831e-6, rtol=1e-4)
    m22 = camera.limiting_vmag(54.0 * camera.NL_CD_M2)
    assert 6.3 < m22 < 6.8, m22
    m17 = camera.limiting_vmag(garstang_nl(17.3) * camera.NL_CD_M2)
    assert 3.5 < m17 < 4.5, m17
    # dark-sky limit near 8 (Schaefer's B = 0 limit), monotone fainter -> brighter backgrounds
    lb = np.geomspace(1e-7, 1e3, 200)
    m = camera.limiting_vmag(lb)
    assert np.all(np.diff(m) < 0) and 7.5 < m[0] < 8.2
    # visibility_factor 2.5 -> limit 1 mag brighter
    assert np.isclose(camera.limiting_vmag(1e-3, 10 ** 0.4), camera.limiting_vmag(1e-3) - 1)
    # the package's V = 0 illuminance matches Schaefer's zero point (-16.57) to ~0.05 mag
    grid = spectral.default_grid()
    e0 = spectral.to_xyz(spectral.vmag_irradiance(0.0, 5800.0, grid), grid)[1]
    assert abs(-16.57 - 2.5 * np.log10(e0 / camera.FC_LUX)) < 0.05


def test_probability_monotone():
    grid = spectral.default_grid()
    lb = 1e-2
    v = np.linspace(-2, 10, 400)
    e = spectral.to_xyz(spectral.vmag_irradiance(v, 5800.0, grid), grid)[:, 1]
    p = camera.detection_probability(e, lb)
    assert np.all(np.diff(p) <= 0) and p[0] > 0.999 and p[-1] < 1e-6
    mlim = camera.limiting_vmag(lb)
    assert np.isclose(np.interp(0.5, p[::-1], v[::-1]), mlim, atol=0.06)
    # brighter background -> lower probability for the same star
    pb = camera.detection_probability(e, 10 * lb)
    assert np.all(pb <= p + 1e-12)


def _toy(n=3000, seed=0):
    grid = spectral.default_grid(6)
    v = camera.make_view({"kind": "perspective", "width": 96, "height": 64, "azimuth_deg": 0, "pitch_deg": 40, "fov_deg": 40})
    rng = np.random.default_rng(seed)
    alt, az = rng.uniform(25, 55, n), rng.uniform(-22, 22, n)
    d = frames.altaz_to_vec(alt, az)
    e = spectral.vmag_irradiance(rng.uniform(-1, 9, n), rng.uniform(3500, 20000, n), grid)
    bg = np.broadcast_to(spectral.surface_brightness_radiance(17.3, 5000.0, grid), (64, 96, grid.n)).astype(np.float32)
    return grid, v, d, e, bg


def test_flux_conservation_and_counts():
    grid, v, d, e, bg = _toy()
    pc = config.DEFAULTS["points"]
    ref = bg.copy()
    n0 = camera.deposit_points(ref, v, d, e, pc, grid)
    img = bg.copy()
    n1, info = camera.deposit_points_eye(img, v, d, e, pc, grid, bg)
    assert n0 == n1 == info["n_points_in_frame"] > 0
    f0 = (ref * v.omega[..., None]).sum((0, 1))
    f1 = (img * v.omega[..., None]).sum((0, 1))
    assert np.allclose(f1, f0, rtol=1e-3), f1 / f0
    # the sky dominates f0: check the star light alone
    s0 = ((ref.astype(float) - bg) * v.omega[..., None]).sum((0, 1))
    s1 = ((img.astype(float) - bg) * v.omega[..., None]).sum((0, 1))
    assert np.allclose(s1, s0, rtol=1e-5), s1 / s0
    # with a 17.3 V/arcsec^2 sky only the stars brighter than ~V 4 are points
    vm = np.atleast_1d(spectral.vmag_of(e, grid))
    row, col = v.project(d)
    inside = np.isfinite(row)
    assert 0 < info["n_points_visible"] < 0.6 * inside.sum()
    assert abs(info["n_points_visible"] - (vm[inside] < info["naked_eye_limit_V_star_bg_median"]).sum()) < 0.15 * inside.sum()
    # the peak of a faint star is gone, the unresolved light is smooth
    assert info["unresolved_blur_px"] >= 1.5
    faint = np.flatnonzero(vm > 6)[:20]  # isolated faint stars: the peak is spread out
    ref, img = bg.copy(), bg.copy()
    camera.deposit_points(ref, v, d[faint], e[faint], pc, grid)
    camera.deposit_points_eye(img, v, d[faint], e[faint], pc, grid, bg)
    assert (img - bg).max() < 0.25 * (ref - bg).max()  # 0.6 px PSF -> 1.5 px blur: peak ratio ~0.14


def test_allsky_mask_conservation():
    grid = spectral.default_grid(4)
    v = camera.make_view({"kind": "allsky", "width": 64, "height": 64})
    rng = np.random.default_rng(2)
    d = rng.normal(size=(2000, 3))
    d[:, 2] = np.abs(d[:, 2])
    d /= np.linalg.norm(d, axis=1)[:, None]
    e = spectral.vmag_irradiance(rng.uniform(0, 9, len(d)), 6000.0, grid)
    bg = np.broadcast_to(spectral.surface_brightness_radiance(18.0, 5000.0, grid), (64, 64, grid.n)).astype(np.float32) * v.valid[..., None]
    pc = {**config.DEFAULTS["points"], "unresolved_blur_deg": 5.0}
    ref, img = bg.copy(), bg.copy()
    camera.deposit_points(ref, v, d, e, pc, grid)
    camera.deposit_points_eye(img, v, d, e, pc, grid, bg)
    om = np.where(v.valid, v.omega, 0)[..., None]
    assert np.allclose((img * om).sum((0, 1)), (ref * om).sum((0, 1)), rtol=1e-3)
    assert not img[~v.valid].any()


def test_bright_stars_unchanged():
    """Stars far above the threshold deposit exactly as before (p = 1)."""
    grid, v, d, e, bg = _toy(50)
    e = e * 1e4
    pc = config.DEFAULTS["points"]
    ref, img = bg.copy(), bg.copy()
    camera.deposit_points(ref, v, d, e, pc, grid)
    _, info = camera.deposit_points_eye(img, v, d, e, pc, grid, bg)
    assert np.isclose(info["n_points_visible"], info["n_points_in_frame"])
    assert np.allclose(img, ref, rtol=1e-6, atol=1e-6 * float(ref.max()))


from test_nightsky_smoke import fakes, tiny_cfg  # noqa: E402,F401
from nightsky import compose  # noqa: E402


def test_compose_modes(tmp_path, fakes):  # noqa: F811
    """camera mode ignores points.visibility (identical output); eye mode records the counts."""
    xyz = {}
    for vis in (True, False):
        cfg = tiny_cfg(tmp_path, foreground=None, camera={"mode": "camera"}, points={"visibility": vis},
                       output_dir=str(tmp_path / f"cam{vis}"))
        cfg["foreground"] = None
        r = compose.render(cfg, views=["persp"], cpus=1)["views"]["persp"]
        xyz[vis] = np.load(r["paths"][1])["xyz"]
        assert "visibility" not in r["info"]
    assert np.array_equal(xyz[True], xyz[False])
    cfg = tiny_cfg(tmp_path, foreground=None, camera={"glare": False}, output_dir=str(tmp_path / "eye"))
    cfg["foreground"] = None
    res = compose.render(cfg, cpus=1)
    n_in = 0
    for name, r in res["views"].items():
        vi = r["info"]["visibility"]
        assert 0 <= vi["n_points_visible"] <= vi["n_points_in_frame"] == r["info"]["points_deposited"], name
        assert np.isfinite(vi["naked_eye_limit_V_sky_median"])
        n_in += vi["n_points_in_frame"]
    assert n_in > 0
    assert 5 < res["scene"]["naked_eye_limit_V_zenith"] < 8
