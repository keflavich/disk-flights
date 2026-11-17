"""Smoke tests for the nightsky integration layer (config, camera, compose, cli).

The source / atmosphere / foreground modules are replaced by tiny stand-ins (monkeypatched
into sys.modules) so the whole pipeline runs on the login node in seconds:

    cd /orange/adamginsburg/gc/visualization && python3 -m pytest -q tests/test_nightsky_smoke.py

NIGHTSKY_REAL=1 additionally runs the real modules at very low resolution (worker only).
"""
import os
import sys
import types

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from nightsky import camera, compose, config, frames, sky, spectral  # noqa: E402


# ----------------------------------------------------------------------------- stand-ins
def _fake_starfield():
    m = types.ModuleType("nightsky.starfield")

    def build_starfield(cfg, grid, galaxy=None):
        rng = np.random.default_rng(0)
        n = 200
        d = rng.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        irr = spectral.vmag_irradiance(rng.uniform(-3, 6, n), 6000.0, grid)
        pts = sky.PointSources(d, irr, [f"s{i}" for i in range(n)])
        diffuse = np.zeros((12 * 4 * 4, grid.n))
        obs, R = compose.scenario_geometry({"scenario": {**config.DEFAULTS["scenario"], "mode": cfg["mode"]}})
        return sky.TOASky(grid, pts, diffuse), {"R_gc_to_cel": R, "observer_pc": obs, "inner_factor": 0.5, "r_res_pc": 10.0}

    m.build_starfield = build_starfield
    m.resolve_config = lambda cfg: dict(cfg or {})
    return m


def _fake_galaxy():
    m = types.ModuleType("nightsky.galaxy")

    class GalaxyModel:
        def __init__(self, grid, preset="milky_way", fields=(), **kw):
            self.grid = grid

        def render_map(self, observer_pc, nside, R, r_res_pc=0.0, inner_factor=1.0, **kw):
            d = frames.healpix_dirs(nside) @ R  # galactocentric directions of the celestial pixels
            band = np.exp(-0.5 * (d[:, 2] / 0.1) ** 2)
            L = spectral.surface_brightness_radiance(19.0, 4500.0, self.grid)
            return sky.TOASky(self.grid, sky.PointSources.empty(self.grid.n), band[:, None] * L[None])

    m.GalaxyModel = GalaxyModel
    return m


def _fake_gas():
    m = types.ModuleType("nightsky.gas")
    m.gc_ionized_gas = lambda grid, seed=0, resolution="default": []
    m.gc_molecular_clouds = lambda grid, seed=0, resolution="default": []
    m.load_cached = lambda grid=None, seed=0, resolution="default", **kw: []
    return m


def _fake_atmosphere():
    m = types.ModuleType("nightsky.atmosphere")
    m.v_mag_arcsec2 = spectral.radiance_to_mag_arcsec2

    class Atmosphere:
        def __init__(self, preset, grid):
            self.grid, self.name, self.preset = grid, str(preset), {}
            self.tau = 0.1 * (550.0 / grid.wl) ** 4

        def transmittance(self, d):
            d = np.asarray(d, float).reshape(-1, 3)
            mu = d[:, 2:3]
            return np.where(mu > 0, np.exp(-self.tau[None] / np.maximum(mu, 0.05)), 0.0)

        def sky(self, toa, R, d):
            d = np.asarray(d, float).reshape(-1, 3)
            T = self.transmittance(d)
            ins = np.broadcast_to(spectral.surface_brightness_radiance(22.0, 5000.0, self.grid), T.shape) * (1 - T + 0.2)
            return {"T": T, "inscatter": ins, "airglow": 0.1 * ins}

        def point_irradiance(self, toa, R):
            p = toa.points.rotated(R)
            return sky.PointSources(p.dirs, p.irr * self.transmittance(p.dirs), list(p.names))

        def ground_irradiance(self, toa, R):
            d = frames.healpix_dirs(8)
            d = d[d[:, 2] > 0]
            r = self.sky(toa, R, d)
            L = r["T"] * toa.diffuse(d @ R) + r["inscatter"] + r["airglow"]
            p = self.point_irradiance(toa, R)
            return (L * d[:, 2:3]).sum(0) * 4 * np.pi / (12 * 64) + (p.irr * np.clip(p.dirs[:, 2:3], 0, None)).sum(0)

        def aerial_perspective(self, d, dist_m, toa, R):
            t = np.exp(-np.asarray(dist_m, float)[:, None] / 3e4 * self.tau[None])
            return t, (1 - t) * spectral.surface_brightness_radiance(22.0, 5000.0, self.grid)[None]

    m.Atmosphere = Atmosphere
    return m


def _fake_foreground():
    m = types.ModuleType("nightsky.foreground")

    def render_foreground(scene_cfg, view, sky_fn, lights, grid, spp=64, cpus=None):
        el = view.dirs[..., 2]
        alpha = (el < 0).astype(np.float32)
        depth = np.where(el < 0, 2.0 / np.maximum(-el, 1e-3), np.inf)
        up = frames.healpix_dirs(8)
        up = up[up[:, 2] > 0]
        e = (sky_fn(up) * up[:, 2:3]).sum(0) * 4 * np.pi / (12 * 64)
        if lights is not None and len(lights):
            e = e + (lights.irr * np.clip(lights.dirs[:, 2:3], 0, None)).sum(0)
        rad = alpha[..., None] * 0.2 * e[None, None] / np.pi
        return {"radiance": rad, "alpha": alpha, "depth": depth, "normal": np.zeros(view.dirs.shape)}

    m.render_foreground = render_foreground
    m.resolve_scene = lambda scene_cfg: dict(scene_cfg or {})
    return m


@pytest.fixture
def fakes(monkeypatch):
    for name, mk in (("starfield", _fake_starfield), ("galaxy", _fake_galaxy), ("gas", _fake_gas),
                     ("atmosphere", _fake_atmosphere), ("foreground", _fake_foreground)):
        monkeypatch.setitem(sys.modules, f"nightsky.{name}", mk())


def tiny_cfg(tmp_path, **over):
    base = {
        "name": "smoke", "output_dir": str(tmp_path / "out"),
        "grid": {"n_bins": 4, "lo": 380.0, "hi": 780.0},
        "galaxy": {"nside": 8, "use_gas": True},
        "foreground": {"type": "meadow"},
        "foreground_render": {"spp": 4, "env_nside": 8},
        "views": [{"name": "persp", "kind": "perspective", "width": 64, "height": 48, "fov_deg": 60,
                   "azimuth_deg": "auto", "pitch_deg": "auto", "target": "sgr_a_star"},
                  {"name": "allsky", "kind": "allsky", "width": 48, "height": 48},
                  {"name": "pano", "kind": "panorama", "width": 96, "height": 24}],
        "sky": {"subsample": 2},
    }
    return config.load_config(None, (), config.deep_merge(base, over))


# ----------------------------------------------------------------------------- tests
def test_config_merge_set_hash(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("camera:\n  mode: camera\nviews:\n  - {name: a, kind: allsky, width: 32, height: 32}\n")
    cfg = config.load_config(str(p), ["camera.iso=6400", "views[0].width=40", "foreground={type: forest}"])
    assert cfg["name"] == "c" and cfg["camera"]["mode"] == "camera" and cfg["camera"]["iso"] == 6400
    assert cfg["camera"]["f_number"] == config.DEFAULTS["camera"]["f_number"]
    assert cfg["views"][0]["width"] == 40 and cfg["foreground"] == {"type": "forest"}
    h1 = config.section_hash(cfg, "camera")
    cfg2 = config.load_config(str(p), ["camera.iso=6400.0", "views[0].width=40"])
    assert config.section_hash(cfg2, "camera") == h1
    assert config.section_hash(cfg2, "foreground") != config.section_hash(cfg, "foreground")


def test_deposit_conserves_flux():
    grid = spectral.default_grid(6)
    v = camera.make_view({"kind": "perspective", "width": 80, "height": 60, "azimuth_deg": 0, "pitch_deg": 45, "fov_deg": 40})
    d = frames.altaz_to_vec(np.array([45.0, 50.0]), np.array([0.0, 5.0]))
    e = spectral.vmag_irradiance(np.array([0.0, 2.0]), 6000.0, grid)
    img = np.zeros((60, 80, grid.n), np.float32)
    camera.deposit_points(img, v, d, e, {"psf_sigma_px": 1.0, "radius_px": 4}, grid)
    assert np.allclose((img * v.omega[..., None]).sum((0, 1)), e.sum(0), rtol=1e-4)


def test_glare_kernel_and_uniform_field():
    v = camera.make_view({"kind": "perspective", "width": 64, "height": 48, "azimuth_deg": 0, "pitch_deg": 30, "fov_deg": 40})
    k = camera.glare_kernel(v, {})
    assert 0.9 < k.sum() <= 1.0 and k.min() >= 0
    img = np.ones((48, 64, 2), np.float32)
    out = camera.convolve(img, k, v.valid)
    assert np.allclose(out, 1.0, rtol=1e-3)


def test_pipeline_end_to_end(tmp_path, fakes):
    cfg = tiny_cfg(tmp_path)
    res = compose.render(cfg, cpus=1)
    assert set(res["views"]) == {"persp", "allsky", "pano"}
    for name, r in res["views"].items():
        png, npz, js = r["paths"]
        assert os.path.exists(png) and os.path.exists(npz) and os.path.exists(js)
        z = np.load(npz)
        assert z["xyz"].shape[-1] == 3 and np.isfinite(z["xyz"]).all() and (z["xyz"][..., 1] >= 0).all()
        info = r["info"]
        assert info["sky_luminance"]["median_cd_m2"] > 0
        assert 0 <= info["camera"]["photopic_fraction"] <= 1
    s = res["scene"]
    assert s["ground_illuminance_lux"] > 0 and np.isfinite(s["zenith_V_mag_arcsec2"])
    # second run: every stage from the cache, identical output
    xyz1 = np.load(res["views"]["persp"]["paths"][1])["xyz"]
    res2 = compose.render(cfg, cpus=1)
    assert np.allclose(np.load(res2["views"]["persp"]["paths"][1])["xyz"], xyz1)
    assert any(f.startswith("foreground_persp") for f in os.listdir(os.path.join(cfg["output_dir"], "cache")))


def test_camera_mode_and_no_foreground(tmp_path, fakes):
    cfg = tiny_cfg(tmp_path, camera={"mode": "camera", "spikes": 4, "bloom": 0.05}, foreground=None)
    cfg["foreground"] = None
    res = compose.render(cfg, views=["allsky"], cpus=1)
    info = res["views"]["allsky"]["info"]
    assert info["camera"]["mode"] == "camera" and info["foreground_fraction"] == 0


def test_foreground_light_fallback(tmp_path, fakes, monkeypatch):
    """KeyError on light_k.irradiance.value (Mitsuba plugin merging) -> retry with one light."""
    fg = sys.modules["nightsky.foreground"]
    real, seen = fg.render_foreground, []

    def flaky(scene_cfg, view, sky_fn, lights, grid, spp=64, cpus=None):
        seen.append(len(lights))
        if len(lights) > 1:
            raise KeyError("light_1.irradiance.value")
        return real(scene_cfg, view, sky_fn, lights, grid, spp, cpus)

    monkeypatch.setattr(fg, "render_foreground", flaky)
    cfg = tiny_cfg(tmp_path)
    res = compose.render(cfg, views=["allsky"], cpus=1)
    assert seen[0] > 1 and seen[-1] == 1 and res["views"]["allsky"]["info"]["foreground_fraction"] > 0


def test_stage_cache_keys(tmp_path, fakes, monkeypatch):
    """Camera-only changes reuse every cached stage; galaxy.nside reuses the starfield;
    sky.subsample reuses the foreground."""
    calls = {"starfield": 0, "galaxy": 0, "foreground": 0}
    sf, gm, fg = (sys.modules[f"nightsky.{m}"] for m in ("starfield", "galaxy", "foreground"))
    real_sf, real_rm, real_fg = sf.build_starfield, gm.GalaxyModel.render_map, fg.render_foreground

    def count(name, fn):
        def wrapped(*a, **k):
            calls[name] += 1
            return fn(*a, **k)
        return wrapped

    monkeypatch.setattr(sf, "build_starfield", count("starfield", real_sf))
    monkeypatch.setattr(gm.GalaxyModel, "render_map", count("galaxy", real_rm))
    monkeypatch.setattr(fg, "render_foreground", count("foreground", real_fg))
    cfg = tiny_cfg(tmp_path)
    compose.render(cfg, views=["persp"], cpus=1)
    assert calls == {"starfield": 1, "galaxy": 1, "foreground": 1}
    compose.render(config.deep_merge(cfg, {"camera": {"mode": "camera", "iso": 12800, "bloom": 0.1}}), views=["persp"], cpus=1)
    assert calls == {"starfield": 1, "galaxy": 1, "foreground": 1}
    compose.render(config.deep_merge(cfg, {"sky": {"subsample": 1}}), views=["persp"], cpus=1)
    assert calls == {"starfield": 1, "galaxy": 1, "foreground": 1}
    compose.render(config.deep_merge(cfg, {"galaxy": {"nside": 16}}), views=["persp"], cpus=1)
    assert calls["starfield"] == 1 and calls["galaxy"] == 2


def test_uniform_sky_photometry(tmp_path, fakes, monkeypatch):
    """A uniform sky of luminance L above the horizon (black below, no point sources) comes out
    as L in the JSON and, within the glare lost to the black band, in the npz; the adaptation
    luminance equals the sky's mesopic luminance."""
    atm = sys.modules["nightsky.atmosphere"].Atmosphere
    grid = spectral.default_grid(4)
    Lsp = spectral.surface_brightness_radiance(20.0, 5000.0, grid)
    Lv = float(spectral.luminance(Lsp, grid))

    def sky_uniform(self, toa, R, d):
        d = np.asarray(d, float).reshape(-1, 3)
        z = np.zeros((len(d), self.grid.n))
        return {"T": z, "inscatter": z + Lsp[None] * (d[:, 2:3] > 0), "airglow": z}

    monkeypatch.setattr(atm, "sky", sky_uniform)
    monkeypatch.setattr(atm, "point_irradiance", lambda self, toa, R: sky.PointSources.empty(self.grid.n))
    x = camera.to_xyzs(Lsp, grid)
    # exact chain: per-pixel sky (subsample 1), no glare -> every lit pixel is Lv, La is the
    # mesopic luminance of the sky (the black band below the horizon is excluded)
    # with glare and subsample 2: the horizon step is blurred and glare leaks into the black
    # band, so only the median of the lit pixels and an upper bound are checked
    for sub, glare, tol in ((1, False, 1e-4), (2, True, 1e-2)):
        cfg = tiny_cfg(tmp_path, foreground=None, sky={"subsample": sub}, camera={"glare": glare},
                       output_dir=str(tmp_path / f"out{sub}"))
        cfg["foreground"] = None
        res = compose.render(cfg, cpus=1)
        for name, r in res["views"].items():
            assert np.isclose(r["info"]["sky_luminance"]["median_cd_m2"], Lv, rtol=1e-5), name
            c = r["info"]["camera"]
            m = c["photopic_fraction"]
            if not glare:
                assert np.isclose(c["La_cd_m2"], m * x[1] + (1 - m) * x[3], rtol=tol), (name, c["La_cd_m2"])
            z = np.load(r["paths"][1])
            y = z["xyz"][..., 1][z["valid"]]
            up = y[y > 0.5 * Lv]
            assert np.isclose(np.median(up), Lv, rtol=tol) and up.max() < (1 + tol) * Lv, \
                (name, sub, np.median(up) / Lv, up.max() / Lv)


def test_eye_adaptation_monotonic():
    grid = spectral.default_grid(10)
    ccfg = config.deep_merge(config.DEFAULTS["camera"], {})
    L1 = spectral.surface_brightness_radiance(21.0, 5000.0, grid)
    las, ms = [], []
    for lt in np.geomspace(1e-3, 10, 41):
        x = camera.to_xyzs(np.broadcast_to(L1 * lt / spectral.luminance(L1, grid), (4, 4, grid.n)), grid)
        ad = camera.eye_adaptation(x, np.ones((4, 4), bool), ccfg)
        las.append(ad["La_cd_m2"])
        ms.append(ad["photopic_fraction"])
        m = ad["photopic_fraction"]
        assert np.isclose(ad["La_cd_m2"], m * x[0, 0, 1] + (1 - m) * x[0, 0, 3], rtol=1e-5)
    assert np.all(np.diff(las) > 0) and np.all(np.diff(ms) >= 0) and np.max(np.array(las[1:]) / las[:-1]) < 1.3


def test_cli_info(tmp_path, capsys):
    p = tmp_path / "x.yaml"
    p.write_text(f"output_dir: {tmp_path / 'o'}\n")
    from nightsky import cli
    cli.main(["info", str(p), "--set", "scenario.lat_deg=30"])
    out = capsys.readouterr().out
    assert "lat_deg: 30" in out and "section hashes" in out


CONFIGS = os.path.join(os.path.dirname(__file__), "..", "configs")


@pytest.mark.parametrize("name", sorted(f for f in os.listdir(CONFIGS) if f.endswith(".yaml")) if os.path.isdir(CONFIGS) else [])
def test_shipped_configs_resolve(name):
    cfg = config.load_config(os.path.join(CONFIGS, name))
    assert cfg["scenario"]["mode"] in ("gc", "earth")
    assert cfg["camera"]["mode"] in ("eye", "camera")
    for v in cfg["views"]:
        assert v["kind"] in ("perspective", "allsky", "panorama")


@pytest.mark.skipif(not os.environ.get("NIGHTSKY_REAL"), reason="set NIGHTSKY_REAL=1 (worker) to run the real modules")
def test_real_modules_low_res(tmp_path):
    cfg = config.load_config(os.path.join(CONFIGS, "gc_arches.yaml"), [
        "grid.n_bins=8", "galaxy.nside=16", "foreground_render.spp=4",
        "views=[{name: persp, kind: perspective, width: 96, height: 54, fov_deg: 60, azimuth_deg: auto, pitch_deg: auto, target: sgr_a_star}]",
        f"output_dir={tmp_path / 'real'}"])
    res = compose.render(cfg)
    assert res["views"]["persp"]["info"]["sky_luminance"]["median_cd_m2"] > 0
