"""Fast tests of nightsky.animate (frame list, time axis, adaptation smoothing, re-development).

    cd /orange/adamginsburg/gc/visualization && python3 -m pytest -q tests/test_nightsky_animate.py
"""
import os
import sys

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from nightsky import animate, camera, config, frames, spectral  # noqa: E402


def test_parse_hours_forms():
    assert np.allclose(animate.parse_hours("0:24:6"), [0, 6, 12, 18])
    assert np.allclose(animate.parse_hours("18:20:0.5"), [18, 18.5, 19, 19.5])
    assert np.allclose(animate.parse_hours("0:24", 4), [0, 6, 12, 18])  # closed cycle: stop excluded
    assert np.allclose(animate.parse_hours("12:24", 4), [12, 16, 20, 24])  # open span: stop included
    assert np.allclose(animate.parse_hours(None, 3), [0, 8, 16])
    assert np.allclose(animate.parse_hours("12, 18,18.5"), [12, 18, 18.5])
    assert len(animate.parse_hours()) == 96
    assert len(animate.parse_hours("0:24:0.0833333333")) == 288


def test_closed_cycle():
    assert animate.is_closed_cycle(animate.parse_hours("0:24:1"))
    assert animate.is_closed_cycle(animate.parse_hours("6:30", 12))
    assert not animate.is_closed_cycle([12, 18, 18.5, 19, 20, 24])
    assert not animate.is_closed_cycle(animate.parse_hours("0:12:1"))


def test_time_axis():
    gc = {"scenario": {"lat_deg": 40.0, "sun": {"enabled": True, "ecliptic_longitude_deg": 180.0}}}
    earth = {"scenario": {"lat_deg": 40.0, "sun": {"enabled": True, "ecliptic_longitude_deg": 0.0}}}
    for cfg in (gc, earth):
        ra = animate.sun_ra_deg(cfg)
        H = lambda t: float(animate.solar_to_hour_angle(t, ra))
        assert animate.sun_altitude(cfg, H(12)) == pytest.approx(50.0, abs=1e-6)  # 90 - lat on the equinox
        assert animate.sun_altitude(cfg, H(0)) == pytest.approx(-50.0, abs=1e-6)
        assert animate.sun_altitude(cfg, H(18)) == pytest.approx(0.0, abs=1e-6)
        assert animate.sun_altitude(cfg, H(6)) == pytest.approx(0.0, abs=1e-6)
    assert animate.solar_to_hour_angle(20.0, animate.sun_ra_deg(gc)) % 360 == pytest.approx(300.0)  # gc_arches H = -60
    assert animate.solar_to_hour_angle(20 + 1 / 3, 0.0) == pytest.approx(125.0)  # earth_darksky H = 125
    assert animate.clock(18.5) == "18:30" and animate.clock(24.0) == "00:00"


def test_smoothing_constant_and_asymmetric():
    t = np.arange(0, 2, 1 / 60)  # 2 h at 1-min frames
    assert np.allclose(animate.smooth_log_la(t, np.full(len(t), 3.0)), 3.0)
    step_up = np.where(t < 1, 1e-3, 1e3)
    up = animate.smooth_log_la(t, step_up, tau_light_min=1.0, tau_dark_min=10.0)
    i = np.searchsorted(t, 1.0)
    # 5 min after a brightening step the adapted state is within e^-4 (in ln) of the target
    assert np.log(up[i + 5] / 1e3) > -np.log(1e6) * np.exp(-4.5)
    down = animate.smooth_log_la(t, step_up[::-1], tau_light_min=1.0, tau_dark_min=10.0)
    lag_dn = np.log(down[i + 10] / 1e-3) / np.log(1e6)  # fraction of the ln step left after 10 min
    assert 0.3 < lag_dn < 0.45  # ~ e^-1 for tau_dark = 10 min (linear ramp over the first minute)
    assert np.all(np.diff(np.log(down[i:])) <= 1e-12)


def test_smoothing_exact_exponential_and_periodic():
    # one coarse step: the ODE is stepped exactly, so the result is independent of the frame spacing
    t = np.array([0.0, 0.5, 1.0])
    la = np.array([1.0, 1.0, np.e ** -6])
    a = animate.smooth_log_la(t, la, 10.0, 10.0)
    fine = np.linspace(0, 1, 121)
    b = animate.smooth_log_la(fine, np.exp(np.interp(fine, t, np.log(la))), 10.0, 10.0)
    assert a[-1] == pytest.approx(b[-1], rel=1e-3)
    # a closed 24 h cycle reaches the periodic steady state: the result does not depend on the
    # phase at which the list starts
    tt = animate.parse_hours("0:24:0.5")
    day = np.exp(8 * np.cos(2 * np.pi * (tt - 12) / 24))
    s = animate.smooth_log_la(tt, day)
    s2 = animate.smooth_log_la(tt, np.roll(day, 10))
    assert np.allclose(np.roll(s, 10), s2, rtol=1e-6)
    # adapted luminance lags the evening decline (higher than the instantaneous La)
    ev = (tt > 13) & (tt < 23)
    assert np.all(s[ev] >= day[ev] * (1 - 1e-9))


def test_exposure_modes():
    t = np.array([12.0, 18.0, 24.0])
    la = np.array([1e3, 1.0, 1e-3])
    acfg = {**animate.ANIM_DEFAULTS}
    eye = {"mode": "eye"}
    assert np.allclose(animate.exposure_series(t, la, "per-frame", acfg, eye), la)
    assert np.allclose(animate.exposure_series(t, la, "fixed", acfg, eye), 1.0)
    assert np.allclose(animate.exposure_series(t, la, "fixed", {**acfg, "fixed_la_cd_m2": 0.1}, eye), 0.1)
    assert np.isnan(animate.exposure_series(t, la, "fixed", acfg, {"mode": "camera"})).all()
    with pytest.raises(ValueError):
        animate.exposure_series(t, la, "auto", acfg, eye)


def test_hours_wrap_single_and_errors():
    assert np.allclose(animate.parse_hours("22:2:1"), [22, 23, 24, 25])  # wraps through midnight
    assert np.allclose(animate.parse_hours("23,1,5.5,6.5"), [23, 25, 29.5, 30.5])
    assert np.allclose(animate.parse_hours("12"), [12])
    assert animate.frame_tag(1.0) == animate.frame_tag(25.0) == "h001.000"
    assert animate.frame_tag(24.0) == animate.frame_tag(23.99999) == "h000.000"
    for bad in ("0:24:0", "0:24:-1", "", 1440.25, 390):  # numbers: YAML base-60 reads of 0:24:0.25, 6:30
        with pytest.raises(ValueError):
            animate.parse_hours(bad)
    assert np.allclose(animate.parse_hours([18, 18.5]), [18, 18.5]) and np.allclose(animate.parse_hours(12), [12])
    assert np.allclose(animate.smooth_log_la([12.0], [5.0]), 5.0)
    gc = {"scenario": {"lat_deg": 40.0, "sun": {"ecliptic_longitude_deg": 180.0}}}
    H = animate.solar_to_hour_angle(np.array([18.0, 42.0]), animate.sun_ra_deg(gc))
    assert H[1] - H[0] == pytest.approx(360.0)
    assert animate.sun_altitude(gc, H[0]) == pytest.approx(animate.sun_altitude(gc, H[1]), abs=1e-9)


def test_sun_rises_east_sets_west():
    for lon in (0.0, 180.0):
        cfg = {"scenario": {"lat_deg": 40.0, "sun": {"ecliptic_longitude_deg": lon}}}
        for t, az in ((6.0, 90.0), (9.0, None), (18.0, 270.0)):
            H = float(animate.solar_to_hour_angle(t, animate.sun_ra_deg(cfg)))
            alt, a = frames.vec_to_altaz(frames.celestial_to_horizon(H, 40.0) @ frames.sun_celestial(lon))
            if az is None:
                assert 90 < a < 180 and alt > 0  # morning Sun in the south-east
            else:
                assert a == pytest.approx(az, abs=1e-6) and alt == pytest.approx(0.0, abs=1e-6)


def test_smoothing_twilight_monotonic_and_floor():
    # 5-min frames through evening twilight; the per-frame La of the GC validation run, extended
    # below the absolute threshold of vision
    t = np.arange(17.0, 22.0, 1 / 12)
    knots_t, knots = [17, 18, 18.5, 19, 20, 21, 22], [3e3, 192, 1.22, 0.0472, 0.0433, 1e-8, 1e-9]
    la = np.exp(np.interp(t, knots_t, np.log(knots)))
    s = animate.smooth_log_la(t, la)
    d = np.diff(np.log(s))
    assert np.all(d <= 1e-12)  # monotonic: no flicker on a monotonic scene
    assert np.all(s >= np.maximum(la, camera.ABS_THRESHOLD_CD_M2) * (1 - 1e-9))  # dark adaptation lags
    assert s.min() >= camera.ABS_THRESHOLD_CD_M2 * (1 - 1e-9)
    assert np.abs(d).max() <= np.abs(np.diff(np.log(np.maximum(la, camera.ABS_THRESHOLD_CD_M2)))).max() + 1e-12
    fixed = animate.exposure_series(t, la, "fixed", {**animate.ANIM_DEFAULTS, "fixed_la_cd_m2": 1e-9}, {"mode": "eye"})
    assert np.allclose(fixed, camera.ABS_THRESHOLD_CD_M2)


def _frames(d, n, w=33, h=21):
    from PIL import Image
    os.makedirs(d, exist_ok=True)
    for k in range(n):
        Image.fromarray(np.full((h, w, 3), 20 + 40 * k, np.uint8)).save(os.path.join(d, f"f_{k:04d}.png"))
    return os.path.join(d, "f_%04d.png")


def test_stitch_order_fps_even_size_and_stale_frames(tmp_path):
    exe, _ = animate.find_ffmpeg()
    probe_exe = os.path.join(os.path.dirname(exe or ""), "ffprobe")
    if exe is None or not os.access(probe_exe, os.X_OK):
        pytest.skip("no ffmpeg / ffprobe")
    pat = _frames(str(tmp_path), 6)
    os.remove(pat % 5)  # frames 0-3 belong to this run; 4 is a stale frame of an earlier run
    out = animate.stitch(pat, str(tmp_path / "v"), 4, 4)
    assert out.endswith(".mp4")
    import subprocess
    probe = subprocess.run([probe_exe, "-v", "error", "-count_frames", "-show_entries",
                            "stream=width,height,r_frame_rate,nb_read_frames", "-of", "csv=p=0", out],
                           capture_output=True, text=True).stdout.strip().split(",")
    assert probe[:2] == ["34", "22"] and probe[2] == "4/1" and probe[3] == "4"
    subprocess.run([exe, "-v", "error", "-i", out, str(tmp_path / "d_%02d.png")], check=True)
    from PIL import Image
    m = [np.asarray(Image.open(tmp_path / f"d_{k:02d}.png"))[2:-2, 2:-2].mean() for k in range(1, 5)]
    assert np.all(np.diff(m) > 20)  # frame order kept


def test_stitch_without_ffmpeg_writes_gif(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(animate, "find_ffmpeg", lambda: (None, None))
    out = animate.stitch(_frames(str(tmp_path), 3), str(tmp_path / "v"), 5, 3)
    assert out.endswith(".gif")
    with Image.open(out) as im:
        assert im.n_frames == 3 and im.info["duration"] == 200


def test_run_rejects_unknown_view_and_pointing(tmp_path):
    cfgp = os.path.join(os.path.dirname(__file__), "..", "configs", "gc_arches_timelapse.yaml")
    with pytest.raises(ValueError, match="unknown view"):
        animate.run(cfgp, hours="12", vnames=["nope"], sets=[f"output_dir={tmp_path}"])
    with pytest.raises(ValueError, match="pointing"):
        animate.run(cfgp, hours="12", sets=[f"output_dir={tmp_path}", "animation.pointing=auto"])


def test_frame_la_camera_ignores_unrendered():
    xyzs = np.zeros((4, 4, 4), np.float32)
    xyzs[:2, :, 1] = 2.0  # upper half rendered, lower half unrendered (0)
    assert animate.frame_la(xyzs, np.ones((4, 4), bool), {"camera": {"mode": "camera"}}) == pytest.approx(2.0)


def test_develop_xyzs_matches_camera_develop():
    grid = spectral.default_grid()
    v = camera.make_view({"kind": "perspective", "width": 64, "height": 48, "azimuth_deg": 0, "pitch_deg": 20, "fov_deg": 40})
    rng = np.random.default_rng(1)
    img = (spectral.surface_brightness_radiance(19.0, 5000.0, grid)[None, None] * rng.lognormal(0, 1.5, (48, 64, 1))).astype(np.float32)
    for cc in ({"mode": "eye"}, {"mode": "eye", "mesopic": False, "glare": False}, {"mode": "camera"}):
        ccfg = config.deep_merge(config.DEFAULTS["camera"], cc)
        rgb, info, xyzs = camera.develop(img, v, grid, ccfg)
        mine, _ = animate.develop_xyzs(xyzs, v.valid, ccfg, info.get("La_cd_m2"))
        assert np.abs(mine.astype(int) - rgb.astype(int)).max() <= 1
