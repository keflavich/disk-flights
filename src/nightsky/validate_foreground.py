"""Validation of the foreground renderer: quantitative checks and diagnostic previews.

    python3 -m nightsky.validate_foreground checks                 # small, fast
    python3 -m nightsky.validate_foreground scenes --types mountains meadow --width 1920 --height 1080 --spp 16
    python3 -m nightsky.validate_foreground noise                  # spp vs noise

Checks (references in brackets):
  envmap   camera sees the envmap: rendered / sky_fn(view.dirs)                  [1]
  envpos   compact sky blob: rendered vs sky_fn centroid [0 deg]; point deposited near the
           zenith lights a plane with E sin(el) / pi [1]
  shadow   box lit from the NW at 30 deg: shadow azimuth from the box [135 deg]
  albedo   Lambertian plane under a uniform sky L: radiance / (albedo L), per bin [1]
  sun      plane lit by 1 or 3 directional lights (E_k, elevation el_k), also with
           lights folded into the envmap: radiance / (albedo sum E_k sin(el_k) / pi)  [1]
  marker   centroid of a sphere marker vs views.View.project (perspective, allsky) [0 px]
Units: W m^-2 sr^-1 nm^-1 (radiance), W m^-2 nm^-1 (irradiance); horizon frame x=N, y=W, z=Up.
Synthetic sky: twilight-like gradient (zenith 18 V mag/arcsec^2 with a 9000 K shape,
horizon glow towards the west with a 3500 K shape) plus a full Moon (V = -12.7, 4100 K;
Allen's Astrophysical Quantities) at az 120, el 25 deg; the "gc" sky adds 40 directional
lights within 6 deg of the north celestial pole with total V = -13.5 (README Stage 3).
"""
import argparse
import os
import time

import numpy as np
from PIL import Image

import photometry as ph
import views
from . import albedo as alb
from . import foreground as fg
from . import frames, spectral
from .sky import PointSources

OUT = "/orange/adamginsburg/gc/visualization/renders/nightsky/validation/foreground"


def twilight_sky(grid):
    zen = spectral.surface_brightness_radiance(18.0, 9000.0, grid)
    glow = spectral.surface_brightness_radiance(17.0, 3500.0, grid)

    def f(d):
        d = np.asarray(d, float)
        el = np.arcsin(np.clip(d[:, 2], 0, 1))
        alt, az = frames.vec_to_altaz(d)
        west = np.clip(np.cos(np.radians(az - 270.0)), 0, None)
        return (zen[None] * (1 + 2.0 * np.exp(-el / 0.2))[:, None]
                + glow[None] * (west ** 2 * np.exp(-el / 0.12))[:, None])
    return f


def moon(grid, alt=25.0, az=120.0):
    return PointSources(frames.altaz_to_vec(alt, az)[None], spectral.vmag_irradiance(-12.7, 4100.0, grid)[None], ["moon"])


def gc_lights(grid, n=40, seed=3):
    rng = np.random.default_rng(seed)
    alt = 40 + rng.normal(0, 2.5, n)
    az = rng.normal(0, 2.5, n) / np.cos(np.radians(40))
    w = rng.pareto(1.2, n) + 0.2
    w /= w.sum()
    irr = spectral.vmag_irradiance(-13.5, 30000.0, grid)[None] * w[:, None] * 0.7
    return PointSources(frames.altaz_to_vec(alt, az), irr, [f"s{i}" for i in range(n)])


def to_srgb8(rad, grid, exposure):
    xyz = spectral.to_xyz(rad, grid) * exposure
    rgb = np.clip(xyz @ ph.XYZ_TO_SRGB.T, 0, None)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(np.clip(rgb, 0, 1), 1 / 2.4) - 0.055)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def save_previews(name, out, view, sky_fn, grid):
    os.makedirs(OUT, exist_ok=True)
    rad, a = out["radiance"], out["alpha"]
    skyimg = sky_fn(view.dirs.reshape(-1, 3)).reshape(rad.shape) * view.valid[..., None]
    comp = rad + (1 - a[..., None]) * skyimg
    y = spectral.luminance(comp, grid)
    yf = spectral.luminance(rad, grid)[a > 0.99]
    exp_c = 0.18 / max(np.percentile(y[view.valid], 50), 1e-12)
    exp_f = 0.25 / max(np.percentile(yf, 75), 1e-12) if yf.size else exp_c
    Image.fromarray(to_srgb8(comp, grid, exp_c)).save(os.path.join(OUT, f"{name}_composite.png"))
    Image.fromarray(to_srgb8(rad, grid, exp_f)).save(os.path.join(OUT, f"{name}_foreground.png"))
    Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).save(os.path.join(OUT, f"{name}_alpha.png"))
    import matplotlib.cm as cm
    d = out["depth"]
    ld = np.where(np.isfinite(d), np.log10(np.clip(d, 1, 1e5)) / 5, np.nan)
    rgba = cm.viridis(np.nan_to_num(ld, nan=0))
    rgba[~np.isfinite(d)] = [0, 0, 0, 1]
    Image.fromarray((rgba[..., :3] * 255).astype(np.uint8)).save(os.path.join(OUT, f"{name}_depth.png"))
    n = out["normal"]
    Image.fromarray(((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(OUT, f"{name}_normal.png"))
    np.savez_compressed(os.path.join(OUT, f"{name}.npz"), radiance=rad.astype(np.float16), alpha=a,
                        depth=d.astype(np.float32))
    return exp_c, exp_f


def check_envmap(grid, cpus):
    v = views.View("perspective", 192, 108, azimuth_deg=30, pitch_deg=20, fov_deg=90)
    f = lambda d: (1.2 + 0.5 * d[:, 0] + 0.3 * d[:, 1] + 0.8 * d[:, 2])[:, None] * np.linspace(1, 2, grid.n)[None]
    im = fg.render_sky_only(v, f, grid, spp=4, bins=(0, grid.n - 1), cpus=cpus)
    ref = f(v.dirs.reshape(-1, 3)).reshape(v.height, v.width, grid.n)[..., [0, grid.n - 1]]
    e = np.abs(im / ref - 1)[v.dirs[..., 2] > 0.01]
    print(f"[envmap] rendered/sky_fn - 1 (above horizon): median {np.median(e):.2e}, 99th pct {np.percentile(e, 99):.2e} (ref 0)")
    v2 = views.View("perspective", 192, 108, azimuth_deg=250, pitch_deg=60, fov_deg=70)
    im2 = fg.render_sky_only(v2, f, grid, spp=4, bins=(0,), cpus=cpus)[..., 0]
    ref2 = f(v2.dirs.reshape(-1, 3))[:, 0].reshape(v2.height, v2.width)
    print(f"[envmap] second view (az 250, pitch 60): median |ratio-1| {np.median(np.abs(im2 / ref2 - 1)):.2e}")


def check_env_position(grid, cpus):
    """Compact (sigma 0.8 deg) sky blobs: rendered centroid vs sky_fn centroid; points deposited
    into the envmap near the zenith (pole rows) still light a plane with E sin(el) / pi."""
    for alt, az in ((20.0, 60.0), (55.0, 250.0), (80.0, 10.0)):
        c = frames.altaz_to_vec(alt, az)
        f = lambda d: (1e-3 + np.exp(-0.5 * (np.degrees(np.arccos(np.clip(d @ c, -1, 1))) / 0.8) ** 2))[:, None] * np.ones(grid.n)
        v = views.View("perspective", 480, 270, azimuth_deg=az + 7, pitch_deg=alt - 4, fov_deg=40)
        rr, cc = np.mgrid[:v.height, :v.width]
        ref = np.clip(f(v.dirs.reshape(-1, 3))[:, 0].reshape(v.height, v.width) - 1e-3, 0, None)
        for res in (512, 128):
            im = np.clip(fg.render_sky_only(v, f, grid, spp=16, env_res=res, cpus=cpus)[..., 0] - 1e-3, 0, None)
            dr_, dc_ = [((im * g).sum() / im.sum() - (ref * g).sum() / ref.sum()) * 40.0 / 480 for g in (rr, cc)]
            print(f"[envpos] blob alt {alt:.0f} az {az:.0f}, env_res {res}: centroid offset (row {dr_:+.3f}, col {dc_:+.3f}) deg "
                  f"(texel {180 / (res - 1):.3f} deg; ref 0); flux ratio {im.sum() / ref.sum():.4f} (ref 1)")
    v = views.View("perspective", 160, 90, azimuth_deg=10, pitch_deg=-25, fov_deg=60)
    for alt in (89.97, 88.0, 10.0):
        ls = PointSources(frames.altaz_to_vec(alt, 33.0)[None], np.ones((1, grid.n)), ["x"])
        cfg = dict(type="plane", n_theta=256, verbose=False, n_lights=0, albedo_overrides=dict(grass=0.5))
        out = fg.render_foreground(cfg, v, lambda d: np.full((len(d), grid.n), 1e-9), ls, grid, spp=64, cpus=cpus)
        r = out["radiance"][out["alpha"] > 0.999] / (0.5 * np.sin(np.radians(alt)) / np.pi)
        print(f"[envpos] point at alt {alt} deposited in the envmap: radiance/(a E sin el/pi) {r.mean():.4f} (ref 1)")


def check_shadow(grid, cpus):
    """4 m tall box 6 m north of the observer, light from the NW (az 315) at 30 deg: the shadow
    must extend towards the SE (az 135) from the box."""
    import mitsuba as mi
    fg._mitsuba(cpus)
    box = {"type": "cube", "to_world": mi.ScalarTransform4f().translate([6.0, 0.0, 2.0]).scale([0.5, 0.5, 2.0]),
           "bsdf": {"type": "diffuse", "reflectance": {"type": "uniform", "value": 0.3}}}
    v = views.View("perspective", 400, 400, azimuth_deg=0.0, pitch_deg=-60, fov_deg=100)
    for nl, tag in ((1, "directional"), (0, "in envmap")):
        ls = PointSources(frames.altaz_to_vec(30.0, 315.0)[None], np.ones((1, grid.n)), ["nw"])
        cfg = dict(type="plane", n_theta=256, verbose=False, n_lights=nl, extra_shapes=[box], albedo_overrides=dict(grass=0.5))
        out = fg.render_foreground(cfg, v, lambda d: np.full((len(d), grid.n), 1e-9), ls, grid, spp=32, cpus=cpus)
        P = np.array(out["info"]["eye"]) + out["depth"][..., None] * v.dirs
        ground = (out["alpha"] > 0.999) & (np.abs(P[..., 2]) < 0.05)
        rel = out["radiance"][..., 0] / (0.5 * np.sin(np.radians(30)) / np.pi)
        cen = P[ground & (rel < 0.1)][:, :2].mean(0) - [6.0, 0.0]
        print(f"[shadow] {tag}: shadow centroid azimuth from the box {np.degrees(np.arctan2(-cen[1], cen[0])) % 360:.1f} deg "
              f"(ref 135); lit ground radiance ratio {rel[ground & (rel > 0.5)].mean():.4f} (ref 1)")


def check_albedo(grid, cpus):
    v = views.View("perspective", 160, 90, azimuth_deg=10, pitch_deg=-25, fov_deg=60)
    L = np.linspace(1.0, 3.0, grid.n)
    for mat in ("grass", "rock", "snow"):
        cfg = dict(type="plane", n_theta=256, albedo_overrides=dict(grass=mat), aov_spp=4, verbose=False)
        out = fg.render_foreground(cfg, v, lambda d: np.ones((len(d), grid.n)) * L, None, grid, spp=16, cpus=cpus)
        m = out["alpha"] > 0.999
        ratio = out["radiance"][m] / (alb.spectrum(mat, grid) * L)
        print(f"[albedo] {mat:5s} plane, uniform sky: radiance/(albedo L) mean {ratio.mean():.4f} "
              f"(bins {ratio.mean(0).min():.4f}-{ratio.mean(0).max():.4f}), pixel rms {ratio.std():.3f} (ref 1)")


def check_sun(grid, cpus):
    v = views.View("perspective", 160, 90, azimuth_deg=10, pitch_deg=-25, fov_deg=60)
    e = np.linspace(1.0, 2.0, grid.n)
    alts = np.array([30.0, 60.0, 45.0])
    lights = PointSources(frames.altaz_to_vec(alts, np.array([200.0, 90.0, 300.0])),
                          np.stack([e, 0.5 * e, 0.25 * e[::-1]]), ["a", "b", "c"])
    rho = alb.spectrum("grass", grid)
    cases = (("1 light ", lights.select([0]), 1, e * np.sin(np.radians(30))),
             ("3 lights", lights, 3, (lights.irr * np.sin(np.radians(alts))[:, None]).sum(0)),
             ("3, n_lights=1 (2 in envmap)", lights, 1, (lights.irr * np.sin(np.radians(alts))[:, None]).sum(0)))
    for name, ls, nl, eh in cases:
        cfg = dict(type="plane", n_theta=256, aov_spp=4, verbose=False, n_lights=nl)
        out = fg.render_foreground(cfg, v, lambda d: np.full((len(d), grid.n), 1e-12), ls, grid, spp=64, cpus=cpus)
        m = out["alpha"] > 0.999
        ratio = out["radiance"][m] / (rho * eh / np.pi)
        print(f"[sun] {name}: radiance/(albedo sum(E sin el)/pi) mean {ratio.mean():.4f} "
              f"(bins {ratio.mean(0).min():.4f}-{ratio.mean(0).max():.4f}), pixel rms {ratio.std():.3f} (ref 1)")


def check_marker(grid, cpus):
    for v in (views.View("perspective", 320, 180, azimuth_deg=20, pitch_deg=3, fov_deg=60),
              views.View("allsky", 400, 400)):
        d = frames.altaz_to_vec(12.0, 37.0)
        eye = np.array([0, 0, 1.7])
        c = eye + 300.0 * d
        cfg = dict(type="plane", n_theta=256, aov_spp=64, verbose=False,
                   extra_shapes=[{"type": "sphere", "center": c.tolist(), "radius": 4.0}])
        out = fg.render_foreground(cfg, v, lambda q: np.full((len(q), grid.n), 1e-6), None, grid, spp=1, cpus=cpus)
        a = out["alpha"] * (v.dirs[..., 2] > 0.02)
        rr, cc = np.mgrid[:v.height, :v.width]
        r0, c0 = (a * rr).sum() / a.sum(), (a * cc).sum() / a.sum()
        pr, pc = v.project(d[None])
        k = np.argmax(v.dirs.reshape(-1, 3) @ d)
        i, j = divmod(int(k), v.width)
        J = np.stack([v.dirs[i + 1, j] - v.dirs[i, j], v.dirs[i, j + 1] - v.dirs[i, j]], 1)
        di, dj = np.linalg.lstsq(J, d - v.dirs[i, j], rcond=None)[0]
        print(f"[marker] {v.kind:11s}: alpha centroid ({r0:.2f}, {c0:.2f}); View.project ({pr[0]:.2f}, {pc[0]:.2f}) "
              f"offset {np.hypot(r0 - pr[0], c0 - pc[0]):.2f} px; inverse of View.dirs ({i + di:.2f}, {j + dj:.2f}) "
              f"offset {np.hypot(r0 - i - di, c0 - j - dj):.2f} px (ref 0); depth {np.median(out['depth'][a > 0.5]):.1f} m "
              f"(ref {300 - 4:.0f})")


def run_checks(cpus):
    grid = spectral.default_grid()
    t = time.time()
    check_envmap(grid, cpus)
    check_env_position(grid, cpus)
    check_shadow(grid, cpus)
    check_albedo(grid, cpus)
    check_sun(grid, cpus)
    check_marker(grid, cpus)
    print(f"checks done in {time.time() - t:.1f} s")


def run_scenes(types, width, height, spp, cpus, sky="twilight", n_theta=1536, az=None, pitch=4.0, fov=100.0,
               allsky=False, denoise=False):
    grid = spectral.default_grid()
    sky_fn = twilight_sky(grid)
    lights = moon(grid) if sky != "gc" else gc_lights(grid).concat(moon(grid))
    default_az = dict(mountains=200.0, meadow=110.0, forest=150.0, lakeshore=0.0)
    for t in types:
        view = (views.View("allsky", height, height) if allsky else
                views.View("perspective", width, height, azimuth_deg=default_az.get(t, 0.0) if az is None else az,
                           pitch_deg=pitch, fov_deg=fov))
        cfg = dict(type=t, n_theta=n_theta, denoise=denoise)
        t0 = time.time()
        out = fg.render_foreground(cfg, view, sky_fn, lights, grid, spp=spp, cpus=cpus)
        name = f"{t}_{sky}_{'allsky' if allsky else 'persp'}"
        exp_c, exp_f = save_previews(name, out, view, sky_fn, grid)
        a = out["alpha"]
        yfg = spectral.luminance(out["radiance"], grid)
        inf = out["info"]
        print(f"[scene] {name}: {view.width}x{view.height}, {spp} spp, {grid.n} bins: total {time.time() - t0:.1f} s "
              f"(geometry {inf['t_geometry']:.1f}, load {inf['t_scene']:.1f}, aov {inf['t_aov']:.1f}, "
              f"radiance {inf['t_radiance']:.1f}); coverage {a.mean():.3f}; ground luminance median "
              f"{np.median(yfg[a > 0.99]) if (a > 0.99).any() else 0:.3e} cd/m^2; trees {inf['n_trees']}; "
              f"depth range {np.nanmin(out['depth']):.1f}-{np.nanmax(np.where(np.isfinite(out['depth']), out['depth'], np.nan)):.0f} m")
        e = spectral.to_xyz(lights.irr * np.clip(lights.dirs[:, 2:3], 0, None), grid)[:, 1].sum()
        print(f"        horizontal illuminance: lights {e:.3e} lx, sky {spectral.to_xyz(inf['sky_irradiance_h'], grid)[1]:.3e} lx")


def run_noise(cpus, t="meadow"):
    grid = spectral.default_grid(4)
    view = views.View("perspective", 480, 270, azimuth_deg=110, pitch_deg=-5, fov_deg=60)
    sky_fn, lights = twilight_sky(grid), moon(grid)
    ref = fg.render_foreground(dict(type=t, verbose=False), view, sky_fn, lights, grid, spp=256, cpus=cpus)
    m = ref["alpha"] > 0.999
    yr = spectral.luminance(ref["radiance"], grid)
    for spp in (4, 16, 64):
        tt = time.time()
        o = fg.render_foreground(dict(type=t, verbose=False, seed_render=7), view, sky_fn, lights, grid, spp=spp, cpus=cpus)
        dt = time.time() - tt
        y = spectral.luminance(o["radiance"], grid)
        od = fg._denoise(o["radiance"], o["alpha"], o["depth"], o["normal"])
        yd = spectral.luminance(od, grid)
        rel = np.sqrt(np.mean((y[m] - yr[m]) ** 2)) / yr[m].mean()
        reld = np.sqrt(np.mean((yd[m] - yr[m]) ** 2)) / yr[m].mean()
        chrom = np.std((o["radiance"][m] / np.maximum(y[m, None], 1e-30)) / (ref["radiance"][m] / np.maximum(yr[m, None], 1e-30)))
        print(f"[noise] {t} {spp:3d} spp: luminance rms error {rel:.3f} (denoised {reld:.3f}), chroma rms {chrom:.3f}, "
              f"{dt:.1f} s for {grid.n} bins at 480x270 (ref 256 spp)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["checks", "scenes", "noise"])
    ap.add_argument("--types", nargs="+", default=["mountains", "meadow", "forest", "lakeshore"])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--cpus", type=int, default=int(os.environ.get("NUMBA_NUM_THREADS", 2)))
    ap.add_argument("--sky", default="twilight", choices=["twilight", "gc"])
    ap.add_argument("--n-theta", type=int, default=1536)
    ap.add_argument("--az", type=float, default=None)
    ap.add_argument("--pitch", type=float, default=4.0)
    ap.add_argument("--fov", type=float, default=100.0)
    ap.add_argument("--allsky", action="store_true")
    ap.add_argument("--denoise", action="store_true")
    a = ap.parse_args()
    if a.what == "checks":
        run_checks(a.cpus)
    elif a.what == "scenes":
        run_scenes(a.types, a.width, a.height, a.spp, a.cpus, a.sky, a.n_theta, a.az, a.pitch, a.fov, a.allsky, a.denoise)
    else:
        run_noise(a.cpus)


if __name__ == "__main__":
    main()
