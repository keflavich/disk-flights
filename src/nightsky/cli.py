"""Command line: python3 -m nightsky {render,info,validate} ...

    python3 -m nightsky render ../configs/gc_arches.yaml [--set camera.mode=camera ...]
            [--views allsky ...] [--stage starfield|galaxy|sky|points|foreground|camera]
            [--force galaxy ...] [--no-cache] [--scale 0.25] [--cpus 32]
    python3 -m nightsky info ../configs/gc_arches.yaml
    python3 -m nightsky validate [--out DIR]

`--stage X` stops after stage X; `--force X` recomputes the cached stage X. Heavy stages
(galaxy map, foreground) belong on a worker:  CPUS=32 ../slurm/onworker.sh python3 -m nightsky render ...
`validate` runs the observer-model checks (no other nightsky modules needed) and writes
diagnostic PNGs to renders/nightsky/validation/compose/.
"""
import argparse
import os
import sys
import time

import numpy as np

from . import camera, compose, config, spectral

VALIDATION_DIR = os.path.join(config.ROOT, "renders", "nightsky", "validation", "compose")


def cmd_render(a):
    cfg = config.load_config(a.config, a.set)
    if a.cpus:
        os.environ["NUMBA_NUM_THREADS"] = str(a.cpus)
    compose.log(f"config {cfg['name']} -> {cfg['output_dir']} ({a.cpus or compose.default_cpus()} cpus)")
    t0 = time.time()
    res = compose.render(cfg, views=a.views, stop_after=a.stage, force=a.force or (), use_cache=not a.no_cache,
                         cpus=a.cpus, scale=a.scale)
    compose.log(f"done in {time.time() - t0:.1f} s")
    return res


def cmd_info(a):
    cfg = config.load_config(a.config, a.set)
    print(config.dump(cfg))
    print("section hashes:")
    for s in ("scenario", "grid", "starfield", "galaxy", "atmosphere", "foreground", "views", "camera"):
        print(f"  {s:12s} {config.section_hash(cfg, s)}")
    print("cache:", os.path.join(cfg["output_dir"], "cache"))
    st = compose.cache_state(cfg)
    for k, v in st.items():
        print(f"  {k}: {len(v)} file(s): {', '.join(v)}")
    if not st:
        print("  (empty)")
    print("modules:")
    import importlib.util
    for m in ("starfield", "galaxy", "gas", "atmosphere", "foreground"):
        print(f"  nightsky.{m}: {'present' if importlib.util.find_spec(f'nightsky.{m}') else 'MISSING'}")


def _row(name, value, ref, tol=None, unit=""):
    ok = "" if tol is None else ("  PASS" if abs(value / ref - 1) <= tol else "  FAIL")
    print(f"  {name:58s} {value:12.5g} {unit:9s} ref {ref:10.4g}{ok}")
    return tol is None or abs(value / ref - 1) <= tol


def validate(out_dir=VALIDATION_DIR):
    """Quantitative checks of the views, point deposit, photometry and observer model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    grid = spectral.default_grid()
    ok = []
    print("photometry (spectral grid, 20 bins):")
    e0 = spectral.vmag_irradiance(0.0, 9600.0, grid)
    ok.append(_row("V=0 A0 star illuminance (Allen 2000: V = -13.98 - 2.5 log E)", spectral.to_xyz(e0, grid)[1], 10 ** (-13.98 / 2.5), 0.05, "lx"))
    L22 = spectral.surface_brightness_radiance(22.0, 5800.0, grid)
    ok.append(_row("22.0 V mag/arcsec^2 sky (Falchi+2016: 174 ucd/m^2)", spectral.luminance(L22, grid), 1.74e-4, 0.15, "cd/m^2"))
    print("views:")
    a = camera.make_view({"kind": "allsky", "width": 512, "height": 512})
    ok.append(_row("allsky: sum omega(valid), theta<95 deg", a.omega[a.valid].sum(), 2 * np.pi * (1 - np.cos(np.radians(95))), 0.01, "sr"))
    p = camera.make_view({"kind": "panorama", "width": 720, "height": 140, "el_min_deg": -10, "el_max_deg": 60})
    ok.append(_row("panorama: sum omega, -10..60 deg band", p.omega.sum(), 2 * np.pi * (np.sin(np.radians(60)) + np.sin(np.radians(10))), 1e-3, "sr"))
    r, c = p.project(p.dirs.reshape(-1, 3))
    rr, cc = np.meshgrid(np.arange(140), np.arange(720), indexing="ij")
    print(f"  panorama project() max pixel error {max(np.nanmax(np.abs(r - rr.ravel())), np.nanmax(np.abs(c - cc.ravel()))):.2e} px")
    print("point-source deposit (flux conservation):")
    rng = np.random.default_rng(3)
    for v in (camera.make_view({"kind": "perspective", "width": 400, "height": 300, "azimuth_deg": 30, "pitch_deg": 40, "fov_deg": 50}), a, p):
        alt = rng.uniform(5, 85, 300)
        az = rng.uniform(0, 360, 300)
        from . import frames
        d = frames.altaz_to_vec(alt, az)
        rw, cl = v.project(d)
        inner = np.isfinite(rw) & (rw > 6) & (rw < v.height - 7) & (cl > 6) & (cl < v.width - 7)
        if v.kind == "allsky":
            inner &= alt > 3
        e = spectral.vmag_irradiance(rng.uniform(-2, 6, inner.sum()), 6000.0, grid)
        img = np.zeros((v.height, v.width, grid.n), np.float32)
        camera.deposit_points(img, v, d[inner], e, {"psf_sigma_px": 0.8, "radius_px": 4}, grid)
        ok.append(_row(f"{v.kind}: sum(L omega) / sum(E), {inner.sum()} stars", float((img * v.omega[..., None]).sum() / e.sum()), 1.0, 1e-4))
    print("glare PSF (CIE 2002, age 25, p 0.5):")
    ok.append(_row("integral over 0-100 deg", camera.encircled_glare(100.0), 1.0, 0.1))
    gen = lambda t, A=25.0, pp=0.5: 10 / t ** 3 + (5 / t ** 2 + 0.1 * pp / t) * (1 + (A / 62.5) ** 4) + 0.0025 * pp
    for t in (1.0, 3.0, 10.0, 30.0):
        _row(f"L_veil/E at {t:4.0f} deg vs CIE 'general' 0.1-100 deg form", float(camera.cie_glare_psf(t)), gen(t), None, "sr^-1")
    v = camera.make_view({"kind": "perspective", "width": 320, "height": 240, "azimuth_deg": 0, "pitch_deg": 30, "fov_deg": 40})
    k = camera.glare_kernel(v, {})
    half = k.shape[0] // 2
    ok.append(_row("glare kernel sum = EE(kernel radius)/EE(100 deg)", float(k.sum()),
                   camera.encircled_glare(half * camera.view_pixel_deg(v)) / camera.encircled_glare(100.0), 1e-6))
    print(f"  unscattered core fraction in the centre pixel ({camera.view_pixel_deg(v) * 60:.1f} arcmin): {k[half, half]:.3f}")
    print("exposure / tone:")
    ex = camera.camera_exposure({"iso": 100, "f_number": 1.0, "exposure_s": 1.0})
    ok.append(_row("ISO 100 f/1 1 s saturation luminance (78/(0.65*100))", ex["L_sat_cd_m2"], 1.2, 1e-3, "cd/m^2"))
    ex = camera.camera_exposure({"iso": 100, "f_number": 16.0, "exposure_s": 0.01})
    ok.append(_row("sunny-16: 18% card in 1e5 lx -> fraction of saturation", 0.18 * 1e5 / np.pi * ex["scale"], 0.18, 0.15))
    ok.append(_row("sRGB encode(0.18)", float(camera.srgb_encode(np.array(0.18))), 0.4614, 1e-3))
    for la, ref in ((0.005, 0.0), (0.158, 0.5), (5.0, 1.0)):
        print(f"  mesopic photopic fraction at {la:6.3f} cd/m^2: {camera.mesopic_weight(la):.3f} (ref {ref})")

    # diagnostic renders: a uniform 21.7 mag/arcsec^2 sky with a ladder of stars V = -12 ... 6
    v = camera.make_view({"kind": "perspective", "width": 640, "height": 360, "azimuth_deg": 0, "pitch_deg": 30, "fov_deg": 60})
    L = np.broadcast_to(spectral.surface_brightness_radiance(21.7, 5000.0, grid), (360, 640, grid.n)).astype(np.float32).copy()
    mags = np.arange(-12.0, 6.1, 1.5)
    from . import frames
    d = frames.altaz_to_vec(np.full(len(mags), 30.0), np.linspace(-25, 25, len(mags)))
    camera.deposit_points(L, v, d, spectral.vmag_irradiance(mags, 7000.0, grid), {"psf_sigma_px": 0.6}, grid)
    panels = []
    for name, cc in (("eye", {"mode": "eye"}), ("eye_fixed_La", {"mode": "eye", "adaptation_cd_m2": 0.03}),
                     ("camera", {"mode": "camera", "iso": 3200, "f_number": 2.8, "exposure_s": 20, "bloom": 0.02, "spikes": 4})):
        ccfg = config.deep_merge(config.DEFAULTS["camera"], cc)
        t0 = time.time()
        rgb, info, _ = camera.develop(L, v, grid, ccfg)
        dt = time.time() - t0
        pth = os.path.join(out_dir, f"star_ladder_{name}.png")
        from PIL import Image
        Image.fromarray(rgb).save(pth)
        sky_srgb = rgb[300:340, 20:60].mean() / 255
        print(f"  {name:13s} sky sRGB {sky_srgb:.3f}; " + ", ".join(f"{k} {v:.4g}" for k, v in info.items() if isinstance(v, float))
              + f"; develop {dt:.2f} s -> {pth}")
        panels.append((name, rgb))
    th = np.geomspace(0.003, 100, 400)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for A, pp in ((25, 0.5), (70, 1.0)):
        ax[0].loglog(th, camera.cie_glare_psf(th, A, pp), label=f"CIE 2002, age {A}, p {pp}")
    ax[0].loglog(th[th > 0.1], gen(th[th > 0.1]), "k--", lw=1, label="CIE general (0.1-100 deg)")
    ax[0].loglog(th[th > 1], 10 / th[th > 1] ** 2, "k:", lw=1, label="Stiles-Holladay 10/theta^2")
    ax[0].set(xlabel="angle (deg)", ylabel="L_veil / E_glare (sr$^{-1}$)", title="eye glare PSF")
    ax[0].legend(fontsize=8)
    ls = np.geomspace(1e-5, 100, 200)
    ax[1].semilogx(ls, [camera.mesopic_weight(x) for x in ls])
    ax[1].set(xlabel="adaptation luminance (cd/m$^2$)", ylabel="photopic fraction m", title="mesopic blend (CIE 191 range)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "glare_mesopic.png"), dpi=110)
    print(f"  wrote {os.path.join(out_dir, 'glare_mesopic.png')}")
    print(f"validation: {sum(ok)}/{len(ok)} checks pass")
    return all(ok)


SCENE_REFS = {
    "earth": [("zenith_V_mag_arcsec2", (21.3, 22.0), "natural dark-sky zenith, V mag/arcsec^2 (Falchi et al. 2016: 22.0; "
               "Benn & Ellison 1998, La Palma: 21.9)"),
              ("ground_illuminance_lux", (5e-4, 3e-3), "moonless clear-sky horizontal illuminance, lx (starlight + airglow)")],
    "gc": [("zenith_luminance_cd_m2", (None, None), "previous toy GC sky: 17 V mag/arcsec^2 = 0.017 cd/m^2 (README)")],
}


def check_scenes(dirs):
    """Print the photometric summary of rendered output directories against reference ranges."""
    import json
    for d in dirs:
        p = os.path.join(d, "scene.json")
        if not os.path.exists(p):
            print(f"  {d}: no scene.json")
            continue
        with open(p) as f:
            s = json.load(f)
        mode = s["config"]["scenario"]["mode"]
        sc = s["scene"]
        print(f"scene {s['config']['name']} ({mode}, atmosphere {s['config']['atmosphere']}):")
        for k in ("zenith_luminance_cd_m2", "zenith_V_mag_arcsec2", "zenith_airglow_cd_m2", "ground_illuminance_lux",
                  "ground_scotopic_lux", "point_sources_horizontal_lux", "n_point_sources_above_horizon"):
            print(f"  {k:32s} {sc[k]:.4g}")
        for k, (lo, hi), txt in SCENE_REFS.get(mode, []):
            flag = "" if lo is None else ("  in range" if lo <= sc[k] <= hi else "  OUTSIDE")
            print(f"  ref {k}: {'' if lo is None else f'{lo:g}-{hi:g} '}{txt}{flag}")
        for b in sc["brightest"][:5]:
            print(f"  source {b['name'][:24]:24s} V {b['V']:7.2f} alt {b['alt_deg']:6.1f} az {b['az_deg']:6.1f} E_h {b['horizontal_lux']:.3g} lx")
        for f in sorted(os.listdir(d)):
            if f.endswith(".json") and f != "scene.json":
                with open(os.path.join(d, f)) as fh:
                    v = json.load(fh)
                c, sl = v["camera"], v["sky_luminance"]
                print(f"  view {f[:-5]:12s} sky median {sl['median_cd_m2']:.3g} log-avg {sl['log_average_cd_m2']:.3g} cd/m^2; "
                      f"La {c.get('La_cd_m2', float('nan')):.3g} photopic {c.get('photopic_fraction', float('nan')):.2f}; "
                      f"fg {v['foreground_fraction']:.2f}; " + ", ".join(f"{k} {t:.1f}s" for k, t in v["timing_s"].items()))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m nightsky", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("render", help="render the views of a config")
    r.add_argument("config")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override (YAML value), repeatable")
    r.add_argument("--views", nargs="+", default=None)
    r.add_argument("--stage", choices=compose.STAGES, default=None, help="stop after this stage")
    r.add_argument("--force", nargs="+", default=None, help="recompute these cached stages (starfield galaxy sky_<view> foreground_<view>)")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--scale", type=float, default=1.0, help="multiply view pixel sizes (previews)")
    r.add_argument("--cpus", type=int, default=None)
    i = sp.add_parser("info", help="print the resolved config and cache state")
    i.add_argument("config")
    i.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    v = sp.add_parser("validate", help="observer-model checks and diagnostic PNGs")
    v.add_argument("--out", default=VALIDATION_DIR)
    v.add_argument("--scene", nargs="+", default=None, help="also summarise these render output dirs")
    a = ap.parse_args(argv)
    if a.cmd == "render":
        cmd_render(a)
    elif a.cmd == "info":
        cmd_info(a)
    else:
        good = validate(a.out)
        if a.scene:
            check_scenes(a.scene)
        sys.exit(0 if good else 1)


if __name__ == "__main__":
    main()
