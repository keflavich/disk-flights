"""Validation of nightsky.atmosphere against the Mitsuba path-traced reference and literature.

(a) Mitsuba: renders/day/data/sun_0XX.npz (src/render_day.py: 24 shells of pure Rayleigh,
    exponential density with H = 8.43 km evaluated at shell mid-altitudes, sigma_550 =
    4.53e-27 cm^2 x (550/lambda)^4, Rayleigh phase without depolarisation, ground albedo 0.3,
    camera at 1.7 m, volpath max_depth 12, Sun = delta light with blackbody 5772 K, 1361 W/m^2
    irradiance at the bin centre; faces are radiance W m^-2 sr^-1 nm^-1 on 8 bins 400-700 nm).
    An equivalent Rayleigh-only Atmosphere (same shells, cross section, albedo, bins) is
    compared along the solar and anti-solar azimuths (bilinear cube samples, plotted) and per
    pixel (nightsky averaged over each pixel's Gaussian footprint; ratios of band sums).
(b) Literature: clear-sky zenith luminance vs solar elevation (0.9-2.5e3 cd/m^2 at sun 30 deg
    from the Krochmann & Seidl 1974 formula,
    1-3 cd/m^2 at -6 deg, ~1e-2 at -12 deg, night level at -18 deg; Rozenberg 1966, Patat et
    al. 2006), V extinction k_V (0.12-0.2 mag/airmass clear sea level, 0.1-0.13 at
    observatories), dark-sky zenith V ~21.8-22.0 mag/arcsec^2 (Benn & Ellison 1998).
(e) Energy: isotropic unit TOA radiance over a non-absorbing Rayleigh atmosphere, T + inscatter
    vs a plane-parallel backward Monte Carlo (mc_slab) for ground albedo 0, 0.3 and 1.

    python3 -m nightsky.validate_atmosphere [--out DIR]
"""
import argparse
import os
import time

import numpy as np

import photometry as ph
import views
from . import atmosphere as A
from . import frames, sky, spectral

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "renders", "nightsky", "validation", "atmosphere")
DAY = os.path.join(os.path.dirname(__file__), "..", "..", "renders", "day", "data")
LAT = 40.0
NSIDE = 16


def mitsuba_shells(h_m):
    """Relative density of render_day's 24 quadratic shells (value at each shell's mid-altitude)."""
    t = np.linspace(0, 1, 25)
    alt = 100e3 * (1 - (1 - t) ** 2)
    mid = 0.5 * (alt[:-1] + alt[1:])
    i = np.clip(np.searchsorted(alt, h_m, side="right") - 1, 0, 23)
    return np.exp(-mid[i] / 8.43e3)


MITSUBA_PRESET = dict(base="earth_clear", name="mitsuba_rayleigh", top_m=100e3, altitude_m=0.0, eye_m=1.7,
                      rayleigh_density=mitsuba_shells, rayleigh_n0_m3=2.546e25,
                      rayleigh_xsec=lambda wl: 4.53e-27 * (550.0 / wl) ** 4, depolarization=0.0,
                      ozone_du=0.0, aerosol_tau550=0.0, strat_tau550=0.0, ground_albedo=0.3, airglow_scale=0.0)


def sun_toa(grid, sun_dir_h, nside=NSIDE, irr=None):
    irr = spectral.blackbody_irradiance(1361.0, 5772.0, grid)[None] if irr is None else irr
    return sky.TOASky(grid, sky.PointSources(np.atleast_2d(sun_dir_h), irr, ["sun"]),
                      np.zeros((12 * nside ** 2, grid.n)))


def night_toa(grid, zl_v=23.1, isl_v=23.6, nside=NSIDE):
    m = spectral.surface_brightness_radiance(zl_v, 5800.0, grid) + spectral.surface_brightness_radiance(isl_v, 4000.0, grid)
    return sky.TOASky(grid, sky.PointSources.empty(grid.n), np.tile(m, (12 * nside ** 2, 1)))


def total(atm, toa, dirs):
    return atm.sky_radiance(toa, np.eye(3), dirs)


def srgb(L, grid, valid, pct=99.0):
    xyz = spectral.to_xyz(L, grid)
    rgb = np.clip(xyz @ ph.XYZ_TO_SRGB.T, 0, None)
    y = xyz[..., 1][valid]
    s = np.percentile(y, pct) if y.size else 1.0
    rgb = np.clip(rgb / max(s, 1e-30), 0, 1)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.where(valid[..., None], rgb, 0.0)


def check_mitsuba(out, plt):
    grid = spectral.SpectralGrid(np.linspace(400.0, 700.0, 9))
    t0 = time.time()
    atm = A.Atmosphere(MITSUBA_PRESET, grid)
    atm_exp = A.Atmosphere({**MITSUBA_PRESET, "rayleigh_density": None, "rayleigh_profile": "exponential",
                            "rayleigh_scale_height_m": 8.43e3, "name": "rayleigh_exponential"}, grid)
    print(f"[mitsuba] built Rayleigh-only atmospheres in {time.time() - t0:.1f} s")
    frames_k = [30, 32, 34, 36, 38, 40, 44, 50, 60]
    els = np.array([1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 75, 89.5])
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True)
    rows = []
    for ax, k in zip(axes.flat, frames_k):
        path = os.path.join(DAY, f"sun_{k:03d}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        H, alt = float(z["hour_angle"]), float(z["altitude"])
        sd = frames.equator_vec(H, LAT)
        az_s = frames.vec_to_altaz(sd)[1]
        toa = sun_toa(grid, sd, irr=np.array([[ph.irradiance_bb(1361.0, 5772.0, w) for w in z["wavelengths"]]]))
        faces = np.moveaxis(z["faces"], 0, -1)
        for azn, az, ls in (("solar", az_s, "-"), ("anti", az_s + 180.0, "--")):
            d = frames.altaz_to_vec(els, np.full_like(els, az))
            ref = views.sample_cube(faces, d)
            r = atm.sky(toa, np.eye(3), d)
            mine = r["T"] * 0 + r["inscatter"]
            mexp = atm_exp.sky(toa, np.eye(3), d)["inscatter"]
            Lr, Lm, Le = (spectral.luminance(x, grid) for x in (ref, mine, mexp))
            ax.semilogy(els, Lr, "k" + ls, label=f"Mitsuba {azn}")
            ax.semilogy(els, Lm, "C3" + ls, label=f"nightsky {azn}")
            ratio, rb, rr, re = (np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan) for a, b in
                                 ((Lm, Lr), (mine[:, 1], ref[:, 1]), (mine[:, 6], ref[:, 6]), (Le, Lr)))
            rows.append((alt, azn, ratio, rb, rr, re))
        ax.set_title(f"sun {alt:+.1f} deg (frame {k})")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("view elevation (deg)")
    for ax in axes[:, 0]:
        ax.set_ylabel("luminance (cd/m$^2$)")
    fig.suptitle("Rayleigh-only sky: nightsky.atmosphere vs Mitsuba path tracing (render_day)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mitsuba_compare.png"), dpi=110)
    plt.close(fig)
    print("[mitsuba] luminance ratio nightsky/Mitsuba at view elevations", list(els))
    for alt, azn, ratio, rb, rr, re in rows:
        print(f"  sun {alt:+6.1f} {azn:5s} L: " + " ".join(f"{x:5.2f}" for x in ratio)
              + f" | median L {np.nanmedian(ratio):.2f}, 456nm {np.nanmedian(rb):.2f}, 681nm {np.nanmedian(rr):.2f},"
              + f" exponential-profile L {np.nanmedian(re):.2f}")
    allr = np.concatenate([r[2] for r in rows if r[0] > 0])
    twl = np.concatenate([r[2] for r in rows if r[0] <= 0])
    print(f"[mitsuba] bilinear cube samples (blend the ground into el < 3 deg; Mitsuba zeros skipped, and single"
          f" samples at sun < -10 deg are dominated by Monte Carlo noise): median ratio day (sun>0): "
          f"{np.nanmedian(allr):.3f} (16-84%: {np.nanpercentile(allr, 16):.2f}-{np.nanpercentile(allr, 84):.2f}); "
          f"twilight (sun<=0): {np.nanmedian(twl):.3f} ({np.nanpercentile(twl, 16):.2f}-{np.nanpercentile(twl, 84):.2f})")
    check_mitsuba_pixels(atm, grid, frames_k)


def _face_dirs(size, sub):
    """Pixel-centre (6-face subset) directions and Gaussian-footprint subsamples (sigma 0.5 px,
    Mitsuba's default reconstruction filter) for the four horizontal faces and the up face."""
    t = np.tan(np.radians(views.FACE_FOV_DEG) / 2)
    o = np.linspace(-1.5, 1.5, sub)
    ox, oy = np.meshgrid(o, o)
    w = np.exp(-0.5 * (ox ** 2 + oy ** 2) / 0.25).ravel()
    rows, cols = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5, indexing="ij")
    out = []
    for k in range(5):
        fwd, right, up = views.face_basis(k)
        rr, cc = rows[..., None] + oy.ravel(), cols[..., None] + ox.ravel()
        d = fwd + ((cc / size * 2 - 1) * t)[..., None] * right + ((1 - rr / size * 2) * t)[..., None] * up
        out.append(d / np.linalg.norm(d, axis=-1, keepdims=True))
    return np.stack(out), w / w.sum()


def check_mitsuba_pixels(atm, grid, frames_k):
    """Mitsuba pixel values vs nightsky averaged over the same pixel footprints; ratios of sums
    over elevation bands (sums, since per-pixel Monte Carlo noise biases medians of ratios)."""
    bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 45), (45, 90)]
    ib = [0, 3, 7]
    print("[mitsuba-px] ratio of sums nightsky/Mitsuba per elevation band (deg), luminance and "
          f"{', '.join(f'{grid.wl[i]:.0f}' for i in ib)} nm; pixels > 3 deg from the sun")
    day, twl = [], []
    for k in frames_k:
        path = os.path.join(DAY, f"sun_{k:03d}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        H, alt = float(z["hour_angle"]), float(z["altitude"])
        sd = frames.equator_vec(H, LAT)
        toa = sun_toa(grid, sd, irr=np.array([[ph.irradiance_bb(1361.0, 5772.0, w) for w in z["wavelengths"]]]))
        ds, w = _face_dirs(z["faces"].shape[-1], 7)
        mine = (atm.sky(toa, np.eye(3), ds)["inscatter"] * w[:, None]).sum(-2)
        ref = np.moveaxis(z["faces"], 0, -1)[:5]
        dc = ds[..., len(w) // 2, :]
        el = np.degrees(np.arcsin(dc[..., 2]))
        far = np.degrees(np.arccos(np.clip(dc @ sd, -1, 1))) > 3
        Lm, Lr = spectral.luminance(mine, grid), spectral.luminance(ref, grid)
        line = f"  sun {alt:+6.1f}:"
        for lo, hi in bands:
            m = (el > lo) & (el <= hi) & far
            rl = Lm[m].sum() / max(Lr[m].sum(), 1e-30)
            rs = [mine[m][:, i].sum() / max(ref[m][:, i].sum(), 1e-30) for i in ib]
            line += f" [{lo}-{hi}] {rl:.2f} (" + "/".join(f"{x:.2f}" for x in rs) + ")"
            (day if alt > 0 else twl).append(rl)
        print(line)
    print("[mitsuba-px] band ratios: " + "; ".join(
        f"{lab} median {np.median(v):.3f} (range {min(v):.2f}-{max(v):.2f})" for lab, v in (("day", day), ("twilight", twl)) if v))


def mc_slab(tau_star, mu_view, albedo, n=100000, seed=1):
    """Backward Monte Carlo reference: radiance / L0 seen upward (cos zenith mu_view) under a
    conservative plane-parallel Rayleigh slab (no depolarisation) lit by isotropic L0 from
    above, Lambertian ground of the given albedo."""
    rng = np.random.default_rng(seed)
    tau = np.full(n, tau_star)
    d = np.tile([np.sqrt(1 - mu_view ** 2), 0.0, mu_view], (n, 1))
    alive = np.ones(n, bool)
    esc = np.zeros(n)
    for _ in range(500):
        idx = np.nonzero(alive)[0]
        if not len(idx):
            break
        tn = tau[idx] + np.log(rng.random(len(idx))) * d[idx, 2]
        up, gnd = tn <= 0, tn >= tau_star
        esc[idx[up]] = 1
        alive[idx[up]] = False
        gi = idx[gnd]
        refl = rng.random(len(gi)) < albedo
        alive[gi[~refl]] = False
        gr = gi[refl]
        mz, ph_ = np.sqrt(rng.random(len(gr))), 2 * np.pi * rng.random(len(gr))
        d[gr] = np.column_stack([np.sqrt(1 - mz ** 2) * np.cos(ph_), np.sqrt(1 - mz ** 2) * np.sin(ph_), mz])
        tau[gr] = tau_star
        sc = idx[~up & ~gnd]
        tau[sc] = tn[~up & ~gnd]
        mu, todo = np.empty(len(sc)), np.arange(len(sc))
        while len(todo):
            c = 2 * rng.random(len(todo)) - 1
            ok = 2 * rng.random(len(todo)) < 1 + c * c
            mu[todo[ok]] = c[ok]
            todo = todo[~ok]
        phi = 2 * np.pi * rng.random(len(sc))
        old = d[sc]
        e1 = np.cross(old, np.where(np.abs(old[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]]))
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = np.cross(old, e1)
        st = np.sqrt(1 - mu ** 2)
        d[sc] = mu[:, None] * old + (st * np.cos(phi))[:, None] * e1 + (st * np.sin(phi))[:, None] * e2
    return esc.mean()


def check_energy():
    """Isotropic unit TOA radiance over a pure-Rayleigh (no absorption) atmosphere: T + inscatter
    vs a plane-parallel Monte Carlo; albedo 1 must return 1 everywhere (radiative equilibrium)."""
    g = spectral.default_grid()
    iso = sky.TOASky(g, sky.PointSources.empty(g.n), np.ones((12 * NSIDE ** 2, g.n)))
    els = np.array([89.9, 60.0, 30.0, 10.0])
    dirs = frames.altaz_to_vec(els, np.full_like(els, 37.0))
    print(f"[energy] isotropic unit TOA field, Rayleigh only; nightsky T+inscatter vs plane-parallel MC at el {list(els)}")
    for scale in (0.1, 1.0, 3.0):
        for alb in (0.0, 0.3, 1.0):
            pre = dict(base="earth_clear", name="energy", rayleigh_profile="exponential", rayleigh_scale_height_m=8e3,
                       rayleigh_xsec=lambda wl, s=scale: s * 1e4 * A.rayleigh_xsec_air(wl), depolarization=0.0,
                       ozone_du=0.0, aerosol_tau550=0.0, strat_tau550=0.0, ground_albedo=alb, airglow_scale=0.0)
            a = A.Atmosphere(pre, g)
            r = a.sky(iso, np.eye(3), dirs)
            tot = r["T"] + r["inscatter"]
            tz = a.tau(np.array([[0, 0, 1.0]]))[0]
            for b in (0, 9):
                ref = [mc_slab(tz[b], np.sin(np.radians(e)), alb) for e in els]
                print(f"  x{scale:<4} albedo {alb:.1f} {g.wl[b]:.0f} nm tau {tz[b]:.3f}: nightsky "
                      + " ".join(f"{x:.3f}" for x in tot[:, b]) + " | MC " + " ".join(f"{x:.3f}" for x in ref))


def check_literature(out, plt):
    g = spectral.default_grid()
    names = ["earth_clear", "earth_hazy", "earth_observatory", "mars"]
    atms = {}
    for n in names + ["none"]:
        t0 = time.time()
        atms[n] = A.Atmosphere(n, g)
        s = atms[n].summary()
        print(f"[preset] {n:18s} build {time.time() - t0:5.1f} s  k_V = {s['k_V']:.3f} mag/airmass"
              + (f"  tau550: total {s['tau_zenith_550']:.3f} Rayleigh {s['tau_rayleigh_550']:.3f} ozone "
                 f"{s['tau_ozone_550']:.4f} aerosol {s['tau_aerosol_550']:.3f}" if "tau_zenith_550" in s else ""))
    print("[ref] k_V: 0.12-0.2 clear sea level, 0.1-0.13 observatory (2-3 km)")
    zen = np.array([[0.0, 0.0, 1.0]])
    sun_el = np.array([60, 45, 30, 20, 10, 5, 2, 0, -2, -4, -6, -8, -10, -12, -14, -16, -18])
    curves = {}
    night = night_toa(g)
    for n in names:
        a = atms[n]
        Ln = spectral.luminance(total(a, night, zen), g)[0]
        curves[n] = np.array([spectral.luminance(total(a, sun_toa(g, frames.altaz_to_vec(e, 90.0)), zen), g)[0]
                              for e in sun_el]) + Ln
    print("[zenith luminance cd/m^2 incl. night sky]  sun el: " + " ".join(f"{e:8d}" for e in sun_el))
    for n in names:
        print(f"  {n:18s}" + " ".join(f"{x:8.2e}" for x in curves[n]))
    lit = {30: (0.92e3, 2.5e3), -6: (1.0, 3.0), -12: (5e-3, 2e-2), -18: (1e-4, 4e-4)}
    print("[ref] zenith: sun 30 deg 0.9e3-2.5e3 (clear-sky formula L_z = [(1.376 T - 1.81) tan(el) + 0.38] kcd/m^2"
          " for Linke turbidity T = 2-4, Krochmann & Seidl 1974, as used by Radiance gensky), -6 deg 1-3,"
          " -12 deg ~1e-2, -18 deg night (1e-4-4e-4)")
    for e, (lo, hi) in lit.items():
        v = curves["earth_clear"][list(sun_el).index(e)]
        print(f"  earth_clear sun {e:+3d}: {v:.2e} cd/m^2  (ref {lo:.1e}-{hi:.1e}) -> factor {v / np.sqrt(lo * hi):.2f} of ref geometric mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in names:
        ax.semilogy(sun_el, curves[n], "-o", ms=3, label=n)
    for e, (lo, hi) in lit.items():
        ax.plot([e, e], [lo, hi], "k-", lw=4, alpha=0.5)
    ax.plot([], [], "k-", lw=4, alpha=0.5, label="literature range")
    ax.set_xlabel("solar elevation (deg)")
    ax.set_ylabel("zenith luminance (cd/m$^2$)")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "zenith_luminance.png"), dpi=120)
    plt.close(fig)

    for n in ["earth_clear", "earth_observatory"]:
        a = atms[n]
        r = a.sky(night, np.eye(3), zen)
        tot = r["T"] * night.diffuse_map[0] + r["inscatter"] + r["airglow"]
        print(f"[dark sky] {n:18s} zenith V = {A.v_mag_arcsec2(tot, g)[0]:.2f} mag/arcsec^2 (airglow alone "
              f"{A.v_mag_arcsec2(r['airglow'], g)[0]:.2f}, TOA ZL 23.1 + ISL 23.6; spectral.radiance_to_mag_arcsec2 "
              f"gives {spectral.radiance_to_mag_arcsec2(tot, g)[0]:.2f} because its 550-nm bin holds OI 557.7)")
    print("[ref] dark-sky zenith V 21.8-22.0, airglow alone 22.5-22.8")
    vac = atms["none"]
    d = frames.altaz_to_vec(np.array([45.0, -5.0]), np.array([0.0, 0.0]))
    r = vac.sky(sun_toa(g, frames.altaz_to_vec(30, 90)), np.eye(3), d)
    print(f"[none] T above/below horizon {r['T'][0, 0]:.1f}/{r['T'][1, 0]:.1f}, max inscatter {r['inscatter'].max():.1e}")
    return atms


def check_consistency(atms):
    """Ground illuminance in twilight, aerial perspective limits, diffuse-light bookkeeping."""
    g = spectral.default_grid()
    a = atms["earth_clear"]
    R = np.eye(3)
    lit = {30: (6e4, 9e4), 0: (400.0, 750.0), -6: (2.0, 4.0), -12: (5e-3, 1.2e-2), -18: (4e-4, 1e-3)}
    print("[illuminance] horizontal, earth_clear incl. night sky (ref: Bond & Henderson 1963 / USNO: "
          "sun 0 ~400-750 lx, -6 ~3.4 lx, -12 ~0.008 lx, -18 ~6e-4 lx)")
    night = night_toa(g)
    for e, (lo, hi) in lit.items():
        E = a.ground_irradiance(sun_toa(g, frames.altaz_to_vec(e, 100.0)).merged(night), R)
        ev = spectral.luminance(E, g)
        print(f"  sun {e:+3d}: {ev:9.3e} lx  (ref {lo:.1e}-{hi:.1e})")
    toa = sun_toa(g, frames.altaz_to_vec(20.0, 100.0))
    d = frames.altaz_to_vec(np.array([2.0, 10.0, 30.0, -3.0]), np.array([280.0, 190.0, 100.0, 10.0]))
    sk = a.sky(toa, R, d)
    for dist in (300.0, 3e3, 3e4, 1e5, np.inf):
        T, ins = a.aerial_perspective(d, np.full(len(d), dist), toa, R)
        Ly = spectral.luminance(ins, g)
        print(f"[aerial] d={dist:8.0f} m  T550 " + " ".join(f"{x:.3f}" for x in T[:, g.bin_index(550)])
              + "  inscatter/sky_inscatter " + " ".join(f"{x:.3f}" for x in Ly / np.maximum(spectral.luminance(sk["inscatter"], g), 1e-30)))
    print("[aerial] (directions: el 2 anti-solar, el 10, el 30 toward the sun, el -3; ratios -> 1 for long paths above the"
          " horizon; at el -3 the sky path ends on the ground sphere after ~40 m while the terrain path continues"
          " through ground-level air, so that ratio grows with d)")
    up = sky.TOASky(g, sky.PointSources.empty(g.n), np.where(frames.healpix_dirs(16)[:, 2:3] > 0, 1.0, 0.0) * np.ones(g.n))
    for n in ("earth_clear", "earth_observatory"):
        r = atms[n].sky(up, R, np.array([[0, 0, 1.0], frames.altaz_to_vec(10.0, 0.0)]))
        tot = r["T"] + r["inscatter"]
        print(f"[diffuse] {n}: unit-radiance upper-hemisphere TOA field -> zenith {tot[0, [2, 8, 17]].round(3)}, "
              f"el 10 {tot[1, [2, 8, 17]].round(3)} (420/550/730 nm; <1 from absorption and the dark ground)")


def check_maps(out, plt, atms):
    g = spectral.default_grid()
    v = views.View("allsky", 320, 320)
    dirs = v.dirs.reshape(-1, 3)
    valid = v.valid & (v.dirs[..., 2] > 0)
    a = atms["earth_clear"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, e in zip(axes, [30, -6, -12]):
        L = spectral.luminance(total(a, sun_toa(g, frames.altaz_to_vec(e, 100.0)), dirs), g).reshape(v.dirs.shape[:2])
        im = ax.imshow(np.where(valid, np.log10(np.maximum(L, 1e-8)), np.nan), cmap="inferno")
        ax.set_title(f"earth_clear, sun {e:+d} deg at az 100 (E-SE, left)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, label="log10 luminance (cd/m$^2$)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "allsky_falsecolor.png"), dpi=110)
    plt.close(fig)

    sun_els = [30, 4, -2, -5, -9]
    names = ["earth_clear", "earth_hazy", "earth_observatory", "mars"]
    fig, axes = plt.subplots(len(names), len(sun_els), figsize=(3.1 * len(sun_els), 3.2 * len(names)))
    night = night_toa(g)
    for i, n in enumerate(names):
        for j, e in enumerate(sun_els):
            toa = sun_toa(g, frames.altaz_to_vec(e, 100.0)).merged(night)
            t0 = time.time()
            L = total(atms[n], toa, dirs).reshape(v.dirs.shape[:2] + (g.n,))
            dt = time.time() - t0
            img = srgb(L, g, valid)
            axes[i, j].imshow(img)
            axes[i, j].set_title(f"{n} sun {e:+d}  ({dt:.2f} s)", fontsize=8)
            axes[i, j].axis("off")
            import imageio.v3 as iio
            iio.imwrite(os.path.join(out, f"allsky_srgb_{n}_sun{e:+03d}.png"), (img * 255).astype(np.uint8))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "allsky_srgb_presets.png"), dpi=100)
    plt.close(fig)

    m = atms["mars"]
    sd = frames.altaz_to_vec(2.0, 100.0)
    toa = sun_toa(g, sd)
    near = frames.altaz_to_vec(np.array([4.0, 30.0]), np.array([100.0, 280.0]))
    L = total(m, toa, near)
    br = L[:, g.bin_index(450)] / L[:, g.bin_index(650)]
    xyz = spectral.to_xyz(L, g)
    xy = xyz[:, :2] / xyz.sum(1, keepdims=True)
    print(f"[mars] sun +2 deg: B/R (450/650 nm) radiance ratio 2 deg from sun {br[0]:.2f}, anti-solar el 30 {br[1]:.2f}; "
          f"chromaticity xy near sun {xy[0].round(3)}, anti-solar {xy[1].round(3)} (blue glow if near-sun B/R exceeds anti-solar)")

    e = atms["earth_clear"]
    sdir = frames.altaz_to_vec(-4.0, 100.0)
    toa = sun_toa(g, sdir)
    els = np.linspace(0.5, 89.5, 90)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for az, ls, lab in ((100.0, "-", "solar az"), (280.0, "--", "anti-solar az")):
        L = total(e, toa, frames.altaz_to_vec(els, np.full_like(els, az)))
        axs[0].semilogy(els, spectral.luminance(L, g), ls, label=f"sun -4, {lab}")
    toa30 = sun_toa(g, frames.altaz_to_vec(30.0, 100.0))
    for az, ls, lab in ((100.0, "-", "solar az"), (280.0, "--", "anti-solar az")):
        L = total(e, toa30, frames.altaz_to_vec(els, np.full_like(els, az)))
        axs[0].semilogy(els, spectral.luminance(L, g), ls, label=f"sun +30, {lab}")
    axs[0].set_xlabel("view elevation (deg)")
    axs[0].set_ylabel("luminance (cd/m$^2$)")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)
    ag = e.airglow0
    axs[1].step(g.wl, ag, where="mid", label="airglow zenith (TOA)")
    axs[1].step(g.wl, night.diffuse_map[0], where="mid", label="ZL + ISL (TOA)")
    axs[1].set_xlabel("wavelength (nm)")
    axs[1].set_ylabel("W m$^{-2}$ sr$^{-1}$ nm$^{-1}$")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "luminance_vs_elevation.png"), dpi=110)
    plt.close(fig)


def check_timing(atms, out=None, plt=None):
    """GC-like night: 17 mag/arcsec^2 band (nside 64) + 2000 points near the pole, 1920x1080."""
    g = spectral.default_grid()
    a = atms["earth_clear"]
    v = views.View("perspective", 1920, 1080, azimuth_deg=0.0, pitch_deg=30.0, fov_deg=90.0)
    d = v.dirs.reshape(-1, 3)
    rng = np.random.default_rng(1)
    hd = frames.healpix_dirs(64)
    band = np.exp(-0.5 * (hd[:, 0] / 0.15) ** 2)[:, None]
    m = band * spectral.surface_brightness_radiance(17.0, 4000.0, g)[None, :]
    pole = np.array([0.0, 0.0, 1.0])
    off = rng.normal(scale=0.03, size=(2000, 3))
    pd = (pole + off) / np.linalg.norm(pole + off, axis=1, keepdims=True)
    irr = spectral.vmag_irradiance(rng.uniform(-6, 8, 2000), 30000.0, g)
    toa = sky.TOASky(g, sky.PointSources(pd, irr), m)
    R = frames.celestial_to_horizon(180.0, LAT)
    t0 = time.time()
    r = a.sky(toa, R, d)
    t1 = time.time()
    a.sky(toa, R, d)
    t2 = time.time()
    a.aerial_perspective(d, np.full(len(d), 2e4), toa, R)
    t3 = time.time()
    E = a.ground_irradiance(toa, R)
    t4 = time.time()
    pi = a.point_irradiance(toa, R)
    print(f"[timing] 1920x1080, 2000 points + nside-64 map: sky first call {t1 - t0:.2f} s (source binning + grids), "
          f"repeat {t2 - t1:.2f} s; aerial_perspective {t3 - t2:.2f} s; ground_irradiance {t4 - t3:.2f} s")
    tot = spectral.luminance(r["T"] * toa.diffuse(d @ R) + r["inscatter"] + r["airglow"], g)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(np.log10(tot.reshape(v.dirs.shape[:2])), cmap="magma")
        fig.colorbar(im, ax=ax, label="log10 sky luminance (cd/m$^2$), point sources excluded")
        ax.set_title("GC-like night, earth_clear: scattered cluster light (2000 points at the pole) + 17 mag/arcsec$^2$ band")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "gc_like_aureole.png"), dpi=100)
        plt.close(fig)
    print(f"[gc-like] sky luminance in view: median {np.median(tot):.2e} cd/m^2, max {tot.max():.2e}; ground "
          f"illuminance {spectral.luminance(E, g):.3e} lx; points summed V(TOA) "
          f"{spectral.vmag_of(toa.points.irr.sum(0), g):.2f} -> after extinction {spectral.vmag_of(pi.irr.sum(0), g):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--skip-mitsuba", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not a.skip_mitsuba:
        check_mitsuba(a.out, plt)
    check_energy()
    atms = check_literature(a.out, plt)
    check_consistency(atms)
    check_maps(a.out, plt, atms)
    check_timing(atms, a.out, plt)
    print("wrote PNGs to", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
