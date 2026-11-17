"""Validation of nightsky.galaxy: Earth and Galactic Center observers, synthetic voxel fields.

    cd src && CPUS=16 MEM=48G ../slurm/onworker.sh python3 -m nightsky.validate_galaxy [--nside 512]

Prints quantitative checks against reference values and writes Mollweide PNGs and a JSON
summary to renders/nightsky/validation/galaxy/. Surface brightness is V mag/arcsec^2 from
spectral.radiance_to_mag_arcsec2 (550 nm, Vega); luminance in cd/m^2 (photopic, CIE 1931).
References: Leinert et al. 1998 (integrated starlight, ISL), Masana et al. 2021 (Gaia ISL),
Bland-Hawthorn & Gerhard 2016 (masses), Flynn et al. 2006 (L_V).
"""
import argparse
import json
import os
import time

import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import fields, frames, galaxy, spectral
import photometry as ph

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "renders", "nightsky", "validation", "galaxy")
R_GAL = np.diag([-1.0, 1.0, 1.0])  # galactocentric direction -> (l, b) frame seen from the observer
F_B0_440 = 6.32e-11  # Vega B zero point, W m^-2 nm^-1 (Bessell 1998)


def mag(m, grid):
    return spectral.radiance_to_mag_arcsec2(m, grid)


def b_minus_v(m, grid):
    fb = np.array([np.interp(440.0, grid.wl, r) for r in np.atleast_2d(m)])
    fv = np.array([np.interp(550.0, grid.wl, r) for r in np.atleast_2d(m)])
    return -2.5 * np.log10(np.maximum(fb, 1e-300) / F_B0_440) + 2.5 * np.log10(np.maximum(fv, 1e-300) / ph.F_V0_550NM)


def at(m, nside, l, b):
    return m[hp.ang2pix(nside, l, b, lonlat=True)]


def moll(values, nside, xsize=1600, rot=None):
    proj = hp.projector.MollweideProj(xsize=xsize, rot=rot)
    return proj.projmap(values, lambda x, y, z: hp.vec2pix(nside, x, y, z))


def srgb_image(m, grid, nside, xsize=1600, white_pct=99.5, stretch="log", rot=None):
    xyz = spectral.to_xyz(m, grid)
    rgb = np.clip(xyz @ ph.XYZ_TO_SRGB.T, 0, None)
    Y = xyz[:, 1]
    if stretch == "log":
        lo, hi = np.percentile(Y, 1), np.percentile(Y, white_pct)
        t = np.clip(np.log(Y / lo) / np.log(hi / lo), 0, 1)
        rgb = rgb / np.maximum(Y, 1e-300)[:, None] * t[:, None]
    else:
        rgb = rgb / np.percentile(Y, white_pct)
    rgb = np.clip(rgb / max(rgb.max(), 1e-30), 0, 1) ** (1 / 2.2)
    chans = [moll(rgb[:, i], nside, xsize, rot) for i in range(3)]
    img = np.stack(chans, -1)
    return np.where(np.isfinite(img), img, 1.0)


def save_maps(prefix, m, grid, nside, title, lon_label="l", vrange=None, rot=None):
    sb = mag(m, grid)
    fig, ax = plt.subplots(2, 1, figsize=(11, 11))
    img = moll(sb, nside, rot=rot)
    lo, hi = vrange if vrange else (np.nanpercentile(sb, 0.5), np.nanpercentile(sb, 99.5))
    im = ax[0].imshow(img, cmap="magma_r", vmin=lo, vmax=hi, origin="lower")
    plt.colorbar(im, ax=ax[0], shrink=0.7, label="V mag arcsec$^{-2}$")
    c = 0 if rot is None else rot[0]
    ax[0].set_title(f"{title}: V surface brightness (centre {lon_label}={c:.1f}, {lon_label} increases to the left)")
    ax[1].imshow(srgb_image(m, grid, nside, rot=rot), origin="lower")
    ax[1].set_title(f"{title}: colour (log stretch, sRGB)")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(prefix + ".png", dpi=110)
    plt.close(fig)


def stats(m, grid):
    sb = mag(m, grid)
    L = spectral.luminance(m, grid)
    return {"sb_min": float(sb.min()), "sb_median": float(np.median(sb)), "sb_max": float(sb.max()),
            "cd_m2_min": float(L.min()), "cd_m2_median": float(np.median(L)), "cd_m2_max": float(L.max()),
            "cd_m2_mean": float(L.mean())}


def report(name, value, ref, unit=""):
    print(f"  {name:48s} {value:12.4g} {unit:14s} ref: {ref}", flush=True)


def earth(model, grid, nside, out):
    print("\n== Earth observer (R0, 0, +20.8 pc) ==", flush=True)
    obs = np.array(galaxy.EARTH_OBSERVER_PC)
    t = time.time()
    sk = model.render_map(obs, nside, R_GAL)
    dt = time.time() - t
    sk0 = galaxy.GalaxyModel(grid, "no_dust").render_map(obs, nside, R_GAL)
    m, m0 = sk.diffuse_map, sk0.diffuse_map
    sb, sb0 = mag(m, grid), mag(m0, grid)
    print(f"  render_map nside={nside}: {dt:.1f} s")
    l, b = hp.pix2ang(nside, np.arange(len(sb)), lonlat=True)
    lw = np.where(l > 180, l - 360, l)
    plane = (np.abs(b) < 1.0) & (np.abs(lw) > 60)
    pole = np.abs(b) > 80
    res = {"render_s": dt}
    res["plane_|l|>60_|b|<1_median"] = float(np.median(sb[plane]))
    report("plane |l|>60, |b|<1 median (mag/arcsec2)", res["plane_|l|>60_|b|<1_median"], "20.5-22 (Leinert98, Masana21)")
    for (ll, bb) in [(90, 0), (180, 0), (270, 0), (30, 0), (0, 0), (0, 5), (0, -5), (1.0, -3.9), (0, 10), (0, 90), (0, -90)]:
        v, v0 = float(at(sb, nside, ll, bb)), float(at(sb0, nside, ll, bb))
        res[f"sb_l{ll}_b{bb}"] = v
        res[f"sb0_l{ll}_b{bb}"] = v0
        print(f"  (l, b) = ({ll:5.1f}, {bb:5.1f}): {v:6.2f} mag/arcsec2 (no dust {v0:6.2f}), B-V {float(b_minus_v(at(m, nside, ll, bb), grid)[0]):.2f}")
    res["pole_median"] = float(np.median(sb[pole]))
    report("|b|>80 median (mag/arcsec2)", res["pole_median"], "23-24 (Leinert98 ISL)")
    res["gc_lane_contrast_mag"] = float(at(sb, nside, 0, 0) - 0.5 * (at(sb, nside, 0, 4) + at(sb, nside, 0, -4)))
    report("dust lane: sb(0,0) - mean sb(0,+-4)", res["gc_lane_contrast_mag"], "> 0 (dark lane)")
    res["bulge_glow"] = float(at(sb, nside, 30, 4) - at(sb, nside, 0, 4))
    report("bulge glow: sb(30,4) - sb(0,4)", res["bulge_glow"], "> 0 (bulge brighter)")
    E = spectral.to_xyz(sk.total_irradiance_on(np.array([0.0, 0.0, 1.0])), grid)[1]
    Eh = 0.5 * (E + spectral.to_xyz(sk.total_irradiance_on(np.array([0.0, 0.0, -1.0])), grid)[1])
    res["illuminance_hemisphere_mean_lux"] = float(Eh)
    report("starlight illuminance (mean of both hemispheres)", Eh, "~1-3e-4 lux (Roach & Gordon 1973)", "lux")
    save_maps(os.path.join(out, "earth_galactic"), m, grid, nside, "Earth, extincted")
    save_maps(os.path.join(out, "earth_galactic_nodust"), m0, grid, nside, "Earth, no dust")
    return res


def gc(model, grid, nside, out):
    print("\n== GC observer: 3 pc from the Arches along (1, 0, 0) ==", flush=True)
    obs = galaxy.gc_observer_pc()
    arches, sgra = galaxy.arches_pc(), galaxy.sgr_a_star_pc()
    R_cel = frames.galactocentric_to_celestial(obs, arches, reference_pc=sgra)
    print(f"  observer {np.round(obs, 3)} pc, distance to Sgr A* {np.linalg.norm(obs - sgra):.2f} pc")
    res = {}
    t = time.time()
    sk = model.render_map(obs, nside, R_GAL)
    res["render_s"] = time.time() - t
    print(f"  render_map nside={nside} (galactic-like frame): {res['render_s']:.1f} s", flush=True)
    t = time.time()
    skc = model.render_map(obs, nside, R_cel)
    res["render_cel_s"] = time.time() - t
    nod = galaxy.GalaxyModel(grid, "no_dust")
    sk0 = nod.render_map(obs, nside, R_GAL)
    m, m0 = sk.diffuse_map, sk0.diffuse_map
    for k, mm in (("extincted", m), ("unextincted", m0)):
        s = stats(mm, grid)
        res[k] = s
        print(f"  {k:12s} V sb min/median/max = {s['sb_min']:.2f} / {s['sb_median']:.2f} / {s['sb_max']:.2f} mag/arcsec2; "
              f"luminance min/median/max/mean = {s['cd_m2_min']:.3g} / {s['cd_m2_median']:.3g} / {s['cd_m2_max']:.3g} / {s['cd_m2_mean']:.3g} cd/m2")
    L = spectral.luminance(m, grid)
    ib = int(np.argmax(L))
    lb, bb = hp.pix2ang(nside, ib, lonlat=True)
    d_sgra = R_GAL @ ((sgra - obs) / np.linalg.norm(sgra - obs))
    sep = np.degrees(np.arccos(np.clip(hp.pix2vec(nside, ib) @ d_sgra, -1, 1)))
    res["brightest_lb"] = [float(lb), float(bb)]
    res["brightest_sep_from_sgra_deg"] = float(sep)
    l_s, b_s = hp.vec2ang(d_sgra, lonlat=True)
    print(f"  brightest pixel at (l', b') = ({lb:.2f}, {bb:.2f}), {sep:.2f} deg from Sgr A* (at ({l_s[0]:.2f}, {b_s[0]:.2f})), "
          f"{mag(m[ib], grid):.2f} mag/arcsec2, {L[ib]:.3g} cd/m2")
    # apparent sizes from component-only maps
    ns2 = min(nside, 128)
    vec = np.stack(hp.pix2vec(ns2, np.arange(hp.nside2npix(ns2))), -1)
    ang = np.degrees(np.arccos(np.clip(vec @ d_sgra, -1, 1)))
    for comp, kw in (("nsc", dict(M_nsd=0, M_bulge=0, M_longbar=0, rho_local=0)),
                     ("nsd", dict(M_nsc=0, M_bulge=0, M_longbar=0, rho_local=0))):
        mc = galaxy.GalaxyModel(grid, "no_dust", **kw).render_map(obs, ns2, R_GAL).diffuse_map
        Lc = spectral.luminance(mc, grid)
        o = np.argsort(ang)
        cum = np.cumsum(Lc[o]) / Lc.sum()
        th50, th90 = float(ang[o][np.searchsorted(cum, 0.5)]), float(ang[o][np.searchsorted(cum, 0.9)])
        peak_half = float(np.max(ang[Lc > 0.5 * Lc.max()]))
        res[f"{comp}_theta50_deg"], res[f"{comp}_theta90_deg"], res[f"{comp}_halfmax_radius_deg"] = th50, th90, peak_half
        res[f"{comp}_flux_fraction_of_total"] = float(Lc.sum() / spectral.luminance(sk0.diffuse_map, grid).sum() * (hp.nside2npix(nside) / hp.nside2npix(ns2)))
        print(f"  {comp.upper()} (unextincted): half-light radius {th50:.1f} deg, 90% radius {th90:.1f} deg, "
              f"half-max radius {peak_half:.1f} deg around Sgr A*; fraction of total sky light {res[f'{comp}_flux_fraction_of_total']:.2f}")
        if comp == "nsd":
            lat = np.linspace(-90, 90, 721)
            for dl in (0.0, 45.0, 90.0, 180.0):
                prof = spectral.luminance(mc[hp.ang2pix(ns2, np.full_like(lat, l_s[0] + dl), lat, lonlat=True)], grid)
                above = lat[prof > 0.5 * prof.max()]
                res[f"nsd_lat_fwhm_dl{int(dl)}"] = float(above.max() - above.min())
                print(f"    NSD latitude FWHM at l' = l'(Sgr A*) + {dl:5.1f}: {res[f'nsd_lat_fwhm_dl{int(dl)}']:.1f} deg "
                      f"(peak at b' = {lat[np.argmax(prof)]:.1f})")
    print(f"  (NSC R_eff 4.2 pc at {np.linalg.norm(obs - sgra):.1f} pc subtends {np.degrees(np.arctan(4.2 / np.linalg.norm(obs - sgra))):.1f} deg)")
    # illuminance on a horizontal surface (lat 40, equinox) at a few hour angles
    for H in (180.0, 90.0, 0.0):
        Rh = frames.celestial_to_horizon(H, 40.0)
        zen = Rh.T @ np.array([0.0, 0.0, 1.0])
        Ev = spectral.to_xyz(skc.total_irradiance_on(zen), grid)[1]
        Ev0 = spectral.to_xyz(galaxy.GalaxyModel(grid, "no_dust").render_map(obs, 64, R_cel).total_irradiance_on(zen), grid)[1]
        res[f"E_horizontal_H{int(H)}_lux"] = float(Ev)
        res[f"E_horizontal_H{int(H)}_lux_nodust"] = float(Ev0)
        print(f"  horizontal illuminance at H={H:5.1f} deg: {Ev:.4g} lux (unextincted {Ev0:.4g} lux; full Moon ~0.2-0.3 lux)")
    Esph = float(spectral.luminance(m, grid).mean() * np.pi)
    res["E_mean_horizontal_lux"] = Esph
    print(f"  4pi-mean-equivalent horizontal illuminance (pi <L>): {Esph:.4g} lux")
    save_maps(os.path.join(out, "gc_galactic"), m, grid, nside, "GC observer, extincted", "l'")
    save_maps(os.path.join(out, "gc_galactic_nodust"), m0, grid, nside, "GC observer, no dust", "l'")
    save_maps(os.path.join(out, "gc_galactic_sgra_centred"), m, grid, nside, "GC observer, extincted, centred on Sgr A*", "l'",
              rot=(float(l_s[0]), 0.0))
    save_maps(os.path.join(out, "gc_celestial"), skc.diffuse_map, grid, nside, "GC observer, celestial frame (pole = Arches)", "RA")
    return res, obs, R_cel


def timing_views(model, obs, R_cel, out):
    import views
    v = views.View("perspective", 1920, 1080, azimuth_deg=0.0, pitch_deg=20.0, fov_deg=90.0)
    fn = model.diffuse_fn(obs, R_cel)
    t = time.time()
    img = fn(v.dirs.reshape(-1, 3)).reshape(1080, 1920, -1)
    dt = time.time() - t
    print(f"\n  diffuse_fn on a 1920x1080 view: {dt:.1f} s, median {np.median(spectral.luminance(img, model.grid)):.3g} cd/m2", flush=True)
    return dt


def gc_views(model, obs, out):
    """Perspective diagnostics in the GC-observer galactic-like frame (plane horizontal)."""
    import views
    sgra = galaxy.sgr_a_star_pc()
    d = R_GAL @ ((sgra - obs) / np.linalg.norm(sgra - obs))
    l_s, b_s = np.degrees(np.arctan2(d[1], d[0])), np.degrees(np.arcsin(d[2]))
    fn = model.diffuse_fn(obs, R_GAL)
    fig, ax = plt.subplots(2, 1, figsize=(12, 14))
    for a, (dl, name) in zip(ax, ((0.0, "towards Sgr A*"), (180.0, "away from Sgr A*"))):
        v = views.View("perspective", 1200, 675, azimuth_deg=-(l_s + dl), pitch_deg=b_s if dl == 0 else 0.0, fov_deg=120.0)
        img = fn(v.dirs.reshape(-1, 3)).reshape(v.height, v.width, -1)
        xyz = spectral.to_xyz(img, model.grid)
        Y = xyz[..., 1]
        rgb = np.clip(xyz @ ph.XYZ_TO_SRGB.T, 0, None) / np.maximum(Y, 1e-300)[..., None]
        lo, hi = 0.01, 3.0
        t = np.clip(np.log(Y / lo) / np.log(hi / lo), 0, 1)
        a.imshow(np.clip(rgb * t[..., None] / 1.5, 0, 1) ** (1 / 2.2))
        a.set_title(f"GC observer, {name}: 120 deg FOV, log luminance 0.01-3 cd/m$^2$ (median {np.median(Y):.3g})")
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "gc_views.png"), dpi=100)
    plt.close(fig)


def synthetic_fields(grid):
    """Nested synthetic fields: checks of trilinear sampling, opacity and emission integration."""
    print("\n== synthetic VoxelFields ==", flush=True)
    obs = galaxy.gc_observer_pc()
    ccm = grid.bin_average(lambda lam: galaxy.ccm89(lam)) / 1.0857
    centre = obs + np.array([-20.0, 0.0, 0.0])
    sig, av_peak = 2.0, 1.0  # A_V per pc at the cloud centre
    big_vox, fine_vox = 2.0, 0.25
    def blob(origin, vox, n, s, amp):
        ax = [origin[i] + vox * np.arange(n) for i in range(3)]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        r2 = (X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2
        return (amp * np.exp(-0.5 * r2 / s ** 2))[..., None].astype(np.float32)
    o_big = centre - 32.0
    o_fine = centre - 8.0
    ext = [fields.VoxelField(o_big, big_vox, blob(o_big, big_vox, 33, 8.0, 0.05), ccm[None], "extinction", "ext_big"),
           fields.VoxelField(o_fine, fine_vox, blob(o_fine, fine_vox, 65, sig, av_peak), ccm[None], "extinction", "ext_fine")]
    line = grid.line(656.3, 1.0e-12)  # W m^-2 sr^-1 per pc of H-alpha-like emission per unit data
    emi = [fields.VoxelField(o_fine, fine_vox, blob(o_fine, fine_vox, 65, sig, 1.0), line[None], "emission", "emi_fine")]
    m = galaxy.GalaxyModel(grid, "no_dust", fields=ext + emi)
    target = centre + np.array([-30.0, 0.0, 0.0])
    tau = m.optical_depth(obs, target[None])[0]
    av = np.interp(550.0, grid.wl, tau) / np.interp(550.0, grid.wl, ccm)
    av_ref = av_peak * sig * np.sqrt(2 * np.pi) + 0.05 * 8.0 * np.sqrt(2 * np.pi)
    print(f"  A_V through both nested clouds: {av:.4f} mag (analytic {av_ref:.4f}, rel. err {av / av_ref - 1:+.2e})")
    d = (centre - obs) / np.linalg.norm(centre - obs)
    I_with = m.march(obs, d[None])[0]
    m_noext = galaxy.GalaxyModel(grid, "no_dust", fields=emi, M_nsc=0, M_nsd=0, M_bulge=0, M_longbar=0, rho_local=0)
    I_emis = m_noext.march(obs, d[None])[0]
    ib = grid.bin_index(656.3)
    I_ref = 1e-12 / grid.dwl[ib] * sig * np.sqrt(2 * np.pi)
    print(f"  emission-field radiance at 656 nm: {I_emis[ib]:.4e} (analytic {I_ref:.4e}, rel. err {I_emis[ib] / I_ref - 1:+.2e})")
    pt = m_noext.emissivity(centre[None])[0][ib] * grid.dwl[ib]
    print(f"  emissivity() at the blob centre: {pt:.4e} W m^-2 sr^-1 per pc (expected 1e-12)")
    k = m.opacity(centre[None])[0]
    print(f"  opacity() A_V/pc at the blob centre: {np.interp(550.0, grid.wl, k) / np.interp(550.0, grid.wl, ccm):.4f} (expected {av_peak + 0.05:.4f})")
    # same-cloud emission is self-absorbed: T * I compared to an embedded-source analytic estimate
    print(f"  stellar + cloud: radiance ratio (cloud / no cloud) at 550 nm along the cloud axis: "
          f"{np.interp(550.0, grid.wl, I_with) / np.interp(550.0, grid.wl, galaxy.GalaxyModel(grid, 'no_dust').march(obs, d[None])[0]):.3f}")
    return {"AV_fields": float(av), "AV_fields_ref": float(av_ref), "I_emis": float(I_emis[ib]), "I_emis_ref": float(I_ref)}


def inner_check(model, grid):
    """r_res / inner_factor: removing the stellar light within r_res lowers I by ~ j(observer) r_res."""
    obs = galaxy.gc_observer_pc()
    d = frames.healpix_dirs(2)
    nod = galaxy.GalaxyModel(grid, "no_dust")
    dI = nod.march(obs, d) - nod.march(obs, d, r_res_pc=1.0, inner_factor=0.0)
    j = nod.emissivity(obs[None])[0]
    r = dI[:, 10] / j[10]
    print(f"\n  inner_factor=0, r_res=1 pc: removed radiance / j(observer) = {r.min():.4f}-{r.max():.4f} pc (expected ~1)")
    return {"inner_removed_over_j_min": float(r.min()), "inner_removed_over_j_max": float(r.max())}


def _blob(grid, centre, sig, n=41, kind="emission"):
    vox = sig / 3
    o = np.asarray(centre, float) - vox * (n - 1) / 2
    X, Y, Z = np.meshgrid(*[o[i] + vox * np.arange(n) for i in range(3)], indexing="ij")
    r2 = (X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2
    return fields.VoxelField(o, vox, np.exp(-0.5 * r2 / sig ** 2)[..., None].astype(np.float32),
                             np.full((1, grid.n), 1e-12), kind, "blob")


def review_checks(grid):
    """Frames, numba field sampling, an independent line-of-sight quadrature, step convergence,
    pixel supersampling and the bar orientation."""
    print("\n== review checks ==", flush=True)
    res = {}
    i5 = grid.bin_index(550.0)
    nostars = dict(M_nsc=0, M_nsd=0, M_bulge=0, M_longbar=0, rho_local=0, dust=False)
    obs, sgra, arches = galaxy.gc_observer_pc(), galaxy.sgr_a_star_pc(), galaxy.arches_pc()
    R_cel = frames.galactocentric_to_celestial(obs, arches, reference_pc=sgra)
    for name, tgt in (("sgra", sgra), ("arches", arches)):
        dist = np.linalg.norm(tgt - obs)
        mm = galaxy.GalaxyModel(grid, fields=[_blob(grid, tgt, 0.002 * dist)], **nostars).render_map(obs, 512, R_cel)
        w = mm.diffuse_map[:, i5].astype(float)
        c = (w[:, None] * frames.healpix_dirs(512)).sum(0)
        sep = np.degrees(np.arccos(np.clip(c / np.linalg.norm(c) @ (R_cel @ ((tgt - obs) / dist)), -1, 1))) * 3600
        res[f"frame_{name}_offset_arcsec"] = float(sep)
        print(f"  emitter at {name}: map centroid {sep:.1f} arcsec from R_cel @ direction (pixel 412 arcsec)")
    print(f"  Arches celestial direction {np.round(R_cel @ ((arches - obs) / np.linalg.norm(arches - obs)), 6)} (pole), "
          f"Sgr A* {np.round(R_cel @ ((sgra - obs) / np.linalg.norm(sgra - obs)), 4)} (RA 0)")
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sun = np.array([frames.R0_PC, 0.0, 0.0])
    gc_pt = lambda c: frames.lb_to_galactocentric(c.l.deg, c.b.deg, 1000.0)
    ncp, eqx = SkyCoord(ra=0 * u.deg, dec=90 * u.deg).galactic, SkyCoord(ra=0 * u.deg, dec=0 * u.deg).galactic
    tgt = SkyCoord(ra=266.41684 * u.deg, dec=-29.00781 * u.deg).galactic
    for lab, fn in (("frames.galactocentric_to_celestial", frames.galactocentric_to_celestial),
                    ("galaxy.galactocentric_to_celestial_rh", galaxy.galactocentric_to_celestial_rh)):
        Re = fn(sun, gc_pt(ncp), gc_pt(eqx))
        v = Re @ ((gc_pt(tgt) - sun) / 1000.0)
        ra = float(np.degrees(np.arctan2(v[1], v[0])) % 360)
        res[f"handedness_{lab.split('.')[0]}_sgra_ra"] = ra
        print(f"  {lab} built for Earth (pole = NCP, reference = equinox): Sgr A* at RA {ra:.2f} (true 266.42), det R = {np.linalg.det(Re):+.0f}")
    rng = np.random.default_rng(3)
    f_e = fields.VoxelField(np.array([1.0, -3.0, 2.0]), 0.7, rng.random((7, 9, 5, 3)).astype(np.float32),
                            rng.random((3, grid.n)), "emission", "e")
    f_x = fields.VoxelField(f_e.origin_pc, f_e.voxel_pc, f_e.data, f_e.spectra, "extinction", "x")
    lo, hi = f_e.bounds_pc
    pts = lo - 1.05 + rng.random((20000, 3)) * (hi - lo + 2.1)
    mf = galaxy.GalaxyModel(grid, fields=[f_e, f_x], **nostars)
    ref = f_e.spectral(pts)
    res["voxel_sample_maxdiff"] = float(max(np.abs(mf.emissivity(pts) - ref).max(), np.abs(mf.opacity(pts) - ref).max()) / ref.max())
    print(f"  numba trilinear vs fields.VoxelField.spectral at 20000 random points: max |diff|/max {res['voxel_sample_maxdiff']:.1e}")
    d = np.array([[1.0, 0.0, 0.0]])
    o = np.array([lo[0] - 5, 0.3 * lo[1] + 0.7 * hi[1], 0.5 * (lo[2] + hi[2]) + 0.11])
    s = np.linspace(0, 30, 300001)
    p = o + 0.5 * (s[1:] + s[:-1])[:, None] * d
    j, k, ds = f_e.spectral(p), f_x.spectral(p), np.diff(s)[:, None]
    Iref = (j * np.exp(-(np.cumsum(k * ds, 0) - 0.5 * k * ds)) * ds).sum(0)
    res["voxel_march_relerr"] = float(np.abs(mf.march(o, d, s_max=30.0)[0] / Iref - 1).max())
    print(f"  marcher through an emitting + absorbing field vs 1e-4 pc quadrature: max rel err {res['voxel_march_relerr']:.1e}")
    m = galaxy.GalaxyModel(grid)
    dsg = (sgra - obs) / np.linalg.norm(sgra - obs)
    off = np.cross(dsg, [0.0, 0.0, 1.0])
    off /= np.linalg.norm(off)
    dirs = [dsg, (arches - obs) / np.linalg.norm(arches - obs), -dsg, [0, 0, 1.0], [0, 0, -1.0], [0, 1.0, 0]]
    dirs += [dsg + np.radians(a) * off for a in (0.001, 0.01, 0.1, 1.0, 5.0)] + list(rng.normal(size=(12, 3)))
    dirs = np.array([v / np.linalg.norm(v) for v in dirs])
    keep = dirs.copy()
    Im = m.march(obs, dirs)
    errs, near = [], []
    for i, v in enumerate(dirs):
        sc = max(float((sgra - obs) @ v), 1e-3)
        s = np.unique(np.concatenate([np.logspace(-5, np.log10(25000), 60000), sc + np.logspace(-6, 3.5, 30000),
                                      sc - np.logspace(-6, np.log10(sc), 30000)]))
        s = s[(s >= 0) & (s <= 25000)]
        sm, ds = 0.5 * (s[1:] + s[:-1]), np.diff(s)
        j, k = m.emissivity(obs + sm[:, None] * v), m.opacity(obs + sm[:, None] * v)
        dt = k * ds[:, None]
        Iq = (j * np.exp(-(np.cumsum(dt, 0) - dt)) * np.where(dt > 1e-8, -np.expm1(-dt) / np.maximum(k, 1e-300), ds[:, None])).sum(0)
        errs.append(Im[i, i5] / Iq[i5] - 1)
        near.append((j[sm < 3, i5] * ds[sm < 3]).sum() / Iq[i5])
    res["los_quadrature_max_relerr"], res["los_quadrature_median_relerr"] = float(np.max(np.abs(errs))), float(np.median(np.abs(errs)))
    print(f"  marcher vs independent fine quadrature on {len(dirs)} GC rays (incl. 0-5 deg from Sgr A*): max |rel err| "
          f"{res['los_quadrature_max_relerr']:.1e}, median {res['los_quadrature_median_relerr']:.1e}; first 3 pc carry up to "
          f"{max(near):.0%} of a ray's light")
    res["march_input_unchanged"] = bool(np.array_equal(keep, dirs))
    fine = dict(f_obs=0.015, f_c=0.05, ds_min=0.025, ds_core=0.0025, ds_max=250.0, step_vox=0.35)
    for name, ob, R in (("gc", obs, R_cel), ("earth", np.array(galaxy.EARTH_OBSERVER_PC), R_GAL)):
        a = m.render_map(ob, 64, R).diffuse_map[:, i5]
        b = galaxy.GalaxyModel(grid, **fine).render_map(ob, 64, R).diffuse_map[:, i5]
        c = m.render_map(ob, 64, R, s_max=50000.0).diffuse_map[:, i5]
        res[f"conv_{name}_halfstep_max"], res[f"conv_{name}_2smax_max"] = float(np.abs(b / a - 1).max()), float(np.abs(c / a - 1).max())
        print(f"  {name}: halved steps change the nside-64 map by max {res[f'conv_{name}_halfstep_max']:.1e}; "
              f"s_max 25 -> 50 kpc by max {res[f'conv_{name}_2smax_max']:.1e}")
    a = m.render_map(obs, 64, R_cel, supersample=4).diffuse_map.astype(float)
    b = m.render_map(obs, 256, R_cel).diffuse_map.astype(float)
    bd = np.stack([hp.ud_grade(b[:, i], 64) for i in (i5,)], 1)[:, 0]
    res["supersample_vs_udgrade"] = float(np.abs(a[:, i5] / bd - 1).max())
    print(f"  render_map(nside 64, supersample=4) vs nside-256 map averaged to 64: max rel diff {res['supersample_vs_udgrade']:.1e}")
    sb = mag(galaxy.GalaxyModel(grid, "no_dust").render_map(np.array(galaxy.EARTH_OBSERVER_PC), 128, R_GAL).diffuse_map, grid)
    asym = [float(at(sb, 128, -ll, bb) - at(sb, 128, ll, bb)) for ll, bb in ((10, 5), (10, -5), (20, 5))]
    res["bar_asymmetry_mag"] = asym
    print(f"  bar orientation (no dust): sb(-l) - sb(+l) at (10,5), (10,-5), (20,5) = {np.round(asym, 2)} mag "
          f"(> 0: near end at l > 0, as in COBE/DIRBE, Weiland et al. 1994)")
    return res


def gas_check(grid, nside, resolution, out):
    """GC render with the gas agent's VoxelFields (gas.gc_ionized_gas + gas.gc_molecular_clouds)."""
    from . import gas
    print(f"\n== GC observer with gas fields (resolution={resolution}) ==", flush=True)
    t = time.time()
    flds = gas.gc_ionized_gas(grid, resolution=resolution, verbose=False) + gas.gc_molecular_clouds(grid, resolution=resolution, verbose=False)
    tb = time.time() - t
    model = galaxy.GalaxyModel(grid, fields=flds)
    obs = galaxy.gc_observer_pc()
    t = time.time()
    m = model.render_map(obs, nside, R_GAL).diffuse_map
    dt = time.time() - t
    s = stats(m, grid)
    av = model.extinction_av(obs, np.array([galaxy.sgr_a_star_pc(), galaxy.arches_pc()]))
    print(f"  {len(flds)} fields built in {tb:.1f} s; render_map nside={nside}: {dt:.1f} s")
    print(f"  V sb min/median/max = {s['sb_min']:.2f} / {s['sb_median']:.2f} / {s['sb_max']:.2f} mag/arcsec2; "
          f"luminance median/mean {s['cd_m2_median']:.3g} / {s['cd_m2_mean']:.3g} cd/m2")
    print(f"  A_V observer -> Sgr A*: {av[0]:.1f} mag, -> Arches centre: {av[1]:.2f} mag")
    sgra = galaxy.sgr_a_star_pc()
    d = R_GAL @ ((sgra - obs) / np.linalg.norm(sgra - obs))
    l_s = float(np.degrees(np.arctan2(d[1], d[0])) % 360)
    save_maps(os.path.join(out, f"gc_gas_{resolution}_sgra_centred"), m, grid, nside, f"GC observer + gas ({resolution})", "l'", rot=(l_s, 0.0))
    return {"render_s": dt, "build_s": tb, **s, "AV_sgra": float(av[0]), "AV_arches": float(av[1])}


def dgl_check(grid, nside, out):
    """Diffuse galactic light (dgl=True) relative to the dust-extincted starlight."""
    print("\n== diffuse galactic light (single scattering, albedo 0.6, g 0.6) ==", flush=True)
    t = time.time()
    m0, m1 = galaxy.GalaxyModel(grid), galaxy.GalaxyModel(grid, dgl=True)
    res = {"build_s": m1.info["dgl_build_s"]}
    print(f"  DGL grids built in {res['build_s']:.1f} s")
    for name, obs in (("earth", np.array(galaxy.EARTH_OBSERVER_PC)), ("gc", galaxy.gc_observer_pc())):
        a = m0.render_map(obs, nside, R_GAL).diffuse_map
        t = time.time()
        b = m1.render_map(obs, nside, R_GAL).diffuse_map
        dt = time.time() - t
        La, Lb = spectral.luminance(a, grid), spectral.luminance(b, grid)
        r = Lb / La - 1
        _, bb = hp.pix2ang(nside, np.arange(len(r)), lonlat=True)
        res[name] = {"render_s": dt, "dgl_over_isl_median": float(np.median(r)), "plane": float(np.median(r[np.abs(bb) < 2])),
                     "highlat": float(np.median(r[np.abs(bb) > 60])), "mean_L_ext": float(La.mean()), "mean_L_dgl": float(Lb.mean())}
        ref = "plane ~0.2-0.3, |b|>60 ~0.05-0.2 (Leinert+98; Brandt & Draine 2012)" if name == "earth" else "no reference"
        print(f"  {name}: render {dt:.1f} s; DGL/starlight median {np.median(r):.3f}, |b|<2 {res[name]['plane']:.3f}, "
              f"|b|>60 {res[name]['highlat']:.3f}; mean luminance {La.mean():.3g} -> {Lb.mean():.3g} cd/m2  [ref: {ref}]")
        rot = None
        if name == "gc":
            sgra = galaxy.sgr_a_star_pc()
            d = R_GAL @ ((sgra - obs) / np.linalg.norm(sgra - obs))
            rot = (float(np.degrees(np.arctan2(d[1], d[0])) % 360), 0.0)
        save_maps(os.path.join(out, f"{name}_galactic_dgl"), b, grid, nside, f"{name}, extincted + DGL", "l'" if name == "gc" else "l", rot=rot)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument("--gas", default="low", help="gas field resolution for the combined check, or 'none'")
    ap.add_argument("--earth-nside", type=int, default=256)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    grid = spectral.default_grid()
    t = time.time()
    model = galaxy.GalaxyModel(grid)
    print(model.describe())
    print(f"  init {time.time() - t:.1f} s")
    res = {"model": {k: v for k, v in model.info.items()}}
    report("total stellar mass", model.info["M_total"], "5-6e10 Msun (BHG16: 5 +- 1e10)", "Msun")
    report("total L_V", model.info["L_V_total"], "2-3e10 Lsun (Flynn+06)", "Lsun")
    report("A_V Sun -> Sgr A*", model.info["AV_sun_sgra"], "~30 mag (A_Ks 2.5-3)", "mag")
    report("local dust", model.info["dust_local_mag_per_kpc"], "0.7-1 mag/kpc", "mag/kpc")
    rho_sun = model.stellar_density(np.array([[frames.R0_PC, 0.0, 0.0]]))[0]
    report("stellar density at the Sun", rho_sun, "0.04-0.05 Msun/pc3", "Msun/pc3")
    obs = galaxy.gc_observer_pc()
    report("stellar density at the GC observer", model.stellar_density(obs[None])[0], "~1e2-1e3 (Launhardt+02 M(<30pc)~1e8)", "Msun/pc3")
    av_obs = model.optical_depth(obs, galaxy.sgr_a_star_pc()[None])[0]
    for name, tgt, ref in (("A_V to the NGP (z = 3 kpc)", [frames.R0_PC, 0.0, 3000.0], "~0.03-0.1 (Schlegel+98 E(B-V)~0.01-0.03)"),
                           ("A_V Sun -> 2 kpc at l=90 (plane)", [frames.R0_PC, 2000.0, 0.0], "~1.5 (0.7-1 mag/kpc)")):
        t_ = model.optical_depth(np.array(galaxy.EARTH_OBSERVER_PC), np.array([tgt]))[0]
        report(name, float(np.interp(550, grid.wl, t_) / np.interp(550, grid.wl, model.C)), ref, "mag")
    report("A_V GC observer -> Sgr A*", float(np.interp(550, grid.wl, av_obs) / np.interp(550, grid.wl, model.C)), "~1 (smooth CMZ only)", "mag")
    res["earth"] = earth(model, grid, a.earth_nside, a.out)
    res["gc"], obs, R_cel = gc(model, grid, a.nside, a.out)
    res["gc"]["diffuse_fn_1920x1080_s"] = timing_views(model, obs, R_cel, a.out)
    gc_views(model, obs, a.out)
    res["fields"] = synthetic_fields(grid)
    res["inner"] = inner_check(model, grid)
    res["review"] = review_checks(grid)
    res["dgl"] = dgl_check(grid, min(a.nside, 256), a.out)
    if a.gas != "none":
        res["gas"] = gas_check(grid, a.nside, a.gas, a.out)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"\nwrote {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
