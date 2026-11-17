"""Validation of nightsky.starfield: magnitude counts, integrated light and illuminance per
component, Teff distributions, all-sky dot plots (horizon frame, lat +40, H = 180) and frame checks.

    python3 -m nightsky.validate_starfield [--out DIR]

Reference values: Yale BSC has 9110 entries (Hoffleit & Jaschek 1991), ~8400 with V < 6.5;
22 stars brighter than V = 1.5 (V = 1.50 is Adhara); old solar-metallicity Kroupa-IMF
populations have M/L_V ~ 3-4 at 9-10 Gyr (Maraston 2005); red-clump M_V ~ +0.8, Teff ~ 4800 K
(Girardi 2016); Galactic Centre at RA 266.40 deg, Dec -28.94 deg (ICRS).
"""
import argparse
import os
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import frames, spectral  # noqa: E402
from . import starfield as sf  # noqa: E402
from . import stellar_pops as sp  # noqa: E402

OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "renders", "nightsky", "validation",
                                   "starfield"))
LAT, H = 40.0, 180.0


class MockGalaxy:
    """Stand-in for galaxy.GalaxyModel: density rising towards Sgr A*, uniform A_V/pc."""

    def __init__(self, grid, av_per_pc=0.02):
        self.grid, self.av = grid, av_per_pc

    def stellar_density(self, xyz):
        return 300.0 * (25.0 / np.maximum(np.linalg.norm(xyz, axis=1), 1.0))

    def optical_depth(self, obs, tgt):
        d = np.linalg.norm(np.asarray(tgt) - obs, axis=1)
        return (self.av * d / 1.0857)[:, None] * sf.ccm89(self.grid)[None]


def horizon_illuminance(irr, dirs_cel, grid, R):
    up = (dirs_cel @ R.T)[:, 2]
    e = (irr * np.clip(up, 0, None)[:, None]).sum(0)
    return float(spectral.to_xyz(e, grid)[1])


def allsky_plot(dirs_cel, v, teff, R, title, path, vmax=6.5):
    dh = dirs_cel @ R.T
    alt, az = frames.vec_to_altaz(dh)
    s = (alt > 0) & (v < vmax)
    r = 90 - alt[s]
    x, y = -r * np.sin(np.radians(az[s])), r * np.cos(np.radians(az[s]))
    size = np.clip(0.12 * (vmax + 1 - v[s]) ** 2, 0.02, 80)
    o = np.argsort(-v[s])
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="k")
    ax.set_facecolor("k")
    ax.scatter(x[o], y[o], s=size[o], c=np.log10(teff[s][o]), cmap="RdYlBu", vmin=3.45, vmax=4.4, lw=0)
    ax.add_patch(plt.Circle((0, 0), 90, fill=False, color="0.5"))
    for lab, (tx, ty) in dict(N=(0, 93), E=(-95, 0), S=(0, -97), W=(93, 0)).items():
        ax.text(tx, ty, lab, color="w", ha="center", va="center")
    ax.set_xlim(-100, 100), ax.set_ylim(-100, 100), ax.set_aspect("equal"), ax.axis("off")
    ax.set_title(title, color="w", fontsize=9)
    fig.savefig(path, dpi=130, facecolor="k")
    plt.close(fig)


def radec(v):
    return np.degrees(np.arctan2(v[1], v[0])) % 360, np.degrees(np.arcsin(np.clip(v[2], -1, 1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=OUT)
    p.add_argument("--real-galaxy", action="store_true", help="also run with nightsky.galaxy.GalaxyModel(grid)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    grid = spectral.default_grid()
    Rh = frames.celestial_to_horizon(H, LAT)
    rep = []
    say = lambda s="": (print(s), rep.append(s))

    # ---------------- spectral helper accuracy
    tt = np.geomspace(2500, 1e5, 300)
    vv = np.random.default_rng(0).uniform(-10, 10, 300)
    ex, fa = spectral.vmag_irradiance(vv, tt, grid), sf.vmag_irr_fast(vv, tt, grid)
    say(f"vmag_irr_fast vs spectral.vmag_irradiance: max rel err {np.max(np.abs(fa / ex - 1)):.2e}")
    say(f"v550 round trip (V=3, 4800 K): {sf.v550(spectral.vmag_irradiance([3.0], [4800.0], grid), grid)[0]:.4f}")
    say(f"CCM89 A_lambda/A_V at bins: {np.round(sf.ccm89(grid)[[0, 8, 17]], 3)} (wl {grid.wl[[0, 8, 17]]} nm)")
    import companion as cp
    d1000 = 1000 * sf.AU_M / sf.PC_M
    v_sun = sf.v550(sf.vmag_irr_fast(np.array([sp.mv_from_lum(1.0, 5772.0) + 5 * np.log10(d1000 / 10)]),
                                     np.array([5772.0]), grid), grid)[0]
    vega = sf.vmag_irr_fast(np.zeros(1), np.array([9602.0]), grid)[0]
    say(f"flux units: 1 Lsun, 5772 K at 1000 au through the starfield path V {v_sun:.3f} (companion.vmag() "
        f"{cp.vmag():.3f}); V=0 9602 K: max |fast/spectral.vmag_irradiance - 1| "
        f"{np.max(np.abs(vega / spectral.vmag_irradiance(0.0, 9602.0, grid) - 1)):.1e}, 550 nm bin "
        f"{vega[grid.bin_index(550.0)]:.4e} W/m2/nm (F_V0 3.63e-11)")

    # ---------------- populations
    say("\n== stellar populations (per Msun of present-day stellar mass)")
    K = sp.KINDS.index
    for age, feh in ((10.0, 0.0), (9.0, 0.15), (9.0, 0.3), (1.0, 0.15)):
        t = sp.old_population(age, feh)
        rc, rgb = t.kind == K("RC"), t.kind == K("RGB")
        agb = (t.kind == K("EAGB")) | (t.kind == K("TPAGB"))
        mv_rc = np.average(t.mv[rc], weights=t.w[rc])
        up = rgb & (t.mv < mv_rc)
        say(f"{age:g} Gyr [Fe/H]={feh:+.2f}: M/L_V={1 / t.light_v():.2f} (ref ~3-4 at 9-10 Gyr, Maraston 2005); "
            f"N(M_V<4)={t.number(4):.2e}, N(M_V<1)={t.number(1):.2e}; per 1e6 Msun: red clump {1e6 * t.w[rc].sum():.0f} "
            f"(fuel consumption B ~2.2e-11 /yr/Lsun x 1e8 yr / (M/L_bol ~2.3) ~ 950 at 10 Gyr), RGB {1e6 * t.w[rgb].sum():.0f}, "
            f"RGB above the clump {1e6 * t.w[up].sum():.0f} (R = N_RC / N_RGB,>RC {t.w[rc].sum() / t.w[up].sum():.2f}; "
            f"globular-cluster R ~1.2-1.5, Buzzoni et al. 1983), AGB/RC {t.w[agb].sum() / t.w[rc].sum():.2f}; "
            f"RC at M_V {mv_rc:.2f}, Teff {t.teff[rc][0]:.0f} K; V light from M_V<4: {t.light_v(4) / t.light_v():.2f}")
    tab = sf.local_population_table(sf.DEFAULTS["local_old_population"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    be = np.arange(-8, 16.01, 0.25)
    for i, k in enumerate(sp.KINDS):
        s = tab.kind == i
        if s.any() and tab.w[s].sum() > 0:
            h, _ = np.histogram(tab.mv[s], be, weights=tab.w[s])
            if h.max() > 0:
                ax.step(be[:-1], h / 0.25, where="post", label=k, color=plt.cm.tab20(i))
    ax.set_yscale("log"), ax.set_ylim(1e-8, 10), ax.set_xlabel("M_V"), ax.set_ylabel("dN/dM_V per Msun")
    ax.legend(fontsize=7, ncol=2), ax.set_title("local field population LF (default cfg mix)")
    fig.tight_layout(), fig.savefig(os.path.join(a.out, "local_luminosity_function.png"), dpi=120), plt.close(fig)

    # ---------------- GC mode
    say("\n== GC mode (defaults)")
    t0 = time.time()
    toa, info = sf.build_starfield(dict(mode="gc", return_star_table=True), grid)
    info.update(info["stars"])
    say(f"build_starfield(gc): {time.time() - t0:.1f} s, {len(toa.points)} points, "
        f"v_point_lim used {info['v_point_lim_used']:.2f}, inner_factor {info['inner_factor']:.3f}")
    say(f"Arches centre (celestial) = {np.round(info['arches_cel'], 6)}; angle from pole "
        f"{np.degrees(np.arccos(np.clip(info['arches_cel'][2], -1, 1))):.2e} deg")
    ar = info["component"] == "arches"
    w = 10 ** (-0.4 * info["vmag"][ar])
    pts_dirs = np.zeros((len(info["vmag"]), 3))
    pts_dirs[info["is_point"]] = toa.points.dirs
    cen = (pts_dirs[ar & info["is_point"]] * w[info["is_point"][ar], None]).sum(0)
    say(f"Arches V-flux-weighted centroid {np.degrees(np.arccos(cen[2] / np.linalg.norm(cen))):.2f} deg from the pole "
        f"(Plummer a=0.25 pc at 3 pc = {np.degrees(0.25 / 3):.1f} deg core)")
    ra, dec = radec(info["sgr_a_cel"])
    obs, arc = info["observer_pc"], frames.lb_to_galactocentric(*sf.ARCHES_LB, frames.R0_PC)
    sg = frames.lb_to_galactocentric(*sf.SGRA_LB, frames.R0_PC)
    u, v_ = (arc - obs) / np.linalg.norm(arc - obs), (sg - obs) / np.linalg.norm(sg - obs)
    say(f"Sgr A* celestial RA {ra:.3f} deg (frames: equinox = Sgr A* projection -> 0), Dec {dec:.3f} deg "
        f"(independent: 90 - angle(Arches, Sgr A*) = {90 - np.degrees(np.arccos(u @ v_)):.3f}); "
        f"distance {np.linalg.norm(sg - obs):.1f} pc")
    ra, dec = radec(info["quintuplet_cel"])
    say(f"Quintuplet RA {ra:.1f} Dec {dec:.1f} deg; observer {np.round(obs, 2)} pc")
    lo = info["local"]
    say(f"local population: {lo['mass_msun']:.3g} Msun within {info['r_res_pc']} pc "
        f"({lo['density_msun_pc3']:.0f} Msun/pc^3), M/L_V {lo['mass_to_light_v']:.2f}, {lo['n_dwarfs']} dwarfs "
        f"(M_V >= mv_lim, drawn to V = v_draw_lim), resolved share of the r_res light {lo['resolved_light_frac']:.3f}, "
        f"kinds {lo['kind_counts']}")
    tab_u = sf.local_population_table(sf.DEFAULTS["local_old_population"])
    r_min = sf.DEFAULTS["min_dist_au"] * sf.AU_M / sf.PC_M
    rb = np.where(tab_u.mv < 4, 25.0, np.clip(10 ** (0.2 * (6.5 - tab_u.mv) + 1), r_min, 25.0))
    n_an = (tab_u.w * 300.0 * 4 / 3 * np.pi * (rb ** 3 - r_min ** 3)).sum()
    lv_u = tab_u.w * sp.lum_v_lsun(tab_u.mv)
    say(f"  analytic (uniform 300 Msun/pc^3): N drawn {n_an:.4g} vs {info['components']['local']['n_stars']}; resolved "
        f"light share {(lv_u * (rb - r_min) / (25.0 - r_min)).sum() / lv_u.sum():.3f}")
    vb_seed = []
    for s in (2, 3, 4):
        _, i_s = sf.build_starfield(dict(mode="gc", seed=s, arches=False, quintuplet=False, nsc_young=False,
                                         field_massive=False, companion=False), grid)
        vb_seed.append(round(i_s["components"]["local"]["V_brightest"], 2))
    say(f"  brightest local star V for seeds 2, 3, 4: {vb_seed}")
    say(f"{'component':15s} {'N':>7s} {'pts':>7s} {'V_tot':>7s} {'V_min':>6s} {'E_h(lux)':>9s} "
        f"N(<-4) N(<-2) N(<0) N(<1) N(<2) N(<4) N(<6.5)")
    allpts = toa.points
    vp = info["vmag"][info["is_point"]]
    comp_p = info["component"][info["is_point"]]
    for name, c in info["components"].items():
        s = comp_p == name
        eh = horizon_illuminance(allpts.irr[s], allpts.dirs[s], grid, Rh)
        k = c["counts_lt"]
        say(f"{name:15s} {c['n_stars']:7d} {c['n_points']:7d} {c['V_total']:7.2f} {c['V_brightest']:6.2f} {eh:9.2e} "
            f"{k['-4']:6d} {k['-2']:6d} {k['0']:5d} {k['1']:6d} {k['2']:6d} {k['4']:6d} {k['6.5']:7d}")
    ncel = Rh.T @ [0, 0, 1.0]
    e_map = spectral.to_xyz(TOAmap_irr(toa, ncel), grid)[1]
    e_all = spectral.to_xyz(toa.total_irradiance_on(ncel), grid)[1]
    say(f"binned faint resolved stars (diffuse_map): V_total {info['diffuse_V_total']:.2f}, E_h {e_map:.2e} lux; "
        f"TOASky.total_irradiance_on(up) = {e_all:.3e} lux (full Moon ~0.1-0.3 lux)")
    vtot_pts = sf.v550(toa.points.irr.sum(0)[None], grid)[0]
    import healpy as hp
    vtot_map = sf.v550((toa.diffuse_map * 4 * np.pi / hp.nside2npix(toa.nside)).sum(0)[None], grid)[0]
    vtot_all = -2.5 * np.log10(sum(10 ** (-0.4 * c["V_total"]) for c in info["components"].values()))
    say(f"light conservation: points V {vtot_pts:.3f} + map V {vtot_map:.3f} -> "
        f"{-2.5 * np.log10(10 ** (-0.4 * vtot_pts) + 10 ** (-0.4 * vtot_map)):.4f}; sum of components {vtot_all:.4f}")
    loc = info["component"] == "local"
    say(f"local stars brighter than V=1: {int((info['vmag'][loc] < 1).sum())} "
        f"(task expectation: thousands of first-magnitude orange giants); of these Teff<5200 K: "
        f"{int(((info['vmag'][loc] < 1) & (info['teff'][loc] < 5200)).sum())}")

    fig, ax = plt.subplots(figsize=(7, 5))
    vb = np.arange(-14, 9.01, 0.1)
    cum = lambda v: np.searchsorted(np.sort(v), vb)
    for name in info["components"]:
        ax.step(vb, cum(info["vmag"][info["component"] == name]), where="post", label=name)
    ax.step(vb, cum(info["vmag"]), "k", where="post", label="all")
    ax.axvline(info["v_point_lim_used"], color="0.5", ls=":")
    ax.set_yscale("log"), ax.set_xlabel("V (apparent, after extinction)"), ax.set_ylabel("N(<V)")
    ax.legend(fontsize=8), ax.set_title("GC sky: cumulative star counts (dotted: point / map split)")
    fig.tight_layout(), fig.savefig(os.path.join(a.out, "gc_cumulative_counts.png"), dpi=120), plt.close(fig)
    allsky_plot(toa.points.dirs, vp, info["teff"][info["is_point"]], Rh,
                "GC sky, lat +40, H=180 (point stars V<4; colour = Teff)", os.path.join(a.out, "allsky_gc_lat40_H180.png"),
                vmax=4.0)

    # ---------------- GC with mock galaxy + legacy box
    t0 = time.time()
    toa2, info2 = sf.build_starfield(dict(mode="gc", local_zams_box=True), grid, galaxy=MockGalaxy(grid))
    say(f"\nGC + MockGalaxy + local_zams_box: {time.time() - t0:.1f} s, {len(toa2.points)} points, local mass "
        f"{info2['local']['mass_msun']:.3g} Msun (mean density {info2['local']['density_msun_pc3']:.0f}), "
        f"v_point_lim used {info2['v_point_lim_used']:.2f}")
    for name, c in info2["components"].items():
        say(f"   {name:15s} N {c['n_stars']:7d} pts {c['n_points']:7d} V_tot {c['V_total']:7.2f} V_min {c['V_brightest']:6.2f}")
    off = dict(arches=False, quintuplet=False, nsc_young=False, field_massive=False, local_old_population=False,
               companion=False)
    for lab, c in (("all components off", off), ("field_massive with n_wr = n_osg = 0",
                                                  dict(field_massive=dict(n_wr=0, n_osg=0), local_old_population=False)),
                   ("r_res_pc = 0", dict(r_res_pc=0.0)), ("v_draw_lim = None (no dwarfs)", dict(v_draw_lim=None)),
                   ("8-bin 400-700 nm grid", dict(local_old_population=dict(density_msun_pc3=30.0)))):
        g = spectral.default_grid(8, 400, 700) if lab.startswith("8-bin") else grid
        te, ii = sf.build_starfield(c, g)
        say(f"edge case {lab}: {len(te.points)} points, irr {te.points.irr.shape}, map {te.diffuse_map.shape}, "
            f"inner_factor {ii['inner_factor']:.3f}")

    if a.real_galaxy:
        from . import galaxy as gx
        t0 = time.time()
        gm = gx.GalaxyModel(grid)
        t1 = time.time()
        toa3, info3 = sf.build_starfield(dict(mode="gc"), grid, galaxy=gm)
        l3 = info3["local"]
        say(f"\nGC + galaxy.GalaxyModel(grid): model {t1 - t0:.1f} s, starfield {time.time() - t1:.1f} s, "
            f"{len(toa3.points)} points; local {l3['mass_msun']:.3g} Msun (mean density "
            f"{l3['density_msun_pc3']:.0f} Msun/pc^3), {l3['n_dwarfs']} dwarfs, inner_factor {info3['inner_factor']:.3f} "
            f"(unclipped {l3['inner_factor_unclipped']:.3f}), M/L_V population {l3['mass_to_light_v']:.3f} vs galaxy "
            f"{l3['mass_to_light_v_galaxy']:.3f}, v_point_lim used {info3['v_point_lim_used']:.2f}; rejection sampling "
            f"{l3['rejection_iterations']} iterations, {l3['n_unconverged']} unconverged, "
            f"{l3['n_envelope_violations']} envelope violations")
        for name, c in info3["components"].items():
            say(f"   {name:15s} N {c['n_stars']:7d} pts {c['n_points']:7d} V_tot {c['V_total']:7.2f} "
                f"V_min {c['V_brightest']:6.2f} N(<1) {c['counts_lt']['1']}")
        tau = gm.optical_depth(info3["observer_pc"], np.vstack([frames.lb_to_galactocentric(*sf.ARCHES_LB, frames.R0_PC),
                                                                 frames.lb_to_galactocentric(*sf.QUINTUPLET_LB, frames.R0_PC),
                                                                 frames.lb_to_galactocentric(*sf.SGRA_LB, frames.R0_PC)]))
        say(f"   galaxy A_V (550 nm) to Arches / Quintuplet / Sgr A*: {np.round(1.0857 * tau[:, grid.bin_index(550)], 2)}")
        import healpy as hp
        obs3, R3 = info3["observer_pc"], info3["R_gc_to_cel"]
        m0 = gm.render_map(obs3, 32, R3, r_res_pc=info3["r_res_pc"], inner_factor=0.0).diffuse_map
        m1 = gm.render_map(obs3, 32, R3, r_res_pc=info3["r_res_pc"], inner_factor=1.0).diffuse_map
        v_in = sf.v550(((m1 - m0).sum(0) * 4 * np.pi / hp.nside2npix(32))[None], grid)[0]
        say(f"   light budget within r_res: galaxy V {v_in:.2f}; galaxy x (1 - inner_factor) V "
            f"{v_in - 2.5 * np.log10(1 - info3['inner_factor']):.2f} vs starfield resolved local V "
            f"{info3['components']['local']['V_total']:.2f} (should agree up to sampling noise)")

    # ---------------- Earth mode
    say("\n== Earth mode")
    t0 = time.time()
    toe, ie = sf.build_starfield(dict(mode="earth", return_star_table=True), grid)
    ie.update(ie["stars"])
    v = ie["vmag"]
    say(f"build_starfield(earth): {time.time() - t0:.1f} s, catalogue: {ie['catalog']}")
    say(f"N(V<6.5) = {(v < 6.5).sum()} (BSC: 9110 entries in total, ~8400 with V<6.5); N(V<1.5) = {(v < 1.5).sum()} "
        f"(22 known; task note '~50' matches N(V<2.0) = {(v < 2.0).sum()}); brightest V {v.min():.2f}")
    say(f"integrated V of V<6.5 stars: {ie['components']['bsc']['V_total']:.2f}; horizontal illuminance at lat 40, "
        f"H=180: {horizon_illuminance(toe.points.irr, toe.points.dirs, grid, Rh):.2e} lux "
        f"(all-sky starlight incl. faint stars ~2-3e-4 lux, Roach & Gordon 1973)")
    Re = ie["R_gc_to_cel"]
    ra, dec = radec(Re @ [-1.0, 0, 0])
    say(f"Earth R_gc_to_cel: direction to Sgr A* (-x_gc) -> RA {(ra + 180) % 360:.2f}, Dec {dec:.2f} (ICRS, ref "
        f"266.40, -28.94); celestial RA {ra:.2f} (= ICRS RA - sun_ra 180)")
    i_sir = int(np.argmin(v))
    ra, dec = radec(toe.points.dirs[i_sir])
    say(f"brightest star {toe.points.names[i_sir]} at celestial RA {ra:.2f} (ICRS {(ra + 180) % 360:.2f}, ref 101.29), "
        f"Dec {dec:.2f} (ref -16.72)")
    nm = np.array(toe.points.names)
    for lab, key, ref in (("Polaris", "Alp UMi", "Dec 89.26"), ("Betelgeuse", "Alp Ori", "RA 5.92 h"),
                          ("Rigel", "Bet Ori", "RA 5.24 h"), ("Alnitak", "Zet Ori", "RA 5.68 h")):
        j = np.flatnonzero(np.char.find(nm.astype(str), key) >= 0)
        j = j[np.argmin(v[j])]
        ra, dec = radec(toe.points.dirs[j])
        alt, az = frames.vec_to_altaz((Rh @ toe.points.dirs[j])[None])
        say(f"{lab} ({nm[j]}): ICRS RA {((ra + 180) % 360) / 15:.2f} h, Dec {dec:.2f} (ref {ref}); lat 40, H=180: "
            f"alt {alt[0]:.1f} az {az[0]:.1f}")
    from astropy.coordinates import get_sun
    from astropy.time import Time
    sun = get_sun(Time("2026-09-23T00:05:00", scale="utc"))
    ra_s, dec_s = sun.ra.deg, sun.dec.deg
    dsun = np.array([np.cos(np.radians(dec_s)) * np.cos(np.radians(ra_s - 180)),
                     np.cos(np.radians(dec_s)) * np.sin(np.radians(ra_s - 180)), np.sin(np.radians(dec_s))])
    say(f"Sun at the 2026 Sept. equinox (astropy GCRS): RA {ra_s:.2f}, Dec {dec_s:.2f} -> celestial "
        f"{np.round(dsun, 4)} (convention: Sun on +x); Sun altitude at H=180: {frames.vec_to_altaz((Rh @ dsun)[None])[0][0]:.1f}")
    t_fb, src = sf.load_bsc(99.0, fetch=False)
    say(f"offline fallback check (vmax=99, no fetch): {len(t_fb)} rows from {src}")
    import shutil
    import tempfile
    import requests
    from astroquery.vizier import VizierClass
    cat_dir, q0 = sf.CATALOG_DIR, VizierClass.query_constraints
    with tempfile.TemporaryDirectory(dir=a.out) as tmp:
        shutil.copy(os.path.join(cat_dir, "bsc5_vlt2.05.csv"), tmp)
        sf.CATALOG_DIR = tmp

        def fail(*args, **kw):
            raise requests.exceptions.ConnectionError("simulated network failure")
        VizierClass.query_constraints = fail
        try:
            tf, i_f = sf.build_starfield(dict(mode="earth"), grid)
        finally:
            sf.CATALOG_DIR, VizierClass.query_constraints = cat_dir, q0
    say(f"simulated VizieR failure with an empty cache: {len(tf.points)} points, catalogue {i_f['catalog']}, "
        f"notes {[n for n in i_f['notes'] if 'fallback' in n]}")
    if a.real_galaxy:
        toe3, ie3 = sf.build_starfield(dict(mode="earth"), grid, galaxy=gm)
        om = 4 * np.pi / hp.nside2npix(32)
        m = gm.render_map(ie3["observer_pc"], 32, ie3["R_gc_to_cel"]).diffuse_map
        m0 = gm.render_map(ie3["observer_pc"], 32, ie3["R_gc_to_cel"], r_res_pc=ie3["r_res_pc"],
                           inner_factor=ie3["inner_factor"]).diffuse_map
        say(f"Earth + galaxy: r_res_pc {ie3['r_res_pc']:.0f}, inner_factor {ie3['inner_factor']:.2f}; galaxy all-sky V "
            f"{sf.v550((m.sum(0) * om)[None], grid)[0]:.2f}, light removed inside r_res V "
            f"{sf.v550(((m - m0).sum(0) * om)[None], grid)[0]:.2f} vs BSC V<6.5 total "
            f"{ie3['components']['bsc']['V_total']:.2f}")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    vb = np.arange(-2, 6.6, 0.05)
    ax.step(vb, np.searchsorted(np.sort(v), vb), where="post", label="BSC (model input)")
    ax.plot([1.5, 6.5], [22, 8400], "ro", label="reference points")
    ax.set_yscale("log"), ax.set_xlabel("V"), ax.set_ylabel("N(<V)"), ax.legend()
    ax.set_title("Earth sky: cumulative counts")
    fig.tight_layout(), fig.savefig(os.path.join(a.out, "earth_cumulative_counts.png"), dpi=120), plt.close(fig)
    allsky_plot(toe.points.dirs, v, ie["teff"], Rh, "Earth sky, lat +40, H=180 (Sept. equinox midnight)",
                os.path.join(a.out, "allsky_earth_lat40_H180.png"))

    # ---------------- Teff distributions
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tb = np.geomspace(2500, 1.2e5, 60)
    for name in ("local", "arches", "quintuplet", "nsc_young"):
        s = info["component"] == name
        ax.hist(info["teff"][s], tb, histtype="step", label=f"GC {name}", weights=None)
    ax.hist(ie["teff"], tb, histtype="step", color="k", label="Earth BSC V<6.5")
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlabel("Teff (K)"), ax.set_ylabel("N"), ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(os.path.join(a.out, "teff_distribution.png"), dpi=120), plt.close(fig)
    for name, s in (("GC local V<1", (info["component"] == "local") & (info["vmag"] < 1)),
                    ("Earth V<6.5", np.ones(len(ie["teff"]), bool))):
        tsel = (info["teff"] if name.startswith("GC") else ie["teff"])[s]
        say(f"Teff quantiles {name}: 10/50/90% = {np.percentile(tsel, [10, 50, 90]).round(0)}")
    with open(os.path.join(a.out, "report.txt"), "w") as f:
        f.write("\n".join(rep) + "\n")
    print("wrote", a.out)


def TOAmap_irr(toa, normal):
    import healpy as hp
    d = frames.healpix_dirs(toa.nside)
    cos = np.clip(d @ normal, 0, None)
    return (toa.diffuse_map * cos[:, None]).sum(0) * 4 * np.pi / hp.nside2npix(toa.nside)


if __name__ == "__main__":
    main()
