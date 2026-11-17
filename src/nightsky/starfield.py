"""Resolved stars as point sources on the observer planet's celestial sphere.

build_starfield(cfg, grid, galaxy=None) -> (TOASky, info). Points are in the CELESTIAL frame
(frames.py); irradiance in W m^-2 nm^-1 (bin averages on `grid`) at the top of the planet's
atmosphere, after interstellar extinction. Resolved stars fainter than cfg v_point_lim (6.5),
or fainter than the max_points-th brightest star (5e6) when that is brighter, are binned into
TOASky.diffuse_map (HEALPix RING, cfg nside, radiance W m^-2 sr^-1 nm^-1): their light is kept
while the point list stays bounded. info["v_point_lim_used"] reports the split.
cfg: see DEFAULTS; a component key may be a bool (enable/disable) or a dict of overrides.

GC mode (solar system 3 pc from the Arches; positions in galactocentric pc, frames.py):
* Arches: cluster.Cluster physics (IMF 1.5e4 Msun, 1-120 Msun, alpha 1.85, Plummer
  a = 0.25 pc, rmax 1.5 pc, seed 1, ZAMS L/Teff), placed around the Arches in 3D. ZAMS is kept
  although the cluster age is ~2.5 Myr (Clark et al. 2019); the most massive stars would be
  WNh stars of similar luminosity.
* Quintuplet: 1e4 Msun, 4.8 +- 1 Myr (Clark et al. 2018), stellar_pops.young_cluster recipe
  plus the Pistol star (1.6e6 Lsun, 11800 K) and qF362 (~1.3e6 Lsun, ~10000 K) after
  Najarro et al. (2009), and a few red supergiants. Placed at R0 (line-of-sight offset unknown).
* NSC young stars: 1.5e4 Msun, 6 +- 1 Myr, IMF alpha 1.7 (Lu et al. 2013), dN/dr = const
  between 0.04 and 0.5 pc of Sgr A* (Paumard et al. 2006), isotropic; IRS 7 (RSG, ~1.5e5 Lsun,
  ~3600 K; approximate, after Paumard et al. 2014).
* Isolated massive CMZ stars: WR / O supergiants scattered over |l| < 0.8 deg (Mauerhan et
  al. 2010), log L 5.3-6.0.
* Local field population within r_res_pc of the observer: stellar_pops luminosity functions
  (94.5% old 9 Gyr split between [Fe/H] = 0 and +0.3, 5% at 1 Gyr, 0.5% formed at a constant
  rate over the last 100 Myr; NSD star-formation history after Nogueras-Lara et al. 2020; the
  young fraction corresponds to ~1.5e-8 Msun/yr/pc^3, about half the CMZ mean), density from
  galaxy.stellar_density or cfg (300 Msun/pc^3). Stars with M_V < mv_lim (4) are drawn out to
  r_res; fainter dwarfs out to the distance where they reach V = v_draw_lim (6.5, extinction
  neglected). With a galaxy, positions follow galaxy.stellar_density by rejection sampling
  (envelope from probe maxima x 1.5-2; info['local'] counts envelope violations, so peaks
  above the envelope, e.g. the NSC side of the sphere, are slightly under-sampled).
  info['inner_factor'] = 1 - (resolved share of the population's irradiance from
  inside r_res) x (galaxy M/L_V / population M/L_V); galaxy.render_map keeps that fraction of its
  stellar emission inside r_res as diffuse light. The galaxy M/L_V comes from
  galaxy.luminosity_density_v when present (a galaxy.py helper outside INTERFACES.md).
* Optional: cluster.LocalStars box (legacy), companion.py companion (its frame, x = equinox
  and z = celestial pole, equals ours; obliquity 23.44 deg assumed for the planet).
Extinction: galaxy.optical_depth if a galaxy is given; else A_V = local_av_per_pc * d with the
CCM89 curve (Cardelli, Clayton & Mathis 1989, R_V = 3.1).

EARTH mode: Yale Bright Star Catalogue (VizieR V/50) to V < 6.5, cached in
data/star_catalogs/, offline fallback bsc5_vlt2.05.csv. RA is measured from cfg sun_ra_deg
(default 180 deg, the September equinox), so the Sun (RA = sun_ra, Dec = 0) lies on celestial
+x. J2000 positions, no precession. With a galaxy (earth_subtract_bsc), info r_res_pc is the
radius whose galaxy stellar light equals the catalogue's integrated light and inner_factor = 0,
so the catalogue stars replace that light in galaxy.render_map instead of adding to it.

Spectra: blackbody shapes at Teff scaled to the star's V (spectral.vmag_irradiance); V from
L_bol with the Flower (1996) bolometric corrections (stellar_pops.bc_v); Earth stars use the
catalogue V and Teff(B-V) (Ballesteros 2012). Legacy components (LocalStars, companion) use
bolometric blackbodies as in the earlier pipeline.
"""
import copy
import os

import numpy as np

import cluster as cl
from . import frames, spectral
from . import stellar_pops as sp
from .sky import PointSources, TOASky

PC_M, AU_M = cl.PC, cl.AU
CATALOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "star_catalogs"))
ARCHES_LB = (0.1217, 0.0188)
QUINTUPLET_LB = (0.1604, -0.0591)
SGRA_LB = (359.9442, -0.0462)
F_V0 = 3.63e-11

DEFAULTS = dict(
    mode="gc", seed=1, r_res_pc=25.0, observer_offset_dir=(1.0, 0.0, 0.0), offset_pc=3.0, roll_deg=0.0,
    mv_lim=4.0, v_draw_lim=6.5, v_point_lim=6.5, nside=128, local_av_per_pc=0.02, rv=3.1, min_dist_au=2000.0,
    local_density_msun_pc3=300.0, sun_ra_deg=180.0, max_points=5_000_000, return_star_table=False,
    earth_subtract_bsc=True, max_local_stars=1e7,
    arches=dict(enabled=True, mass=1.5e4, m_min=1.0, m_max=120.0, alpha=1.85, a_pc=0.25, rmax_pc=1.5, seed=1,
                extra_av=0.0, zams=False, age_myr=2.5),  # evolved: top members ~WNh, log L 6.5, Teff 34 kK
    quintuplet=dict(enabled=True, mass=1.0e4, m_min=1.0, m_max=120.0, alpha=1.85, age_myr=4.8, age_spread_myr=1.0,
                    post_ms_frac=0.2, a_pc=0.6, rmax_pc=3.0, seed=2, named=True, n_rsg=3, extra_av=0.0,
                    los_offset_pc=0.0),
    nsc_young=dict(enabled=True, mass=1.5e4, m_min=1.0, m_max=120.0, alpha=1.7, age_myr=6.0, age_spread_myr=1.0,
                   post_ms_frac=0.2, r_in_pc=0.04, r_out_pc=0.5, seed=3, irs7=True, extra_av=0.0),
    field_massive=dict(enabled=True, n_wr=30, n_osg=40, l_range=(-0.8, 0.8), b_range=(-0.15, 0.15),
                       los_sigma_pc=60.0, seed=5, extra_av=0.0),
    local_old_population=dict(enabled=True, density_msun_pc3=None, seed=4,
                              components=[dict(kind="old", age_gyr=9.0, feh=0.0, frac=0.47),
                                          dict(kind="old", age_gyr=9.0, feh=0.3, frac=0.475),
                                          dict(kind="old", age_gyr=1.0, feh=0.15, frac=0.05),
                                          dict(kind="young", age_max_myr=100.0, feh=0.1, frac=0.005)]),
    local_zams_box=dict(enabled=False, mass=1000.0, m_min=0.3, m_max=4.0, box_pc=1.0, min_dist_au=50.0, seed=7),
    companion=dict(enabled=True, sep_au=1000.0, inclination_deg=25.0, node_deg=0.0, phase_deg=90.0),
    bsc=dict(enabled=True, vmax=6.5, fetch=True),
    planets=[],  # Earth mode: [dict(name, ecl_lon_deg, ecl_lat_deg=0, vmag, bv=0.8)], longitudes from the March equinox
)
EARTH_OBSERVER_PC = (frames.R0_PC, 0.0, 20.8)
MAG_EDGES = (-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 6.5, 7, 8, 9)


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, bool):
            out[k] = {**out[k], "enabled": v}
        elif isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config(cfg):
    """cfg merged with DEFAULTS: what build_starfield actually uses (compose hashes this, so a
    change of a default here invalidates cached starfields)."""
    return _merge(DEFAULTS, cfg)


def ccm89(grid, rv=3.1):
    """(nbins,) A_lambda / A_V, CCM89 optical/NIR curve, bin-averaged."""
    def f(wl):
        x = 1e3 / wl
        y = x - 1.82
        a = 1 + 0.17699 * y - 0.50447 * y ** 2 - 0.02427 * y ** 3 + 0.72085 * y ** 4 + 0.01979 * y ** 5 \
            - 0.77530 * y ** 6 + 0.32999 * y ** 7
        b = 1.41338 * y + 2.28305 * y ** 2 + 1.07233 * y ** 3 - 5.38434 * y ** 4 - 0.62251 * y ** 5 \
            + 5.30260 * y ** 6 - 2.09002 * y ** 7
        return a + b / rv
    return grid.bin_average(f)


def _e550(irr, grid):
    """Bin-average spectrum (..., nbins) linearly interpolated to 550 nm."""
    wl, irr = grid.wl, np.asarray(irr, float)
    if len(wl) == 1:
        return irr[..., 0]
    i = int(np.clip(np.searchsorted(wl, 550.0) - 1, 0, len(wl) - 2))
    t = (550.0 - wl[i]) / (wl[i + 1] - wl[i])
    return (1 - t) * irr[..., i] + t * irr[..., i + 1]


def v550(irr, grid):
    """V magnitude (550 nm, Vega) of (N, nbins) irradiance, vectorised."""
    return -2.5 * np.log10(np.maximum(_e550(irr, grid), 1e-300) / F_V0)


def earth_r_res(galaxy, observer, e550_target, grid, rng, r_lo=1.0, r_hi=3000.0, n=100000):
    """Radius (pc) inside which the galaxy's stellar emission gives the 4 pi-integrated 550 nm
    irradiance e550_target (W m^-2 nm^-1), extinction neglected: E(<r) = 4 pi int_0^r <j> ds,
    estimated with log-uniform radii. Returns (r_pc, V of the galaxy light inside r)."""
    r = np.sort(r_lo * (r_hi / r_lo) ** rng.random(n))
    j = _e550(np.asarray(galaxy.emissivity(np.asarray(observer, float) + _iso(n, rng) * r[:, None]), float), grid)
    cum = 4 * np.pi * np.log(r_hi / r_lo) * np.cumsum(j * r) / n
    r_eq = float(np.interp(e550_target, cum, r)) if e550_target < cum[-1] else r_hi
    return r_eq, float(-2.5 * np.log10(np.interp(r_eq, r, cum) / F_V0))


def vmag_irr_fast(v, teff, grid, ntab=1500):
    """vmag_irradiance via a log-Teff table of blackbody shapes (linear interpolation, < 1e-4 error)."""
    lt = np.linspace(np.log10(1500.0), np.log10(3e5), ntab)
    tab = spectral.vmag_irradiance(np.zeros(ntab), 10 ** lt, grid)
    x = np.clip((np.log10(np.asarray(teff, float)) - lt[0]) / (lt[1] - lt[0]), 0, ntab - 1.000001)
    i = x.astype(int)
    f = (x - i)[:, None]
    return ((1 - f) * tab[i] + f * tab[i + 1]) * 10 ** (-0.4 * np.asarray(v, float))[:, None]


def _iso(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def _comp(name, pos=None, mv=None, teff=None, labels=None, irr=None, dirs_cel=None, dist_pc=None, extra_av=0.0,
          kind=None):
    return dict(name=name, pos=pos, mv=mv, teff=teff, labels=labels, irr=irr, dirs_cel=dirs_cel, dist_pc=dist_pc,
                extra_av=extra_av, kind=kind)


def _labels(prefix, kind):
    return [f"{prefix}:{sp.KINDS[k]}" for k in kind]


# ---------------------------------------------------------------- GC components
def arches(c, center):
    rng = np.random.default_rng(c["seed"])
    mass = cl.sample_imf(c["mass"], c["m_min"], c["m_max"], c["alpha"], rng)
    lum, teff = cl.lum_teff(mass)
    pos = center + cl.plummer_positions(len(mass), c["a_pc"], c["rmax_pc"], rng)
    lum, kind = lum / cl.LSUN, np.zeros(len(mass), int)
    if not c["zams"]:
        lum, teff, kind, alive = sp.evolve(mass, c["age_myr"], post_ms_frac=0.2)
        lum, teff, kind, pos = lum[alive], teff[alive], kind[alive], pos[alive]
    return _comp("arches", pos, sp.mv_from_lum(lum, teff), teff, _labels("arches", kind), extra_av=c["extra_av"],
                 kind=kind)


def quintuplet(c, center):
    rng = np.random.default_rng(c["seed"])
    s = sp.young_cluster(c["mass"], c["age_myr"], rng, c["alpha"], c["m_min"], c["m_max"], c["age_spread_myr"],
                         post_ms_frac=c["post_ms_frac"])
    lum, teff, kind = s["lum"], s["teff"], s["kind"]
    labels = _labels("quintuplet", kind)
    pos = center + cl.plummer_positions(len(lum), c["a_pc"], c["rmax_pc"], rng)
    if c["named"]:
        K = sp.KINDS.index
        n_rsg = max(0, c["n_rsg"] - int((kind == K("RSG")).sum()))
        extra_l = [1.6e6, 1.3e6] + [1.5e5] * n_rsg
        extra_t = [11800.0, 10000.0] + [3600.0] * n_rsg
        extra_k = [K("LBV"), K("LBV")] + [K("RSG")] * n_rsg
        labels += ["Pistol star", "qF362"] + ["quintuplet:RSG"] * n_rsg
        off = rng.normal(0, 0.4, size=(len(extra_l), 3))
        lum, teff, kind = (np.concatenate([a, b]) for a, b in ((lum, extra_l), (teff, extra_t), (kind, extra_k)))
        pos = np.vstack([pos, center + off])
    return _comp("quintuplet", pos, sp.mv_from_lum(lum, teff), teff, labels, extra_av=c["extra_av"], kind=kind)


def nsc_young(c, sgra):
    rng = np.random.default_rng(c["seed"])
    s = sp.young_cluster(c["mass"], c["age_myr"], rng, c["alpha"], c["m_min"], c["m_max"], c["age_spread_myr"],
                         post_ms_frac=c["post_ms_frac"])
    lum, teff, kind = s["lum"], s["teff"], s["kind"]
    r = c["r_in_pc"] + rng.random(len(lum)) * (c["r_out_pc"] - c["r_in_pc"])
    pos = sgra + _iso(len(lum), rng) * r[:, None]
    labels = _labels("nsc", kind)
    if c["irs7"]:
        lum, teff = np.append(lum, 1.5e5), np.append(teff, 3600.0)
        kind = np.append(kind, sp.KINDS.index("RSG"))
        pos = np.vstack([pos, sgra + [0.0, 0.0, 0.22]])
        labels.append("IRS 7")
    return _comp("nsc_young", pos, sp.mv_from_lum(lum, teff), teff, labels, extra_av=c["extra_av"], kind=kind)


def field_massive(c, r0):
    rng = np.random.default_rng(c["seed"])
    n = c["n_wr"] + c["n_osg"]
    l = rng.uniform(*c["l_range"], n)
    b = rng.uniform(*c["b_range"], n)
    d = r0 + rng.normal(0, c["los_sigma_pc"], n)
    pos = frames.lb_to_galactocentric(l, b, d, r0)
    lum = 10 ** rng.uniform(5.3, 6.0, n)
    K = sp.KINDS.index
    wr = np.arange(n) < c["n_wr"]
    wc = wr & (rng.random(n) < 0.4)
    kind = np.where(wc, K("WC"), np.where(wr, K("WN"), K("BSG")))
    teff = np.where(wc, 70000.0, np.where(wr, rng.uniform(35000, 55000, n), rng.uniform(25000, 35000, n)))
    return _comp("field_massive", pos, sp.mv_from_lum(lum, teff), teff, _labels("cmz", kind), extra_av=c["extra_av"],
                 kind=kind)


def local_population_table(c):
    tabs, fr = [], []
    for comp in c["components"]:
        if comp.get("kind", "old") == "old":
            tabs.append(sp.old_population(comp["age_gyr"], comp.get("feh", 0.0)))
        else:
            tabs.append(sp.continuous_population(comp["age_max_myr"], comp.get("feh", 0.0)))
        fr.append(comp["frac"])
    fr = np.asarray(fr) / np.sum(fr)
    return sp.combine(tabs, fr)


def local_population(c, cfg, observer, galaxy, rng):
    """Resolved field stars within r_res of the observer; returns (component, stats).

    Luminosity-function bins with M_V < mv_lim are drawn throughout r_res. Fainter bins (dwarfs)
    are drawn out to the distance where they reach V = v_draw_lim (extinction neglected), so the
    nearby K/M dwarfs appear as points; v_draw_lim=None drops them. inner_factor is the fraction
    of the galaxy's stellar irradiance from inside r_res that is not resolved here: per bin the
    resolved share is int_rmin^rb <rho> dr / int_rmin^rres <rho> dr (shell-averaged density).
    """
    table = local_population_table(c)
    r_res, r_min = float(cfg["r_res_pc"]), cfg["min_dist_au"] * AU_M / PC_M
    if r_res <= r_min:
        return None, dict(inner_factor=1.0, mass_msun=0.0, note="r_res_pc <= min_dist_au: no local stars")
    rb = np.full(len(table.mv), r_res)
    dwarf = table.mv >= cfg["mv_lim"]
    vdl = cfg.get("v_draw_lim")
    rb[dwarf] = r_min if vdl is None else np.clip(10 ** (0.2 * (float(vdl) - table.mv[dwarf]) + 1), r_min, r_res)

    def draw_pos(rmax):
        r = (r_min ** 3 + rng.random(len(rmax)) * (rmax ** 3 - r_min ** 3)) ** (1 / 3)
        return observer + _iso(len(rmax), rng) * r[:, None]

    vol = 4 / 3 * np.pi * (rb ** 3 - r_min ** 3)
    v_res = 4 / 3 * np.pi * (r_res ** 3 - r_min ** 3)
    ml_gal = None
    if galaxy is not None:
        probe = draw_pos(np.full(50000, r_res))
        rho = np.asarray(galaxy.stellar_density(probe), float)
        rho_mean = float(rho.mean())
        lr = r_min * (r_res / r_min) ** rng.random(40000)  # log-uniform radii for the radial profile
        rl = np.asarray(galaxy.stellar_density(observer + _iso(len(lr), rng) * lr[:, None]), float)
        rho_max = 1.2 * max(float(rho.max()), float(rl.max()))
        o = np.argsort(lr)
        run_max = np.maximum.accumulate(rl[o])
        bound = lambda r: np.where(r >= r_res, rho_max,
                                   np.minimum(rho_max, 1.5 * np.interp(r, lr[o], run_max, left=run_max[0])))
        cum = lambda f: np.interp(rb, lr[o], np.cumsum(f[o]), left=0.0) / np.sum(f)
        frac_light = np.where(rb >= r_res, 1.0, cum(rl * lr))  # int <rho> dr, relative
        mass_b = rho_mean * v_res * np.where(rb >= r_res, 1.0, cum(rl * lr ** 3))
        if hasattr(galaxy, "luminosity_density_v"):  # galaxy.py helper outside INTERFACES.md
            ml_gal = float(rho.sum() / np.asarray(galaxy.luminosity_density_v(probe), float).sum())
    else:
        rho_mean = float(c["density_msun_pc3"] or cfg["local_density_msun_pc3"])
        rho_max = rho_mean
        frac_light = (rb - r_min) / (r_res - r_min)
        mass_b = rho_mean * vol
    n_exp = float((table.w * mass_b).sum())
    if n_exp > cfg["max_local_stars"]:
        raise ValueError(f"starfield: {n_exp:.3g} local stars expected (max_local_stars {cfg['max_local_stars']:.3g}); "
                         "lower r_res_pc, mv_lim or v_draw_lim")
    idx, mv, teff, kind = table.draw(table.w * mass_b, rng)
    rmax = rb[idx]
    pos = draw_pos(rmax)
    n_iter, todo, n_viol = 0, np.zeros(0, int), 0
    if galaxy is not None:
        big = rmax >= r_res
        todo, bmax = np.flatnonzero(~big), bound(rmax)
        while len(todo) and n_iter < 500:
            n_iter += 1
            ok = rng.random(len(todo)) * bmax[todo] < np.asarray(galaxy.stellar_density(pos[todo]), float)
            todo = todo[~ok]
            pos[todo] = draw_pos(rmax[todo])
        # r_res stars: piecewise-constant envelope on a cubic grid (2 x the max of 8 probes per cell)
        h = 2 * r_res / 24
        c1 = (np.arange(24) + 0.5) * h - r_res
        cen = np.stack(np.meshgrid(c1, c1, c1, indexing="ij"), -1).reshape(-1, 3)
        cen = cen[np.linalg.norm(np.clip(0.0, cen - h / 2, cen + h / 2), axis=1) < r_res]
        pr = observer + (cen[:, None] + (rng.random((len(cen), 8, 3)) - 0.5) * h).reshape(-1, 3)
        env = np.maximum(2 * np.asarray(galaxy.stellar_density(pr), float).reshape(len(cen), 8).max(1), 0.01 * rho_mean)
        todo_b, it = np.flatnonzero(big), 0
        while len(todo_b) and it < 500:
            it += 1
            k = rng.choice(len(cen), len(todo_b), p=env / env.sum())
            x = cen[k] + (rng.random((len(k), 3)) - 0.5) * h
            r = np.linalg.norm(x, axis=1)
            rho_x = np.zeros(len(k))
            ins = (r > r_min) & (r < r_res)
            rho_x[ins] = np.asarray(galaxy.stellar_density(observer + x[ins]), float)
            ok = rng.random(len(k)) * env[k] < rho_x
            n_viol += int((rho_x[ok] > env[k][ok]).sum())
            pos[todo_b[ok]] = observer + x[ok]
            todo_b = todo_b[~ok]
        n_iter, todo = max(n_iter, it), np.concatenate([todo, todo_b])
    lv = table.w * sp.lum_v_lsun(table.mv)
    lv_res = float((lv * frac_light).sum())
    ml_used = ml_gal if ml_gal is not None else 1 / lv.sum()
    inner = 1 - ml_used * lv_res
    stats = dict(mass_msun=rho_mean * v_res, density_msun_pc3=rho_mean,
                 mass_to_light_v=1 / lv.sum(), mass_to_light_v_galaxy=ml_gal, inner_factor=float(np.clip(inner, 0, 1)),
                 inner_factor_unclipped=float(inner), resolved_light_frac=lv_res / lv.sum(),
                 expected_n=float((table.w * mass_b).sum()), n_dwarfs=int((table.mv[idx] >= cfg["mv_lim"]).sum()),
                 table=table.name, kind_counts=sp.kind_counts(kind), rejection_iterations=n_iter,
                 n_unconverged=int(len(todo)), n_envelope_violations=n_viol)
    return _comp("local", pos, mv, teff, _labels("local", kind), kind=kind), stats


def local_zams_box(c):
    """Legacy cluster.LocalStars box (directions already celestial); bolometric blackbodies."""
    ls = cl.LocalStars(c["mass"], c["m_min"], c["m_max"], c["box_pc"], c["min_dist_au"], c["seed"])
    out = _comp("local_zams_box", dirs_cel=ls.direction, dist_pc=ls.dist_m / PC_M, teff=ls.teff,
                labels=["zams_box"] * len(ls.teff), kind=np.zeros(len(ls.teff), int))
    out["f_bol"] = ls.f_bol
    return out


def companion_comp(c, grid):
    import companion as cp
    d = cp.direction(c["inclination_deg"], c["node_deg"], c["phase_deg"])  # x = equinox, z = pole: our frame
    irr = spectral.blackbody_irradiance(1361.0 / c["sep_au"] ** 2, 5772.0, grid)[None]
    return _comp("companion", irr=irr, dirs_cel=d[None], dist_pc=np.array([c["sep_au"] * AU_M / PC_M]),
                 teff=np.array([5772.0]), labels=["companion"], kind=np.zeros(1, int))


# ---------------------------------------------------------------- Earth
def galactocentric_to_icrs_matrix():
    """v_icrs = M @ v_gc for directions (galactic Cartesian = diag(-1, 1, 1) v_gc)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    g = SkyCoord(l=[0, 90, 0] * u.deg, b=[0, 0, 90] * u.deg, frame="galactic").icrs
    m_gal_to_icrs = np.array(g.cartesian.xyz.value)  # columns: images of the galactic basis
    return m_gal_to_icrs @ np.diag([-1.0, 1.0, 1.0])


def rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def load_bsc(vmax=6.5, fetch=True):
    """(table, source string). Cache: bsc5_vlt{vmax}.csv, or any cached bsc5_vlt{X}.csv with X >= vmax
    (rows are filtered by the caller); else VizieR V/50; else the offline fallback bsc5_vlt2.05.csv."""
    import glob
    from astropy.table import Table
    path = os.path.join(CATALOG_DIR, f"bsc5_vlt{vmax:g}.csv")
    if os.path.exists(path):
        return Table.read(path, format="csv"), f"cache {path}"
    cached = {}
    for p in glob.glob(os.path.join(CATALOG_DIR, "bsc5_vlt*.csv")):
        try:
            cached[float(os.path.basename(p)[8:-4])] = p
        except ValueError:
            continue
    deeper = sorted(x for x in cached if x >= vmax)
    if deeper:
        return Table.read(cached[deeper[0]], format="csv"), f"cache {cached[deeper[0]]}"
    if fetch:
        import requests
        from astroquery import exceptions as aqe
        from astroquery.vizier import Vizier
        v = Vizier(columns=["HR", "Name", "_RAJ2000", "_DEJ2000", "Vmag", "B-V", "SpType"],
                   column_filters={"Vmag": f"<{vmax}"}, row_limit=-1)
        try:
            t = v.query_constraints(catalog="V/50/catalog")[0]
        except (requests.exceptions.RequestException, OSError, IndexError, ValueError, aqe.TableParseError,
                aqe.TimeoutError, aqe.RemoteServiceError) as err:
            print(f"starfield: VizieR query failed ({err}); using the offline fallback")
        else:
            t.sort("Vmag")
            os.makedirs(CATALOG_DIR, exist_ok=True)
            t.write(path, format="csv", overwrite=True)
            return t, f"VizieR V/50 -> {path}"
    path = os.path.join(CATALOG_DIR, "bsc5_vlt2.05.csv")
    return Table.read(path, format="csv"), f"offline fallback {path} (V < 2.05 only)"


def earth_stars(cfg, grid):
    import bright_stars as bs
    c = cfg["bsc"]
    t, source = load_bsc(c["vmax"], c.get("fetch", True))
    vm = np.ma.filled(np.ma.asarray(t["Vmag"], float), np.nan)
    ok = np.isfinite(vm) & (vm < c["vmax"])
    t, vm = t[ok], vm[ok]
    ra = np.radians(np.asarray(t["_RAJ2000"], float) - cfg["sun_ra_deg"])
    dec = np.radians(np.asarray(t["_DEJ2000"], float))
    dirs = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)
    bv = np.ma.filled(np.ma.asarray(t["B-V"], float), 0.6)
    teff = bs.teff_from_bv(np.clip(bv, -0.4, 2.2))
    irr = vmag_irr_fast(vm, teff, grid)
    names = [str(n).strip() if str(n).strip() not in ("", "--") else f"HR {h}" for n, h in zip(t["Name"], t["HR"])]
    return dirs, irr, names, teff, source


def planets(plist, sun_ra_deg, grid, obliquity_deg=23.44):
    """User-specified planets (ecliptic coordinates, V, B-V) as PointSources in the celestial frame."""
    import bright_stars as bs
    lon = np.radians([p["ecl_lon_deg"] for p in plist])
    lat = np.radians([p.get("ecl_lat_deg", 0.0) for p in plist])
    e = np.radians(obliquity_deg)
    x, y, z = np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)
    eq = np.stack([x, y * np.cos(e) - z * np.sin(e), y * np.sin(e) + z * np.cos(e)], axis=1)
    teff = bs.teff_from_bv(np.array([p.get("bv", 0.8) for p in plist]))
    irr = vmag_irr_fast(np.array([p["vmag"] for p in plist], float), teff, grid)
    return PointSources(eq @ rot_z(-sun_ra_deg).T, irr, [p["name"] for p in plist])


# ---------------------------------------------------------------- assembly
def _mag_counts(v):
    return {f"{e:g}": int((v < e).sum()) for e in MAG_EDGES}


def build_starfield(cfg, grid, galaxy=None):
    """Point-source starfield (TOASky with points + binned faint resolved stars) and an info dict."""
    import healpy as hp
    cfg = _merge(DEFAULTS, cfg)
    nside = int(cfg["nside"])
    dmap = np.zeros((hp.nside2npix(nside), grid.n))
    omega_pix = 4 * np.pi / hp.nside2npix(nside)
    info = dict(mode=cfg["mode"], nside=nside, components={}, notes=[])

    if cfg["mode"] == "earth":
        R = rot_z(-cfg["sun_ra_deg"]) @ galactocentric_to_icrs_matrix()
        observer = np.array(EARTH_OBSERVER_PC)
        dirs, irr, names, teff, source = earth_stars(cfg, grid)
        v = v550(irr, grid)
        info.update(R_gc_to_cel=R, observer_pc=observer, inner_factor=1.0, r_res_pc=0.0, catalog=source,
                    n_points=len(v), counts_lt=_mag_counts(v),
                    components={"bsc": dict(n_points=len(v), n_binned=0, V_total=_vsum(irr, grid),
                                            counts_lt=_mag_counts(v))})
        if source.startswith("offline fallback"):
            info["notes"].append(f"Earth: {source}; the V < {cfg['bsc']['vmax']:g} catalogue was unavailable.")
        if galaxy is not None and cfg.get("earth_subtract_bsc", True) and len(v):
            r, e_in = earth_r_res(galaxy, observer, _e550(irr.sum(0)[None], grid)[0], grid,
                                  np.random.default_rng(cfg["seed"]))
            info.update(r_res_pc=r, inner_factor=0.0)
            info["notes"].append(f"Earth: galaxy stellar light inside r_res_pc = {r:.0f} pc (V {e_in:.2f}, extinction "
                                 f"neglected) is replaced by the catalogue stars (inner_factor 0).")
        else:
            info["notes"].append("Earth: catalogue stars are also present in any galaxy diffuse map (r_res_pc = 0).")
        if cfg.get("return_star_table"):
            info["stars"] = dict(vmag=v, teff=teff, component=np.full(len(v), "bsc"), is_point=np.ones(len(v), bool))
        pts = PointSources(dirs, irr, names)
        if cfg["planets"]:
            pts = pts.concat(planets(cfg["planets"], cfg["sun_ra_deg"], grid))
            info["n_points"] = len(pts)
        return TOASky(grid, pts, dmap, meta=dict(starfield=_meta(info))), info

    arches_c = frames.lb_to_galactocentric(*ARCHES_LB, frames.R0_PC)
    quint_c = frames.lb_to_galactocentric(*QUINTUPLET_LB, frames.R0_PC + cfg["quintuplet"]["los_offset_pc"])
    sgra = frames.lb_to_galactocentric(*SGRA_LB, frames.R0_PC)
    odir = np.asarray(cfg["observer_offset_dir"], float)
    observer = arches_c + cfg["offset_pc"] * odir / np.linalg.norm(odir)
    R = frames.galactocentric_to_celestial(observer, arches_c, sgra, cfg["roll_deg"])
    rng = np.random.default_rng(cfg["seed"])
    tau_v_curve = ccm89(grid, cfg["rv"]) / 1.0857

    comps = []
    if cfg["arches"]["enabled"]:
        comps.append(arches(cfg["arches"], arches_c))
    if cfg["quintuplet"]["enabled"]:
        comps.append(quintuplet(cfg["quintuplet"], quint_c))
    if cfg["nsc_young"]["enabled"]:
        comps.append(nsc_young(cfg["nsc_young"], sgra))
    if cfg["field_massive"]["enabled"]:
        comps.append(field_massive(cfg["field_massive"], frames.R0_PC))
    info["inner_factor"] = 1.0
    if cfg["local_old_population"]["enabled"]:
        lc, stats = local_population(cfg["local_old_population"], cfg, observer, galaxy, rng)
        if lc is not None:
            comps.append(lc)
        info["inner_factor"] = stats["inner_factor"]
        info["local"] = stats
        if stats.get("inner_factor_unclipped", 0.0) < -1e-3:
            info["notes"].append(f"resolved local light exceeds the galaxy's inside r_res (inner_factor "
                                 f"{stats['inner_factor_unclipped']:.3f} clipped to 0): population M/L_V "
                                 f"{stats['mass_to_light_v']:.2f} vs galaxy {stats['mass_to_light_v_galaxy']}")
    if cfg["local_zams_box"]["enabled"]:
        comps.append(local_zams_box(cfg["local_zams_box"]))
    if cfg["companion"]["enabled"]:
        comps.append(companion_comp(cfg["companion"], grid))

    D, E, V, T, N, C = [np.zeros((0, 3))], [np.zeros((0, grid.n))], [np.zeros(0)], [np.zeros(0)], [], []
    for c in comps:
        n_c = len(c["teff"])
        if n_c == 0:
            info["components"][c["name"]] = dict(n_stars=0, V_total=np.inf, V_brightest=np.inf,
                                                 counts_lt=_mag_counts(np.zeros(0)), dist_pc_median=np.nan, kind_counts={})
            continue
        if c["pos"] is not None:
            rel = c["pos"] - observer
            dist = np.linalg.norm(rel, axis=1)
            dirs = (rel / dist[:, None]) @ R.T
        else:
            dirs, dist = c["dirs_cel"], c["dist_pc"]
        if c["irr"] is not None:
            irr = c["irr"]
        elif c.get("f_bol") is not None:
            irr = spectral.blackbody_irradiance(c["f_bol"], c["teff"], grid)
        else:
            irr = vmag_irr_fast(c["mv"] + 5 * np.log10(dist / 10.0), c["teff"], grid)
        if c["pos"] is not None:
            if galaxy is not None:
                tau = np.vstack([np.asarray(galaxy.optical_depth(observer, c["pos"][s:s + 20000]), float)
                                 for s in range(0, len(dist), 20000)])
                tau = tau + c["extra_av"] * tau_v_curve
            else:
                tau = (cfg["local_av_per_pc"] * dist + c["extra_av"])[:, None] * tau_v_curve
            irr = irr * np.exp(-tau)
        v = v550(irr, grid)
        D.append(dirs), E.append(irr), V.append(v), T.append(np.asarray(c["teff"], float))
        N += list(c["labels"])
        C += [c["name"]] * len(v)
        info["components"][c["name"]] = dict(
            n_stars=len(v), V_total=_vsum(irr, grid), V_brightest=float(v.min()), counts_lt=_mag_counts(v),
            dist_pc_median=float(np.median(dist)),
            kind_counts=sp.kind_counts(c["kind"]) if c.get("kind") is not None else {})
    D, E, V, T, C = np.vstack(D), np.vstack(E), np.concatenate(V), np.concatenate(T), np.array(C, dtype=str)
    vlim = float(cfg["v_point_lim"])
    if len(V) > cfg["max_points"]:
        vlim = min(vlim, float(np.partition(V, int(cfg["max_points"]))[int(cfg["max_points"])]))
    faint = V >= vlim
    np.add.at(dmap, hp.vec2pix(nside, *D[faint].T), E[faint] / omega_pix)
    keep = np.flatnonzero(~faint)
    for name, d in info["components"].items():
        d["n_points"] = int(((C == name) & ~faint).sum())
        d["n_binned"] = int(((C == name) & faint).sum())
    pts = PointSources(D[keep], E[keep], [N[i] for i in keep])
    to_cel = lambda p: R @ ((p - observer) / np.linalg.norm(p - observer))
    if cfg.get("return_star_table"):
        info["stars"] = dict(vmag=V, teff=T, component=C, is_point=~faint)
    info.update(R_gc_to_cel=R, observer_pc=observer, r_res_pc=float(cfg["r_res_pc"]), n_points=len(pts),
                v_point_lim_used=vlim, counts_lt=_mag_counts(V), arches_cel=to_cel(arches_c),
                sgr_a_cel=to_cel(sgra), quintuplet_cel=to_cel(quint_c),
                diffuse_V_total=_vsum((dmap * omega_pix).sum(0)[None], grid) if faint.any() else np.inf)
    info["notes"] += ["Arches kept at ZAMS (age ~2.5 Myr).",
                      "Quintuplet placed at R0; NSC young stars isotropic (disc ignored).",
                      "Blackbody spectral shapes normalised to V via empirical BC_V."]
    return TOASky(grid, pts, dmap, meta=dict(starfield=_meta(info))), info


def _vsum(irr, grid):
    return float(v550(np.asarray(irr).sum(0)[None], grid)[0])


def _meta(info):
    return {k: v for k, v in info.items() if k in ("mode", "inner_factor", "r_res_pc", "n_points", "counts_lt",
                                                   "catalog")}


def main():
    import argparse
    import time
    p = argparse.ArgumentParser(description="Build a starfield and print a summary.")
    p.add_argument("--mode", default="gc")
    a = p.parse_args()
    t0 = time.time()
    toa, info = build_starfield(dict(mode=a.mode), spectral.default_grid())
    print(f"{len(toa.points)} points in {time.time() - t0:.1f} s; inner_factor={info['inner_factor']:.3f}")
    for k, c in info["components"].items():
        print(f"  {k}: {c}")
    print("N(<V):", info["counts_lt"])


if __name__ == "__main__":
    main()
