"""Simple stellar-population recipes for the starfield: lifetimes, crude evolutionary phases,
bolometric corrections, and luminosity functions per unit stellar mass.

Units: masses Msun, luminosities Lsun (bolometric, M_bol,sun = 4.74), ages Myr unless named
_gyr, absolute V magnitudes M_V (Vega). Spectra downstream are blackbodies at Teff scaled to V.

Approximations (order-of-magnitude stellar astrophysics, no isochrone tables available offline):
* H-burning lifetimes t_H(M): interpolated from solar-metallicity Geneva/MIST-like values
  (Ekstrom et al. 2012; Choi et al. 2016), lengthened by 10^(0.25 [Fe/H]).
* Main-sequence brightening L = L_ZAMS (1 + 1.2 f^1.5), f = age / t_H; ZAMS L/Teff from
  cluster.lum_teff (extended below 0.3 Msun with L ~ M^2.5).
* Massive post-MS phases (young clusters): a fixed fraction of t_H after the MS, split by mass
  and fractional phase age into LBV / WN / WC (M > 30), BSG / YSG / RSG (8-30), He-burning
  giants (2.2-8). Conti-scenario ordering; luminosities scaled from the TAMS value.
* Old populations: fuel-consumption theorem (Renzini & Buzzoni 1986). Evolutionary flux
  B = phi(M_TO) |dM_TO/dt|; the RGB uses the core-mass--luminosity relation
  L = 2.3e5 Mc^6 (Refsdal & Weigert 1970) with dMc/dt = 1.37e-11 L Msun/yr, so
  dt/dlnL = Mc / (8.2e-11 L) yr; RGB tip L = 2500 Lsun; red clump 1e8 yr at ~50 Lsun,
  Teff ~ 4800 K (Girardi 2016); E-AGB 1.2e7 yr (100-2000 Lsun), TP-AGB 1.5e6 yr
  (2000-6000 Lsun, Teff 3000-3500 K).
* IMF: Kroupa (2001) 0.08-100 Msun for field populations; remnant masses from Kalirai et al.
  (2008) (WD), 1.4 Msun (NS, 8-25 Msun), 10 Msun (BH) for the present-day mass.
* BC_V(Teff): Flower (1996) as tabulated by Torres (2010), extended above 50 kK with the
  Rayleigh-Jeans slope (-7 mag/dex); the same table is used for dwarfs, giants and WR stars.
"""
from dataclasses import dataclass

import numpy as np

import cluster as cl

MBOL_SUN = 4.74
MV_SUN = 4.83

_TH_MASS = np.array([0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 7.0, 10, 15, 20, 25, 40, 60, 85, 120])
_TH_MYR = np.array([6.0e4, 2.5e4, 1.0e4, 5.0e3, 2.7e3, 1.2e3, 380, 100, 45, 24, 12.5, 9.0, 7.2, 4.8, 3.7, 3.2, 2.8])

_BC_LOGT = np.array([3.45, 3.50, 3.55, 3.60, 3.65, 3.70, 3.75, 3.80, 3.85, 3.90, 4.00, 4.10, 4.20, 4.30,
                     4.40, 4.50, 4.60, 4.70])
_BC_V = np.array([-4.2, -2.9, -1.75, -1.0, -0.55, -0.27, -0.11, -0.04, 0.0, -0.02, -0.35, -0.80, -1.30, -1.85,
                  -2.40, -2.95, -3.55, -4.15])

# dwarf M_V(M) and Teff(M) (Pecaut & Mamajek 2013 table), used for the field main sequence
_MS_M = np.array([0.08, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0,
                  2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 40.0, 100.0])
_MS_MV = np.array([19.0, 16.0, 13.5, 12.2, 11.0, 10.2, 9.2, 8.1, 7.2, 6.3, 5.5, 4.8, 4.3, 3.9, 3.1, 2.6, 2.1,
                   1.6, 0.8, 0.2, -0.6, -1.1, -1.8, -2.6, -3.4, -4.0, -5.2, -6.0])
_MS_T = np.array([2600, 2900, 3100, 3200, 3400, 3550, 3750, 4050, 4400, 5000, 5400, 5770, 6000, 6200, 6600,
                  7000, 7600, 8600, 10500, 12500, 15000, 16500, 19500, 24000, 30000, 34000, 42000, 48000.0])

KINDS = ("MS", "SGB", "RGB", "RC", "EAGB", "TPAGB", "WD", "BSG", "YSG", "RSG", "LBV", "WN", "WC", "HeG")


def bc_v(teff):
    lt = np.log10(np.asarray(teff, float))
    bc = np.interp(lt, _BC_LOGT, _BC_V)
    return np.where(lt > _BC_LOGT[-1], _BC_V[-1] - 7.0 * (lt - _BC_LOGT[-1]), bc)


def mv_from_lum(lum_lsun, teff):
    return MBOL_SUN - 2.5 * np.log10(lum_lsun) - bc_v(teff)


def lum_v_lsun(mv):
    """V-band luminosity in solar V units."""
    return 10 ** (-0.4 * (np.asarray(mv, float) - MV_SUN))


def t_h_myr(mass, feh=0.0):
    return 10 ** (np.interp(np.log10(mass), np.log10(_TH_MASS), np.log10(_TH_MYR))) * 10 ** (0.25 * feh)


def turnoff_mass(age_myr, feh=0.0):
    lt = np.log10(age_myr) - 0.25 * feh
    return 10 ** np.interp(-lt, -np.log10(_TH_MYR), np.log10(_TH_MASS))


def zams(mass):
    """ZAMS (L Lsun, Teff K); cluster.lum_teff above 0.3 Msun, L ~ M^2.5 and the dwarf Teff table below."""
    m = np.asarray(mass, float)
    lum, teff = cl.lum_teff(np.maximum(m, 0.3))
    lum = lum / cl.LSUN * np.where(m < 0.3, (m / 0.3) ** 2.5, 1.0)
    teff = np.where(m < 0.3, np.interp(m, _MS_M, _MS_T), teff)
    return lum, teff


def kroupa_phi(m):
    """Kroupa (2001) dN/dM (unnormalised, continuous at 0.5 Msun)."""
    m = np.asarray(m, float)
    return np.where(m < 0.5, m ** -1.3, 0.5 * m ** -2.3)


def _kroupa_norm(m_lo=0.08, m_hi=100.0):
    g = np.geomspace(m_lo, m_hi, 4001)
    return np.trapezoid(g * kroupa_phi(g), g)


def sample_kroupa_range(n_expected, m_lo, m_hi, rng):
    """Poisson(n_expected) masses from Kroupa between m_lo and m_hi."""
    g = np.geomspace(m_lo, m_hi, 4001)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (kroupa_phi(g[1:]) + kroupa_phi(g[:-1])) * np.diff(g))])
    n = rng.poisson(n_expected)
    return np.interp(rng.random(n), cdf / cdf[-1], g)


def remnant_mass(m):
    m = np.asarray(m, float)
    return np.where(m < 8, 0.109 * m + 0.394, np.where(m < 25, 1.4, 10.0))


def evolve(mass, age_myr, feh=0.0, post_ms_frac=0.12):
    """Crude single-star state at age_myr: (L Lsun, Teff K, kind index, alive bool)."""
    m = np.asarray(mass, float)
    age = np.broadcast_to(np.asarray(age_myr, float), m.shape)
    th = t_h_myr(m, feh)
    f = age / th
    l0, t0 = zams(m)
    cool = np.interp(np.log10(m), [0.0, 0.3, 1.0, 2.0], [-0.03, 0.05, 0.25, 0.4])
    fm = np.minimum(f, 1.0)
    lum = l0 * (1 + 1.2 * fm ** 1.5)
    teff = t0 * (1 - cool * fm ** 2)
    kind = np.zeros(m.shape, int)
    pfrac = np.where(m >= 8, post_ms_frac, 0.25)
    tau = (f - 1) / pfrac
    post = (f >= 1) & (tau < 1)
    alive = f < 1 + pfrac
    ltams = lum.copy()
    K = {k: i for i, k in enumerate(KINDS)}

    def put(sel, k, lfac, t):
        kind[sel] = K[k]
        lum[sel] = ltams[sel] * lfac
        teff[sel] = t if np.ndim(t) == 0 else t[sel]

    hi, mid, lo8, im = post & (m >= 30), post & (m >= 15) & (m < 30), post & (m >= 8) & (m < 15), post & (m < 8)
    put(hi & (tau < 0.1), "LBV", 1.0, 10 ** (4.0 + 0.3 * np.clip(tau / 0.1, 0, 1)))
    put(hi & (tau >= 0.1) & (tau < 0.4), "WN", 0.9, 45000.0)
    put(hi & (tau >= 0.4), "WC", 0.6, 70000.0)
    put(mid & (tau < 0.1), "BSG", 1.1, 18000.0 - 1e5 * np.clip(tau, 0, 0.1))
    put(mid & (tau >= 0.1) & (tau < 0.15), "YSG", 1.2, 6500.0)
    put(mid & (tau >= 0.15), "RSG", 1.3, 3600.0)
    put(lo8 & (tau < 0.1), "BSG", 1.1, 12000.0)
    put(lo8 & (tau >= 0.1), "RSG", 1.5, 3900.0)
    put(im & (tau < 0.1), "SGB", 1.2, 7000.0)
    put(im & (tau >= 0.1) & (tau < 0.9), "HeG", 1.0, 4900.0)
    put(im & (tau >= 0.9), "EAGB", 3.0, 3800.0)
    return lum, teff, kind, alive


@dataclass
class PopTable:
    """Discretised luminosity function: w = expected number of stars per Msun of present-day
    stellar mass in each bin, with absolute V magnitude mv, Teff and phase index kind."""
    w: np.ndarray
    mv: np.ndarray
    teff: np.ndarray
    kind: np.ndarray
    name: str = ""

    def light_v(self, mv_max=np.inf, mv_min=-np.inf):
        """V luminosity (Lsun_V per Msun) of the bins with mv_min <= M_V < mv_max."""
        s = (self.mv < mv_max) & (self.mv >= mv_min)
        return float((self.w[s] * lum_v_lsun(self.mv[s])).sum())

    def number(self, mv_max=np.inf):
        return float(self.w[self.mv < mv_max].sum())

    def scaled(self, frac):
        return PopTable(self.w * frac, self.mv, self.teff, self.kind, self.name)

    def draw(self, n_exp, rng, jitter_mag=0.08):
        """Poisson draw with n_exp (per bin) expected stars: (bin index, M_V, Teff, kind)."""
        n_exp = np.asarray(n_exp, float)
        tot = float(n_exp.sum())
        n = rng.poisson(tot) if tot > 0 else 0
        idx = rng.choice(len(n_exp), size=n, p=n_exp / tot) if n else np.zeros(0, int)
        mv = self.mv[idx] + rng.normal(0, jitter_mag, n)
        teff = self.teff[idx] * np.exp(rng.normal(0, 0.02, n))
        return idx, mv, teff, self.kind[idx]

    def sample(self, mass_msun, rng, mv_max=np.inf, jitter_mag=0.08):
        """Poisson draw of the stars brighter than mv_max in mass_msun: (M_V, Teff, kind)."""
        return self.draw(mass_msun * self.w * (self.mv < mv_max), rng, jitter_mag)[1:]


def combine(tables, fracs):
    t = [tb.scaled(f) for tb, f in zip(tables, fracs)]
    return PopTable(*(np.concatenate([getattr(x, a) for x in t]) for a in ("w", "mv", "teff", "kind")),
                    name="+".join(x.name for x in tables))


def _rgb_teff(lum, feh):
    return 10 ** (3.69 - 0.065 * np.log10(np.maximum(lum, 1) / 10.0) - 0.06 * feh)


def old_population(age_gyr=9.0, feh=0.15, nbin=300):
    """Fuel-consumption luminosity function of a single-age population (age >~ 1 Gyr)."""
    K = {k: i for i, k in enumerate(KINDS)}
    age = age_gyr * 1e3
    mto = float(turnoff_mass(age, feh))
    norm = _kroupa_norm()
    phi = lambda m: kroupa_phi(m) / norm  # per Msun formed
    g = np.geomspace(0.08, 100, 6001)
    living = np.trapezoid(np.where(g < mto, g * phi(g), 0), g)
    rem = np.trapezoid(np.where(g >= mto, remnant_mass(g) * phi(g), 0), g)
    pdm = living + rem  # present-day mass per Msun formed
    w, mv, teff, kind = [], [], [], []

    def add(wi, mvi, ti, k):
        w.append(np.asarray(wi, float) / pdm)
        mv.append(np.broadcast_to(np.asarray(mvi, float), np.shape(wi)))
        teff.append(np.broadcast_to(np.asarray(ti, float), np.shape(wi)))
        kind.append(np.full(np.shape(wi), K[k]))

    me = np.geomspace(0.08, mto, nbin + 1)
    mc = np.sqrt(me[1:] * me[:-1])
    f = age / t_h_myr(mc, feh)
    add(phi(mc) * np.diff(me), np.interp(mc, _MS_M, _MS_MV) - 0.75 * f ** 2,
        np.interp(mc, _MS_M, _MS_T) * 10 ** (-0.03 * feh), "MS")
    dmdt =(float(turnoff_mass(age * 0.99, feh)) - float(turnoff_mass(age * 1.01, feh))) / (0.02 * age * 1e6)
    B = phi(mto) * dmdt  # stars per yr per Msun formed leaving the MS
    mv_tams = float(np.interp(mto, _MS_M, _MS_MV)) - 0.75
    t_tams = float(np.interp(mto, _MS_M, _MS_T))
    th_yr = float(t_h_myr(mto, feh)) * 1e6
    x = (np.arange(40) + 0.5) / 40
    add(np.full(40, B * 0.07 * th_yr / 40), mv_tams - 0.4 * x, t_tams - (t_tams - 5000.0) * x, "SGB")
    degenerate = mto < 1.9
    l_rc = 50.0 if degenerate else 50.0 * (mto / 1.9) ** 3.5
    l_tip = 2500.0 if degenerate else 2.0 * l_rc
    le = np.geomspace(3.0 * mto ** 2, l_tip, nbin + 1)
    lc = np.sqrt(le[1:] * le[:-1])
    mcore = (lc / 2.3e5) ** (1 / 6)
    dt = mcore / (8.2e-11 * lc) * np.diff(np.log(le))
    if not degenerate:
        dt *= 0.05 * th_yr / dt.sum()
    t_rgb = _rgb_teff(lc, feh)
    add(B * dt, mv_from_lum(lc, t_rgb), t_rgb, "RGB")
    t_rc = 4800.0 - 300 * feh
    mv_rc = MBOL_SUN - 2.5 * np.log10(l_rc) - bc_v(t_rc)
    dmv = np.linspace(-0.4, 0.4, 21)
    wrc = np.exp(-0.5 * (dmv / 0.15) ** 2)
    add(B * 1e8 * (1 if degenerate else (mto / 1.9) ** -2) * wrc / wrc.sum(), mv_rc + dmv, t_rc, "RC")
    le = np.geomspace(1.5 * l_rc, 2000.0 * max(1, mto / 1.9), 101)
    lc = np.sqrt(le[1:] * le[:-1])
    wt = np.diff(np.log(le)) / lc
    t_e = _rgb_teff(lc, feh) - 50
    add(B * 1.2e7 * wt / wt.sum(), mv_from_lum(lc, t_e), t_e, "EAGB")
    le = np.geomspace(2000.0, 6000.0, 41) * max(1, mto / 1.9) ** 1.5
    lc = np.sqrt(le[1:] * le[:-1])
    t_tp = 3500.0 - 500.0 * np.log10(lc / le[0]) / np.log10(3.0) - 100 * feh
    add(np.full(40, B * 1.5e6 / 40), mv_from_lum(lc, t_tp), t_tp, "TPAGB")
    n_wd = np.trapezoid(np.where((g >= mto) & (g < 8), phi(g), 0), g)
    add(np.array([n_wd]), np.array([13.0]), np.array([7000.0]), "WD")
    return PopTable(*(np.concatenate(a) for a in (w, mv, teff, kind)), name=f"old{age_gyr:g}Gyr")


def continuous_population(age_max_myr=100.0, feh=0.1, nm=500, nt=60):
    """Constant star formation over the last age_max_myr; evolve() recipe on a (mass, age) grid."""
    norm = _kroupa_norm()
    me = np.geomspace(0.08, 100, nm + 1)
    mc = np.sqrt(me[1:] * me[:-1])
    wm = kroupa_phi(mc) * np.diff(me) / norm
    ages = (np.arange(nt) + 0.5) / nt * age_max_myr
    M, A = np.meshgrid(mc, ages)
    W = np.broadcast_to(wm, M.shape) / nt
    lum, teff, kind, alive = evolve(M, A, feh)
    pdm = (W * np.where(alive, M, remnant_mass(M))).sum()
    s = alive.ravel()
    return PopTable((W.ravel() / pdm)[s], mv_from_lum(lum, teff).ravel()[s], teff.ravel()[s], kind.ravel()[s],
                    name=f"young{age_max_myr:g}Myr")


def young_cluster(total_mass, age_myr, rng, alpha=1.85, m_min=1.0, m_max=120.0, age_spread_myr=0.0, feh=0.0,
                  post_ms_frac=0.12):
    """IMF-sampled cluster with the evolve() recipe. Returns dict(mass, lum, teff, kind) of living stars."""
    m = cl.sample_imf(total_mass, m_min, m_max, alpha, rng)
    age = np.clip(age_myr + age_spread_myr * rng.standard_normal(len(m)), 0.05, None)
    lum, teff, kind, alive = evolve(m, age, feh, post_ms_frac)
    return dict(mass=m[alive], lum=lum[alive], teff=teff[alive], kind=kind[alive], n_dead=int((~alive).sum()))


def kind_counts(kind):
    k = np.asarray(kind)
    return {name: int((k == i).sum()) for i, name in enumerate(KINDS) if (k == i).any()}
