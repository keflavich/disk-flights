"""Naked-eye Bortle-3 sky over the Chiefland Astronomy Village, 2026-11-08 00:00 UTC, looking
south-west (frames and projection in geometry.py).

Model (V band throughout; colour only as a mild tint):
- Points: Hipparcos (V < 9.5, propagated from J1991.25) plus Gaia DR3 G < 9.5 sources without
  a Hipparcos-2 counterpart (propagated from J2016.0, V from G and BP-RP, Riello et al. 2021).
  Exact apparent alt/az from astropy (no refraction).
- Integrated starlight: Gaia DR3 G >= 9.5 flux summed in HEALPix cells (fetch_diffuse.py),
  converted to V with the cell's flux-weighted BP-RP, deposited flux-conserving into the image
  and smoothed to an eye-like resolution (`eye_fwhm_arcmin`).
- Atmosphere: nightsky.atmosphere (spherical shell, Rayleigh + aerosol + ozone, single and
  multiple scattering, airglow layer with van Rhijn brightening), aerosol set for
  `k_v` mag/airmass. Its scattering input is the Gaia starlight map (outside the frame footprint
  filled by Galactic-latitude averages) and the airglow. The airglow is scaled so that the
  zenith (airglow + scattered light + light pollution + transmitted starlight) is
  `zenith_mag` mag/arcsec^2. Light pollution: a faint uniform term plus a dome toward the
  Tampa Bay / Nature Coast towns at the left (south-south-east) edge.
- Eye: stars are detected with the Hecht / Schaefer (1990) threshold on the local background
  (factor set for limiting magnitude `zenith_limit` at the zenith); the undetected fraction of
  each star's flux joins the smoothed glow.
- Display: fixed PSF in pixels (same in every frame), log stretch for the glow, compressed
  power law for the stars.

Usage (from src/):  python3 -m chiefland.render <outdir> [--frames main,teapot] [--space]
"""
import argparse
import json
import time
from pathlib import Path

import healpy as hp
import numpy as np
from astropy.table import Table
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.special import erf

from astropy.coordinates import ICRS, SkyCoord, get_body

from nightsky import atmosphere as atm_mod
from nightsky import frames, spectral
from nightsky.sky import PointSources, TOASky

from . import geometry as geo

ROOT = Path(__file__).resolve().parents[2]
CAT = ROOT / "data" / "chiefland_catalogs"
EPOCH = 2026.853  # 2026-11-08
ARCSEC2_SR = (np.pi / 180 / 3600) ** 2
CD_M2_PER_V0_ARCSEC2 = 1.08e5  # luminance (cd/m^2) of V = 0 mag/arcsec^2
SCATTER_NSIDE = 64

PARAMS = dict(
    k_v=0.25,                 # V extinction, mag/airmass
    zenith_mag=21.7,          # total zenith sky brightness, V mag/arcsec^2
    zenith_limit=6.8,         # naked-eye limiting magnitude at the zenith
    lp_zenith_frac=0.15,      # light-pollution share of the zenith sky
    lp_dome_az=158.0, lp_dome_width=28.0, lp_dome_amp=6.0, lp_dome_alt=8.0,
    lp_horizon_amp=3.0, lp_horizon_alt=6.0,
    horizon_monotonic=25.0,   # below this altitude the sky never dims toward the horizon (see SkyModel)
    eye_fwhm_arcmin=15.0,     # smoothing of the integrated starlight (Gaia G >= 9.5)
    spike_clip=0.0,           # >0: integrated-light cells above this x the 30' mean are spread over 30' (off: it also dims M22)
    undetected_fwhm_arcmin=120.0,  # undetected share of the catalogued stars (V ~ 7-9.5, a few per deg^2)
    psf_sigma_main=2.8,       # star core (pixels) in the 4096 x 3072 wide frame (shown at ~0.35 device px/px)
    psf_sigma_teapot=1.4,     # star core (pixels) in the close-ups (~0.7 device px/px)
    sized_stars=0,            # 1: star sigma grows with brightness (STAR_SIZE), else the fixed psf_sigma_*
    star_size_gamma=1.0,
    star_wing=1.0,            # < 1: keep the psf_sigma_* core and give this share of the peak to a sized wing      # steepness: sigma = s_faint (STAR_SIZE / s_faint)^gamma, s_faint = STAR_SIZE_SIGMA[-1]
    halo_vmax=1.5,            # soft halo only on stars brighter than this (fades in over 1 mag)
    halo_r_sigma=4.0,         # halo core radius in units of the star sigma
    twilight=1,               # include the Sun (at -17.6 deg) as a scattering source
    sigma_mag=0.3,            # psychometric width of the detection
    min_alt=-3.0,
)

# Display: glow encoded value D = clip(g0 + g1 * log10(S / S_zenith)); stars added in linear light
STRETCHES = {
    "": dict(g0=0.135, g1=0.62, gfloor=0.0, star_ref=1.2, star_alpha=0.62, star_gain=1.0,
             halo=0.15),
    "_bright": dict(g0=0.175, g1=0.70, gfloor=0.0, star_ref=1.6, star_alpha=0.60, star_gain=1.15,
                    halo=0.2),
}
# brightness-sized stars (sized_stars=1): sigma in close-up px vs apparent V, log-linear between
# the anchors and constant outside; the wide frame uses psf_sigma_main / psf_sigma_teapot (= 2)
# times these so a star has the same size on the slide in both frames
STAR_SIZE_V = np.array([1.0, 2.0, 3.5, 6.0])
STAR_SIZE_SIGMA = np.array([5.0, 4.0, 3.0, 1.4])
SKY_TINT = np.array([0.93, 0.97, 1.0])
LP_TINT = np.array([1.0, 0.93, 0.82])
STAR_SAT = 0.45


# ----------------------------------------------------------------------------- catalogues
def g_minus_v(c):
    """Riello et al. (2021) G - V as a function of BP - RP (valid -0.5 .. 5)."""
    c = np.clip(c, -0.5, 5.0)
    return -0.02704 + 0.01424 * c - 0.2156 * c ** 2 + 0.01426 * c ** 3


def teff_from_bv(bv):
    """Ballesteros (2012)."""
    bv = np.clip(bv, -0.35, 2.0)
    return 4600.0 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62))


def teff_from_bprp(c):
    """Mucciarelli & Bellazzini (2020) dwarf relation (colour only)."""
    c = np.clip(c, -0.3, 3.5)
    return 5040.0 / (0.4929 + 0.5092 * c - 0.0353 * c ** 2)


def _col(t, name, fill=np.nan):
    c = t[name]
    return np.asarray(c.filled(fill) if hasattr(c, "filled") else c, float)


def load_points():
    """Point-source list: dict(ra, dec, v, teff, src) at EPOCH."""
    hip = Table.read(CAT / "hipparcos.fits")
    gpath = CAT / "gaia_bright.fits"
    if not gpath.exists():  # test runs before the Gaia download finished
        print("WARNING: no gaia_bright.fits, Hipparcos only")
        gai = Table(dict(hip2=[1], ra=[0.0], dec=[0.0], pmra=[0.0], pmdec=[0.0], bp_rp=[0.0],
                         phot_g_mean_mag=[20.0], source_id=[0]))
    else:
        gai = Table.read(gpath)
    dt = EPOCH - 1991.25
    dec = _col(hip, "de")
    ra = _col(hip, "ra") + np.nan_to_num(_col(hip, "pmra")) * dt / 3.6e6 / np.cos(np.radians(dec))
    dec = dec + np.nan_to_num(_col(hip, "pmde")) * dt / 3.6e6
    bv = _col(hip, "b_v")
    h = dict(ra=ra, dec=dec, v=_col(hip, "vmag"), teff=np.where(np.isfinite(bv), teff_from_bv(bv), 5800.0),
             src=np.zeros(len(hip), int), id=np.asarray(hip["hip"], int))
    hip2 = np.asarray(gai["hip2"].filled(0) if hasattr(gai["hip2"], "filled") else gai["hip2"], float)
    # Gaia G < 9.5 sources not in the V < 9.5 Hipparcos list (keeps red stars with V > 9.5, whose
    # flux the G >= 9.5 integrated light would otherwise miss)
    only = ~np.isin(hip2, h["id"])
    gg = gai[only]
    dt = EPOCH - 2016.0
    dec = _col(gg, "dec")
    ra = _col(gg, "ra") + np.nan_to_num(_col(gg, "pmra")) * dt / 3.6e6 / np.cos(np.radians(dec))
    dec = dec + np.nan_to_num(_col(gg, "pmdec")) * dt / 3.6e6
    c = _col(gg, "bp_rp")
    G = _col(gg, "phot_g_mean_mag")
    v = G - np.where(np.isfinite(c), g_minus_v(c), 0.0)
    g = dict(ra=ra, dec=dec, v=v, teff=np.where(np.isfinite(c), teff_from_bprp(c), 5800.0),
             src=np.ones(len(gg), int), id=np.asarray(gg["source_id"], np.int64))
    out = {k: np.concatenate([h[k], g[k]]) for k in h}
    ok = np.isfinite(out["v"]) & np.isfinite(out["ra"]) & np.isfinite(out["dec"])
    return {k: a[ok] for k, a in out.items()}


def load_diffuse(order):
    """Gaia G >= 9.5 cells of one order: (nested index, V flux in V=0 units, BP-RP)."""
    files = sorted((CAT / "gaia_diffuse_sync").glob(f"o{order}_c*.npz"))
    idx, fg, fb, fr = [], [], [], []
    for f in files:
        d = np.load(f)
        if d["hpx"].size == 0:
            continue
        idx.append(d["hpx"].astype(np.int64))
        fg.append(d["fg"]); fb.append(d["fbp"] * d["n"] / np.maximum(d["nbp"], 1))
        fr.append(d["frp"] * d["n"] / np.maximum(d["nrp"], 1))
    idx, fg, fb, fr = (np.concatenate(a) for a in (idx, fg, fb, fr))
    idx, first = np.unique(idx, return_index=True)  # chunks do not overlap within an order
    fg, fb, fr = fg[first], fb[first], fr[first]
    c = -2.5 * np.log10(np.maximum(fb, 1e-30) / np.maximum(fr, 1e-30))
    c = np.where((fb > 0) & (fr > 0), c, 1.5)
    fv = fg * 10 ** (0.4 * g_minus_v(c))  # V flux = G flux * 10^(0.4 (G - V))
    return idx, fv, c


# ----------------------------------------------------------------------------- geometry
def icrs_to_horizon_matrix():
    """R with v_horizon(x=N, y=W, z=Up) = R @ v_icrs (astropy, no refraction; aberration makes
    it slightly non-orthogonal, so it is orthonormalised)."""
    basis = SkyCoord(x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1], representation_type="cartesian", frame=ICRS())
    basis = SkyCoord(ra=basis.spherical.lon, dec=basis.spherical.lat, frame=ICRS())
    aa = basis.transform_to(geo.altaz_frame())
    cols = frames.altaz_to_vec(aa.alt.deg, aa.az.deg).T
    u_, _, vt = np.linalg.svd(cols)
    return u_ @ vt


def frame_dirs(frame, step=1):
    """Horizon-frame unit vectors (H', W', 3), alt, az and pixel solid angle (sr) for pixel
    centres every `step` px."""
    W, H, f, x0, y0 = geo.FRAMES[frame]
    py, px = np.mgrid[0:H:step, 0:W:step].astype(float)
    px += 0.5 * step
    py += 0.5 * step
    alt, az = geo.pix_to_altaz(px, py, f, x0, y0)
    return frames.altaz_to_vec(alt, az), alt, az, geo.pixel_scale_sr(px, py, f, x0, y0)


# ----------------------------------------------------------------------------- atmosphere
def make_atmosphere(k_v, grid):
    """earth_clear at 15 m with the aerosol depth tuned for k_v mag/airmass in V."""
    build = lambda tau: atm_mod.Atmosphere(dict(base="earth_clear", altitude_m=15.0, aerosol_tau550=tau,
                                                k_exact=4), grid)
    tau = 0.06
    for _ in range(3):  # k_V is linear in the aerosol depth (slope ~1.086 x the V-weighted Angstrom factor)
        a = build(tau)
        k = a.summary()["k_V"]
        if abs(k - k_v) < 2e-3:
            break
        tau = max(tau + (k_v - k) / 1.05, 0.0)
    return a, tau


def scatter_map(order, idx, fv, grid):
    """RING nside-64 TOA radiance of the Gaia starlight for the scattering source: footprint cells
    summed; empty cells filled with the mean surface brightness at their |b| (1-deg bins)."""
    ns = SCATTER_NSIDE
    npix = hp.nside2npix(ns)
    fac = (2 ** order // ns) ** 2
    nest = idx // fac
    flux = np.bincount(nest, weights=fv, minlength=npix)
    cnt = np.bincount(nest, minlength=npix)
    full = cnt == fac
    sb = np.where(full, flux, 0.0) / (4 * np.pi / npix / ARCSEC2_SR)  # V flux per arcsec^2
    v = np.array(hp.pix2vec(ns, np.arange(npix), nest=True)).T
    gal = SkyCoord(x=v[:, 0], y=v[:, 1], z=v[:, 2], representation_type="cartesian", frame=ICRS()).galactic
    ab = np.minimum(np.abs(gal.b.deg).astype(int), 89)
    prof = np.bincount(ab[full], weights=sb[full], minlength=90) / np.maximum(np.bincount(ab[full], minlength=90), 1)
    for i in range(1, 90):  # bins with no footprint cells take the neighbour value
        if prof[i] == 0:
            prof[i] = prof[i - 1]
    sb = np.where(full, sb, prof[ab])
    ring = hp.nest2ring(ns, np.arange(npix))
    sb_ring = np.zeros(npix)
    sb_ring[ring] = sb
    mu = -2.5 * np.log10(np.maximum(sb_ring, 1e-30))
    rad = spectral.surface_brightness_radiance(mu, 4500.0, grid)
    return rad, sb_ring


def v_band(spec, grid, weight=None):
    """V-filter weighted mean of (..., n); for radiance this is in the units of
    atmosphere.v_mag_arcsec2's flux (linear, before the log)."""
    s = atm_mod.v_filter_bins(grid) * grid.dwl
    if weight is not None:
        s = s * weight
    return np.asarray(spec) @ s / s.sum()


def rad_to_sb(rad_v):
    """V-weighted radiance -> V flux per arcsec^2 (V=0 units)."""
    return rad_v * ARCSEC2_SR / spectral.ph.F_V0_550NM


def light_pollution(alt, az, p):
    """Relative light-pollution profile (1 at the zenith)."""
    a = np.maximum(alt, 0.0)
    daz = (az - p["lp_dome_az"] + 180) % 360 - 180
    dome = p["lp_dome_amp"] * np.exp(-0.5 * (daz / p["lp_dome_width"]) ** 2) * np.exp(-a / p["lp_dome_alt"])
    return 1.0 + p["lp_horizon_amp"] * np.exp(-a / p["lp_horizon_alt"]) + dome, dome


class SkyModel:
    """Low-resolution background (airglow + scattered light + light pollution) and the
    transmittance, interpolable at any (alt, az)."""

    def __init__(self, p, grid, R, cells, log=print):
        self.p, self.grid, self.R = p, grid, R
        t0 = time.time()
        self.atm, self.tau_aer = make_atmosphere(p["k_v"], grid)
        log(f"atmosphere: aerosol tau550 {self.tau_aer:.3f}, k_V {self.atm.summary()['k_V']:.3f} ({time.time() - t0:.0f} s)")
        rad, self.sb_scatter = scatter_map(*cells, grid)
        n = grid.n
        self.toa_isl = TOASky(grid, PointSources.empty(n), rad)
        self.toa_none = TOASky(grid, PointSources.empty(n), np.zeros_like(rad))
        sun = get_body("sun", geo.TIME, geo.LOCATION).transform_to(geo.altaz_frame())
        self.sun_altaz = (float(sun.alt.deg), float(sun.az.deg))
        sun_c = R.T @ frames.altaz_to_vec(*self.sun_altaz)
        sun_irr = spectral.vmag_irradiance(-26.74, 5772.0, grid)[None, :]
        self.toa_sun = TOASky(grid, PointSources(sun_c[None, :], sun_irr, ["Sun"]), np.zeros_like(rad))
        self.w_isl = spectral.ph.planck_shape(grid.wl, 4500.0)
        # (alt, az) table: alt -3..90 (fine near the horizon), az 0..360 every 2 deg
        self.alts = np.concatenate([np.arange(-3.0, 0.0, 0.5), np.arange(0.0, 6.0, 0.1), np.arange(6.0, 20.0, 0.5),
                                    np.arange(20.0, 90.01, 2.0)])
        self.azs = np.arange(0.0, 362.01, 2.0)
        A, Z = np.meshgrid(self.alts, self.azs, indexing="ij")
        d = frames.altaz_to_vec(np.maximum(A, 0.05), Z).reshape(-1, 3)  # below the horizon: hold alt 0.05
        s_isl = self.atm.sky(self.toa_isl, R, d)
        s_ag = self.atm.sky(self.toa_none, R, d)
        tw = np.zeros(len(d))
        if p["twilight"]:
            s_sun = self.atm.sky(self.toa_sun, R, d)
            tw = rad_to_sb(v_band(s_sun["inscatter"] - s_ag["inscatter"], grid))
        tw = np.maximum(tw, 0.0).reshape(A.shape)
        shp = A.shape
        self.T = v_band(s_isl["T"], grid, self.w_isl).reshape(shp)
        ins_isl = rad_to_sb(v_band(s_isl["inscatter"] - s_ag["inscatter"], grid)).reshape(shp)
        ag = rad_to_sb(v_band(s_ag["airglow"] + s_ag["inscatter"], grid)).reshape(shp)
        lp, dome = light_pollution(A, Z, p)
        # zenith calibration: s * ag_z + ins_isl_z + T_z * isl_z + lp0 = target, lp0 = frac * target
        zd = R.T @ np.array([0.0, 0.0, 1.0])
        isl_z = self._isl_scatter_sb(zd)
        target = 10 ** (-0.4 * p["zenith_mag"])
        iz = len(self.alts) - 1
        lp0 = p["lp_zenith_frac"] * target
        s = (target - lp0 - tw[iz].mean() - ins_isl[iz].mean() - self.T[iz].mean() * isl_z) / ag[iz].mean()
        if s <= 0:
            raise ValueError("zenith target brighter than airglow + starlight can reach")
        self.airglow_scale = float(s)
        self.bg = s * ag + ins_isl + lp0 * lp + tw
        if p["horizon_monotonic"] > 0:
            # the RT natural sky dims toward the horizon below ~10 deg (extinction wins over the
            # longer path); a Bortle 3 sky brightens toward the horizon, so below
            # horizon_monotonic deg hold the brightest value met higher up at that azimuth
            top = np.flatnonzero(self.alts <= p["horizon_monotonic"])[-1]
            self.bg[: top + 1] = np.maximum.accumulate(self.bg[top::-1], axis=0)[::-1]
        self.twilight_frac = tw / self.bg
        self.lp_frac = lp0 * lp / self.bg
        self.dome = dome
        self.target = target
        log(f"airglow scale {s:.2f}; zenith: airglow+scatter {s * ag[iz].mean() / target:.2f}, "
            f"scattered starlight {ins_isl[iz].mean() / target:.3f}, starlight {self.T[iz].mean() * isl_z / target:.3f}, "
            f"LP {lp0 / target:.2f}")
        ih = np.argmin(np.abs(self.alts - 3.0))
        log(f"Sun alt {self.sun_altaz[0]:.2f} az {self.sun_altaz[1]:.1f}: twilight share zenith {tw[iz].mean() / self.bg[iz].mean():.4f}, "
            f"alt 3 max over az {self.twilight_frac[ih].max():.3f}")

    def _isl_scatter_sb(self, d_icrs):
        pix = hp.vec2pix(SCATTER_NSIDE, *np.atleast_2d(d_icrs).T)
        return float(self.sb_scatter[pix][0])

    def sample(self, name, alt, az):
        tab = getattr(self, name)
        ia = np.interp(np.clip(alt, self.alts[0], self.alts[-1]), self.alts, np.arange(len(self.alts)))
        iz = np.asarray(az, float) % 360 / 2.0
        if name == "T":  # steep near the horizon: linear
            return map_coordinates(tab, [ia.ravel(), iz.ravel()], order=1, mode="nearest").reshape(np.shape(alt))
        # cubic spline: linear interpolation of the 2-deg azimuth grid shows as banding in smooth gradients
        out = map_coordinates(tab, [ia.ravel(), iz.ravel()], order=3, mode="nearest").reshape(np.shape(alt))
        return np.maximum(out, 0.0)


# ----------------------------------------------------------------------------- image layers
def smooth_varying(img, sigma_center_px, kmap, kmin, kmax, nlev=4):
    """Gaussian blur with sigma = sigma_center_px * k (stereographic scale factor k per pixel),
    interpolated between nlev fixed-sigma blurs."""
    ks = np.linspace(kmin, kmax, nlev)
    outs = [gaussian_filter(img, sigma_center_px * k, mode="nearest") for k in ks]
    t = np.interp(kmap, ks, np.arange(nlev))
    i0 = np.clip(np.floor(t).astype(int), 0, nlev - 2)
    w = t - i0
    out = np.zeros_like(img)
    for i in range(nlev - 1):
        m = i0 == i
        out[m] = (1 - w[m]) * outs[i][m] + w[m] * outs[i + 1][m]
    return out


UNDETECTED_NSIDE = 128


def undetected_map(vec_icrs, flux, p):
    """All-sky RING map (V flux per arcsec^2) of the undetected share of the catalogued stars,
    smoothed to p['undetected_fwhm_arcmin']. Only a few V ~ 7-9.5 stars fall in a square degree,
    so at the eye-like widths used for the Gaia integrated light each one would show up as a
    separate blob brighter than the sky; a single sub-threshold star is not seen as a patch."""
    npix = hp.nside2npix(UNDETECTED_NSIDE)
    pix = hp.vec2pix(UNDETECTED_NSIDE, *vec_icrs.T)
    m = np.bincount(pix, weights=flux, minlength=npix) / (4 * np.pi / npix / ARCSEC2_SR)
    m = hp.smoothing(m, fwhm=np.radians(p["undetected_fwhm_arcmin"] / 60.0))
    return np.maximum(m, 0.0)


def glow_layer(frame, R, cells, und_map, p):
    """Integrated starlight above the atmosphere, V flux per arcsec^2, (H, W): the HEALPix cell
    map (surface brightness) interpolated bilinearly at each pixel and smoothed to
    p['eye_fwhm_arcmin'] in quadrature with the cell size, plus the smoothed undetected share
    of the catalogued stars (undetected_map, interpolated)."""
    W, H, f, x0, y0 = geo.FRAMES[frame]
    order, idx, fv = cells
    nside = 2 ** order
    cell_as2 = 4 * np.pi / hp.nside2npix(nside) / ARCSEC2_SR
    m = np.zeros(hp.nside2npix(nside), np.float64)
    m[idx] = fv / cell_as2
    py_, px_ = np.mgrid[0:H, 0:W].astype(float) + 0.5
    alt, az = geo.pix_to_altaz(px_, py_, f, x0, y0)
    d = frames.altaz_to_vec(alt, az).reshape(-1, 3) @ R  # horizon -> ICRS (R orthonormal)
    theta = np.arccos(np.clip(d[:, 2], -1, 1))
    phi = np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)
    sb_cells = hp.get_interp_val(m, theta, phi, nest=True).reshape(H, W)
    sb_und = hp.get_interp_val(und_map, theta, phi).reshape(H, W)
    del d, theta, phi
    k = 1.0 + (((px_ - x0) / f) ** 2 + ((y0 - py_) / f) ** 2) / 4  # stereographic scale (px per rad = f k)
    rad_px = np.pi / 180.0 / 60.0 * f  # px per arcmin at k = 1
    cell_arcmin = np.sqrt(cell_as2) / 60.0
    if p["spike_clip"] > 0:
        # a G ~ 10-11 star alone in a cell makes it stand out against the local mean; spread
        # the cell excess above spike_clip x the 30' mean over 30' (flux conserving) so the
        # integrated light does not show the single-star grain
        sig30 = 30.0 / 2.3548 * rad_px
        ref = smooth_varying(sb_cells, sig30, k, k.min(), k.max())
        excess = np.maximum(sb_cells - p["spike_clip"] * ref, 0.0)
        sb_cells = sb_cells - excess + smooth_varying(excess, sig30, k, k.min(), k.max())
    extra = np.sqrt(max(p["eye_fwhm_arcmin"] ** 2 - cell_arcmin ** 2, 0.0))
    if extra > 0:
        sb_cells = smooth_varying(sb_cells, extra / 2.3548 * rad_px, k, k.min(), k.max())
    return sb_cells + sb_und, k


def render_stars(img_lin, px, py, phi, tints, sigma, halo, H, W, halo_w=None, halo_r=2.0, wing_sigma=None,
                 wing=0.0):
    """Add stars in linear light: peak phi, profile (1 - wing) * Gaussian(sigma) + wing *
    Gaussian(wing_sigma) (wing_sigma per star, None: a single Gaussian), plus a soft circular halo
    halo * halo_w * phi * (1 + r^2 / (halo_r sigma)^2)^-1.5 centred on the same point (halo_w
    in [0, 1] per star selects the bright ones). sigma is a scalar or one value per star."""
    n = len(px)
    if halo_w is None:
        halo_w = np.ones(n)
    sigmas = np.broadcast_to(np.asarray(sigma, float), (n,))
    if wing_sigma is None:
        wing_sigma, wing = sigmas, 0.0
    wsig = np.broadcast_to(np.asarray(wing_sigma, float), (n,))
    for x, y, a, c, hw, sigma, sw in zip(px, py, phi, tints, halo_w, sigmas, wsig):
        rc = max(sigma, sw if wing > 0 else 0.0) * np.sqrt(2 * np.log(max(a / 1e-4, 1.0)))
        ah = halo * hw * a
        rh = halo_r * sigma * np.sqrt(max((ah / 1e-3) ** (2 / 3) - 1, 0.0)) if ah > 1e-3 else 0.0
        r = int(np.ceil(max(rc, rh, 3 * sigma))) + 1
        i0, i1 = max(int(x) - r, 0), min(int(x) + r + 1, W)
        j0, j1 = max(int(y) - r, 0), min(int(y) + r + 1, H)
        if i0 >= i1 or j0 >= j1:
            continue
        yy, xx = np.mgrid[j0:j1, i0:i1].astype(float) + 0.5
        r2 = (xx - x) ** 2 + (yy - y) ** 2
        prof = a * (1 - wing) * np.exp(-0.5 * r2 / sigma ** 2)
        if wing > 0:
            prof = prof + a * wing * np.exp(-0.5 * r2 / sw ** 2)
        if rh > 0:
            prof = prof + ah * (1 + r2 / (halo_r * sigma) ** 2) ** -1.5
        img_lin[j0:j1, i0:i1] += prof[..., None] * c[None, None, :]


def star_tint(teff):
    """Mildly saturated blackbody colour, unit luminance (linear sRGB)."""
    from nightsky.spectral import default_grid, to_xyz
    g = default_grid()
    t = np.atleast_1d(teff)
    e = spectral.blackbody_irradiance(np.ones_like(t), t, g)
    xyz = to_xyz(e, g)
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    rgb = np.clip(xyz @ m.T, 0, None)
    rgb /= np.maximum(rgb @ np.array([0.2126, 0.7152, 0.0722]), 1e-30)[:, None]
    rgb = 1 + STAR_SAT * (rgb - 1)
    rgb = np.clip(rgb, 0, None)
    return rgb / (rgb @ np.array([0.2126, 0.7152, 0.0722]))[:, None]


def srgb_encode(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def srgb_decode(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


# ----------------------------------------------------------------------------- frame
def render_frame(frame, pts, sky, cells, p, outdir, space=False, log=print, suffix=""):
    W, H, f, x0, y0 = geo.FRAMES[frame]
    t0 = time.time()
    # all stars above the horizon: extinction, background (low-res sky + glow at the scatter
    # map resolution) and detection; the undetected share goes into the glow
    ia = np.flatnonzero(((pts["alt"] > 0) | space) & np.isfinite(pts["v"]))  # no Earth in the space frame
    T = np.ones(len(ia)) if space else sky.sample("T", pts["alt"][ia], pts["az"][ia])
    v_obs = pts["v"][ia] - 2.5 * np.log10(np.maximum(T, 1e-30))
    bg_star = (0.0 if space else sky.sample("bg", pts["alt"][ia], pts["az"][ia])) + T * sky_star_isl(sky, pts, ia)
    lb = CD_M2_PER_V0_ARCSEC2 * np.maximum(bg_star, 1e-12)
    pdet = detect_prob(v_obs, lb, p, sky.vis_factor)
    flux = 10 ** (-0.4 * pts["v"][ia])  # above the atmosphere (the glow layer gets T below)
    und = undetected_map(pts["vec_icrs"][ia], flux * (1 - pdet), p)
    px, py = geo.altaz_to_pix(pts["alt"][ia], pts["az"][ia], f, x0, y0)
    inside = (px > -20) & (px < W + 20) & (py > -20) & (py < H + 20)
    ii = ia[inside]
    px, py, T, v_obs, pdet = px[inside], py[inside], T[inside], v_obs[inside], pdet[inside]
    # glow: integrated starlight plus the undetected share of the catalogued stars
    sb_top, k = glow_layer(frame, sky.R, cells, und, p)
    py_, px_ = np.mgrid[0:H, 0:W].astype(float) + 0.5
    galt, gaz = geo.pix_to_altaz(px_, py_, f, x0, y0)
    if space:
        sb = sb_top
        bg = np.zeros_like(sb)
        lpf = np.zeros_like(sb)
    else:
        Tm = sky.sample("T", np.maximum(galt, 0.0), gaz)
        bg = sky.sample("bg", galt, gaz)
        lpf = sky.sample("lp_frac", galt, gaz)
        sb = np.where(galt > 0, Tm * sb_top, 0.0) + bg
    log(f"{frame}{' space' if space else ''}: {len(ii)} catalogue stars in frame, {pdet.sum():.0f} detected, "
        f"p50 {int((pdet > 0.5).sum())}, layers {time.time() - t0:.0f} s")
    stats = dict(frame=frame, space=space, n_in_frame=int(len(ii)), n_detected=float(pdet.sum()),
                 n_p50=int((pdet > 0.5).sum()), sb_min_mag=float(-2.5 * np.log10(sb[galt > 5].min() + 1e-30)),
                 sb_max_mag=float(-2.5 * np.log10(sb[galt > 5].max() + 1e-30)))
    tints = star_tint(pts["teff"][ii])
    vis = pdet > 1e-3
    outs = {}
    for tag, st in STRETCHES.items():
        x = sb / sky.target
        d = np.clip(st["g0"] + st["g1"] * np.log10(np.maximum(x, 1e-6)), st["gfloor"], 1.0)
        if space:
            d = np.where(sb > 0, d, 0.0)
        tint = SKY_TINT[None, None, :] * (1 - lpf[..., None]) + (SKY_TINT * LP_TINT)[None, None, :] * lpf[..., None]
        lin = srgb_decode(d)[..., None] * tint
        phi = st["star_gain"] * pdet * 10 ** (-0.4 * st["star_alpha"] * (v_obs - st["star_ref"]))
        hw = np.clip(p["halo_vmax"] - v_obs + 0.5, 0.0, 1.0)  # 0 fainter than halo_vmax + 0.5, 1 brighter than - 0.5
        sigma = p["psf_sigma_main"] if frame == "main" else p["psf_sigma_teapot"]
        wsig, wing = None, 0.0
        if p["sized_stars"]:
            scale = sigma / p["psf_sigma_teapot"]
            lns = np.log(STAR_SIZE_SIGMA[-1]) + p["star_size_gamma"] * np.log(STAR_SIZE_SIGMA / STAR_SIZE_SIGMA[-1])
            sized = scale * np.exp(np.interp(v_obs, STAR_SIZE_V, lns))[vis]
            if p["star_wing"] >= 1.0:  # the whole star takes the brightness-dependent width
                sigma = sized
            else:  # fixed sharp core, brightness-dependent wing
                wsig, wing = sized, p["star_wing"]
        render_stars(lin, px[vis], py[vis], phi[vis], tints[vis], sigma, st["halo"], H, W, hw[vis],
                     p["halo_r_sigma"], wsig, wing)
        # triangular dither (+-1 LSB) so the smooth horizon gradients do not band in 8 bits
        rng = np.random.default_rng(1)
        dith = rng.random(lin.shape[:2], np.float32) - rng.random(lin.shape[:2], np.float32)
        enc = srgb_encode(lin) * 255
        dith *= enc.max(axis=-1) > 2.0  # keep black black
        rgb8 = np.clip(enc + 0.5 + dith[..., None], 0, 255).astype(np.uint8)
        name = {"main": "chiefland_sky_sw_bortle3", "teapot": "chiefland_sky_sw_bortle3_teapot"}[frame]
        if space:
            name = "chiefland_sky_teapot_space" if frame == "teapot" else "chiefland_sky_sw_space"
        path = Path(outdir) / f"{name}{suffix}{tag}.jpg"
        Image.fromarray(rgb8).save(path, quality=95, subsampling=0, optimize=True)
        Image.fromarray(rgb8).save(path.with_suffix(".png"))
        outs[tag] = str(path)
    np.savez_compressed(Path(outdir) / f"{frame}{'_space' if space else ''}{suffix}_linear.npz",
                        sb=sb.astype(np.float32), sb_top=sb_top.astype(np.float32))
    stats["files"] = outs
    stats["seconds"] = time.time() - t0
    return stats


def sky_star_isl(sky, pts, ii):
    """Starlight surface brightness at the star positions from the nside-64 map (V=0/arcsec^2)."""
    v = pts["vec_icrs"][ii]
    return sky.sb_scatter[hp.vec2pix(SCATTER_NSIDE, *v.T)]


def detect_prob(v_obs, lb, p, factor):
    from nightsky.camera import limiting_vmag
    vlim = limiting_vmag(lb, factor)
    return 0.5 * (1 + erf((vlim - v_obs) / (np.sqrt(2) * p["sigma_mag"])))


def calibrate_factor(p, sky):
    """Detection factor for limiting magnitude p['zenith_limit'] on the zenith sky."""
    from nightsky.camera import limiting_vmag
    lb = CD_M2_PER_V0_ARCSEC2 * sky.target
    return float(10 ** (-0.4 * (p["zenith_limit"] - limiting_vmag(lb, 1.0))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--frames", default="main,teapot")
    ap.add_argument("--space", action="store_true", help="also render the no-atmosphere teapot frame")
    ap.add_argument("--set", action="append", default=[], help="PARAMS override key=value")
    ap.add_argument("--suffix", default="", help="appended to the output names (before the stretch tag)")
    a = ap.parse_args(argv)
    p = dict(PARAMS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        p[k] = type(PARAMS[k])(v)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "render.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    grid = spectral.default_grid()
    R = icrs_to_horizon_matrix()
    pts = load_points()
    alt, az = geo.icrs_to_altaz(pts["ra"], pts["dec"])
    pts["alt"], pts["az"] = alt, az
    pts["vec_icrs"] = np.array(hp.ang2vec(pts["ra"], pts["dec"], lonlat=True))
    log(f"{len(pts['v'])} catalogue stars (V < 9.5), {(alt > 0).sum()} above the horizon ({time.time() - t0:.0f} s)")
    i9, f9, _ = load_diffuse(9)
    log(f"order-9 cells {len(i9)}, total V {-2.5 * np.log10(f9.sum()):.2f}")
    sky = SkyModel(p, grid, R, (9, i9, f9), log=log)
    sky.vis_factor = calibrate_factor(p, sky)
    log(f"detection factor {sky.vis_factor:.3f}")
    stats = []
    for fr in a.frames.split(","):
        cells = (9, i9, f9) if fr == "main" else (10,) + load_diffuse(10)[:2]
        stats.append(render_frame(fr, pts, sky, cells, p, out, log=log, suffix=a.suffix))
        if a.space and fr == "teapot":
            stats.append(render_frame(fr, pts, sky, cells, p, out, space=True, log=log, suffix=a.suffix))
    meta = dict(params=p, airglow_scale=sky.airglow_scale, aerosol_tau550=sky.tau_aer, detection_factor=sky.vis_factor,
                stats=stats, R_icrs_to_horizon=R.tolist())
    (out / f"render_meta{a.suffix}.json").write_text(json.dumps(meta, indent=1))
    log(f"done {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
