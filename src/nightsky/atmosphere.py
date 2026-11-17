"""Numerical spherical-shell planetary atmosphere: extinction, scattered light and airglow.

Units: radiance W m^-2 sr^-1 nm^-1, irradiance W m^-2 nm^-1 (bin averages on the caller's
SpectralGrid), lengths m. Directions are unit vectors in the HORIZON frame (x=N, y=W, z=Up)
unless stated; TOASky inputs are rotated with R = frames.celestial_to_horizon(H, lat).

Model
- Planet: sphere of radius R; ground (Lambertian, spectral albedo) at the preset altitude,
  observer `eye_m` above it; atmosphere up to `top_m`. Density profiles use absolute
  altitude above the planet radius.
- Rayleigh: Bodhaine et al. (1999, eq. 29) air cross section and King factor (their
  eqs. 5, 13-15) giving the depolarisation rho; Chandrasekhar phase function with
  gamma = rho / (2 - rho). Density: US Standard Atmosphere 1976 (piecewise lapse rates,
  geopotential = geometric altitude) or exponential. CO2 (Mars): air cross section x 2.4
  and rho = 0.075 (Sneep & Ubachs 2005).
- Ozone: Chappuis/Huggins cross sections (Serdyuchenko et al. 2014, 233 K, 10-nm averages as
  tabulated by Bruneton 2017), Gaussian layer at 22.5 km (sigma 6.5 km) scaled to the column
  in Dobson units.
- Aerosol: Angstrom law tau(lambda) = tau_550 (lambda/550)^-alpha, exponential profile above
  the ground (scale height ~1.2 km), spectral single-scattering albedo, double
  Henyey-Greenstein phase function (forward g1, backward g2, weight w1); optional thin
  stratospheric layer.
- Radiative transfer (Hillaire 2020, EGSR "A scalable and production ready sky and atmosphere
  rendering technique"; parameterisation of Bruneton 2017): transmittance LUT T(r, mu);
  multiple-scattering LUT Psi(h, mu_s) (mean first-order radiance over `ms_dirs` directions
  with an isotropic-phase geometric series 1/(1 - f_ms)) plus the l=2 (quadrupole) moments of
  the first-order field, so that second-order Rayleigh scattering uses the Rayleigh phase
  function (this lowers the zenith sky at sun -8 deg by ~6% relative to the isotropic form);
  the Lambertian ground reflects the direct beam and the diffuse sky irradiance (from a
  black-ground first pass on coarse grids) with the interreflection factor 1/(1 - albedo S),
  S = spherical albedo of the atmosphere; kernel LUT of
  the phase-free single-scattering integrals K_R, K_M and the multiple-scattering integral
  K_MS for unit TOA irradiance as a function of (view elevation, source elevation,
  delta-azimuth). The scattering angle is constant along a ray, so the inscatter of a source
  is E * [P_R(theta) K_R + P_M(theta) K_M + K_MS] with the phase functions evaluated exactly.
  Psi uses a 0.5-deg grid in local solar elevation and the kernel a 0.25-deg source-elevation
  grid for -20..10 deg: the twilight sky falls by ~3x per degree of solar depression, and linear
  interpolation on the earlier 2.5-deg / 1-deg grids put the twilight sky 10-45% too bright.
  Below sun -10 deg the lit atmosphere seen from a shadowed point is a thin sliver near its
  horizon, which 64 gathering directions undersample (deep-twilight red low by up to 2x; raise
  `ms_dirs` at the cost of build time).
  Sources: points are merged into 0.9-deg clumps (HEALPix nside 64, flux-weighted
  direction); the k_exact brightest clumps get the exact phase function per pixel (sharp
  aureole); the other clumps, the diffuse map and the airglow are binned into HEALPix virtual
  sources (nside 16) evaluated on a coarse internal sky grid and interpolated. Sources down to -20 deg elevation light the
  upper atmosphere (twilight).
- Airglow: thin layer at ~95 km, van Rhijn factor 1/sqrt(1 - (r_obs/r_layer)^2 cos^2 el),
  times the transmittance; lines OI 557.7 (250 R), Na D 589 (50 R), OI 630.0/636.4 (100/33 R),
  OH Meinel bands in the red, and a pseudo-continuum rising to the red (Leinert et al. 1998;
  Hanuschik 2003 for the relative OH strengths). Calibrated so the zenith airglow above the
  atmosphere is V = 22.55 mag/arcsec^2 (Bessell 1990 V filter): ~22.65 observed at 2.5 km,
  ~22.8 at sea level. The airglow also enters the scattering as a diffuse source.

Approximations: no refraction, no polarisation, no ground shadowing of the sky by terrain,
isotropic third and higher scattering orders (Hillaire 2020), sharp planetary shadow
(point-like sources),
the sky below the horizon contains only path radiance (the ground itself is the
foreground's job). Aerial perspective uses a distance-resolved version of the kernel LUT.
"""
import numpy as np
from numba import njit, prange

from . import frames, spectral
from .sky import PointSources

K_B = 1.380649e-23
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
DU_M2 = 2.6867e20  # molecules m^-2 per Dobson unit
RAYLEIGH_PHOTONS = 1e10 / (4 * np.pi)  # photons s^-1 m^-2 sr^-1 per rayleigh
ES_MIN_DEG = -20.0
CLUMP_NSIDE = 64

# Ozone absorption cross sections (m^2), 360-830 nm in 10-nm bins (Serdyuchenko+2014, 233 K,
# binned by Bruneton 2017).
_O3_WL = 365.0 + 10.0 * np.arange(48)
_O3_XS = np.array([
    1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27, 8.451e-27,
    1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26, 1.48e-25, 1.602e-25,
    2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25, 4.672e-25, 4.398e-25, 4.701e-25,
    5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25, 2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25,
    1.209e-25, 9.423e-26, 7.455e-26, 6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26,
    2.451e-26, 2.801e-26, 2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27])

# Bessell (1990) V passband
_V_WL = np.arange(470.0, 701.0, 10.0)
_V_S = np.array([0.0, 0.030, 0.163, 0.458, 0.780, 0.967, 1.000, 0.973, 0.898, 0.792, 0.684, 0.574,
                 0.461, 0.359, 0.270, 0.197, 0.135, 0.081, 0.045, 0.025, 0.017, 0.013, 0.009, 0.0])

# Night airglow: (wavelength nm, zenith intensity R) above the atmosphere.
AIRGLOW_LINES = [(557.7, 250.0), (589.3, 50.0), (630.0, 100.0), (636.4, 33.0),
                 (625.0, 20.0), (650.0, 30.0), (698.0, 80.0), (731.0, 200.0), (775.0, 250.0)]
_AG_CONT_WL = np.array([380., 420., 450., 500., 550., 600., 650., 700., 780.])
_AG_CONT_SHAPE = np.array([0.8, 0.9, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7, 2.2])  # relative R/nm
AIRGLOW_TOA_V = 22.55

_EARTH_ALBEDO = ((380, 450, 500, 550, 600, 650, 700, 750, 780),
                 (0.05, 0.06, 0.08, 0.11, 0.11, 0.12, 0.22, 0.32, 0.33))  # grass/soil mix
_MARS_ALBEDO = ((380, 450, 500, 550, 600, 650, 700, 780), (0.06, 0.08, 0.11, 0.16, 0.24, 0.29, 0.31, 0.32))
_MARS_SSA = ((380, 420, 460, 500, 550, 600, 650, 700, 780),
             (0.72, 0.78, 0.84, 0.89, 0.93, 0.96, 0.97, 0.975, 0.98))  # Wolff et al. 2009
_MARS_G1 = ((380, 500, 600, 780), (0.84, 0.80, 0.74, 0.68))  # stronger forward lobe in the blue

_BASE = dict(
    planet_radius_m=6.371e6, top_m=100e3, altitude_m=0.0, eye_m=2.0, gravity=9.80665,
    rayleigh_profile="us76", rayleigh_scale_height_m=8.0e3, rayleigh_n0_m3=2.5469e25,
    rayleigh_gas="air", rayleigh_xsec=None, depolarization=None, rayleigh_density=None,
    ozone_du=300.0, ozone_peak_m=22.5e3, ozone_width_m=6.5e3,
    aerosol_tau550=0.06, aerosol_alpha=1.3, aerosol_scale_height_m=1.2e3, aerosol_ssa=0.93,
    aerosol_g1=0.80, aerosol_g2=-0.30, aerosol_w1=0.95,
    strat_tau550=0.005, strat_peak_m=20e3, strat_width_m=4e3,
    ground_albedo=_EARTH_ALBEDO, airglow_scale=1.0, airglow_height_m=95e3,
    k_exact=16, scatter_nside=16, ms_dirs=64,
)

PRESETS = {
    "none": dict(vacuum=True),
    "earth_clear": dict(),
    "earth_hazy": dict(aerosol_tau550=0.4, aerosol_alpha=1.0, aerosol_ssa=0.88, aerosol_scale_height_m=1.5e3,
                       aerosol_g1=0.78),
    "earth_observatory": dict(altitude_m=2500.0, aerosol_tau550=0.015, aerosol_ssa=0.95, ozone_du=280.0),
    "mars": dict(planet_radius_m=3.3895e6, top_m=120e3, gravity=3.71, rayleigh_profile="exponential",
                 rayleigh_scale_height_m=11.1e3, rayleigh_n0_m3=610.0 / (K_B * 210.0), rayleigh_gas="co2",
                 ozone_du=0.0, aerosol_tau550=0.6, aerosol_alpha=0.0, aerosol_scale_height_m=11e3,
                 aerosol_ssa=_MARS_SSA, aerosol_g1=_MARS_G1, aerosol_g2=-0.25, aerosol_w1=0.92,
                 strat_tau550=0.0, ground_albedo=_MARS_ALBEDO, airglow_scale=0.02, airglow_height_m=60e3),
}


# ----------------------------------------------------------------------------- spectra
def rayleigh_xsec_air(wl_nm):
    """Bodhaine et al. (1999) eq. 29: air Rayleigh cross section (m^2), 360 ppm CO2."""
    lam = np.asarray(wl_nm, float) * 1e-3
    l2 = lam ** 2
    return 1e-32 * (1.0455996 - 341.29061 / l2 - 0.90230850 * l2) / (1 + 0.0027059889 / l2 - 85.968563 * l2)


def king_factor_air(wl_nm, co2_percent=0.04):
    lam = np.asarray(wl_nm, float) * 1e-3
    fn2 = 1.034 + 3.17e-4 / lam ** 2
    fo2 = 1.096 + 1.385e-3 / lam ** 2 + 1.448e-4 / lam ** 4
    return (78.084 * fn2 + 20.946 * fo2 + 0.934 + 1.15 * co2_percent) / (78.084 + 20.946 + 0.934 + co2_percent)


def _spec(val, grid):
    """Scalar, (wl, values) table, or callable(wl_nm) -> (n,) bin average."""
    if callable(val):
        return np.asarray(grid.bin_average(val), float)
    if isinstance(val, (tuple, list)) and len(val) == 2 and np.ndim(val[0]) == 1:
        wl, v = np.asarray(val[0], float), np.asarray(val[1], float)
        return grid.bin_average(lambda x: np.interp(x, wl, v))
    return np.full(grid.n, float(val))


def v_filter_bins(grid):
    """(n,) bin-averaged Bessell (1990) V response."""
    return grid.bin_average(lambda x: np.interp(x, _V_WL, _V_S, left=0.0, right=0.0))


def v_mag_arcsec2(spec, grid):
    """V surface brightness (mag/arcsec^2) of radiance (..., n), V-filter weighted mean flux density.

    Lines are weighted by the V response instead of the 550-nm bin value used by
    spectral.radiance_to_mag_arcsec2 (that one reads the 540-560 nm bin holding OI 557.7)."""
    s = v_filter_bins(grid) * grid.dwl
    f = np.asarray(spec) @ s / s.sum() * spectral.ph.ARCSEC2_IN_SR
    return -2.5 * np.log10(np.maximum(f, 1e-300) / spectral.ph.F_V0_550NM)


def airglow_zenith(grid, scale=1.0):
    """(n,) zenith airglow radiance above the atmosphere, calibrated to V = AIRGLOW_TOA_V."""
    fine = spectral.SpectralGrid(np.arange(380.0, 781.0, 1.0))

    def build(g, c0):
        e_ph = lambda wl: H_PLANCK * C_LIGHT / (wl * 1e-9)
        cont = g.bin_average(lambda wl: c0 * np.interp(wl, _AG_CONT_WL, _AG_CONT_SHAPE) * RAYLEIGH_PHOTONS * e_ph(wl))
        lines = sum(g.line(w, r * RAYLEIGH_PHOTONS * e_ph(w)) for w, r in AIRGLOW_LINES)
        return cont + lines

    s = v_filter_bins(fine) * fine.dwl
    lines_v = build(fine, 0.0) @ s / s.sum()
    cont_v = (build(fine, 1.0) - build(fine, 0.0)) @ s / s.sum()
    target = 10 ** (-0.4 * AIRGLOW_TOA_V) * spectral.ph.F_V0_550NM / spectral.ph.ARCSEC2_IN_SR
    c0 = max(target - lines_v, 0.0) / cont_v
    return scale * build(grid, c0)


def us76_number_density(h_m):
    """US Standard Atmosphere 1976 number density (m^-3), exponential above 86 km."""
    hb = np.array([0, 11, 20, 32, 47, 51, 71, 84.852]) * 1e3
    lapse = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0]) * 1e-3
    g0M_R = 9.80665 * 0.0289644 / 8.3144598
    tb, pb = [288.15], [101325.0]
    for i in range(len(lapse)):
        dh = hb[i + 1] - hb[i]
        t1 = tb[i] + lapse[i] * dh
        p1 = pb[i] * (np.exp(-g0M_R * dh / tb[i]) if lapse[i] == 0 else (tb[i] / t1) ** (g0M_R / lapse[i]))
        tb.append(t1)
        pb.append(p1)
    tb, pb = np.array(tb), np.array(pb)
    h = np.clip(np.asarray(h_m, float), 0, None)
    i = np.clip(np.searchsorted(hb, h, side="right") - 1, 0, len(lapse))
    top = i == len(lapse)
    il = np.minimum(i, len(lapse) - 1)
    dh = h - hb[i]
    t = np.where(top, tb[-1], tb[i] + lapse[il] * dh)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_iso = pb[i] * np.exp(-g0M_R * dh / tb[i])
        p_grad = pb[i] * (tb[i] / t) ** (g0M_R / np.where(lapse[il] == 0, 1, lapse[il]))
    p = np.where(top | (lapse[il] == 0), p_iso, p_grad)
    return p / (K_B * t)


# ----------------------------------------------------------------------------- numba core
@njit(inline="always")
def _row(h, h0, dh, nh):
    x = (h - h0) / dh
    if x <= 0.0:
        return 0, 0.0
    if x >= nh - 1:
        return nh - 2, 1.0
    i = int(x)
    return i, x - i


@njit(inline="always")
def _tlut_index(r, mu, rb, rt, nr, nmu):
    """Bruneton (2017) parameterisation; returns (ir, fr, imu, fmu, ok)."""
    if r > rt:
        r = rt
    if r < rb:
        r = rb
    hh = np.sqrt(rt * rt - rb * rb)
    rho = np.sqrt(max(r * r - rb * rb, 0.0))
    disc = r * r * (mu * mu - 1.0) + rb * rb
    if mu < 0.0 and disc >= 0.0:
        return 0, 0.0, 0, 0.0, False
    d = max(-r * mu + np.sqrt(max(r * r * (mu * mu - 1.0) + rt * rt, 0.0)), 0.0)
    dmin = rt - r
    dmax = rho + hh
    xm = (d - dmin) / max(dmax - dmin, 1e-9)
    xr = rho / hh
    xr = min(max(xr, 0.0), 1.0) * (nr - 1)
    xm = min(max(xm, 0.0), 1.0) * (nmu - 1)
    ir = min(int(xr), nr - 2)
    im = min(int(xm), nmu - 2)
    return ir, xr - ir, im, xm - im, True


@njit(inline="always")
def _tau_at(tl, ir, fr, im, fm, b):
    return ((1 - fr) * ((1 - fm) * tl[ir, im, b] + fm * tl[ir, im + 1, b])
            + fr * ((1 - fm) * tl[ir + 1, im, b] + fm * tl[ir + 1, im + 1, b]))


@njit(inline="always")
def _phase_r(c, gam):
    return 3.0 / (16.0 * np.pi * (1 + 2 * gam)) * ((1 + 3 * gam) + (1 - gam) * c * c)


@njit(inline="always")
def _hg(c, g):
    d = 1.0 + g * g - 2.0 * g * c
    return (1.0 - g * g) / (4.0 * np.pi * d * np.sqrt(d))


@njit(parallel=True, cache=True)
def _build_tlut(rb, rt, rplan, h0, dh, ext, nr, nmu, nsteps):
    nb = ext.shape[1]
    nh = ext.shape[0]
    out = np.zeros((nr, nmu, nb))
    hh = np.sqrt(rt * rt - rb * rb)
    for i in prange(nr):
        rho = hh * i / (nr - 1)
        r = np.sqrt(rho * rho + rb * rb)
        for j in range(nmu):
            dmin = rt - r
            dmax = rho + hh
            d = dmin + (dmax - dmin) * j / (nmu - 1)
            mu = 1.0 if d <= 0 else (hh * hh - rho * rho - d * d) / (2 * r * d)
            mu = min(max(mu, -1.0), 1.0)
            dt = d / nsteps
            for k in range(nsteps):
                t = (k + 0.5) * dt
                rr = np.sqrt(r * r + 2 * r * mu * t + t * t)
                ih, f = _row(rr - rplan, h0, dh, nh)
                for b in range(nb):
                    out[i, j, b] += ((1 - f) * ext[ih, b] + f * ext[ih + 1, b]) * dt
    return out


@njit(parallel=True, cache=True)
def _build_ms(rb, rt, rplan, h0, dh, ext, sr, sm, tl, hs, ess, wdirs, albedo, gam, g1, g2, w1, nsteps,
              eg_es, eg, refl):
    """Psi (isotropic multiple-scattering source, Hillaire 2020) and the quadrupole moments
    (Qxx, Qzz, Qxz in the local frame z = up, x = solar azimuth) of the first-order radiance
    field, per unit TOA irradiance. Ground: Lambertian, lit by the direct beam plus the diffuse
    irradiance eg(local source elevation eg_es), times the interreflection factor refl."""
    nb = ext.shape[1]
    nh = ext.shape[0]
    nr, nmu = tl.shape[0], tl.shape[1]
    nhs, nes, nd = hs.shape[0], ess.shape[0], wdirs.shape[0]
    neg = eg_es.shape[0]
    psi = np.zeros((nhs, nes, nb))
    quad = np.zeros((3, nhs, nes, nb))
    for idx in prange(nhs * nes):
        i = idx // nes
        j = idx % nes
        r = min(max(rplan + hs[i], rb + 1.0), rt - 1.0)
        mus = np.sin(np.radians(ess[j]))
        sx, sz = np.sqrt(max(1 - mus * mus, 0.0)), mus
        l2 = np.zeros(nb)
        fms = np.zeros(nb)
        tau = np.zeros(nb)
        lp = np.zeros(nb)
        fp = np.zeros(nb)
        pm = np.zeros(nb)
        qa = np.zeros((3, nb))
        for w in range(nd):
            wx, wy, wz = wdirs[w, 0], wdirs[w, 1], wdirs[w, 2]
            c = wx * sx + wz * sz
            pr = _phase_r(c, gam[0])
            for b in range(nb):
                pm[b] = w1[b] * _hg(c, g1[b]) + (1 - w1[b]) * _hg(c, g2[b])
            disc_g = r * r * (wz * wz - 1) + rb * rb
            hit = wz < 0 and disc_g >= 0
            if hit:
                length = -r * wz - np.sqrt(disc_g)
            else:
                length = -r * wz + np.sqrt(max(r * r * (wz * wz - 1) + rt * rt, 0.0))
            for b in range(nb):
                tau[b] = 0.0
                lp[b] = 0.0
                fp[b] = 0.0
            for k in range(nsteps):
                t0 = length * (k / nsteps) ** 2
                t1 = length * ((k + 1) / nsteps) ** 2
                t = 0.5 * (t0 + t1)
                dt = t1 - t0
                rr = np.sqrt(r * r + 2 * r * wz * t + t * t)
                ih, f = _row(rr - rplan, h0, dh, nh)
                mup = (r * mus + t * c) / rr
                ir, fr, im, fm, ok = _tlut_index(rr, mup, rb, rt, nr, nmu)
                for b in range(nb):
                    e = (1 - f) * ext[ih, b] + f * ext[ih + 1, b]
                    br = (1 - f) * sr[ih, b] + f * sr[ih + 1, b]
                    bm = (1 - f) * sm[ih, b] + f * sm[ih + 1, b]
                    tv = np.exp(-(tau[b] + 0.5 * e * dt))
                    if ok:
                        ts = np.exp(-_tau_at(tl, ir, fr, im, fm, b))
                        lp[b] += tv * (br * pr + bm * pm[b]) * ts * dt
                    fp[b] += tv * (br + bm) * dt
                    tau[b] += e * dt
            if hit:
                mug = (r * mus + length * c) / rb
                eg_el = np.degrees(np.arcsin(min(max(mug, -1.0), 1.0)))
                fe, ke = 0.0, -1
                if eg_el >= eg_es[0]:
                    ke = min(int(np.searchsorted(eg_es, eg_el)) - 1, neg - 2)
                    ke = max(ke, 0)
                    fe = min(max((eg_el - eg_es[ke]) / (eg_es[ke + 1] - eg_es[ke]), 0.0), 1.0)
                ok = False
                ir, fr, im, fm = 0, 0.0, 0, 0.0
                if mug > 0:
                    ir, fr, im, fm, ok = _tlut_index(rb, mug, rb, rt, nr, nmu)
                for b in range(nb):
                    eb = mug * np.exp(-_tau_at(tl, ir, fr, im, fm, b)) if ok else 0.0
                    if ke >= 0:
                        eb += (1 - fe) * eg[ke, b] + fe * eg[ke + 1, b]
                    lp[b] += np.exp(-tau[b]) * albedo[b] / np.pi * refl[b] * eb
            pq0 = 0.5 * (3 * wx * wx - 1)
            pq1 = 0.5 * (3 * wz * wz - 1)
            pq2 = 1.5 * wx * wz
            for b in range(nb):
                l2[b] += lp[b] / nd
                fms[b] += fp[b] / nd
                qa[0, b] += lp[b] * pq0 / nd
                qa[1, b] += lp[b] * pq1 / nd
                qa[2, b] += lp[b] * pq2 / nd
        for b in range(nb):
            psi[i, j, b] = l2[b] / (1.0 - min(fms[b], 0.999))
            for m in range(3):
                quad[m, i, j, b] = qa[m, b]
    return psi, quad


@njit(inline="always")
def _ms_at(psi, hs, ess, h, mus, b):
    nh, ne = hs.shape[0], ess.shape[0]
    i = np.searchsorted(hs, h) - 1
    i = min(max(i, 0), nh - 2)
    fh = min(max((h - hs[i]) / (hs[i + 1] - hs[i]), 0.0), 1.0)
    e = np.degrees(np.arcsin(min(max(mus, -1.0), 1.0)))
    x = (e - ess[0]) / (ess[1] - ess[0])
    j = min(max(int(x), 0), ne - 2)
    fe = min(max(x - j, 0.0), 1.0)
    return ((1 - fh) * ((1 - fe) * psi[i, j, b] + fe * psi[i, j + 1, b])
            + fh * ((1 - fe) * psi[i + 1, j, b] + fe * psi[i + 1, j + 1, b]))


@njit(parallel=True, cache=True)
def _build_kernel(r0, rb, rt, rplan, h0, dh, ext, sr, sm, tl, psi, quad, a_r, hs, ess, ev, es, dp, bps, rel, nsub,
                  store_all):
    """K[iev, ies, idp, ibp, b, comp]: comp 0 Rayleigh, 1 aerosol, 2 multiple scattering.
    The multiple-scattering source is sigma_s Psi plus, for Rayleigh, the l=2 term
    a_r v^T Q v of the first-order field (second order exact for the Rayleigh phase function).
    bps: breakpoint distances (m, rel=False) or fractions of the ray length (rel=True); the
    cumulative integrals are stored at every breakpoint except the first (0) if store_all, else
    only the full-path value in slot 0."""
    nb = ext.shape[1]
    nh = ext.shape[0]
    nr, nmu = tl.shape[0], tl.shape[1]
    nev, nes, ndp, nbp = ev.shape[0], es.shape[0], dp.shape[0], bps.shape[0]
    out = np.zeros((nev, nes, ndp, nbp - 1 if store_all else 1, nb, 3), dtype=np.float32)
    for idx in prange(nev * nes * ndp):
        iv = idx // (nes * ndp)
        js = (idx // ndp) % nes
        kd = idx % ndp
        cev, sev = np.cos(np.radians(ev[iv])), np.sin(np.radians(ev[iv]))
        ces, ses = np.cos(np.radians(es[js])), np.sin(np.radians(es[js]))
        sxx, syy = ces * np.cos(np.radians(dp[kd])), ces * np.sin(np.radians(dp[kd]))
        c = cev * sxx + sev * ses
        mu = sev
        disc_g = r0 * r0 * (mu * mu - 1) + rb * rb
        if mu < 0 and disc_g >= 0:
            # sky mode: the path ends on the ground sphere; distance mode (terrain below the
            # eye): the path continues through air held at ground-level density
            length = max(-r0 * mu - np.sqrt(disc_g), 0.0) if rel else bps[nbp - 1]
        else:
            length = -r0 * mu + np.sqrt(max(r0 * r0 * (mu * mu - 1) + rt * rt, 0.0))
        tau = np.zeros(nb)
        acc = np.zeros((nb, 3))
        for q in range(1, nbp):
            a = bps[q - 1] * length if rel else min(bps[q - 1], length)
            z = bps[q] * length if rel else min(bps[q], length)
            dt = (z - a) / nsub
            if dt > 0:
                for k in range(nsub):
                    t = a + (k + 0.5) * dt
                    rr = np.sqrt(r0 * r0 + 2 * r0 * mu * t + t * t)
                    h = rr - rplan
                    ih, f = _row(h, h0, dh, nh)
                    mup = (r0 * ses + t * c) / rr
                    ir, fr, im, fm, ok = _tlut_index(rr, mup, rb, rt, nr, nmu)
                    # view direction in the local frame (z = up, x = solar azimuth) at the point
                    ux, uz = t * cev / rr, (r0 + t * sev) / rr
                    vz = cev * ux + sev * uz
                    ex0, ey0, ez0 = sxx - mup * ux, syy, ses - mup * uz
                    en = np.sqrt(ex0 * ex0 + ey0 * ey0 + ez0 * ez0)
                    vx = (cev * ex0 + sev * ez0) / en if en > 1e-9 else np.sqrt(max(1 - vz * vz, 0.0))
                    vy2 = max(1 - vx * vx - vz * vz, 0.0)
                    for b in range(nb):
                        e = (1 - f) * ext[ih, b] + f * ext[ih + 1, b]
                        br = (1 - f) * sr[ih, b] + f * sr[ih + 1, b]
                        bm = (1 - f) * sm[ih, b] + f * sm[ih + 1, b]
                        tv = np.exp(-(tau[b] + 0.5 * e * dt))
                        if ok:
                            ts = np.exp(-_tau_at(tl, ir, fr, im, fm, b))
                            acc[b, 0] += tv * br * ts * dt
                            acc[b, 1] += tv * bm * ts * dt
                        qxx = _ms_at(quad[0], hs, ess, h, mup, b)
                        qzz = _ms_at(quad[1], hs, ess, h, mup, b)
                        qxz = _ms_at(quad[2], hs, ess, h, mup, b)
                        vqv = qxx * vx * vx - (qxx + qzz) * vy2 + qzz * vz * vz + 2 * qxz * vx * vz
                        acc[b, 2] += tv * ((br + bm) * _ms_at(psi, hs, ess, h, mup, b) + br * a_r * vqv) * dt
                        tau[b] += e * dt
            if store_all or q == nbp - 1:
                slot = q - 1 if store_all else 0
                for b in range(nb):
                    for m in range(3):
                        out[iv, js, kd, slot, b, m] = acc[b, m]
    return out


@njit(inline="always")
def _bracket(arr, x):
    n = arr.shape[0]
    i = np.searchsorted(arr, x) - 1
    i = min(max(i, 0), n - 2)
    f = (x - arr[i]) / (arr[i + 1] - arr[i])
    return i, min(max(f, 0.0), 1.0)


@njit(inline="always")
def _geom(vx, vy, vz, sx, sy, sz):
    """(view elevation deg, source elevation deg, delta azimuth deg, cos scattering angle)."""
    ev = np.degrees(np.arcsin(min(max(vz, -1.0), 1.0)))
    es = np.degrees(np.arcsin(min(max(sz, -1.0), 1.0)))
    hv = np.sqrt(vx * vx + vy * vy)
    hs = np.sqrt(sx * sx + sy * sy)
    if hv < 1e-9 or hs < 1e-9:
        dpv = 0.0
    else:
        dpv = np.degrees(np.arccos(min(max((vx * sx + vy * sy) / (hv * hs), -1.0), 1.0)))
    return ev, es, dpv, vx * sx + vy * sy + vz * sz


@njit(parallel=True, cache=True)
def _eval_sum(lut, ev, es, dp, q, s, e, gam, g1, g2, w1):
    """Sum over sources with phase functions: (Q, nd, nb)."""
    nq, ns = q.shape[0], s.shape[0]
    nd, nb = lut.shape[3], lut.shape[4]
    out = np.zeros((nq, nd, nb))
    for i in prange(nq):
        for j in range(ns):
            a, bb, cc, cth = _geom(q[i, 0], q[i, 1], q[i, 2], s[j, 0], s[j, 1], s[j, 2])
            i0, fa = _bracket(ev, a)
            j0, fb = _bracket(es, bb)
            k0, fc = _bracket(dp, cc)
            pr = _phase_r(cth, gam)
            for b in range(nb):
                pm = w1[b] * _hg(cth, g1[b]) + (1 - w1[b]) * _hg(cth, g2[b])
                for d in range(nd):
                    acc = 0.0
                    for di in range(2):
                        wa = fa if di else 1 - fa
                        for dj in range(2):
                            wb = fb if dj else 1 - fb
                            for dk in range(2):
                                w = wa * wb * (fc if dk else 1 - fc)
                                acc += w * (pr * lut[i0 + di, j0 + dj, k0 + dk, d, b, 0]
                                            + pm * lut[i0 + di, j0 + dj, k0 + dk, d, b, 1]
                                            + lut[i0 + di, j0 + dj, k0 + dk, d, b, 2])
                    out[i, d, b] += e[j, b] * acc
    return out


@njit(parallel=True, cache=True)
def _eval_kernels(lut, ev, es, dp, q, s):
    """Phase-free kernels per source: (S, Q, nb, 3) (first distance slot)."""
    nq, ns = q.shape[0], s.shape[0]
    nb = lut.shape[4]
    out = np.zeros((ns, nq, nb, 3), dtype=np.float32)
    for i in prange(nq):
        for j in range(ns):
            a, bb, cc, cth = _geom(q[i, 0], q[i, 1], q[i, 2], s[j, 0], s[j, 1], s[j, 2])
            i0, fa = _bracket(ev, a)
            j0, fb = _bracket(es, bb)
            k0, fc = _bracket(dp, cc)
            for di in range(2):
                wa = fa if di else 1 - fa
                for dj in range(2):
                    wb = fb if dj else 1 - fb
                    for dk in range(2):
                        w = wa * wb * (fc if dk else 1 - fc)
                        for b in range(nb):
                            for m in range(3):
                                out[j, i, b, m] += w * lut[i0 + di, j0 + dj, k0 + dk, 0, b, m]
    return out


@njit(parallel=True, cache=True)
def _pixel_eval(q, ea, daz_a, ka, s, e, gam, g1, g2, w1, eb, daz_b, sb):
    """Per-pixel inscatter: exact-phase sum over the k exact sources (kernels ka interpolated
    on grid A) plus bilinear interpolation of the binned-source sum sb on grid B."""
    nq, ns = q.shape[0], s.shape[0]
    nb = sb.shape[2]
    out = np.zeros((nq, nb))
    naz_a, naz_b = ka.shape[2], sb.shape[1]
    for i in prange(nq):
        vx, vy, vz = q[i, 0], q[i, 1], q[i, 2]
        el = np.degrees(np.arcsin(min(max(vz, -1.0), 1.0)))
        az = np.degrees(np.arctan2(-vy, vx)) % 360.0
        ib, fb = _bracket(eb, el)
        xb = az / daz_b
        jb = min(int(xb), naz_b - 2)
        gb = xb - jb
        for b in range(nb):
            out[i, b] = ((1 - fb) * ((1 - gb) * sb[ib, jb, b] + gb * sb[ib, jb + 1, b])
                         + fb * ((1 - gb) * sb[ib + 1, jb, b] + gb * sb[ib + 1, jb + 1, b]))
        if ns == 0:
            continue
        ia, fa = _bracket(ea, el)
        xa = az / daz_a
        ja = min(int(xa), naz_a - 2)
        ga = xa - ja
        w00, w01 = (1 - fa) * (1 - ga), (1 - fa) * ga
        w10, w11 = fa * (1 - ga), fa * ga
        for j in range(ns):
            cth = vx * s[j, 0] + vy * s[j, 1] + vz * s[j, 2]
            pr = _phase_r(cth, gam)
            for b in range(nb):
                pm = w1[b] * _hg(cth, g1[b]) + (1 - w1[b]) * _hg(cth, g2[b])
                acc = 0.0
                for m in range(3):
                    km = (w00 * ka[j, ia, ja, b, m] + w01 * ka[j, ia, ja + 1, b, m]
                          + w10 * ka[j, ia + 1, ja, b, m] + w11 * ka[j, ia + 1, ja + 1, b, m])
                    acc += km * (pr if m == 0 else (pm if m == 1 else 1.0))
                out[i, b] += e[j, b] * acc
    return out


@njit(parallel=True, cache=True)
def _tau_dirs(q, r0, rb, rt, tl):
    """Optical depth to space (inf where the ray hits the ground): (Q, nb)."""
    nq, nb = q.shape[0], tl.shape[2]
    out = np.empty((nq, nb))
    for i in prange(nq):
        ir, fr, im, fm, ok = _tlut_index(r0, q[i, 2], rb, rt, tl.shape[0], tl.shape[1])
        for b in range(nb):
            out[i, b] = _tau_at(tl, ir, fr, im, fm, b) if ok else np.inf
    return out


@njit(parallel=True, cache=True)
def _tau_path(q, dist, r0, rplan, h0, dh, ext, nsteps):
    nq, nb, nh = q.shape[0], ext.shape[1], ext.shape[0]
    out = np.zeros((nq, nb))
    for i in prange(nq):
        d = dist[i]
        if not np.isfinite(d):
            for b in range(nb):
                out[i, b] = np.inf
            continue
        dt = d / nsteps
        mu = q[i, 2]
        for k in range(nsteps):
            t = (k + 0.5) * dt
            rr = np.sqrt(r0 * r0 + 2 * r0 * mu * t + t * t)
            ih, f = _row(rr - rplan, h0, dh, nh)
            for b in range(nb):
                out[i, b] += ((1 - f) * ext[ih, b] + f * ext[ih + 1, b]) * dt
    return out


@njit(parallel=True, cache=True)
def _interp_ap(q, dist, ea, daz, dgrid, ap):
    """Trilinear (elevation, azimuth, distance) interpolation of ap (nel, naz, nd, nb)."""
    nq, nb = q.shape[0], ap.shape[3]
    naz = ap.shape[1]
    out = np.zeros((nq, nb))
    for i in prange(nq):
        el = np.degrees(np.arcsin(min(max(q[i, 2], -1.0), 1.0)))
        az = np.degrees(np.arctan2(-q[i, 1], q[i, 0])) % 360.0
        ia, fa = _bracket(ea, el)
        x = az / daz
        ja = min(int(x), naz - 2)
        ga = x - ja
        d = dist[i] if np.isfinite(dist[i]) else dgrid[-1]
        kd, fd = _bracket(dgrid, d)
        for b in range(nb):
            acc = 0.0
            for di in range(2):
                wa = fa if di else 1 - fa
                for dj in range(2):
                    wb = ga if dj else 1 - ga
                    for dk in range(2):
                        acc += wa * wb * (fd if dk else 1 - fd) * ap[ia + di, ja + dj, kd + dk, b]
            out[i, b] = acc
    return out


# ----------------------------------------------------------------------------- grids
EV = np.concatenate([-np.array([90, 60, 40, 25, 15, 10, 6, 4, 2.5, 1.5, 1.0, 0.6, 0.3, 0.1])[:],
                     90.0 * (np.arange(37) / 36.0) ** 2])
ES = np.concatenate([np.arange(-20.0, 10.1, 0.25), [12.5, 15, 17.5, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90]])
DP = np.linspace(0.0, 180.0, 25)
FRAC = np.array([0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 0.85, 1.0])
MS_H_FRAC = (np.arange(32) / 31.0) ** 2
MS_ES = np.linspace(-90.0, 90.0, 361)
MS_ES_COARSE = np.linspace(-90.0, 90.0, 73)
GRID_A_EL = EV[EV >= -10.0]
GRID_A_DAZ = 2.0
GRID_B_EL = np.array([-10, -5, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12.5, 15, 20,
                      25, 30, 35, 40, 50, 60, 70, 80, 90], float)
GRID_B_DAZ = 5.0
AP_EV = np.array([-90, -60, -40, -30, -20, -15, -10, -7, -5, -3, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 3, 5,
                  7, 10, 15, 20, 30, 45, 60, 90], float)
AP_DP = np.linspace(0.0, 180.0, 13)
AP_D = np.array([0, 100, 250, 500, 1e3, 2e3, 4e3, 7e3, 1e4, 1.5e4, 2.5e4, 4e4, 6e4, 1e5])
AP_DAZ = 10.0
ES_COARSE = np.concatenate([np.arange(-20.0, 10.5, 1.0), [12.5, 15, 17.5, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90]])


def _fingerprint(a):
    """Cheap content key for large arrays: shape, dtype, full sum and a strided sample."""
    a = np.asarray(a)
    flat = a.reshape(-1)
    step = max(1, flat.size // 65536)
    return a.shape, a.dtype.str, float(np.sum(flat, dtype=np.float64)), hash(np.ascontiguousarray(flat[::step]).tobytes())


def _fib_sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    rr = np.sqrt(1 - z * z)
    return np.stack([rr * np.cos(phi), rr * np.sin(phi), z], axis=-1)


def _grid_dirs(els, daz):
    az = np.arange(0.0, 360.0 + 0.5 * daz, daz)
    E, A = np.meshgrid(els, az, indexing="ij")
    return frames.altaz_to_vec(E, A).reshape(-1, 3), len(az)


class Atmosphere:
    """Spherical-shell atmosphere; see the module docstring. preset: name or dict of
    overrides (key "base" selects the preset they modify, default "earth_clear")."""

    def __init__(self, preset="earth_clear", grid=None):
        self.grid = grid if grid is not None else spectral.default_grid()
        if isinstance(preset, str):
            name, over = preset, {}
        else:
            over = dict(preset)
            name = over.pop("base", "earth_clear")
            over.pop("name", None)
        if name not in PRESETS:
            raise ValueError(f"unknown atmosphere preset {name!r}; choose from {sorted(PRESETS)}")
        self.preset = {**_BASE, **PRESETS[name], **over}
        self.name = name if isinstance(preset, str) else preset.get("name", name + "+custom")
        self.vacuum = bool(self.preset.get("vacuum", False))
        self._cache = (None, None)
        self._ap_lut = None
        if not self.vacuum:
            self._build()

    # ------------------------------------------------------------------ setup
    def _profiles(self, h):
        p, g = self.preset, self.grid
        if p.get("rayleigh_density") is not None:
            n = p["rayleigh_n0_m3"] * np.asarray(p["rayleigh_density"](h), float)
        elif p["rayleigh_profile"] == "us76":
            n = us76_number_density(h) * p["rayleigh_n0_m3"] / us76_number_density(0.0)
        else:
            n = p["rayleigh_n0_m3"] * np.exp(-h / p["rayleigh_scale_height_m"])
        if p["rayleigh_xsec"] is not None:
            xs = _spec(p["rayleigh_xsec"], g) * 1e-4
        else:
            xs = g.bin_average(rayleigh_xsec_air) * (2.4 if p["rayleigh_gas"] == "co2" else 1.0)
        if p["depolarization"] is not None:
            rho = float(p["depolarization"])
        elif p["rayleigh_gas"] == "co2":
            rho = 0.075
        else:
            f = float(king_factor_air(550.0))
            rho = 6 * (f - 1) / (3 + 7 * f)
        self.gamma = rho / (2 - rho)
        self.rho = rho
        b_r = n[:, None] * xs[None, :]
        o3 = np.exp(-0.5 * ((h - p["ozone_peak_m"]) / p["ozone_width_m"]) ** 2)
        o3 *= p["ozone_du"] * DU_M2 / max(np.trapezoid(o3, h), 1e-30)
        xo = g.bin_average(lambda x: np.interp(x, _O3_WL, _O3_XS))
        b_o3 = o3[:, None] * xo[None, :]
        lam = (g.wl / 550.0) ** (-p["aerosol_alpha"])
        hg0 = h - p["altitude_m"]
        prof = np.exp(-np.clip(hg0, 0, None) / p["aerosol_scale_height_m"]) / p["aerosol_scale_height_m"]
        strat = np.exp(-0.5 * ((h - p["strat_peak_m"]) / p["strat_width_m"]) ** 2)
        strat /= max(np.trapezoid(strat, h), 1e-30)
        b_a = (p["aerosol_tau550"] * prof + p["strat_tau550"] * strat)[:, None] * lam[None, :]
        ssa = _spec(p["aerosol_ssa"], g)
        return b_r, b_a * ssa[None, :], b_r + b_a + b_o3, b_o3

    def _build(self):
        p, g = self.preset, self.grid
        self.r_plan = float(p["planet_radius_m"])
        self.rb = self.r_plan + p["altitude_m"]
        self.rt = self.r_plan + p["top_m"]
        self.r0 = self.rb + p["eye_m"]
        self.h0 = float(p["altitude_m"])
        h = np.linspace(self.h0, p["top_m"], 1201)
        self.dh = h[1] - h[0]
        self.alt = h
        self.b_r, self.b_ms, self.ext, self.b_o3 = self._profiles(h)
        self.g1 = _spec(p["aerosol_g1"], g)
        self.g2 = _spec(p["aerosol_g2"], g)
        self.w1 = _spec(p["aerosol_w1"], g)
        self.albedo = _spec(p["ground_albedo"], g)
        self.tlut = _build_tlut(self.rb, self.rt, self.r_plan, self.h0, self.dh, self.ext, 64, 256, 256)
        self.ms_h = self.h0 + (p["top_m"] - self.h0) * MS_H_FRAC
        self.a_r = (1 - self.gamma) / (2 * (1 + 2 * self.gamma))
        ms_a = (self.rb, self.rt, self.r_plan, self.h0, self.dh, self.ext, self.b_r, self.b_ms, self.tlut,
                self.ms_h)
        fib = _fib_sphere(int(p["ms_dirs"]))
        ms_b = (np.array([self.gamma]), self.g1, self.g2, self.w1, 40, ES_COARSE)
        self.ground_diffuse, self.refl = np.zeros((len(ES_COARSE), g.n)), np.ones(g.n)
        self.spherical_albedo = np.zeros(g.n)
        if np.any(self.albedo > 0):
            # black-ground pass on coarse grids; the ground then reflects the direct beam plus the
            # diffuse sky irradiance of that pass, times the interreflection factor 1 / (1 - albedo S)
            self.psi, self.quad = _build_ms(*ms_a, MS_ES_COARSE, fib, np.zeros(g.n), *ms_b,
                                            self.ground_diffuse, self.refl)
            k0 = self._kernel(EV, ES_COARSE, DP[::2], FRAC, True, 6, False, MS_ES_COARSE)
            self.ground_diffuse, self.refl = self._ground_terms(k0, ES_COARSE, DP[::2])
        self.psi, self.quad = _build_ms(*ms_a, MS_ES, fib, self.albedo, *ms_b, self.ground_diffuse, self.refl)
        self.kernel = self._kernel(EV, ES, DP, FRAC, True, 10, False)
        self.airglow0 = airglow_zenith(g, p["airglow_scale"]) if p["airglow_scale"] > 0 else np.zeros(g.n)
        self.grid_a, self.naz_a = _grid_dirs(GRID_A_EL, GRID_A_DAZ)
        self.grid_b, self.naz_b = _grid_dirs(GRID_B_EL, GRID_B_DAZ)

    def _kernel(self, ev, es, dp, bps, rel, nsub, store_all, ms_es=MS_ES):
        return _build_kernel(self.r0, self.rb, self.rt, self.r_plan, self.h0, self.dh, self.ext, self.b_r,
                             self.b_ms, self.tlut, self.psi, self.quad, self.a_r, self.ms_h, ms_es, ev, es, dp,
                             bps, rel, nsub, store_all)

    def _ground_terms(self, kernel, es_deg, dp_deg):
        """Diffuse horizontal irradiance at the ground per unit TOA irradiance of a source at
        elevation es_deg (len(es_deg), nbins), integrated from a kernel LUT on (EV, es_deg, dp_deg),
        and the interreflection factor 1 / (1 - albedo S) with the atmosphere's spherical albedo
        S = (1 - isotropic transmittance) x column single-scattering albedo."""
        up = EV >= 0
        ev, es, dp = np.radians(EV[up]), np.radians(es_deg), np.radians(dp_deg)
        c = (np.cos(ev)[:, None, None] * np.cos(es)[None, :, None] * np.cos(dp)[None, None, :]
             + np.sin(ev)[:, None, None] * np.sin(es)[None, :, None])[..., None]
        gm = self.gamma
        pr = 3.0 / (16 * np.pi * (1 + 2 * gm)) * ((1 + 3 * gm) + (1 - gm) * c * c)
        hg = lambda g_: (1 - g_ ** 2) / (4 * np.pi * (1 + g_ ** 2 - 2 * g_ * c) ** 1.5)
        pm = self.w1 * hg(self.g1) + (1 - self.w1) * hg(self.g2)
        k = kernel[up][:, :, :, 0].astype(float)
        rad = pr * k[..., 0] + pm * k[..., 1] + k[..., 2]
        e_dif = 2 * np.trapezoid(np.trapezoid(rad, dp, axis=2) * (np.sin(ev) * np.cos(ev))[:, None, None], ev, axis=0)
        t_dir = np.where(es[:, None] > 0, np.sin(es)[:, None]
                         * self.transmittance(frames.altaz_to_vec(np.clip(es_deg, 0, 90), 0.0)), 0.0)
        t_iso = 2 * np.trapezoid((t_dir + e_dif) * np.cos(es)[:, None], es, axis=0)
        ssa = np.trapezoid(self.b_r + self.b_ms, self.alt, axis=0) / np.trapezoid(self.ext, self.alt, axis=0)
        s = np.clip((1 - t_iso) * ssa, 0.0, 0.95)
        self.spherical_albedo = s
        return np.maximum(e_dif, 0.0), 1.0 / (1.0 - self.albedo * s)

    # ------------------------------------------------------------------ helpers
    def _dirs(self, d):
        d = np.asarray(d, float)
        return d.reshape(-1, 3), d.shape[:-1]

    def _phase_args(self):
        return self.gamma, self.g1, self.g2, self.w1

    def tau(self, view_dirs_h):
        """(N, nbins) optical depth from the observer to space (inf where the ray hits the ground)."""
        q, shp = self._dirs(view_dirs_h)
        if self.vacuum:
            return (np.where(q[:, 2:3] >= 0, 0.0, np.inf) * np.ones(self.grid.n)).reshape(shp + (self.grid.n,))
        return _tau_dirs(q, self.r0, self.rb, self.rt, self.tlut).reshape(shp + (self.grid.n,))

    def transmittance(self, view_dirs_h):
        return np.exp(-self.tau(view_dirs_h))

    def van_rhijn(self, view_dirs_h):
        q, shp = self._dirs(view_dirs_h)
        if self.vacuum:
            return np.zeros(shp)
        ratio = self.r0 / (self.r_plan + self.preset["airglow_height_m"])
        cz = np.clip(q[:, 2], -1, 1)
        return (1.0 / np.sqrt(np.clip(1 - ratio ** 2 * (1 - cz ** 2), 1e-6, None))).reshape(shp)

    def _sources(self, toa, R):
        """Horizon-frame scattering sources, cached per (toa, R): the k_exact brightest point
        clumps (exact phase per pixel), and HEALPix (nside scatter_nside, horizon frame) virtual
        sources holding the other points, the diffuse map and the airglow (radiance x pixel
        solid angle)."""
        import healpy as hp
        R = np.asarray(R, float)
        key = (R.tobytes(), _fingerprint(toa.points.dirs), _fingerprint(toa.points.irr), _fingerprint(toa.diffuse_map))
        if self._cache[0] == key:
            return self._cache[1]
        g, p = self.grid, self.preset
        pts = toa.points.rotated(R)
        smin = np.sin(np.radians(ES_MIN_DEG))
        ns = p["scatter_nside"]
        npix = hp.nside2npix(ns)
        omega = 4 * np.pi / npix
        dh = frames.healpix_dirs(ns)
        vi = np.zeros((npix, g.n))
        sd, si = np.zeros((0, 3)), np.zeros((0, g.n))
        if len(pts):
            y = spectral.to_xyz(pts.irr, g)[:, 1]
            ok = (pts.dirs[:, 2] > smin) & (y > 0)
            # clumps: points merged per HEALPix nside-64 pixel (0.9 deg) at their flux-weighted
            # direction; the k brightest clumps are exact, the rest join the virtual sources
            u, inv = np.unique(hp.vec2pix(CLUMP_NSIDE, *pts.dirs[ok].T), return_inverse=True)
            ci = np.zeros((len(u), g.n))
            cd = np.zeros((len(u), 3))
            cy = np.zeros(len(u))
            np.add.at(ci, inv, pts.irr[ok])
            np.add.at(cd, inv, pts.dirs[ok] * y[ok, None])
            np.add.at(cy, inv, y[ok])
            cd /= np.maximum(np.linalg.norm(cd, axis=1, keepdims=True), 1e-300)
            order = np.argsort(cy)[::-1]
            ex, rest = order[:p["k_exact"]], order[p["k_exact"]:]
            sd, si = cd[ex], ci[ex]
            if len(rest):
                np.add.at(vi, hp.vec2pix(ns, *cd[rest].T), ci[rest])
        m = np.asarray(toa.diffuse_map, float)
        if m.size and np.any(m):
            if hp.npix2nside(m.shape[0]) > ns:
                m = np.stack([hp.ud_grade(m[:, b], ns) for b in range(g.n)], axis=1)
            vi += frames.sample_healpix(m, dh @ R) * omega
        if np.any(self.airglow0):
            up = dh[:, 2] > 0
            vi[up] += self.airglow0[None, :] * self.van_rhijn(dh[up])[:, None] * omega
        keep = (dh[:, 2] > smin) & (vi.max(axis=1) > 0)
        ka = (_eval_kernels(self.kernel, EV, ES, DP, self.grid_a, sd)
              .reshape(len(sd), len(GRID_A_EL), self.naz_a, g.n, 3))
        if keep.any():
            sb = _eval_sum(self.kernel, EV, ES, DP, self.grid_b, dh[keep], vi[keep], *self._phase_args())[:, 0]
        else:
            sb = np.zeros((len(self.grid_b), g.n))
        out = dict(sd=sd, si=si, vi=vi, keep=keep, ka=ka, sb=sb.reshape(len(GRID_B_EL), self.naz_b, g.n))
        self._cache = (key, out)
        return out

    # ------------------------------------------------------------------ public API
    def sky(self, toa, R, view_dirs_h):
        """dict(T, inscatter, airglow), each (..., nbins), for horizon-frame view directions.
        Observed sky radiance = T * toa.diffuse(d @ R) + inscatter + airglow."""
        q, shp = self._dirs(view_dirs_h)
        n = self.grid.n
        T = self.transmittance(q)
        if self.vacuum:
            z = np.zeros((len(q), n))
            return dict(T=T.reshape(shp + (n,)), inscatter=z.reshape(shp + (n,)), airglow=z.reshape(shp + (n,)))
        s = self._sources(toa, R)
        ins = _pixel_eval(q, GRID_A_EL, GRID_A_DAZ, s["ka"], s["sd"], s["si"], *self._phase_args(),
                          GRID_B_EL, GRID_B_DAZ, s["sb"])
        ag = self.airglow0[None, :] * self.van_rhijn(q)[:, None] * T
        return dict(T=T.reshape(shp + (n,)), inscatter=ins.reshape(shp + (n,)), airglow=ag.reshape(shp + (n,)))

    def sky_radiance(self, toa, R, view_dirs_h, exact_diffuse=True):
        """Total observed sky radiance (..., nbins), point sources excluded."""
        d = np.asarray(view_dirs_h, float)
        r = self.sky(toa, R, d)
        dc = d @ np.asarray(R)
        bg = toa.diffuse(dc) if exact_diffuse else frames.sample_healpix(toa.diffuse_map, dc)
        return r["T"] * bg + r["inscatter"] + r["airglow"]

    def point_irradiance(self, toa, R):
        pts = toa.points.rotated(R)
        if not len(pts):
            return pts
        return PointSources(pts.dirs, pts.irr * self.transmittance(pts.dirs), list(pts.names))

    def ground_irradiance(self, toa, R, nside=16):
        """(nbins,) horizontal irradiance at the ground: transmitted points + sky hemisphere."""
        import healpy as hp
        d = frames.healpix_dirs(nside)
        d = d[d[:, 2] > 0]
        L = self.sky_radiance(toa, R, d, exact_diffuse=False)
        e = (L * d[:, 2:3]).sum(0) * 4 * np.pi / hp.nside2npix(nside)
        pts = self.point_irradiance(toa, R)
        if len(pts):
            e = e + (pts.irr * np.clip(pts.dirs[:, 2], 0, None)[:, None]).sum(0)
        return e

    def aerial_perspective(self, view_dirs_h, dist_m, toa, R):
        """(T, inscatter), each (..., nbins), for a path of length dist_m (inf -> to space)."""
        q, shp = self._dirs(view_dirs_h)
        dist = np.broadcast_to(np.asarray(dist_m, float), shp).reshape(-1)
        n = self.grid.n
        if self.vacuum:
            return np.ones(shp + (n,)), np.zeros(shp + (n,))
        fin = np.isfinite(dist)
        tau = _tau_path(q, np.where(fin, dist, 0.0), self.r0, self.r_plan, self.h0, self.dh, self.ext, 24)
        T = np.where(fin[:, None], np.exp(-tau), self.transmittance(q))
        if self._ap_lut is None:
            self._ap_lut = self._kernel(AP_EV, ES_COARSE, AP_DP, AP_D, False, 6, True)
        s = self._sources(toa, R)
        if "ap" not in s:
            import healpy as hp
            ns = self.preset["scatter_nside"]
            ns8 = min(8, ns)
            p8 = hp.nest2ring(ns8, hp.ring2nest(ns, np.arange(hp.nside2npix(ns))) // (ns // ns8) ** 2)
            vi = np.zeros((hp.nside2npix(ns8), n))
            np.add.at(vi, p8, s["vi"] * s["keep"][:, None])
            d8 = frames.healpix_dirs(ns8)
            k8 = (d8[:, 2] > np.sin(np.radians(ES_MIN_DEG))) & (vi.max(axis=1) > 0)
            sd, si = np.vstack([s["sd"], d8[k8]]), np.vstack([s["si"], vi[k8]])
            gdirs, naz = _grid_dirs(AP_EV, AP_DAZ)
            ap = _eval_sum(self._ap_lut, AP_EV, ES_COARSE, AP_DP, gdirs, sd, si, *self._phase_args())
            ap = np.concatenate([np.zeros((len(gdirs), 1, n)), ap], axis=1)
            s["ap"] = ap.reshape(len(AP_EV), naz, len(AP_D), n)
        ins = _interp_ap(q, np.where(fin, dist, np.inf), AP_EV, AP_DAZ, AP_D, s["ap"])
        far = ~fin | (dist > AP_D[-1])
        if far.any():
            ins[far] = self.sky(toa, R, q[far])["inscatter"]
        return T.reshape(shp + (n,)), ins.reshape(shp + (n,))

    def summary(self):
        """Zenith optical depths and V extinction (mag/airmass) for quick checks."""
        g = self.grid
        if self.vacuum:
            return dict(name=self.name, k_V=0.0)
        tz = self.tau(np.array([[0.0, 0.0, 1.0]]))[0]
        tr = np.trapezoid(self.b_r, self.alt, axis=0)
        to = np.trapezoid(self.b_o3, self.alt, axis=0)
        s = v_filter_bins(g) * g.dwl
        kv = -2.5 * np.log10((np.exp(-tz) * s).sum() / s.sum())
        i550 = g.bin_index(550.0)
        return dict(name=self.name, tau_zenith_550=float(tz[i550]), tau_rayleigh_550=float(tr[i550]),
                    tau_ozone_550=float(to[i550]), tau_aerosol_550=float(tz[i550] - tr[i550] - to[i550]),
                    k_V=float(kv), rho=self.rho)


def main():
    import time
    g = spectral.default_grid()
    for name in PRESETS:
        t0 = time.time()
        a = Atmosphere(name, g)
        print(f"{name:18s} build {time.time() - t0:6.1f} s  {a.summary()}", flush=True)


if __name__ == "__main__":
    main()
