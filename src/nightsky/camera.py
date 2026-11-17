"""Views, point-source deposition and the observer model (eye or camera) for nightsky.

Units: input images are (H, W, nbins) spectral radiance W m^-2 sr^-1 nm^-1 (bin averages);
point sources are spectral irradiance W m^-2 nm^-1. Photometric outputs are CIE XYZ with Y in
cd/m^2 and scotopic luminance in scotopic cd/m^2 (spectral.to_xyz, spectral.scotopic_luminance).
Directions are in the horizon frame (x = N, y = W, z = Up).

Observer models
- eye: adaptation luminance La = log-average of the mesopic luminance over the sky/scene pixels
  (floored at the 1e-6 cd/m^2 absolute threshold; La solved self-consistently with m(La)) or
  a fixed value; partial adaptation La_eff = La^g l_ref^(1-g) with g = 0.75 (the earlier
  compose_day.py setting); mesopic weight m log-linear in La between 0.005 and 5 cd/m^2 (the
  CIE 191:2010 mesopic range); scotopic signal shown as a blue-shifted grey (Purkinje tint as in
  compose_day.py, in the spirit of Kirk & O'Brien 2011 and Thompson et al. 2002); disability
  glare from the CIE 2002 general glare equation (Vos & van den Berg 1999; CIE 146:2002),
  applied as an angle-space point-spread function by FFT convolution; extended Reinhard
  (Reinhard et al. 2002) tone curve on luminance.
- camera: saturation luminance L_max = 78 N^2 / (0.65 S t) cd/m^2 (ISO 12232 saturation-based
  speed with lens transmission q = 0.65, as in Lagarde & de Rousiers 2014); the CIE 1931
  observer stands in for the sensor response; optional lens bloom (Gaussian halo) and
  diffraction spikes; ACES-like filmic curve (Narkowicz 2015 fit).
Approximations: the glare and lens kernels use one pixel scale per view: the radial step for
the equidistant all-sky fisheye (tangential pixels subtend sin(z)/z of it, 0.60 at z = 95 deg),
sqrt(median pixel solid angle) otherwise (a 60 deg wide perspective frame has edge pixels
cos^2(30 deg) = 0.75 of the centre size radially). Light from outside the field of view is
approximated by each edge pixel's own value (see convolve).
"""
import json
import os

import numpy as np

import photometry as ph
import views as vw

from . import frames, spectral

TINT_XYZ = np.array([0.933, 1.0, 1.40])  # compose_day.py scotopic (Purkinje) tint, Y = 1
ARCSEC = np.pi / 180 / 3600
ALLSKY_MAX_ZENITH_DEG = 95.0  # fixed in views.View.project
ABS_THRESHOLD_CD_M2 = 1e-6  # absolute (scotopic) threshold of vision, ~1e-6 cd/m^2 (Hood & Finkelstein 1986)


# ----------------------------------------------------------------------------- views
class Panorama(vw.View):
    """Cylindrical panorama: azimuth (N through E) increases left to right, elevation up."""

    def __init__(self, width, height, azimuth_deg=180.0, az_span_deg=360.0, el_min_deg=-10.0, el_max_deg=60.0):
        self.kind, self.width, self.height = "panorama", int(width), int(height)
        self.az0, self.span, self.el_lo, self.el_hi = float(azimuth_deg), float(az_span_deg), float(el_min_deg), float(el_max_deg)
        az = self.az0 + ((np.arange(width) + 0.5) / width - 0.5) * self.span
        el = self.el_hi - (np.arange(height) + 0.5) / height * (self.el_hi - self.el_lo)
        AZ, EL = np.meshgrid(az, el)
        self.dirs = frames.altaz_to_vec(EL, AZ)
        self.valid = np.ones((height, width), bool)
        daz = np.radians(self.span / width)
        de = np.radians((self.el_hi - self.el_lo) / height)
        e = np.radians(EL)
        self.omega = daz * np.abs(np.sin(e + de / 2) - np.sin(e - de / 2))

    def project(self, dirs):
        alt, az = frames.vec_to_altaz(np.atleast_2d(dirs))
        daz = (az - self.az0 + 180.0) % 360.0 - 180.0
        col = (daz / self.span + 0.5) * self.width - 0.5
        row = (self.el_hi - alt) / (self.el_hi - self.el_lo) * self.height - 0.5
        ok = (row > -2) & (row < self.height + 1) & (col > -2) & (col < self.width + 1) & (np.abs(daz) <= self.span / 2 + 1e-9)
        return np.where(ok, row, np.nan), np.where(ok, col, np.nan)


def target_altaz(target, ctx):
    """(alt, az) in deg of a named target ('sgr_a_star', 'nsc', 'galactic_center', 'arches'),
    a galactocentric position [x, y, z] pc, or {'alt':, 'az':}."""
    if isinstance(target, dict):
        return float(target["alt"]), float(target["az"])
    if isinstance(target, str):
        sgr = tuple(frames.lb_to_galactocentric(359.9442, -0.0462, frames.R0_PC))
        pos = {"sgr_a_star": sgr, "nsc": sgr, "galactic_center": sgr, "origin": (0.0, 0.0, 0.0),
               "arches": tuple(frames.lb_to_galactocentric(0.1217, 0.0188, frames.R0_PC))}[target]
    else:
        pos = target
    d = np.asarray(pos, float) - np.asarray(ctx["observer_pc"], float)
    d_h = ctx["R_c2h"] @ (np.asarray(ctx["R_gc_to_cel"]) @ (d / np.linalg.norm(d)))
    alt, az = frames.vec_to_altaz(d_h)
    return float(alt), float(az)


def resolve_pointing(vcfg, ctx):
    """azimuth_deg / pitch_deg for a perspective view; 'auto' points at vcfg['target'] with the
    horizon kept at horizon_frac of the frame height when the target allows it."""
    az, pitch = vcfg.get("azimuth_deg", 0.0), vcfg.get("pitch_deg", 15.0)
    if az != "auto" and pitch != "auto":
        return float(az), float(pitch), None
    alt_t, az_t = target_altaz(vcfg.get("target", "sgr_a_star"), ctx)
    fov = np.radians(vcfg.get("fov_deg", 60.0))
    vfov = 2 * np.degrees(np.arctan(np.tan(fov / 2) * vcfg["height"] / vcfg["width"]))
    if az == "auto":
        az = az_t
    if pitch == "auto":
        f = vcfg.get("horizon_frac", 0.2)
        pitch = np.degrees(np.arctan((1 - 2 * f) * np.tan(np.radians(vfov / 2))))
        if alt_t > pitch + 0.4 * vfov:
            pitch = alt_t - 0.3 * vfov
    return float(az), float(pitch), (alt_t, az_t)


def make_view(vcfg, ctx=None, scale=1.0):
    """views.View / Panorama from a view config dict. scale multiplies the pixel counts."""
    kind = vcfg.get("kind", "perspective")
    w = max(4, int(round(vcfg.get("width", vcfg.get("size", 512)) * scale)))
    h = max(4, int(round(vcfg.get("height", vcfg.get("size", w)) * scale)))
    if kind == "allsky":
        return vw.View("allsky", w, h, max_zenith_deg=ALLSKY_MAX_ZENITH_DEG)
    if kind == "panorama":
        return Panorama(w, h, vcfg.get("azimuth_deg", 180.0), vcfg.get("az_span_deg", 360.0),
                        vcfg.get("el_min_deg", -10.0), vcfg.get("el_max_deg", 60.0))
    if kind == "perspective":
        az, pitch, _ = resolve_pointing({**vcfg, "width": w, "height": h}, ctx or {})
        return vw.View("perspective", w, h, azimuth_deg=az, pitch_deg=pitch, fov_deg=vcfg.get("fov_deg", 60.0))
    raise ValueError(f"unknown view kind {kind!r}")


# ----------------------------------------------------------------------------- point sources
def psf_sigma_px(view, rows, cols, pcfg, grid):
    """Gaussian PSF width (px) per source: max(core, seeing (+) Airy-core) in pixel units."""
    r = np.clip(np.round(rows).astype(int), 0, view.height - 1)
    c = np.clip(np.round(cols).astype(int), 0, view.width - 1)
    pix_rad = np.sqrt(view.omega[r, c])
    s = (pcfg.get("seeing_fwhm_arcsec") or 0.0) * ARCSEC / 2.3548
    if pcfg.get("airy_aperture_m"):
        s = np.hypot(s, 0.42 * 550e-9 / pcfg["airy_aperture_m"])
    return np.maximum(pcfg.get("psf_sigma_px", 0.6), s / pix_rad)


def _project_points(view, dirs, irr, pcfg, grid):
    """(row, col, irr) of the sources above min_irradiance_frac that project into the view."""
    irr = np.asarray(irr, float)
    y = irr @ spectral.cmf_bins(grid)[:, 1]
    keep = y > max(0.0, pcfg.get("min_irradiance_frac", 0.0)) * y.max()
    dirs, irr = np.asarray(dirs)[keep], irr[keep]
    row, col = view.project(dirs)
    ok = np.isfinite(row) & np.isfinite(col)
    return row[ok], col[ok], irr[ok]


def _stencils(view, row, col, pcfg, grid):
    """Per chunk of sources: (slice, flat pixel index, normalised Gaussian weight (0 outside the
    frame / valid mask), source index within the chunk), all (n_chunk * stencil,)."""
    H, W = view.height, view.width
    sig = psf_sigma_px(view, row, col, pcfg, grid)
    R = int(np.clip(np.ceil(3 * sig.max()), 1, pcfg.get("radius_px", 4)))
    off = np.arange(-R, R + 1)
    dr, dc = [a.ravel() for a in np.meshgrid(off, off, indexing="ij")]
    valid = view.valid.ravel()
    n = pcfg.get("chunk", 20000)
    for s0 in range(0, len(row), n):
        rr0, cc0 = row[s0:s0 + n], col[s0:s0 + n]
        rr = np.round(rr0).astype(int)[:, None] + dr[None]
        cc = np.round(cc0).astype(int)[:, None] + dc[None]
        w = np.exp(-0.5 * ((rr - rr0[:, None]) ** 2 + (cc - cc0[:, None]) ** 2) / sig[s0:s0 + n, None] ** 2)
        w /= w.sum(axis=1, keepdims=True)
        inside = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        idx = np.where(inside, rr * W + cc, 0)
        inside &= valid[idx]
        w = np.where(inside, w, 0.0)
        yield slice(s0, s0 + n), idx.ravel(), w.ravel(), np.repeat(np.arange(len(rr0)), len(dr))


def deposit_points(img, view, dirs, irr, pcfg, grid):
    """Add point sources (irradiance (n, nbins)) to img (H, W, nbins) as radiance.

    Each source is spread with a normalised Gaussian stencil and divided by the receiving
    pixel's solid angle, so sum(img * omega) increases by the irradiance falling inside the
    image (flux outside the frame or the valid mask is dropped). Returns the number deposited.
    """
    if len(dirs) == 0:
        return 0
    row, col, irr = _project_points(view, dirs, irr, pcfg, grid)
    if len(row) == 0:
        return 0
    H, W = view.height, view.width
    flat = img.reshape(H * W, -1)
    omega = view.omega.ravel()
    for s, idx, w, k in _stencils(view, row, col, pcfg, grid):
        e = irr[s]
        for b in range(flat.shape[1]):
            flat[:, b] += (np.bincount(idx, weights=w * e[k, b], minlength=H * W) / omega).astype(flat.dtype)
    return int(len(row))


# ----------------------------------------------------------------------------- naked-eye detection
NL_CD_M2 = 1e-5 / np.pi  # 1 nanolambert = 1e-9 L, 1 L = 1e4 / pi cd/m^2
FC_LUX = 10.7639  # 1 foot-candle = 1 lm/ft^2 = 10.7639 lux


def threshold_illuminance(lb_cd_m2):
    """Naked-eye point-source detection threshold (lux) against a background of luminance lb
    (cd/m^2): Hecht (1947) as fitted by Schaefer (1990, PASP 102, 212), I = c1 (1 + sqrt(c2 B))^2
    with B in nL and I in foot-candles; night (c1, c2) = (10^-9.8, 10^-1.9), day (10^-8.35,
    10^-5.9); the smaller branch is used (they cross near B ~ 2000 nL = 6e-3 cd/m^2)."""
    b = np.maximum(np.asarray(lb_cd_m2, float), 0.0) / NL_CD_M2
    night = 10 ** -9.8 * (1 + np.sqrt(10 ** -1.9 * b)) ** 2
    day = 10 ** -8.35 * (1 + np.sqrt(10 ** -5.9 * b)) ** 2
    return np.minimum(night, day) * FC_LUX


def limiting_vmag(lb_cd_m2, factor=1.0):
    """Naked-eye limiting V magnitude on background lb: Schaefer (1990) m = -16.57 - 2.5 log10(I / fc)."""
    return -16.57 - 2.5 * np.log10(factor * threshold_illuminance(lb_cd_m2) / FC_LUX)


def detection_probability(e_lux, lb_cd_m2, factor=1.0, sigma_mag=0.3):
    """Probability of seeing a point of illuminance e_lux on background lb: normal psychometric
    function of the margin m_lim - m = 2.5 log10(E / (factor E_th)) with width sigma_mag."""
    from scipy.special import erf
    dm = 2.5 * np.log10(np.maximum(np.asarray(e_lux, float), 1e-300) / (factor * threshold_illuminance(lb_cd_m2)))
    return 0.5 * (1 + erf(dm / (np.sqrt(2) * sigma_mag)))


def blur_flux(F, view, sigma_px):
    """Gaussian blur of a per-pixel flux image F (H, W, nbins), returned as radiance (F / omega).
    Mode 'reflect' keeps the in-frame sum; weight landing on invalid pixels returns to its source
    pixel (as in convolve), so sum(out * omega) = sum(F[valid])."""
    from scipy.ndimage import gaussian_filter
    m = view.valid.astype(float)
    g = lambda a: gaussian_filter(a, (sigma_px, sigma_px) + (0,) * (a.ndim - 2), mode="reflect")
    miss = np.clip(1.0 - g(m), 0, None) * m
    out = (g(F * m[..., None]) + miss[..., None] * F) * m[..., None]
    om = np.where(view.valid & (view.omega > 0), view.omega, np.inf)
    return out / om[..., None]


def deposit_points_eye(img, view, dirs, irr, pcfg, grid, background):
    """deposit_points with naked-eye detection (camera.mode eye): each source's photopic
    illuminance E (lux) is compared with threshold_illuminance of the background luminance at its
    pixel (background = the sky radiance image before point sources); a fraction p
    (detection_probability, points.visibility_factor / visibility_sigma_mag) is deposited as a
    point, the rest as unresolved light blurred by points.unresolved_blur_deg (>=
    unresolved_min_px pixels; one pixel scale per view, view_pixel_deg), so the frame flux is
    unchanged. Returns (number deposited, info dict)."""
    fac = pcfg.get("visibility_factor", 1.0)
    info = {"n_points_in_frame": 0, "n_points_visible": 0.0, "visibility_factor": fac}
    if len(dirs) == 0:
        return 0, info
    row, col, irr = _project_points(view, dirs, irr, pcfg, grid)
    if len(row) == 0:
        return 0, info
    H, W = view.height, view.width
    r = np.clip(np.round(row).astype(int), 0, H - 1)
    c = np.clip(np.round(col).astype(int), 0, W - 1)
    lb = spectral.luminance(np.asarray(background[r, c], float), grid)
    p = detection_probability(spectral.to_xyz(irr, grid)[:, 1], lb, fac, pcfg.get("visibility_sigma_mag", 0.3))
    flat = img.reshape(H * W, -1)
    diff = np.zeros(flat.shape)
    omega = view.omega.ravel()
    for s, idx, w, k in _stencils(view, row, col, pcfg, grid):
        e, pk = irr[s], p[s][k]
        for b in range(flat.shape[1]):
            we = w * e[k, b]
            flat[:, b] += (np.bincount(idx, weights=we * pk, minlength=H * W) / omega).astype(flat.dtype)
            diff[:, b] += np.bincount(idx, weights=we * (1 - pk), minlength=H * W)
    sig = max(pcfg.get("unresolved_min_px", 1.5), pcfg.get("unresolved_blur_deg", 0.3) / view_pixel_deg(view))
    img += blur_flux(diff.reshape(H, W, -1), view, sig).astype(img.dtype)
    info.update(n_points_in_frame=int(len(row)), n_points_visible=float(p.sum()),
                n_points_p50=int((p >= 0.5).sum()), unresolved_blur_px=float(sig),
                unresolved_lux=float(spectral.to_xyz(diff.sum(0), grid)[1]),
                naked_eye_limit_V_star_bg_median=float(limiting_vmag(np.median(lb), fac)))
    return int(len(row)), info


def deposit_disk(img, view, d, irr, radius_deg, pcfg, grid):
    """Uniform disk (limb darkening ignored) of total irradiance irr (nbins,) centred on d;
    falls back to deposit_points when the disk spans fewer than ~3 pixels. A disk wholly
    inside the frame is normalised by the solid angle of the pixels it covers (flux
    conserving); a disk cut by the frame edge by the analytic cap 2 pi (1 - cos r)."""
    d = np.asarray(d, float)
    pix = np.degrees(np.sqrt(np.median(view.omega[view.valid])))
    if radius_deg < 1.5 * pix:
        return deposit_points(img, view, np.atleast_2d(d), np.atleast_2d(irr), pcfg, grid)
    inside = view.valid & (view.dirs @ d > np.cos(np.radians(radius_deg)))
    if not inside.any():
        return 0
    rim = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
    rim /= np.linalg.norm(rim)
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    a = np.radians(radius_deg + 2 * pix)
    ring = np.cos(a) * d + np.sin(a) * (np.cos(ang)[:, None] * rim + np.sin(ang)[:, None] * np.cross(d, rim))
    rr, cc = view.project(ring)
    whole = np.isfinite(rr).all() and view.valid[np.clip(np.round(rr).astype(int), 0, view.height - 1),
                                                  np.clip(np.round(cc).astype(int), 0, view.width - 1)].all() \
        and (rr >= 0).all() and (rr <= view.height - 1).all() and (cc >= 0).all() and (cc <= view.width - 1).all()
    omega = view.omega[inside].sum() if whole else 2 * np.pi * (1 - np.cos(np.radians(radius_deg)))
    img[inside] += (np.asarray(irr)[None] / omega).astype(img.dtype)
    return 1


# ----------------------------------------------------------------------------- photometry
def to_xyzs(img, grid):
    """(..., nbins) radiance -> (..., 4): X, Y (cd/m^2), Z, scotopic luminance (scotopic cd/m^2)."""
    xyz = spectral.to_xyz(img, grid)
    s = spectral.scotopic_luminance(img, grid)
    return np.concatenate([xyz, s[..., None]], axis=-1).astype(np.float32)


def mesopic_weight(la, lo=0.005, hi=5.0):
    """Photopic fraction m(La): 0 below lo (scotopic), 1 above hi, log-linear between."""
    return float(np.clip((np.log10(max(la, 1e-12)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)), 0, 1))


def log_average(x, mask, floor=ABS_THRESHOLD_CD_M2):
    v = np.maximum(np.asarray(x)[mask], floor)
    return float(np.exp(np.mean(np.log(v)))) if v.size else floor


def eye_adaptation(xyzs, mask, ccfg):
    """dict(La, m, scale) for the eye model; La in cd/m^2 (mesopic), scale multiplies luminance."""
    lo, hi = ccfg.get("mesopic_range_cd_m2", [0.005, 5.0])
    fixed = ccfg.get("adaptation_cd_m2")
    mes = ccfg.get("mesopic", True)
    if fixed:
        la = float(fixed)
    elif not mes:
        la = log_average(xyzs[..., 1], mask)
    else:
        # La = logavg(m(La) Y + (1 - m(La)) S): bisection in log La between the photopic and
        # scotopic log-averages (the residual La - logavg(...) increases monotonically with La)
        y = np.maximum(np.asarray(xyzs[..., 1])[mask], ABS_THRESHOLD_CD_M2).astype(np.float64)
        sc = np.maximum(np.asarray(xyzs[..., 3])[mask], ABS_THRESHOLD_CD_M2).astype(np.float64)
        f = lambda la: float(np.exp(np.mean(np.log(mesopic_weight(la, lo, hi) * y + (1 - mesopic_weight(la, lo, hi)) * sc)))) if y.size else ABS_THRESHOLD_CD_M2
        a, b = sorted((np.log(f(1e-12)), np.log(f(1e12))))
        for _ in range(40):
            c = 0.5 * (a + b)
            a, b = (c, b) if c < np.log(f(np.exp(c))) else (a, c)
        la = float(np.exp(0.5 * (a + b)))
    m = mesopic_weight(la, lo, hi) if mes else 1.0
    g, lref = ccfg.get("adaptation_gamma", 0.75), ccfg.get("l_ref_cd_m2", 30.0)
    la_eff = la ** g * lref ** (1 - g)
    return {"La_cd_m2": la, "photopic_fraction": m, "La_eff_cd_m2": la_eff, "scale": ccfg.get("key", 0.18) / la_eff}


def camera_exposure(ccfg):
    """Saturation luminance (cd/m^2) and EV100 for ISO S, f-number N, exposure time t."""
    s, n, t = float(ccfg["iso"]), float(ccfg["f_number"]), float(ccfg["exposure_s"])
    l_max = 78.0 / (0.65 * s) * n * n / t
    return {"L_sat_cd_m2": l_max, "EV100": float(np.log2(n * n / t * 100.0 / s)), "scale": 1.0 / l_max}


# ----------------------------------------------------------------------------- glare / bloom
def cie_glare_psf(theta_deg, age=25.0, p=0.5):
    """CIE 2002 general disability glare equation (Vos & van den Berg 1999), sr^-1.

    L_veil / E_glare at angle theta (deg) from a glare source; valid 0 - 100 deg.
    """
    t = np.asarray(theta_deg, float)
    a = (age / 70.0) ** 4
    q = 1 + (t / 0.1) ** 2
    return ((1 - 0.08 * a) * (9.2e6 / (1 + (t / 0.0046) ** 2) ** 1.5 + 1.5e5 / (1 + (t / 0.045) ** 2) ** 1.5)
            + (1 + 1.6 * a) * ((400 / q + 3e-8 * t ** 2) + p * (1300 / q ** 1.5 + 0.8 / q ** 0.5))
            + 2.5e-3 * p)


def encircled_glare(theta_max_deg, age=25.0, p=0.5, n=4000):
    """Integral of the CIE 2002 PSF over the cap theta < theta_max (dimensionless)."""
    t = np.geomspace(1e-5, theta_max_deg, n)
    f = cie_glare_psf(t, age, p) * 2 * np.pi * np.sin(np.radians(t)) * np.radians(1.0)
    return float(np.trapezoid(f, t) + cie_glare_psf(0.0, age, p) * np.pi * np.radians(1e-5) ** 2)


def _radial_kernel(fn, pix_deg, half, ss_radius=3, ss=8):
    """Pixel kernel K[i, j] = integral of fn(theta) over pixel (i, j) (fn in sr^-1)."""
    o = np.arange(-half, half + 1)
    Y, X = np.meshgrid(o, o, indexing="ij")
    pix_sr = np.radians(pix_deg) ** 2
    k = fn(np.hypot(X, Y) * pix_deg) * pix_sr
    c = half
    sub = (np.arange(ss) + 0.5) / ss - 0.5
    sy, sx = np.meshgrid(sub, sub, indexing="ij")
    for i in range(-ss_radius, ss_radius + 1):
        for j in range(-ss_radius, ss_radius + 1):
            if (i or j) and abs(i) <= half and abs(j) <= half:
                k[c + i, c + j] = fn(np.hypot(j + sx, i + sy) * pix_deg).mean() * pix_sr
    return k


def view_pixel_deg(view):
    """Angular pixel size (deg): the radial step of the equidistant all-sky fisheye
    (2 x 95 deg across the width), else sqrt of the median pixel solid angle."""
    if view.kind == "allsky":
        return 2 * ALLSKY_MAX_ZENITH_DEG / view.width
    return float(np.degrees(np.sqrt(np.median(view.omega[view.valid]))))


def glare_kernel(view, ccfg):
    """Normalised eye PSF on the view's pixel grid (centre pixel holds the unscattered core)."""
    age, p = ccfg.get("age_years", 25.0), ccfg.get("pigmentation", 0.5)
    pix = view_pixel_deg(view)
    half = int(min(np.ceil(ccfg.get("glare_max_deg", 60.0) / pix), max(view.height, view.width)))
    k = _radial_kernel(lambda t: cie_glare_psf(t, age, p), pix, half)
    # circular support with a cos^2 taper over the outer 30%: a hard square edge shows up as a
    # bright box around the Sun (the wing at 60 deg is a few % of the blue sky)
    o = np.arange(-half, half + 1) / max(half, 1)
    r = np.hypot(*np.meshgrid(o, o, indexing="ij"))
    k *= np.cos(0.5 * np.pi * np.clip((r - 0.7) / 0.3, 0, 1)) ** 2
    total = encircled_glare(100.0, age, p)
    c = half
    k[c, c] = 0.0
    k[c, c] = encircled_glare(half * pix, age, p) - k.sum()
    return k / total


def lens_kernel(view, ccfg):
    """Camera PSF: (1 - b - s) delta + b Gaussian bloom + s diffraction spikes."""
    pix = view_pixel_deg(view)
    b, ns, sf = ccfg.get("bloom", 0.0), int(ccfg.get("spikes", 0)), ccfg.get("spike_fraction", 0.0)
    sig = max(ccfg.get("bloom_sigma_deg", 0.15) / pix, 0.5)
    half = int(min(max(4 * sig, 8 if ns else 0, 1), max(view.height, view.width) // 2))
    if ns:
        half = max(half, min(200, max(view.height, view.width) // 4))
    o = np.arange(-half, half + 1)
    Y, X = np.meshgrid(o, o, indexing="ij")
    k = np.zeros(X.shape)
    if b > 0:
        g = np.exp(-0.5 * (X ** 2 + Y ** 2) / sig ** 2)
        k += b * g / g.sum()
    if ns and sf > 0:
        sp = np.zeros(X.shape)
        r = np.hypot(X, Y)
        for a in np.arange(ns) * 2 * np.pi / ns + np.pi / 4:
            along = X * np.cos(a) + Y * np.sin(a)
            across = -X * np.sin(a) + Y * np.cos(a)
            sp += np.where(along > 0, np.exp(-0.5 * (across / 0.7) ** 2) / (1 + (r / 3.0) ** 2), 0.0)
        sp[half, half] = 0.0
        k += sf * sp / sp.sum()
    k[half, half] += 1.0 - k.sum()
    return k


def convolve(img, kernel, mask=None, workers=1):
    """FFT convolution of (H, W, C) with a (2h+1, 2h+1) kernel; same-size output.

    Light from outside the frame / mask is approximated by each pixel's own value: the kernel
    weight that falls outside (1 - K * mask) is returned to the pixel, so a uniform field stays
    uniform up to the edges. Float64 transforms: float32 round-off along the row and column of
    a source 1e7 times brighter than the sky shows up as streaks.
    """
    import scipy.fft as sfft
    H, W, C = img.shape
    h = kernel.shape[0] // 2
    m = np.ones((H, W)) if mask is None else mask.astype(float)
    sh = (sfft.next_fast_len(H + 2 * h, real=True), sfft.next_fast_len(W + 2 * h, real=True))
    kf = sfft.rfft2(kernel.astype(np.float64), sh, workers=workers)
    conv = lambda a: sfft.irfft2(sfft.rfft2(a, sh, workers=workers) * kf, sh, workers=workers)[h:h + H, h:h + W]
    miss = np.clip(1.0 - conv(m), 0, None) * m
    out = np.empty(img.shape, dtype=np.float32)
    for c in range(C):
        a = img[..., c].astype(np.float64) * m
        out[..., c] = np.maximum(conv(a), 0) + miss * a
    return out


# ----------------------------------------------------------------------------- tone mapping
def reinhard(y, white):
    return y * (1 + y / white ** 2) / (1 + y)


def filmic(x):
    """ACES filmic fit (Narkowicz 2015), input linear with 1 = sensor saturation."""
    x = np.maximum(x, 0)
    return np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def srgb_encode(rgb):
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)


def xyz_to_rgb(xyz):
    return np.asarray(xyz) @ ph.XYZ_TO_SRGB.T


def develop(img, view, grid, ccfg, mask=None, cpus=1):
    """Spectral radiance image -> (uint8 sRGB (H, W, 3), info dict, linear XYZS (H, W, 4)).

    mask: pixels used for the adaptation average (default view.valid).
    """
    mask = view.valid if mask is None else mask
    xyzs = to_xyzs(img, grid)
    xyzs[~view.valid] = 0
    mode = ccfg.get("mode", "eye")
    info = {"mode": mode}
    if mode == "eye":
        ad = eye_adaptation(xyzs, mask, ccfg)
        info.update(ad)
        if ccfg.get("glare", True):
            k = glare_kernel(view, ccfg)
            xyzs = convolve(xyzs, k, view.valid, cpus)
            info["glare_kernel_px"] = int(k.shape[0])
        m = ad["photopic_fraction"]
        if ccfg.get("mesopic", True) and ccfg.get("local_mesopic", True):
            # foveal cones see bright point sources in colour: the photopic fraction of each pixel
            # is at least m(Y) of its own retinal-image luminance (after glare)
            lo, hi = ccfg.get("mesopic_range_cd_m2", [0.005, 5.0])
            mpx = np.clip(np.log10(np.maximum(xyzs[..., 1], 1e-12) / lo) / np.log10(hi / lo), 0, 1)
            m = np.maximum(m, mpx)[..., None]
            info["local_mesopic"] = True
        disp = m * xyzs[..., :3] + (1 - m) * xyzs[..., 3:4] * TINT_XYZ
        rgb = xyz_to_rgb(disp) * ad["scale"]
        tone = ccfg.get("tone") or "reinhard"
    elif mode == "camera":
        ex = camera_exposure(ccfg)
        info.update(ex)
        if ccfg.get("bloom", 0) > 0 or ccfg.get("spikes", 0):
            xyzs = convolve(xyzs, lens_kernel(view, ccfg), view.valid, cpus)
        rgb = xyz_to_rgb(xyzs[..., :3]) * ex["scale"]
        tone = ccfg.get("tone") or "filmic"
    else:
        raise ValueError(f"camera.mode must be eye or camera, got {mode!r}")
    if tone == "reinhard":
        y = np.maximum(rgb @ np.array([0.2126, 0.7152, 0.0722]), 1e-12)
        rgb = rgb * (reinhard(y, ccfg.get("white", 6.0)) / y)[..., None]
    elif tone == "filmic":
        rgb = filmic(rgb)
    elif tone != "linear":
        raise ValueError(f"unknown tone curve {tone!r}")
    info["tone"] = tone
    out = (srgb_encode(rgb) * 255 + 0.5).astype(np.uint8)
    out[~view.valid] = 0
    return out, info, xyzs


def annotate(img8, view, lines=()):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img8)
    dr = ImageDraw.Draw(im)
    for i, t in enumerate(lines):
        dr.text((8, 6 + 12 * i), t, fill=(225, 225, 225), stroke_width=1, stroke_fill=(0, 0, 0))
    if view.kind == "allsky":
        w, h = im.size
        for lab, (x, y) in {"N": (w / 2 - 3, 4), "S": (w / 2 - 3, h - 14), "W": (w - 14, h / 2 - 5), "E": (6, h / 2 - 5)}.items():
            dr.text((x, y), lab, fill=(220, 220, 130), stroke_width=1, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def write_outputs(base, rgb8, xyzs, info, spectral_img=None, extra=None, png=True, npz=True):
    """Write <base>.png, <base>.npz (XYZ, scotopic, extras, optional spectral) and <base>.json."""
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    paths = []
    if png:
        from PIL import Image
        Image.fromarray(rgb8).save(base + ".png")
        paths.append(base + ".png")
    if npz:
        arrs = {"xyz": xyzs[..., :3].astype(np.float32), "scotopic": xyzs[..., 3].astype(np.float32)}
        arrs.update({k: np.asarray(v) for k, v in (extra or {}).items()})
        if spectral_img is not None:
            arrs["spectral"] = spectral_img.astype(np.float32)
        np.savez_compressed(base + ".npz", **arrs)
        paths.append(base + ".npz")
    with open(base + ".json", "w") as f:
        json.dump(_jsonable(info), f, indent=1)
    paths.append(base + ".json")
    return paths
