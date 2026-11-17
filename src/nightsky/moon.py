"""The Moon: position, phase-law brightness, spectrum and a shaded disk for camera views.

Config (cfg['scenario']['moon']):
  {enabled: false, elongation_deg: 180.0, ecliptic_latitude_deg: 0.0, dmag: 0.0}
The Moon sits at ecliptic longitude lambda_sun + elongation (lambda_sun from
scenario.sun.ecliptic_longitude_deg, used even when the Sun itself is disabled), so
elongation 180 is full, 90 first quarter, 270 last quarter. Phase angle alpha = 180 - elongation
(folded to 0..180; lunar parallax ignored).

Brightness: Krisciunas & Schaefer (1991, PASP 103, 1033) V(alpha) = -12.73 + 0.026|alpha|
+ 4e-9 alpha^4 at mean distance, which includes the opposition surge. `dmag` shifts it.
Spectrum: solar blackbody (5772 K) times a lunar reflectance rising as (lambda/550)^1.1,
which reproduces the Moon's B-V = 0.92 against the Sun's 0.65.
The disk is shaded with Lommel-Seeliger mu0 / (mu0 + mu), which has no limb darkening at
full phase, and lit from the Sun direction (terminator), normalised to the total irradiance.
Earthshine and the light of the local cluster on the night side are neglected.
"""
import numpy as np

import photometry as ph

from . import sky, spectral

RADIUS_DEG = 0.2592  # mean angular radius (15.55 arcmin)
REFLECTANCE_SLOPE = 1.1
SUN_TEFF_K = 5772.0


def ecliptic_celestial(lon_deg, lat_deg=0.0, obliquity_deg=23.44):
    lam, beta, eps = np.radians(lon_deg), np.radians(lat_deg), np.radians(obliquity_deg)
    x, y, z = np.cos(beta) * np.cos(lam), np.cos(beta) * np.sin(lam), np.sin(beta)
    return np.array([x, y * np.cos(eps) - z * np.sin(eps), y * np.sin(eps) + z * np.cos(eps)])


def phase_angle_deg(elongation_deg):
    e = np.mod(elongation_deg, 360.0)
    return 180.0 - np.minimum(e, 360.0 - e)


def vmag(alpha_deg, dmag=0.0):
    a = np.abs(alpha_deg)
    return -12.73 + 0.026 * a + 4e-9 * a ** 4 + dmag


def spectrum(v, grid):
    """(n,) irradiance of the Moon with V magnitude v (normalised at 550 nm like spectral.vmag_irradiance)."""
    s = spectral.blackbody_irradiance(1.0, SUN_TEFF_K, grid) * (grid.wl / 550.0) ** REFLECTANCE_SLOPE
    return s * ph.F_V0_550NM * 10 ** (-0.4 * v) / np.interp(550.0, grid.wl, s)


def moon_sources(cfg, grid):
    sc = cfg["scenario"]
    m = sc.get("moon") or {}
    if not m.get("enabled"):
        return sky.PointSources.empty(grid.n)
    lam_sun = (sc.get("sun") or {}).get("ecliptic_longitude_deg", 0.0)
    elong = float(m.get("elongation_deg", 180.0))
    if sc.get("mode") == "earth":
        # Earth mode measures RA from starfield.sun_ra_deg (the Sun on celestial +x), so place the
        # Moon on the true ecliptic and rotate RA by -sun_ra
        ra0 = np.radians((cfg.get("starfield") or {}).get("sun_ra_deg", 180.0))
        lam_true = np.degrees(np.arctan2(np.sin(ra0), np.cos(ra0) * np.cos(np.radians(23.44))))
        e = ecliptic_celestial(lam_true + lam_sun + elong, m.get("ecliptic_latitude_deg", 0.0))
        c, s = np.cos(ra0), np.sin(ra0)
        d = np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1], e[2]])
    else:
        d = ecliptic_celestial(lam_sun + elong, m.get("ecliptic_latitude_deg", 0.0))
    v = vmag(phase_angle_deg(elong), m.get("dmag", 0.0))
    return sky.PointSources(d[None], spectrum(v, grid)[None], ["Moon"])


def disk_weights(dirs, d, sun_dir, radius_deg=RADIUS_DEG):
    """Lommel-Seeliger brightness (unnormalised) of the lunar disk centred on d at view
    directions dirs (..., 3); 0 outside the disk and on the night side."""
    d = np.asarray(d, float)
    s = np.asarray(sun_dir, float) / np.linalg.norm(sun_dir)
    cos_t = dirs @ d
    perp = dirs - cos_t[..., None] * d
    rho = np.linalg.norm(perp, axis=-1) / np.sin(np.radians(radius_deg))
    inside = rho < 1.0
    mu = np.sqrt(np.clip(1 - rho * rho, 0, 1))
    # outward surface normal of the visible hemisphere (faces the observer, i.e. along -d)
    n = perp / np.sin(np.radians(radius_deg)) - mu[..., None] * d
    mu0 = n @ s
    w = np.where(inside & (mu0 > 0), mu0 / np.maximum(mu0 + mu, 1e-9), 0.0)
    return w


def _disk_integral(d, sun_dir, radius_deg, n=401):
    """Integral of disk_weights over solid angle, on a fine tangent-plane grid."""
    r = np.sin(np.radians(radius_deg))
    e1 = np.cross(d, [0.0, 0.0, 1.0] if abs(d[2]) < 0.9 else [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    u = (np.arange(n) + 0.5) / n * 2 - 1
    U, V = np.meshgrid(u * r, u * r, indexing="ij")
    p = d + U[..., None] * e1 + V[..., None] * e2
    p /= np.linalg.norm(p, axis=-1, keepdims=True)
    return float(disk_weights(p, d, sun_dir, radius_deg).sum() * (2 * r / n) ** 2)


def deposit(img, view, d, irr, sun_dir, pcfg, grid, radius_deg=RADIUS_DEG):
    """Add the shaded lunar disk (total irradiance irr) to img; falls back to a point below
    ~1.5 pixels radius. Returns the number of sources deposited (0 or 1)."""
    from . import camera
    d = np.asarray(d, float)
    pix = np.degrees(np.sqrt(np.median(view.omega[view.valid])))
    if radius_deg < 1.5 * pix:
        return camera.deposit_points(img, view, np.atleast_2d(d), np.atleast_2d(irr), pcfg, grid)
    near = view.valid & (view.dirs @ d > np.cos(np.radians(radius_deg + 2 * pix)))
    if not near.any():
        return 0
    w = disk_weights(view.dirs[near], d, sun_dir, radius_deg)
    tot = _disk_integral(d, sun_dir, radius_deg)
    if tot <= 0 or not w.any():
        return 0
    img[near] += (w[:, None] * np.asarray(irr)[None] / tot).astype(img.dtype)
    return 1


def inset(rgb8, view, d, sun_dir, irr, grid, alpha=None, radius_deg=RADIUS_DEG, white=6.0,
          night_veil=1.0):
    """Foveal inset: re-tone the lunar disk in an 8-bit image as seen when the eye adapts to the
    Moon itself (the global adaptation clips the disk to white and hides the phase). The disk
    is shaded by disk_weights, coloured by the Moon's spectrum and blended by pixel coverage.
    No disk pixel ends darker than night_veil x the glare veil already there: the veil lies over
    the night side as much as over the sky next to the limb, and a lit side darker than its own
    halo reads as a dark spot. So the phase shows only where the veil around the Moon is below
    white; a saturated halo (any night-adapted eye, full Moon) leaves a white disk.
    alpha (H, W): foreground coverage."""
    from . import camera
    d = np.asarray(d, float)
    pix = np.sqrt(np.median(view.omega[view.valid]))
    r_px = np.sin(np.radians(radius_deg)) / pix
    if r_px < 2.0:
        return rgb8
    near = view.valid & (view.dirs @ d > np.cos(np.radians(radius_deg) + 2 * pix))
    if not near.any():
        return rgb8
    dirs = view.dirs[near]
    sr = np.sin(np.radians(radius_deg))
    perp = dirs - (dirs @ d)[:, None] * d
    rho = np.linalg.norm(perp, axis=-1) / sr
    cov = np.clip((1 - rho) * r_px + 0.5, 0, 1)
    # shade rim pixels (centres just outside the limb) at the limb itself
    rc = np.minimum(rho, 0.999)
    pin = perp / np.maximum(rho, 1e-9)[:, None] * rc[:, None]
    w = disk_weights(np.sqrt(1 - (rc * sr) ** 2)[:, None] * d + pin, d, sun_dir, radius_deg)
    if alpha is not None:
        cov = cov * (1 - np.clip(alpha[near], 0, 1))
    rgb = np.clip(camera.xyz_to_rgb(spectral.to_xyz(np.asarray(irr, float)[None], grid))[0], 0, None)
    rgb = 0.7 + 0.3 * rgb / max(rgb.max(), 1e-30)  # the Moon at photopic levels: pale, slightly warm
    v = camera.reinhard(8.0 * w / max(w.max(), 1e-30), white)
    lit = camera.srgb_encode(v[:, None] * rgb[None]) * 255
    old = rgb8[near].astype(float)
    new = np.maximum(lit, night_veil * old)
    out = rgb8.copy()
    out[near] = np.clip(cov[:, None] * new + (1 - cov[:, None]) * old + 0.5, 0, 255).astype(np.uint8)
    return out
