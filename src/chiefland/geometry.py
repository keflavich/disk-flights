"""Observer, time and image projection for the Chiefland sky frames (spec from the
colloquia-2026 session, 2026-10-29).

Stereographic projection about the horizon point (az AZ0, alt 0): the horizon is a straight
line, zenith up, azimuth increases to the right. Continuous pixel coordinates: pixel (i, j) has
its centre at (i + 0.5, j + 0.5), j increases downward. No refraction (pressure 0), so the stars
register with the slide overlays.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, ICRS
from astropy.time import Time

AZ0 = 245.0
LOCATION = EarthLocation(lat=29.48 * u.deg, lon=-82.86 * u.deg, height=15 * u.m)
TIME = Time("2026-11-08T00:00:00", scale="utc")

# name -> (width, height, f, x0, y0)
FRAMES = {
    "main": (4096, 3072, 1463.329, 2048.0, 2826.24),
    "teapot": (3072, 3072, 4389.987, 3441.0, 2991.72),
}

# peer check points (continuous px in the main frame)
CHECK_POINTS = {
    "Altair": (97649, 1880.5, 1199.1),
    "Vega": (91262, 3065.3, 1155.9),
    "Deneb": (102098, 2978.4, 146.9),
    "Nunki": (92855, 1390.6, 2219.3),
    "Kaus Australis": (90185, 1371.8, 2507.7),
    "Sgr A*": (None, 1627.3, 2568.9),
}
SGR_A_ICRS = (266.41683667, -29.00781056)


def altaz_frame():
    return AltAz(obstime=TIME, location=LOCATION, pressure=0 * u.hPa)


def pix_to_altaz(px, py, f, x0, y0):
    """Continuous pixel coords -> (alt, az) in degrees (arrays)."""
    x = (np.asarray(px, float) - x0) / f
    y = (y0 - np.asarray(py, float)) / f
    rho = np.maximum(np.hypot(x, y), 1e-12)
    c = 2 * np.arctan(rho / 2)
    alt = np.degrees(np.arcsin(np.clip(y * np.sin(c) / rho, -1, 1)))
    az = (AZ0 + np.degrees(np.arctan2(x * np.sin(c), rho * np.cos(c)))) % 360
    return alt, az


def altaz_to_pix(alt, az, f, x0, y0):
    """(alt, az) degrees -> continuous pixel coords (px, py)."""
    a = np.radians(alt)
    dz = np.radians(np.asarray(az, float) - AZ0)
    k = 2 / (1 + np.cos(a) * np.cos(dz))
    return x0 + f * k * np.cos(a) * np.sin(dz), y0 - f * k * np.sin(a)


def pixel_scale_sr(px, py, f, x0, y0):
    """Solid angle per pixel (sr) at continuous pixel coords: stereographic scale
    k = 2 / (1 + cos c) = 1 + rho^2 / 4, so one pixel spans 1 / (f k) rad."""
    x = (np.asarray(px, float) - x0) / f
    y = (y0 - np.asarray(py, float)) / f
    k = 1 + (x * x + y * y) / 4
    return (1.0 / (f * k)) ** 2


def icrs_to_altaz(ra_deg, dec_deg):
    """Exact apparent (alt, az) in degrees, no refraction."""
    c = SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg, frame="icrs")
    aa = c.transform_to(altaz_frame())
    return aa.alt.deg, aa.az.deg


def altaz_to_unit(alt, az):
    """Horizon unit vector (x = N, y = E, z = Up) for (alt, az) in degrees."""
    a, z = np.radians(alt), np.radians(az)
    return np.stack([np.cos(a) * np.cos(z), np.cos(a) * np.sin(z), np.sin(a)], axis=-1)


def pixel_icrs_grid(frame, step=16):
    """ICRS unit vectors for every pixel centre of `frame`: exact astropy AltAz -> ICRS on a
    coarse grid of continuous pixel coords (every `step` px), bilinearly interpolated and
    renormalised. Interpolation error is far below a pixel. Returns (H, W, 3) float32."""
    W, H, f, x0, y0 = FRAMES[frame]
    gx = np.arange(0, W + step, step, dtype=float)
    gy = np.arange(0, H + step, step, dtype=float)
    PX, PY = np.meshgrid(gx, gy)
    alt, az = pix_to_altaz(PX, PY, f, x0, y0)
    aa = SkyCoord(alt=alt.ravel() * u.deg, az=az.ravel() * u.deg, frame=altaz_frame())
    ic = aa.transform_to(ICRS())
    v = ic.cartesian.xyz.value.T.reshape(len(gy), len(gx), 3)
    # bilinear interpolation to pixel centres
    cx = (np.arange(W) + 0.5) / step
    cy = (np.arange(H) + 0.5) / step
    ix = np.minimum(cx.astype(int), len(gx) - 2)
    iy = np.minimum(cy.astype(int), len(gy) - 2)
    tx = (cx - ix)[None, :, None]
    ty = (cy - iy)[:, None, None]
    out = np.empty((H, W, 3), np.float32)
    for r0 in range(0, H, 256):
        r1 = min(H, r0 + 256)
        yy = iy[r0:r1]
        t = ty[r0:r1]
        v00 = v[yy][:, ix]
        v01 = v[yy][:, ix + 1]
        v10 = v[yy + 1][:, ix]
        v11 = v[yy + 1][:, ix + 1]
        blk = (1 - t) * ((1 - tx) * v00 + tx * v01) + t * ((1 - tx) * v10 + tx * v11)
        blk /= np.linalg.norm(blk, axis=-1, keepdims=True)
        out[r0:r1] = blk
    return out
