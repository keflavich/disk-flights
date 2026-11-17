"""Turn six cube-face renders into arbitrary all-sky / perspective views, and
inject unresolved point-like sources (Sun disk, stars) analytically.

Scene axes: x = North, y = West, z = Up (right-handed; looking north with z up,
East is to the image right).
"""
import numpy as np
from scipy.ndimage import map_coordinates

FACES = [([1, 0, 0], [0, 0, 1]), ([0, 1, 0], [0, 0, 1]), ([-1, 0, 0], [0, 0, 1]),
         ([0, -1, 0], [0, 0, 1]), ([0, 0, 1], [1, 0, 0]), ([0, 0, -1], [1, 0, 0])]
FACE_FOV_DEG = 92.0
BLEND_INNER_TAN = np.tan(np.radians(42.0))


def face_basis(k):
    fwd = np.array(FACES[k][0], float)
    right = np.cross(fwd, FACES[k][1])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return fwd, right, up


def direction_to_face_pixel(dirs, k, size):
    """Fractional (row, col) of unit directions in face k, and a validity mask."""
    fwd, right, up = face_basis(k)
    z = dirs @ fwd
    t = np.tan(np.radians(FACE_FOV_DEG) / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (dirs @ right) / z / t
        v = (dirs @ up) / z / t
    col = (u + 1) / 2 * size - 0.5
    row = (1 - v) / 2 * size - 0.5
    return row, col, z


def sample_cube(faces, dirs, horizon_margin_deg=0.0):
    """Bilinearly sample the cube map. faces: (6, S, S[, C]); dirs: (..., 3) unit vectors.

    horizon_margin_deg > 0 pushes directions away from the horizon (sky side is
    sampled at >= +margin, ground side at <= -margin) so the blur of the sharp
    sky/ground edge in the low-resolution render does not leak across it.
    """
    faces = np.asarray(faces)
    size = faces.shape[1]
    shp = dirs.shape[:-1]
    d = dirs.reshape(-1, 3).astype(float)
    if horizon_margin_deg > 0:
        el = np.arcsin(np.clip(d[:, 2], -1, 1))
        m = np.radians(horizon_margin_deg)
        el2 = np.where(el >= 0, np.maximum(el, m), np.minimum(el, -m))
        horiz = np.linalg.norm(d[:, :2], axis=1)
        scale = np.cos(el2) / np.maximum(horiz, 1e-12)
        d = np.column_stack([d[:, 0] * scale, d[:, 1] * scale, np.sin(el2)])
    chan = faces.shape[3] if faces.ndim == 4 else 1
    out = np.zeros((len(d), chan))
    wsum = np.zeros(len(d))
    tmax = np.tan(np.radians(FACE_FOV_DEG) / 2)
    for k in range(6):
        fwd, right, up = face_basis(k)
        z = d @ fwd
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.maximum(np.abs(d @ right), np.abs(d @ up)) / z
        # full weight inside 42 deg of the face axis, fading to zero at its 46 deg edge, so
        # neighbouring faces cross-fade over their overlap instead of meeting at a hard seam
        x = np.clip((tmax - m) / (tmax - BLEND_INNER_TAN), 0, 1)
        w = np.where(z > 0, x * x * (3 - 2 * x), 0.0)
        sel = w > 0
        if not sel.any():
            continue
        row, col, _ = direction_to_face_pixel(d[sel], k, size)
        for c in range(chan):
            img = faces[k][..., c] if faces.ndim == 4 else faces[k]
            out[sel, c] += w[sel] * map_coordinates(img, [row, col], order=1, mode="nearest")
        wsum[sel] += w[sel]
    out /= np.maximum(wsum, 1e-9)[:, None]
    return out.reshape(*shp, chan) if faces.ndim == 4 else out.reshape(shp)


class View:
    """A named camera model giving unit view directions per output pixel."""

    def __init__(self, kind, width, height, **kw):
        self.kind, self.width, self.height = kind, width, height
        x = (np.arange(width) + 0.5) / width * 2 - 1
        y = 1 - (np.arange(height) + 0.5) / height * 2
        X, Y = np.meshgrid(x, y)
        if kind == "allsky":
            # equidistant fisheye, zenith at centre, horizon on the inscribed circle,
            # north at the top, east to the LEFT (as an all-sky camera looking up sees it)
            r = np.hypot(X, Y)
            theta = r * np.radians(kw.get("max_zenith_deg", 95.0))
            with np.errstate(invalid="ignore", divide="ignore"):
                sx, sy = np.where(r > 0, X / r, 0), np.where(r > 0, Y / r, 0)
            north, east = sy * np.sin(theta), -sx * np.sin(theta)
            self.dirs = np.stack([north, -east, np.cos(theta)], axis=-1)  # y = West = -East
            self.valid = r <= 1.0
        elif kind == "perspective":
            az, pitch, fov = (np.radians(kw.get(k, d)) for k, d in
                              (("azimuth_deg", 0.0), ("pitch_deg", 15.0), ("fov_deg", 100.0)))
            # azimuth measured from north towards east
            fwd = np.array([np.cos(pitch) * np.cos(az), -np.cos(pitch) * np.sin(az), np.sin(pitch)])
            right = np.cross(fwd, [0, 0, 1.0])
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            t = np.tan(fov / 2)
            d = fwd + right * (X[..., None] * t) + up * (Y[..., None] * t * height / width)
            self.dirs = d / np.linalg.norm(d, axis=-1, keepdims=True)
            self.valid = np.ones((height, width), bool)
        else:
            raise ValueError(kind)
        self.dirs = self.dirs / np.linalg.norm(self.dirs, axis=-1, keepdims=True)
        # solid angle per pixel from the direction map
        dx = np.gradient(self.dirs, axis=1)
        dy = np.gradient(self.dirs, axis=0)
        self.omega = np.linalg.norm(np.cross(dx, dy), axis=-1)

    def project(self, dirs):
        """Sub-pixel (row, col) of unit directions, NaN where not in the view."""
        dirs = np.atleast_2d(dirs)
        W, H = self.width, self.height
        if self.kind == "allsky":
            north, east, up = dirs[:, 0], -dirs[:, 1], dirs[:, 2]
            theta = np.arccos(np.clip(up, -1, 1))
            r = theta / np.radians(95.0)
            phi = np.hypot(north, east)
            with np.errstate(invalid="ignore", divide="ignore"):
                X, Y = -r * east / phi, r * north / phi
            X = np.where(phi > 0, X, 0)
            Y = np.where(phi > 0, Y, 0)
            ok = r <= 1.0
        else:
            # invert the perspective map by least squares on the direction grid basis
            c = self.dirs[self.height // 2, self.width // 2]
            fwd = c
            right = self.dirs[self.height // 2, min(self.width - 1, self.width // 2 + 1)] - self.dirs[
                self.height // 2, max(0, self.width // 2 - 1)]
            right -= fwd * (right @ fwd)
            right /= np.linalg.norm(right)
            up = np.cross(right, fwd)
            z = dirs @ fwd
            xa = self.dirs[self.height // 2, -1] @ right / (self.dirs[self.height // 2, -1] @ fwd)
            with np.errstate(invalid="ignore", divide="ignore"):
                X = (dirs @ right) / z / xa
                Y = (dirs @ up) / z / xa * self.width / self.height
            ok = z > 0
        col = (X + 1) / 2 * self.width - 0.5
        row = (1 - Y) / 2 * self.height - 0.5
        ok &= (row > -2) & (row < self.height + 1) & (col > -2) & (col < self.width + 1)
        return np.where(ok, row, np.nan), np.where(ok, col, np.nan)


def add_points(image, view, dirs, flux_maps, sigma_px=0.9, radius_px=4):
    """Add point-like sources to a (H, W, C) linear image (units: per-pixel radiance).

    flux_maps: (n, C) irradiance-like fluxes in the same units as image*steradian;
    each is spread with a Gaussian PSF of width sigma_px and divided by the local
    pixel solid angle, so a source of irradiance E adds E/omega in total.
    """
    row, col = view.project(dirs)
    H, W = image.shape[:2]
    for i in range(len(dirs)):
        if np.isnan(row[i]):
            continue
        r0, c0 = int(round(row[i])), int(round(col[i]))
        rs = np.arange(max(0, r0 - radius_px), min(H, r0 + radius_px + 1))
        cs = np.arange(max(0, c0 - radius_px), min(W, c0 + radius_px + 1))
        if len(rs) == 0 or len(cs) == 0:
            continue
        R, C = np.meshgrid(rs, cs, indexing="ij")
        psf = np.exp(-0.5 * ((R - row[i]) ** 2 + (C - col[i]) ** 2) / sigma_px ** 2)
        psf /= psf.sum()
        # weight in solid angle: each pixel receives psf * E, expressed as radiance E*psf/omega_pix
        image[R, C] += psf[..., None] * flux_maps[i] / view.omega[R, C][..., None]
    return image
