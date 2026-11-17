"""Procedural terrain, vegetation and water geometry for the foreground renderer (numpy only).

Frame and units: horizon frame, metres, x = North, y = West, z = Up; the observer stands at
the origin with the ground surface at z ~ 0 under the feet (eye at z = eye_height).
Heights include the planetary-curvature drop -d^2 / (2 R) (R = 6371 km for an Earth-sized
planet; no refraction correction).

Geometry
  - noise: own 2-D gradient (Perlin 1985/2002-style) noise with a seeded hash table, fBm and
    Musgrave's ridged multifractal (Ebert et al. 2003, "Texturing & Modeling", ch. 16).
  - ground: one polar mesh, ring radii in geometric progression (cell aspect ~1), so the cell
    size grows linearly with distance (about 1 cm at 3 m and 100-150 m at 40 km for the
    default 1536 azimuth samples); a fan closes the centre.
  - scene types: "mountains" (meadow foreground, ridged ranges from ~3 to 40 km reaching
    ~1.5-2.5 km), "meadow" (rolling field, treeline at ~0.3-0.8 km), "forest" (clearing
    of radius ~35 m in mixed conifer/deciduous forest), "lakeshore" (observer on a bank 1.2 m above
    a lake whose shore is ~10-40 m to the north, ranges behind), "plane" (flat Lambertian disc for tests).
  - trees: unit-height prototype meshes (conifer: tapered trunk + stacked star-shaped
    drooping cones; deciduous: trunk + branches + a crown of noisy ellipsoidal clumps),
    placed with random height, yaw, aspect and small tilt. Typical heights: spruce/fir
    12-30 m, broadleaf 10-22 m.
  - surface materials are per-vertex weights over albedo roles (see albedo.py) plus a
    brightness modulation that stands in for grass-blade-scale texture near the camera.
    Snow covers gentle slopes (nz > 0.5-0.7) above snowline_m +- 150 m, shifted by +-250 m of
    km-scale noise, outside the forest texture; snowline_m None (default) takes the albedo
    set's season value (albedo.SNOWLINE_M: summer 2800, autumn 1800, winter 0 m; the default
    ranges peak at ~2.0-2.3 km, so summer has no snow). resolve() fills it in.
"""
import functools

import numpy as np

from . import albedo as alb

R_PLANET_M = 6.371e6
ROLES = ("grass", "grass_dry", "soil", "rock", "snow", "needles", "lakebed")

DEFAULTS = dict(type="mountains", seed=1, extent_m=60000.0, n_theta=1536, r_inner=0.25,
                eye_height=1.7, tree_density=1.0, max_trees=3000, n_conifer_protos=8,
                n_deciduous_protos=6, snowline_m=None, mountain_height_m=3200.0,
                water_level_m=-1.2, planet_radius_m=R_PLANET_M)


def snowline(cfg):
    """Snowline (m): cfg snowline_m, or when None the default of cfg albedo_set (summer if absent)."""
    s = cfg.get("snowline_m")
    return float(alb.snowline(cfg.get("albedo_set", "summer")) if s is None else s)


def resolve(cfg):
    """DEFAULTS merged with cfg, snowline_m resolved to metres (extra keys such as albedo_set kept)."""
    cfg = {**DEFAULTS, **cfg}
    return {**cfg, "snowline_m": snowline(cfg)}


# ----------------------------------------------------------------------------- noise
@functools.lru_cache(maxsize=64)
def _tables(seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(256)
    ang = rng.uniform(0, 2 * np.pi, 256)
    return np.concatenate([perm, perm]), np.cos(ang), np.sin(ang)


def gradient_noise(x, y, seed=0):
    """2-D gradient noise in about [-1, 1], unit lattice spacing (period 256)."""
    perm, gx, gy = _tables(int(seed))
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xf, yf = np.floor(x), np.floor(y)
    fx, fy = x - xf, y - yf
    xi, yi = xf.astype(np.int64) & 255, yf.astype(np.int64) & 255

    def g(ix, iy, dx, dy):
        h = perm[perm[ix & 255] + (iy & 255)]
        return gx[h] * dx + gy[h] * dy

    u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    n0 = g(xi, yi, fx, fy) + u * (g(xi + 1, yi, fx - 1, fy) - g(xi, yi, fx, fy))
    n1 = g(xi, yi + 1, fx, fy - 1) + u * (g(xi + 1, yi + 1, fx - 1, fy - 1) - g(xi, yi + 1, fx, fy - 1))
    return 1.4 * (n0 + v * (n1 - n0))


def _octave_xy(x, y, k, f):
    a = 0.61 + 1.13 * k
    c, s = np.cos(a), np.sin(a)
    return f * (c * x - s * y) + 17.3 * k, f * (s * x + c * y) - 9.1 * k


def fbm(x, y, octaves=5, lacunarity=2.0, gain=0.5, seed=0):
    """Fractional Brownian motion of gradient noise, normalised to about [-1, 1]."""
    out, amp, f, norm = 0.0, 1.0, 1.0, 0.0
    for k in range(octaves):
        u, v = _octave_xy(x, y, k, f)
        out = out + amp * gradient_noise(u, v, seed + k)
        norm += amp
        amp *= gain
        f *= lacunarity
    return out / norm


def ridged(x, y, octaves=8, lacunarity=2.0, gain=2.0, offset=1.0, h=1.0, seed=0):
    """Musgrave ridged multifractal, normalised to about [0, 1] (sharp crests at 1)."""
    u, v = _octave_xy(x, y, 0, 1.0)
    sig = (offset - np.abs(gradient_noise(u, v, seed))) ** 2
    out, norm, f = sig, 1.0, 1.0
    for k in range(1, octaves):
        f *= lacunarity
        w = np.clip(sig * gain, 0, 1)
        u, v = _octave_xy(x, y, k, f)
        sig = (offset - np.abs(gradient_noise(u, v, seed + k))) ** 2 * w
        out = out + sig * f ** -h
        norm += f ** -h
    return out / norm


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------- terrain
def lake_mask(x, y, cfg):
    """1 inside the lake basin, 0 outside (lakeshore scenes only)."""
    if cfg["type"] != "lakeshore":
        return np.zeros(np.shape(x))
    s = cfg["seed"] + 500
    cx, ax, ay = 250.0, 240.0, 600.0
    d = np.hypot((x - cx) / ax, y / ay) + 0.18 * fbm(x / 220, y / 220, 4, seed=s)
    return smoothstep(1.0, 0.85, d)


def height(x, y, cfg):
    """Terrain height (m) at horizon-frame (x, y), including the curvature drop."""
    t, s = cfg["type"], int(cfg["seed"]) * 101
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r = np.hypot(x, y)
    if t == "plane":
        return -r * r / (2 * cfg["planet_radius_m"])
    amp = dict(mountains=(3.0, 40.0), meadow=(3.5, 25.0), forest=(1.5, 15.0), lakeshore=(2.0, 30.0))[t]
    z = np.zeros_like(r)
    near = r < 250
    z[near] = 0.06 * fbm(x[near] / 2.5, y[near] / 2.5, 3, seed=s + 1) * (1 - smoothstep(60, 250, r[near]))
    z += amp[0] * fbm(x / 260, y / 260, 5, seed=s + 2) * smoothstep(4, 60, r)
    z += amp[1] * fbm(x / 1800, y / 1800, 5, seed=s + 3) * smoothstep(200, 1500, r)
    if t in ("mountains", "lakeshore"):
        far = r > 1500
        xf, yf, rf = x[far], y[far], r[far]
        rng_mask = 0.55 + 0.45 * fbm(xf / 22000, yf / 22000, 3, seed=s + 4)
        r0 = 3000.0 if t == "mountains" else 4500.0
        env = smoothstep(r0 - 1000, r0 + 4000, rf) * np.clip(rng_mask, 0.25, 1)
        hm = cfg["mountain_height_m"] * (1.0 if t == "mountains" else 0.7)
        z[far] += hm * env * np.clip(ridged(xf / 9000, yf / 9000, 10, seed=s + 5) - 0.2, 0, None) / 0.8
    if t == "lakeshore":
        m = lake_mask(x, y, cfg)
        zl = cfg["water_level_m"]
        z = np.where(m > 0, (1 - m) * np.maximum(z, zl + 0.25) + m * (zl - 3.5), z)
    return z - r * r / (2 * cfg["planet_radius_m"])


def polar_grid(cfg):
    """Ring radii (nr,) and azimuths (nth,) of the LOD ground mesh."""
    nth = int(cfg["n_theta"])
    dth = 2 * np.pi / nth
    q = 1 + dth
    nr = int(np.ceil(np.log(cfg["extent_m"] / cfg["r_inner"]) / np.log(q))) + 1
    return cfg["r_inner"] * q ** np.arange(nr), np.arange(nth) * dth


def polar_faces(nr, nth):
    j = np.arange(nth)
    fan = np.stack([np.zeros(nth, np.int64), 1 + j, 1 + (j + 1) % nth], axis=1)
    i = np.arange(nr - 1)[:, None]
    a = 1 + i * nth + j[None, :]
    b = 1 + i * nth + ((j + 1) % nth)[None, :]
    c, d = a + nth, b + nth
    quads = np.concatenate([np.stack([a, d, b], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return np.concatenate([fan, quads]).astype(np.uint32)


def materials(x, y, z, nz, cfg):
    """Per-vertex role weights (N, len(ROLES)) and brightness modulation (N,)."""
    t, s = cfg["type"], int(cfg["seed"]) * 101 + 50
    r = np.hypot(x, y)
    w = np.zeros((len(x), len(ROLES)), np.float32)
    if t == "plane":
        w[:, 0] = 1
        return w, np.ones(len(x), np.float32)
    n1 = fbm(x / 35, y / 35, 4, seed=s)
    n2 = fbm(x / 9, y / 9, 3, seed=s + 1)
    dry = np.clip(0.2 + 0.6 * n1, 0, 1) if t != "forest" else np.clip(0.15 + 0.5 * n1, 0, 1)
    soil = smoothstep(0.2, 0.5, n2) * (1 - smoothstep(150, 400, r))
    w[:, 0], w[:, 1], w[:, 2] = (1 - dry) * (1 - soil), dry * (1 - soil), soil
    nf = fbm(x / 1200, y / 1200, 4, seed=s + 2)
    forest = smoothstep(-0.1, 0.15, nf) * smoothstep(700, 1500, r) * (1 - smoothstep(1300, 1700, z))
    if t in ("meadow", "forest"):
        forest = np.maximum(forest, smoothstep(600, 1200, r) * smoothstep(-0.3, 0.0, nf))
    rock = np.maximum(smoothstep(0.82, 0.62, nz), smoothstep(1500, 1900, z + 150 * n1)) * smoothstep(300, 1500, r)
    zs = snowline(cfg)
    snow = smoothstep(zs - 150, zs + 150, z + 250 * nf) * smoothstep(0.5, 0.7, nz) * (1 - forest)
    lake = lake_mask(x, y, cfg)
    for k, f in ((5, forest), (3, rock), (4, snow), (6, lake)):
        w *= (1 - f)[:, None]
        w[:, k] += f
    fine = 1 - smoothstep(20, 150, r)
    mod = np.exp(0.35 * fine * fbm(x / 0.35, y / 0.35, 3, seed=s + 3) + 0.15 * fbm(x / 6, y / 6, 3, seed=s + 4))
    mod *= np.where(w[:, 5] > 0.5, 0.75, 1.0)
    return w, mod.astype(np.float32)


def ground_mesh(cfg):
    """dict(verts (N,3), faces (F,3), normals (N,3), weights (N,nroles), mod (N,)) float32."""
    radii, th = polar_grid(cfg)
    nr, nth = len(radii), len(th)
    R, TH = np.meshgrid(radii, th, indexing="ij")
    X, Y = R * np.cos(TH), R * np.sin(TH)
    Z = height(X, Y, cfg)
    dzdr = np.gradient(Z, radii, axis=0)
    dzdt = (np.roll(Z, -1, 1) - np.roll(Z, 1, 1)) / (2 * (th[1] - th[0]))
    zx = np.cos(TH) * dzdr - np.sin(TH) / R * dzdt
    zy = np.sin(TH) * dzdr + np.cos(TH) / R * dzdt
    N = np.stack([-zx, -zy, np.ones_like(zx)], -1)
    if cfg["type"] != "plane":
        s = int(cfg["seed"]) * 101 + 70
        a = 0.12 * (1 - smoothstep(10, 60, R))
        N[..., 0] += a * gradient_noise(X / 0.08, Y / 0.08, s)
        N[..., 1] += a * gradient_noise(X / 0.08, Y / 0.08, s + 1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    z0 = float(height(np.zeros(1), np.zeros(1), cfg)[0])
    verts = np.concatenate([[[0, 0, z0]], np.stack([X, Y, Z], -1).reshape(-1, 3)])
    normals = np.concatenate([[[0, 0, 1.0]], N.reshape(-1, 3)])
    w, mod = materials(verts[:, 0], verts[:, 1], verts[:, 2], normals[:, 2], cfg)
    return dict(verts=verts.astype(np.float32), faces=polar_faces(nr, nth), normals=normals.astype(np.float32),
                weights=w, mod=mod, roles=ROLES)


def water_mesh(cfg, n=96):
    """Flat (curvature-following) water sheet over the lake basin, or None."""
    if cfg["type"] != "lakeshore":
        return None
    u = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(250 + 300 * u, 760 * u, indexing="ij")
    Z = cfg["water_level_m"] - (X ** 2 + Y ** 2) / (2 * cfg["planet_radius_m"])
    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    a = (i * n + j).ravel()
    faces = np.concatenate([np.stack([a, a + n, a + n + 1], 1), np.stack([a, a + n + 1, a + 1], 1)])
    v = np.stack([X, Y, Z], -1).reshape(-1, 3)
    if np.cross(v[faces[0, 1]] - v[faces[0, 0]], v[faces[0, 2]] - v[faces[0, 0]])[2] < 0:
        faces = faces[:, [0, 2, 1]]
    return dict(verts=v.astype(np.float32), faces=faces.astype(np.uint32))


# ----------------------------------------------------------------------------- trees
def _merge(parts):
    v, f, off = [], [], 0
    for pv, pf in parts:
        v.append(pv)
        f.append(pf + off)
        off += len(pv)
    return np.concatenate(v).astype(np.float32), np.concatenate(f).astype(np.uint32)


def _cylinder(p0, p1, r0, r1, n=8):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    ax /= np.linalg.norm(ax)
    e1 = np.cross(ax, [1.0, 0, 0] if abs(ax[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    a = np.arange(n) * 2 * np.pi / n
    ring = np.cos(a)[:, None] * e1 + np.sin(a)[:, None] * e2
    v = np.concatenate([p0 + r0 * ring, p1 + r1 * ring])
    j = np.arange(n)
    k = (j + 1) % n
    f = np.concatenate([np.stack([j, k, k + n], 1), np.stack([j, k + n, j + n], 1)])
    return v, f


def _cone_tier(z, height_, radius, rng, n_spokes):
    """Star-shaped drooping cone (one whorl of branches), closed underneath."""
    m = 2 * n_spokes
    a = (np.arange(m) + rng.uniform(0, 1)) * 2 * np.pi / m + rng.normal(0, 0.06, m)
    rad = radius * np.where(np.arange(m) % 2 == 0, rng.uniform(0.85, 1.2, m), rng.uniform(0.35, 0.6, m))
    droop = np.where(np.arange(m) % 2 == 0, 0.35, 0.1) * height_ * rng.uniform(0.6, 1.2, m)
    ring = np.stack([rad * np.cos(a), rad * np.sin(a), z - droop], 1)
    apex = [rng.normal(0, 0.05 * radius), rng.normal(0, 0.05 * radius), z + height_]
    bottom = [0, 0, z + 0.15 * height_]
    v = np.concatenate([[apex], ring, [bottom]])
    j = np.arange(m)
    k = (j + 1) % m
    f = np.concatenate([np.stack([np.zeros(m, int), 1 + j, 1 + k], 1),
                        np.stack([np.full(m, m + 1), 1 + k, 1 + j], 1)])
    return v, f


def conifer(rng):
    """Unit-height conifer: dict(bark=(V,F), needles=(V,F))."""
    trunk = _cylinder([0, 0, -0.05], [0, 0, 0.9], 0.014, 0.004, 8)
    z0 = rng.uniform(0.06, 0.2)
    rmax = rng.uniform(0.15, 0.23)
    n = rng.integers(13, 19)
    zs = np.linspace(z0, 0.93, n) + rng.normal(0, 0.008, n)
    dz = (0.93 - z0) / (n - 1)
    tiers = []
    for z in zs:
        frac = (z - z0) / (1 - z0)
        rad = rmax * (1 - frac) ** 0.85 * rng.uniform(0.85, 1.15) + 0.012
        tiers.append(_cone_tier(z, rng.uniform(1.6, 2.4) * dz, rad, rng, int(rng.integers(7, 12))))
    tiers.append(_cone_tier(0.93, 0.08, 0.02, rng, 5))
    return dict(bark=_merge([trunk]), needles=_merge(tiers))


def _blob(c, rad, rng, nlat=9, nlon=14):
    th = np.linspace(0, np.pi, nlat)[1:-1]
    ph = np.arange(nlon) * 2 * np.pi / nlon
    T, P = np.meshgrid(th, ph, indexing="ij")
    d = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1).reshape(-1, 3)
    d = np.concatenate([[[0, 0, 1.0]], d, [[0, 0, -1.0]]])
    u = rng.normal(size=(6, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    disp = 1 + (rng.uniform(0.08, 0.2, 6)[None, :] * np.sin(rng.uniform(3, 8, 6)[None, :] * (d @ u.T)
                                                              + rng.uniform(0, 6.3, 6)[None, :])).sum(1)
    v = c + rad * disp[:, None] * d * [1, 1, 0.8]
    L = nlat - 2
    f = [np.stack([np.zeros(nlon, int), 1 + (np.arange(nlon) + 1) % nlon, 1 + np.arange(nlon)], 1)]
    for i in range(L - 1):
        a = 1 + i * nlon + np.arange(nlon)
        b = 1 + i * nlon + (np.arange(nlon) + 1) % nlon
        f += [np.stack([a, b, b + nlon], 1), np.stack([a, b + nlon, a + nlon], 1)]
    last = len(v) - 1
    a = 1 + (L - 1) * nlon + np.arange(nlon)
    b = 1 + (L - 1) * nlon + (np.arange(nlon) + 1) % nlon
    f.append(np.stack([np.full(nlon, last), a, b], 1))
    f = np.concatenate(f)
    f = f[:, [0, 2, 1]]
    return v, f


def deciduous(rng):
    """Unit-height broadleaf tree: dict(bark=(V,F), leaves=(V,F))."""
    hc = rng.uniform(0.58, 0.66)
    a, c = rng.uniform(0.26, 0.36), rng.uniform(0.25, 0.32)
    parts = [_cylinder([0, 0, -0.05], [0, 0, hc], 0.028, 0.014, 10)]
    for _ in range(rng.integers(3, 6)):
        z = rng.uniform(0.3, 0.48)
        phi = rng.uniform(0, 2 * np.pi)
        tip = np.array([0.6 * a * np.cos(phi), 0.6 * a * np.sin(phi), hc + rng.uniform(-0.05, 0.15)])
        parts.append(_cylinder([0, 0, z], tip, 0.011, 0.004, 6))
    blobs = []
    nb = int(rng.integers(16, 28))
    pts = rng.normal(size=(nb * 3, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * rng.uniform(0, 1, (nb * 3, 1)) ** 0.5
    pts = pts[pts[:, 2] > -0.6][:nb] * [0.72 * a, 0.72 * a, 0.72 * c] + [0, 0, hc]
    for p in pts:
        blobs.append(_blob(p, rng.uniform(0.08, 0.14) * (1 + 0.5 * (a - 0.26)), rng))
    return dict(bark=_merge(parts), leaves=_merge(blobs))


def prototypes(cfg):
    rng = np.random.default_rng(int(cfg["seed"]) + 9000)
    out = [dict(kind="conifer", parts=conifer(rng)) for _ in range(int(cfg["n_conifer_protos"]))]
    out += [dict(kind="deciduous", parts=deciduous(rng)) for _ in range(int(cfg["n_deciduous_protos"]))]
    return out


def _sample_annulus(rng, n, r0, r1):
    r = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return r * np.cos(a), r * np.sin(a)


def tree_positions(cfg):
    """(x, y, height_m, is_conifer) arrays of placed trees."""
    t = cfg["type"]
    rng = np.random.default_rng(int(cfg["seed"]) + 7000)
    s = int(cfg["seed"]) * 101 + 300
    dens = cfg["tree_density"]
    if t == "plane":
        return np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, bool)
    if t == "forest":
        x, y = _sample_annulus(rng, int(40000 * dens), 20, 260)
        th = np.arctan2(y, x)
        rc = 36 + 10 * fbm(np.cos(th) * 1.5, np.sin(th) * 1.5, 3, seed=s)
        r = np.hypot(x, y)
        keep = (r > rc) & (rng.uniform(size=len(x)) < np.exp(-(r - rc) / 70) * 0.5)
        conif = rng.uniform(size=len(x)) < 0.65
    elif t == "meadow":
        x, y = _sample_annulus(rng, int(60000 * dens), 250, 1100)
        th = np.arctan2(y, x)
        rt = 480 + 220 * fbm(np.cos(th) * 2, np.sin(th) * 2, 4, seed=s)
        r = np.hypot(x, y)
        gap = fbm(np.cos(th) * 1.2, np.sin(th) * 1.2, 3, seed=s + 1) > -0.35
        keep = gap & (r > rt) & (r < rt + 90) & (rng.uniform(size=len(x)) < 0.6)
        xs, ys = _sample_annulus(rng, 40, 30, 250)
        x, y, keep = np.concatenate([x, xs]), np.concatenate([y, ys]), np.concatenate([keep, np.ones(40, bool)])
        conif = rng.uniform(size=len(x)) < 0.35
    else:
        x, y = _sample_annulus(rng, int(60000 * dens), 25, 1200)
        r = np.hypot(x, y)
        clump = fbm(x / 120, y / 120, 4, seed=s)
        p = smoothstep(0.05, 0.35, clump) * np.exp(-r / 500) * 0.5
        keep = (rng.uniform(size=len(x)) < p) & (lake_mask(x, y, cfg) < 0.05)
        if t == "mountains":
            keep &= r > 60
        conif = rng.uniform(size=len(x)) < 0.9
    x, y, conif = x[keep], y[keep], conif[keep]
    r = np.hypot(x, y)
    order = np.argsort(r)[: int(cfg["max_trees"])]
    x, y, conif = x[order], y[order], conif[order]
    h = np.where(conif, rng.uniform(12, 28, len(x)), rng.uniform(9, 20, len(x)))
    return x, y, h, conif


def tree_instances(cfg, protos):
    """(proto_index (n,), to_world (n, 4, 4)) for the placed trees."""
    x, y, h, conif = tree_positions(cfg)
    rng = np.random.default_rng(int(cfg["seed"]) + 8000)
    ic = np.array([i for i, p in enumerate(protos) if p["kind"] == "conifer"])
    idc = np.array([i for i, p in enumerate(protos) if p["kind"] == "deciduous"])
    idx = np.where(conif, rng.choice(ic, len(x)), rng.choice(idc, len(x)))
    z = height(x, y, cfg) - 0.05
    mats = np.zeros((len(x), 4, 4))
    yaw = rng.uniform(0, 2 * np.pi, len(x))
    tilt = rng.normal(0, np.radians(2.0), (len(x), 2))
    sxy = h * rng.uniform(0.85, 1.15, len(x))
    for k in range(len(x)):
        c, s_ = np.cos(yaw[k]), np.sin(yaw[k])
        Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        ax, ay = tilt[k]
        Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        mats[k, :3, :3] = Rx @ Ry @ Rz @ np.diag([sxy[k], sxy[k], h[k]])
        mats[k, :3, 3] = [x[k], y[k], z[k]]
        mats[k, 3, 3] = 1
    return idx, mats


def build(cfg):
    """Full foreground geometry: dict(ground, water, prototypes, instances, cfg)."""
    cfg = resolve(cfg)
    protos = prototypes(cfg)
    idx, mats = tree_instances(cfg, protos)
    return dict(ground=ground_mesh(cfg), water=water_mesh(cfg), prototypes=protos,
                instances=(idx, mats), cfg=cfg)
