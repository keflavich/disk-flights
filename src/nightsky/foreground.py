"""Foreground (terrain, trees, water) lit by the night sky, rendered with Mitsuba 3 per bin.

Units: radiance W m^-2 sr^-1 nm^-1 and irradiance W m^-2 nm^-1 as bin averages on the
caller's SpectralGrid; distances in metres. Mitsuba world = horizon frame (x = N, y = W,
z = Up), camera at (0, 0, eye_height) above the ground at the origin.

Method
  - geometry from terrain.py, cached as PLY under CACHE_DIR keyed by the RESOLVED geometry config
    (resolve_scene: snowline_m None -> the albedo set's season default, albedo.SNOWLINE_M);
    trees are prototype meshes (terrain.py) placed a few thousand times and merged into one
    mesh per part (bark / needles / leaves) with a per-vertex albedo attribute; Mitsuba
    shapegroup instances were tried first, but with llvm_ad_mono (3.9.1) the render time
    grew with the instance count (~110 s for a 240x135 AOV pass with 1139 instances).
  - lighting: an `envmap` emitter built from sky_fn on a lat-long grid (default 512 x 1024;
    its local +Y axis is mapped to world +z), plus `directional` emitters for the brightest
    PointSources (their irradiance is the ground-level value perpendicular to the beam);
    fainter sources are deposited into single envmap texels with E / Omega_texel, which keeps
    their total irradiance (texel grid and solid angles in env_dirs follow Mitsuba's lookup:
    azimuth pixel-centred, polar angle corner-aligned). Below the horizon the envmap holds a dim ground radiance
    rho_g E_h / pi (rho_g = 0.1), seen only through gaps at the mesh edge.
  - materials: Lambertian (two-sided) BSDFs with per-bin albedos from albedo.py; the ground
    uses a per-vertex albedo attribute, the lake a smooth dielectric (n = 1.333) over a dark
    lakebed. Per-bin values are written with mi.traverse, the scene is built once.
  - integrator: `path` (max_depth 5, emitters hidden, so sky pixels have zero radiance and
    alpha = 0) per bin with the SAME seed in every bin, so Monte-Carlo noise is strongly
    correlated between bins (low chroma noise); an `aov` pass with the same spp and seed (same
    primary rays) gives alpha, depth and shading normal.
  - perspective views use one Mitsuba perspective sensor (fov_axis = x) reconstructed from
    views.View.dirs; any other view kind (allsky, panorama) renders the six views.FACES cube
    faces and resamples them with views.sample_cube.

Output radiance is the pixel average over all samples, i.e. premultiplied by alpha:
composite as  out = T * radiance + alpha * inscatter + (1 - alpha) * sky.
Not modelled: specular glints of the point lights on water (delta light via a specular
surface), grass blade geometry (grass is albedo/normal texture), wind, dew, leaf
transmission (two-sided reflectance only).
"""
import hashlib
import json
import os
import time

import numpy as np

import views
from . import albedo as alb
from . import spectral, terrain

CACHE_DIR = "/orange/adamginsburg/gc/visualization/renders/nightsky/cache/foreground"
GEOMETRY_KEYS = tuple(terrain.DEFAULTS)
DEFAULTS = dict(albedo_set="summer", albedo_overrides=None, max_depth=5, n_lights=32, env_res=512,
                cube_size=None, aov_spp=None, ground_albedo_far=0.1, denoise=False, seed_render=0,
                verbose=True, extra_shapes=None)
# local envmap frame (+Y up) -> horizon frame (z up): world = (lx, -lz, ly)
ENV_TO_WORLD = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)


def _mitsuba(cpus=None):
    import drjit as dr
    import mitsuba as mi
    if mi.variant() != "llvm_ad_mono":
        mi.set_variant("llvm_ad_mono")
    if cpus:
        dr.set_thread_count(int(cpus))
    return mi, dr


def _log(cfg, *a):
    if cfg.get("verbose", True):
        print("[foreground]", *a, flush=True)


# ----------------------------------------------------------------------------- envmap
def env_dirs(h):
    """(h, 2h, 3) horizon-frame directions of the envmap texels, and (h, 2h) solid angles.

    Texel positions follow Mitsuba 3.9's envmap lookup (measured with a compact test source):
    columns are pixel-centred in azimuth, phi_j = 2 pi (j + 1/2) / w, while rows are
    corner-aligned in polar angle, theta_i = pi i / (h - 1), so rows 0 and h-1 sit on the poles.
    The solid angle of a texel is the integral of its bilinear hat function over the sphere, so
    sum(value * omega) is the exact integral of the interpolated map (and a texel holding
    E / omega carries irradiance E, also on the pole rows)."""
    w = 2 * h
    dth = np.pi / (h - 1)
    th = dth * np.arange(h)
    ph = 2 * np.pi * (np.arange(w) + 0.5) / w
    T, P = np.meshgrid(th, ph, indexing="ij")
    local = np.stack([np.sin(T) * np.sin(P), np.cos(T), -np.sin(T) * np.cos(P)], -1)
    d = local @ ENV_TO_WORLD[:3, :3].T
    hat = np.sin(th) * 2 * (1 - np.cos(dth)) / dth
    hat[[0, -1]] = 1 - np.sin(dth) / dth
    omega = np.broadcast_to(((2 * np.pi / w) * hat)[:, None], T.shape).copy()
    return d, omega


def env_pixel(dirs, h):
    """(row, col) envmap texel nearest to horizon-frame directions (inverse of env_dirs)."""
    local = np.asarray(dirs, float) @ ENV_TO_WORLD[:3, :3]
    th = np.arccos(np.clip(local[:, 1], -1, 1))
    ph = np.arctan2(local[:, 0], -local[:, 2]) % (2 * np.pi)
    row = np.clip(np.rint(th / np.pi * (h - 1)).astype(int), 0, h - 1)
    return row, np.minimum((ph / (2 * np.pi) * 2 * h).astype(int), 2 * h - 1)


def build_envmap(sky_fn, lights, grid, h=512, rho_ground=0.1):
    """(nbins, h, 2h) envmap radiance: sky_fn above the horizon, faint lights deposited, dim
    ground below. Returns (env, horizontal sky irradiance (nbins,))."""
    d, omega = env_dirs(h)
    up = d[..., 2] > 0
    env = np.zeros((grid.n,) + d.shape[:2], np.float32)
    vals = np.asarray(sky_fn(d[up]), float)
    env[:, up] = np.maximum(vals, 0).T
    e_h = (env[:, up] * (omega[up] * d[up][:, 2])[None]).sum(1)
    env[:, ~up] = (rho_ground * e_h / np.pi)[:, None]
    if lights is not None and len(lights):
        deposit_lights(env, lights)
    return env, e_h


def deposit_lights(env, lights):
    """Add PointSources (horizon frame, irradiance (n, nbins)) to an (nbins, h, 2h) envmap in
    place as E / Omega_texel in the nearest texel. The pole rows are skipped (sources within half
    a texel of the zenith go to row 1): Mitsuba's envmap sampling weights texels by sin(theta),
    which is 0 there, so a source on row 0 never reaches the scene."""
    h = env.shape[1]
    _, omega = env_dirs(h)
    r, c = env_pixel(lights.dirs, h)
    r = np.clip(r, 1, h - 2)
    np.add.at(env.transpose(1, 2, 0), (r, c), lights.irr / omega[r, c][:, None])


def _env_tensor(mi, img):
    p = np.concatenate([img[:, -1:], img, img[:, :1]], axis=1)
    return mi.TensorXf(np.repeat(p[..., None], 3, axis=2).astype(np.float32))


# ----------------------------------------------------------------------------- cameras
def perspective_params(view):
    """(fwd, up, horizontal fov deg) of a pinhole views.View, from its corner rays."""
    d = view.dirs
    c = [d[0, 0], d[0, -1], d[-1, 0], d[-1, -1]]
    fwd = np.sum(c, 0)
    fwd /= np.linalg.norm(fwd)
    right = (c[1] + c[3]) - (c[0] + c[2])
    right -= fwd * (right @ fwd)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    p = d[0, -1] / (d[0, -1] @ fwd)
    X = 1 - 1.0 / view.width
    return fwd, up, 2 * np.degrees(np.arctan((p @ right) / X))


def _sensor(mi, eye, fwd, up, fov, w, h, spp):
    return {"type": "perspective", "fov": float(fov), "fov_axis": "x", "near_clip": 0.01, "far_clip": 1e8,
            "to_world": mi.ScalarTransform4f().look_at(origin=list(eye), target=list(np.add(eye, fwd)), up=list(up)),
            "film": {"type": "hdrfilm", "width": int(w), "height": int(h), "pixel_format": "luminance_alpha",
                     "rfilter": {"type": "box"}},
            "sampler": {"type": "independent", "sample_count": int(spp)}}


def sensors_for(mi, view, eye, spp, cube_size=None):
    if view.kind == "perspective":
        fwd, up, fov = perspective_params(view)
        return [_sensor(mi, eye, fwd, up, fov, view.width, view.height, spp)], None
    if cube_size is None:
        pix = np.degrees(np.sqrt(np.median(view.omega[view.valid])))
        cube_size = int(np.clip(np.ceil(views.FACE_FOV_DEG / pix / 8) * 8, 64, 2048))
    out = []
    for k in range(6):
        fwd, right, up = views.face_basis(k)
        out.append(_sensor(mi, eye, fwd, up, views.FACE_FOV_DEG, cube_size, cube_size, spp))
    return out, cube_size


# ----------------------------------------------------------------------------- geometry cache
def resolve_scene(scene_cfg):
    """scene_cfg merged with terrain.DEFAULTS and DEFAULTS, snowline_m resolved from albedo_set when
    None. Stage caches outside this module should hash this (not the raw scene_cfg)."""
    return terrain.resolve({**DEFAULTS, **(scene_cfg or {})})


def _geometry_key(cfg):
    g = {k: cfg[k] for k in GEOMETRY_KEYS}
    g["format"] = 3  # 3: snow excluded from the forest texture (terrain.materials)
    return g["type"] + "_" + hashlib.sha1(json.dumps(g, sort_keys=True, default=float).encode()).hexdigest()[:12]


def _write_ply(mi, path, V, F, N=None, attrs=None):
    m = mi.Mesh(os.path.basename(path), len(V), len(F), has_vertex_normals=N is not None,
                has_vertex_texcoords=False)
    p = mi.traverse(m)
    p["vertex_positions"] = mi.Float(np.ascontiguousarray(V, np.float32).ravel())
    p["faces"] = mi.UInt32(np.ascontiguousarray(F, np.uint32).ravel())
    if N is not None:
        p["vertex_normals"] = mi.Float(np.ascontiguousarray(N, np.float32).ravel())
    p.update()
    for k, a in (attrs or {}).items():
        m.add_attribute(k, 1, mi.Float(np.ascontiguousarray(a, np.float32)))
    m.write_ply(path)


def geometry(mi, cfg):
    """Build (or load from cache) the PLY files and material tables for a geometry config."""
    d = os.path.join(CACHE_DIR, _geometry_key(cfg))
    meta = os.path.join(d, "meta.npz")
    if os.path.exists(meta):
        z = np.load(meta, allow_pickle=False)
        parts = [str(p) for p in z["tree_parts"]]
        return d, dict(weights=z["weights"], mod=z["mod"], tree_parts=parts, inst_idx=z["inst_idx"],
                       has_water=bool(z["has_water"]), tree_factor={p: z[f"tf_{p}"] for p in parts})
    t0 = time.time()
    geo = terrain.build({k: cfg[k] for k in GEOMETRY_KEYS})
    os.makedirs(d, exist_ok=True)
    g = geo["ground"]
    _write_ply(mi, os.path.join(d, "ground.ply"), g["verts"], g["faces"], g["normals"],
               {"vertex_albedo": np.full(len(g["verts"]), 0.1, np.float32)})
    if geo["water"] is not None:
        _write_ply(mi, os.path.join(d, "water.ply"), geo["water"]["verts"], geo["water"]["faces"])
    idx, mats = geo["instances"]
    rng = np.random.default_rng(int(cfg["seed"]) + 4242)
    tf = rng.uniform(0.8, 1.2, len(idx))
    merged = {}
    for k, (i, m) in enumerate(zip(idx, mats)):
        for name, (V, F) in geo["prototypes"][int(i)]["parts"].items():
            merged.setdefault(name, []).append((V @ m[:3, :3].T + m[:3, 3], F, tf[k]))
    factors = {}
    for name, lst in merged.items():
        V, F = terrain._merge([(v, f) for v, f, _ in lst])
        factors[name] = np.concatenate([np.full(len(v), t, np.float32) for v, _, t in lst])
        _write_ply(mi, os.path.join(d, f"trees_{name}.ply"), V, F, None,
                   {"vertex_albedo": np.full(len(V), 0.1, np.float32)})
    tmp = os.path.join(d, f"meta.{os.getpid()}.tmp.npz")
    np.savez(tmp, weights=g["weights"].astype(np.float16), mod=g["mod"], tree_parts=np.array(sorted(factors)),
             inst_idx=idx, has_water=geo["water"] is not None,
             **{f"tf_{k}": v for k, v in factors.items()})
    os.replace(tmp, meta)  # meta.npz marks a complete cache entry, so it appears last and atomically
    _log(cfg, f"geometry {os.path.basename(d)}: {len(g['verts'])} ground vertices, {len(g['faces'])} faces, "
              f"{len(idx)} trees ({sum(len(v) for v in factors.values())} tree vertices), built in {time.time() - t0:.1f} s")
    return geometry(mi, cfg)


# ----------------------------------------------------------------------------- scene
def build_scene(mi, cfg, gdir, gmeta, sensors, env0, lights_sel, light_weights, sky_weight):
    sc = {"type": "scene"}
    sc["bsdf_ground"] = {"type": "twosided", "id": "bsdf_ground",
                         "bsdf": {"type": "diffuse", "reflectance": {"type": "mesh_attribute", "name": "vertex_albedo"}}}
    sc["ground"] = {"type": "ply", "filename": os.path.join(gdir, "ground.ply"),
                    "bsdf": {"type": "ref", "id": "bsdf_ground"}}
    for p in gmeta["tree_parts"]:
        sc[f"trees_{p}"] = {"type": "ply", "filename": os.path.join(gdir, f"trees_{p}.ply"), "face_normals": True,
                            "bsdf": {"type": "ref", "id": "bsdf_ground"}}
    if gmeta["has_water"]:
        sc["water"] = {"type": "ply", "filename": os.path.join(gdir, "water.ply"), "face_normals": True,
                       "bsdf": {"type": "dielectric", "int_ior": 1.333, "ext_ior": 1.0}}
    sc["sky"] = {"type": "envmap", "bitmap": mi.Bitmap(env0[..., None].astype(np.float32)),
                 "to_world": mi.ScalarTransform4f(ENV_TO_WORLD.tolist()), "sampling_weight": float(sky_weight)}
    for k in range(len(lights_sel)):
        sc[f"light_{k}"] = {"type": "directional", "direction": (-lights_sel.dirs[k]).tolist(),
                            "irradiance": {"type": "uniform", "value": float(k + 1)},
                            "sampling_weight": float(light_weights[k])}
    for k, s in enumerate(sensors):
        sc[f"sensor_{k}"] = s
    for k, s in enumerate(cfg.get("extra_shapes") or []):
        sc[f"extra_{k}"] = s
    return mi.load_dict(sc)


def material_tables(cfg, grid):
    """(nroles, nbins) ground role albedos and {tree part: (nbins,)} crown/bark albedos."""
    ov = cfg.get("albedo_overrides")
    tab = alb.role_table(cfg["albedo_set"], grid, terrain.ROLES, ov)
    parts = ("bark", "needles", "leaves")
    return tab, dict(zip(parts, alb.role_table(cfg["albedo_set"], grid, parts, ov)))


def _denoise(img, alpha, depth, normal, sigma_px=1.0):
    """Edge-aware 3x3..5x5 cross-bilateral smoothing of the bin-summed image; chromaticity kept."""
    tot = img.sum(-1)
    ratio = img / np.maximum(tot, 1e-30)[..., None]
    out = np.zeros_like(tot)
    wsum = np.zeros_like(tot)
    logd = np.log(np.where(np.isfinite(depth), np.maximum(depth, 1e-3), 1e9))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            sh = lambda a: np.roll(np.roll(a, dy, 0), dx, 1)
            w = np.exp(-(dx * dx + dy * dy) / (2 * sigma_px ** 2)) * np.exp(-((sh(logd) - logd) / 0.05) ** 2)
            w = w * np.clip((sh(normal) * normal).sum(-1), 0, 1) ** 8 * (sh(alpha) > 0)
            out += w * sh(tot)
            wsum += w
    tot2 = np.where(wsum > 0, out / np.maximum(wsum, 1e-30), tot)
    return (ratio * tot2[..., None]).astype(np.float32)


# ----------------------------------------------------------------------------- main entry
def render_foreground(scene_cfg, view, sky_fn, lights, grid, spp=64, cpus=None):
    """Render the foreground for one view. See the module docstring and INTERFACES.md.

    scene_cfg: terrain.DEFAULTS keys (type, seed, extent_m, n_theta, eye_height, tree_density,
    max_trees, snowline_m (None = albedo_set default), ...) plus DEFAULTS here (albedo_set,
    albedo_overrides, max_depth, n_lights, env_res, cube_size, aov_spp (None = spp), denoise). lights: PointSources in the horizon frame with
    ground-level irradiance (may be None or empty).
    Returns dict(radiance (H,W,nbins) premultiplied by alpha, alpha (H,W), depth (H,W) m with
    inf where no surface, normal (H,W,3), info dict).
    """
    cfg = resolve_scene(scene_cfg)
    t0 = time.time()
    mi, dr = _mitsuba(cpus)
    gdir, gmeta = geometry(mi, cfg)
    t_geo = time.time() - t0

    env, e_h = build_envmap(sky_fn, None, grid, int(cfg["env_res"]), cfg["ground_albedo_far"])
    ybin = spectral.cmf_bins(grid)[:, 1] * grid.dwl
    sel = None
    rest = None
    if lights is not None and len(lights):
        up = lights.dirs[:, 2] > 0
        ey = (lights.irr @ ybin) * np.clip(lights.dirs[:, 2], 0, None)
        lit = up & (lights.irr.max(1) > 0)  # keeps sources with flux outside the V band (ey = 0)
        order = [i for i in np.argsort(ey, kind="stable")[::-1] if lit[i]]
        nl = int(cfg["n_lights"])
        sel = lights.select(order[:nl]) if order[:nl] else None
        rest = lights.select(order[nl:]) if len(order) > nl else None
    if rest is not None:
        deposit_lights(env, rest)
    n_sel = 0 if sel is None else len(sel)
    # envmap emitter-sampling weight: sky plus the lights deposited into it
    sky_y = float(e_h @ ybin) + 1e-30
    if rest is not None:
        sky_y += float(((rest.irr @ ybin) * np.clip(rest.dirs[:, 2], 0, None)).sum())
    lw = [] if sel is None else list((sel.irr @ ybin) * np.clip(sel.dirs[:, 2], 0.02, None) / sky_y + 1e-6)
    eye = (0.0, 0.0, float(terrain.height(np.zeros(1), np.zeros(1), cfg)[0] + cfg["eye_height"]))
    sens, cube = sensors_for(mi, view, eye, spp, cfg["cube_size"])
    scene = build_scene(mi, cfg, gdir, gmeta, sens, env[0], sel if sel is not None else [], lw, 1.0)
    params = mi.traverse(scene)
    t_scene = time.time() - t0 - t_geo
    _log(cfg, f"scene ready: {len(gmeta['inst_idx'])} trees, {n_sel} directional lights, "
              f"{len(sens)} sensor(s){'' if cube is None else f' of {cube}^2'}, geo {t_geo:.1f} s, load {t_scene:.1f} s")

    tab, bs = material_tables(cfg, grid)
    w = gmeta["weights"].astype(np.float32)
    mod = gmeta["mod"]

    aov = mi.load_dict({"type": "aov", "aovs": "dd:depth,nn:sh_normal",
                        "i": {"type": "path", "max_depth": 1, "hide_emitters": True}})
    path = mi.load_dict({"type": "path", "max_depth": int(cfg["max_depth"]), "hide_emitters": True})
    # same spp and seed as the radiance pass: identical primary rays, so alpha is the coverage
    # of exactly the samples that make up the premultiplied radiance (no edge mismatch)
    aov_spp = int(spp if cfg["aov_spp"] is None else cfg["aov_spp"])
    geo_out = [np.array(mi.render(scene, sensor=k, integrator=aov, spp=aov_spp, seed=int(cfg["seed_render"])))
               for k in range(len(sens))]
    t_aov = time.time() - t0 - t_geo - t_scene

    rad = [np.zeros(g.shape[:2] + (grid.n,), np.float32) for g in geo_out]
    for b in range(grid.n):
        tb = time.time()
        params["sky.data"] = _env_tensor(mi, env[b])
        for k in range(n_sel):
            params[f"light_{k}.irradiance.value"] = float(sel.irr[k, b])
        for p in gmeta["tree_parts"]:
            params[f"trees_{p}.vertex_albedo"] = mi.Float(np.clip(gmeta["tree_factor"][p] * bs[p][b], 0, 1))
        params["ground.vertex_albedo"] = mi.Float(np.clip((w @ tab[:, b]) * mod, 0, 1).astype(np.float32))
        params.update()
        for k in range(len(sens)):
            rad[k][..., b] = np.array(mi.render(scene, sensor=k, integrator=path, spp=int(spp),
                                                seed=int(cfg["seed_render"])))[..., 0]
        if b == 0 or b == grid.n - 1:
            _log(cfg, f"bin {b + 1}/{grid.n} ({grid.wl[b]:.0f} nm) {time.time() - tb:.1f} s")
    t_rad = time.time() - t0 - t_geo - t_scene - t_aov

    if view.kind == "perspective":
        g = geo_out[0]
        alpha, ddep, nrm, radiance = g[..., 1], g[..., 2], g[..., 3:6], rad[0]
    else:
        faces_g = np.stack(geo_out)
        faces_r = np.stack(rad)
        dv = view.dirs.reshape(-1, 3)
        sg = views.sample_cube(faces_g[..., 1:6], dv)
        radiance = views.sample_cube(faces_r, dv).reshape(view.height, view.width, grid.n)
        sg = sg.reshape(view.height, view.width, 5)
        alpha, ddep, nrm = sg[..., 0], sg[..., 1], sg[..., 2:5]
        bad = ~view.valid
        alpha[bad] = 0
        radiance[bad] = 0
    alpha = np.clip(alpha, 0, 1).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        depth = np.where(alpha > 1e-3, ddep / np.maximum(alpha, 1e-6), np.inf).astype(np.float32)
    nn = np.linalg.norm(nrm, axis=-1, keepdims=True)
    normal = np.where(nn > 1e-6, nrm / np.maximum(nn, 1e-12), 0).astype(np.float32)
    radiance = np.maximum(radiance, 0).astype(np.float32)
    if cfg["denoise"]:
        radiance = _denoise(radiance, alpha, depth, normal)
    info = dict(t_geometry=t_geo, t_scene=t_scene, t_aov=t_aov, t_radiance=t_rad, t_total=time.time() - t0,
                spp=spp, n_lights=n_sel, n_trees=len(gmeta["inst_idx"]), cube_size=cube, eye=eye,
                sky_irradiance_h=e_h, geometry_dir=gdir, snowline_m=cfg["snowline_m"], premultiplied=True)
    _log(cfg, f"done in {info['t_total']:.1f} s (radiance {t_rad:.1f} s, {spp} spp x {grid.n} bins)")
    return dict(radiance=radiance, alpha=alpha, depth=depth, normal=normal, info=info)


def render_sky_only(view, sky_fn, grid, spp=4, env_res=512, cpus=None, bins=(0,)):
    """Envmap seen directly by the camera (no geometry): checks the envmap orientation."""
    mi, dr = _mitsuba(cpus)
    env, _ = build_envmap(sky_fn, None, grid, env_res)
    fwd, up, fov = perspective_params(view)
    sc = mi.load_dict({"type": "scene", "sky": {"type": "envmap", "bitmap": mi.Bitmap(env[bins[0]][..., None]),
                                                  "to_world": mi.ScalarTransform4f(ENV_TO_WORLD.tolist())},
                       "sensor": _sensor(mi, (0, 0, 0), fwd, up, fov, view.width, view.height, spp)})
    integ = mi.load_dict({"type": "path", "max_depth": 1})
    params = mi.traverse(sc)
    out = []
    for b in bins:
        params["sky.data"] = _env_tensor(mi, env[b])
        params.update()
        out.append(np.array(mi.render(sc, integrator=integ, spp=spp))[..., 0])
    return np.stack(out, -1)
