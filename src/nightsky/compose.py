"""Integration layer: config -> sources -> atmosphere -> foreground -> camera -> files.

Units: spectral radiance W m^-2 sr^-1 nm^-1 and irradiance W m^-2 nm^-1 as bin averages on the
grid built from cfg['grid']; photometric outputs in cd/m^2 and lux. Frames: sources in the
celestial frame (sky.TOASky), images and point sources in the horizon frame (x = N, y = W,
z = Up); R = frames.celestial_to_horizon(H, lat) gives v_h = R v_cel.

Stages (npz cache under <output_dir>/cache/, keyed on hashes of the config sections and
numbers each stage reads):
  1 starfield   starfield.build_starfield(cfg, grid) -> TOASky (points) + info
  2 galaxy      galaxy.GalaxyModel(...).render_map(observer, nside, R_gc_to_cel, r_res, inner)
                merged with the starfield's diffuse map (+ the Sun when scenario.sun.enabled)
  3 sky         per view: T * toa.diffuse(R^T d) + inscatter + airglow (atmosphere.sky; the
                smooth T / inscatter / airglow terms are evaluated on a grid `sky.subsample`
                times coarser and bilinearly upsampled; the diffuse map at full resolution)
  4 points      atmosphere.point_irradiance -> Gaussian PSF deposit, flux conserving
  5 foreground  foreground.render_foreground(scene, view, sky_fn, lights, grid, spp) with
                sky_fn = downwelling sky radiance tabulated on a horizon-frame HEALPix grid
                (point sources not in `lights` are binned into it), lights = the brightest
                n_lights point sources; L = a (L_fg T_ap + S_ap) + (1 - a) L_sky with
                atmosphere.aerial_perspective along the depth map
  6 camera      camera.develop (eye / camera model) -> PNG, npz, JSON sidecar
Earth-mode geometry fallback (used only when starfield info lacks it): Sun at z = +20.8 pc
(Bennett & Bovy 2019), galactic -> ICRS rotation of the Hipparcos catalogue (ESA 1997, vol. 1
sec. 1.5.3), right ascension measured from RA 12h as in bright_stars.py.
"""
import importlib
import json
import os
import time

import numpy as np
from scipy.ndimage import map_coordinates

from . import camera, config, frames, sky, spectral

STAGES = ("starfield", "galaxy", "sky", "points", "foreground", "camera")
SUN_TEFF_K, SOLAR_CONSTANT, SUN_RADIUS_DEG = 5772.0, 1361.0, 0.2666
ARCHES_LB = (0.1217, 0.0188)
SGR_A_LB = (359.9442, -0.0462)
SUN_Z_PC = 20.8
# ICRS -> galactic (Hipparcos A_G); v_gal = A_G @ v_icrs
A_G = np.array([[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
                [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
                [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669]])


def _mod(name):
    return importlib.import_module(f"nightsky.{name}")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def make_grid(cfg):
    g = cfg["grid"]
    return spectral.default_grid(int(g["n_bins"]), float(g["lo"]), float(g["hi"]))


def earth_gc_to_cel():
    """Galactocentric directions -> the Earth celestial frame of bright_stars.py (x = RA 12h)."""
    flip = np.diag([-1.0, 1.0, 1.0])  # galactocentric x points away from l = 0
    rz = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])  # RA -> RA - 12h
    return rz @ A_G.T @ flip


def scenario_geometry(cfg):
    """(observer_pc, R_gc_to_cel) from the scenario section (fallback for starfield info)."""
    sc = cfg["scenario"]
    if sc["mode"] == "earth":
        return np.array([frames.R0_PC, 0.0, SUN_Z_PC]), earth_gc_to_cel()
    arches = frames.lb_to_galactocentric(*ARCHES_LB, frames.R0_PC)
    off = np.asarray(sc["observer_offset_dir"], float)
    obs = arches + sc.get("observer_distance_pc", 3.0) * off / np.linalg.norm(off)
    sgr = frames.lb_to_galactocentric(*SGR_A_LB, frames.R0_PC)
    return obs, frames.galactocentric_to_celestial(obs, arches, sgr, sc.get("roll_deg", 0.0))


# ----------------------------------------------------------------------------- cache
class Cache:
    def __init__(self, root, enabled=True, force=()):
        self.root, self.enabled, self.force = root, enabled, set(force or ())
        os.makedirs(root, exist_ok=True)

    def path(self, stage, key):
        return os.path.join(self.root, f"{stage}_{key}.npz")

    def load(self, stage, key):
        p = self.path(stage, key)
        if not self.enabled or {stage, stage.split("_")[0]} & self.force or not os.path.exists(p):
            return None
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def save(self, stage, key, **arrays):
        if self.enabled:
            p = self.path(stage, key)
            np.savez(p + ".tmp.npz", **arrays)
            os.replace(p + ".tmp.npz", p)

    def listing(self):
        return sorted(f for f in os.listdir(self.root) if f.endswith(".npz")) if os.path.isdir(self.root) else []


def _info_json(info):
    return json.dumps(camera._jsonable(info), default=str)


def _info_load(s):
    info = json.loads(str(s))
    for k in ("R_gc_to_cel", "observer_pc"):
        if k in info:
            info[k] = np.asarray(info[k], float)
    return info


# ----------------------------------------------------------------------------- stages 1-2
def starfield_cfg(cfg):
    sc, s = cfg["scenario"], dict(cfg["starfield"])
    s.setdefault("mode", sc["mode"])
    s.setdefault("observer_offset_dir", sc["observer_offset_dir"])
    s.setdefault("offset_pc", sc.get("observer_distance_pc", 3.0))
    s.setdefault("roll_deg", sc["roll_deg"])
    return s


class LazyGalaxy:
    """Builds galaxy.GalaxyModel (with gas fields) on first use and shares it between stages."""

    def __init__(self, cfg, grid):
        self.cfg, self.grid, self.model = cfg, grid, None

    def get(self):
        if self.model is None:
            t0 = time.time()
            self.model = galaxy_model(self.cfg, self.grid)
            log(f"galaxy model ({'with' if self.cfg['galaxy'].get('use_gas') else 'no'} gas fields): {time.time() - t0:.1f} s")
        return self.model


GALAXY_MODEL_KEYS = ("preset", "overrides", "use_gas", "gas_seed", "gas_resolution", "gas_dust_budget", "extra_fields")


def galaxy_model_cfg(cfg):
    """The galaxy-section keys that define GalaxyModel (map settings such as nside excluded)."""
    return {k: cfg["galaxy"].get(k) for k in GALAXY_MODEL_KEYS}


def stage_starfield(cfg, grid, cache, lazy=None):
    g = cfg["galaxy"]
    use_gal = lazy is not None and g.get("enabled", True) and g.get("pass_to_starfield", True)
    key = config.stable_hash(cfg["grid"], _mod("starfield").resolve_config(starfield_cfg(cfg)),
                             galaxy_model_cfg(cfg) if use_gal else None)
    z = cache.load("starfield", key)
    if z is not None:
        names = list(z["names"]) if z["names"].size else []
        toa = sky.TOASky(grid, sky.PointSources(z["dirs"], z["irr"], [str(n) for n in names]), z["diffuse_map"])
        return toa, _info_load(z["info"]), key
    t0 = time.time()
    toa, info = _mod("starfield").build_starfield(starfield_cfg(cfg), grid, galaxy=lazy.get() if use_gal else None)
    info = dict(info)
    obs, R = scenario_geometry(cfg)
    for k, v in (("observer_pc", obs), ("R_gc_to_cel", R)):
        if k not in info:
            log(f"starfield info lacks {k!r}: using the scenario fallback")
            info[k] = v
    info["observer_pc"], info["R_gc_to_cel"] = np.asarray(info["observer_pc"], float), np.asarray(info["R_gc_to_cel"], float)
    p = toa.points
    names = np.array([str(n) for n in p.names]) if len(p.names) == len(p) else np.array([], dtype="<U1")
    cache.save("starfield", key, dirs=p.dirs, irr=p.irr, names=names, diffuse_map=toa.diffuse_map, info=_info_json(info))
    log(f"starfield: {len(p)} point sources, nside {toa.nside}, {time.time() - t0:.1f} s")
    return toa, info, key


def galaxy_params(cfg, info):
    g = cfg["galaxy"]
    r_res = g.get("r_res_pc")
    r_res = info.get("r_res_pc", cfg["starfield"].get("r_res_pc", 0.0)) if r_res is None else r_res
    inner = g.get("inner_factor")
    inner = info.get("inner_factor", 1.0) if inner is None else inner
    return float(r_res or 0.0), float(inner)


def galaxy_model(cfg, grid):
    g = cfg["galaxy"]
    G, fields, ov = _mod("galaxy"), [], dict(g.get("overrides") or {})
    preset = g.get("preset", "milky_way")
    extra = []
    if g.get("extra_fields"):
        extra = _mod("clouds").extra_fields(g["extra_fields"], grid, scenario_geometry(cfg)[0])
        log(f"extra fields: {', '.join(f.name for f in extra)}")
    if not g.get("use_gas"):
        return G.GalaxyModel(grid, preset=preset, fields=extra, **ov)
    fields = _mod("gas").load_cached(grid, seed=g.get("gas_seed", 0), resolution=g.get("gas_resolution", "default"))
    if not g.get("gas_dust_budget", True) or not any(f.kind == "extinction" for f in fields):
        return G.GalaxyModel(grid, preset=preset, fields=fields + extra, **ov)
    # The gas extinction fields already carry the CMZ dust (clouds + diffuse CMZ + CND), and the
    # disk calibration A_V(Sun -> Sgr A*) = AV_sgra ignores fields. Drop the smooth CMZ term and
    # lower the calibration target by the gas A_V so the total along that sightline stays AV_sgra.
    m = G.GalaxyModel(grid, preset=preset, fields=fields, **ov)
    sun = np.array([m.params["R0"], 0.0, 0.0])
    i550 = int(np.argmin(np.abs(grid.wl - 550.0)))
    tau = m.optical_depth(sun, m.c0[None], fields=True)[0, i550] - m.optical_depth(sun, m.c0[None], fields=False)[0, i550]
    gas_av = float(tau / m.C[i550])
    target = m.params["AV_sgra"] - gas_av / max(m.params["dust_scale"], 1e-9)
    ov.setdefault("cmz_AV", 0.0)
    ov.setdefault("AV_sgra", max(target, 0.0))
    m = G.GalaxyModel(grid, preset=preset, fields=fields + extra, **ov)
    m.info["gas_AV_sun_sgra"] = gas_av
    log(f"gas dust budget: gas A_V(Sun -> Sgr A*) {gas_av:.1f}, smooth-dust target {ov['AV_sgra']:.1f}, cmz_AV {ov['cmz_AV']}")
    return m


def stage_galaxy(cfg, grid, info, cache, lazy):
    g = cfg["galaxy"]
    if not g.get("enabled", True):
        return None, "off"
    r_res, inner = galaxy_params(cfg, info)
    key = config.stable_hash(cfg["grid"], galaxy_model_cfg(cfg), g.get("nside"), info["observer_pc"], info["R_gc_to_cel"], r_res, inner)
    z = cache.load("galaxy", key)
    if z is not None:
        return sky.TOASky(grid, sky.PointSources.empty(grid.n), z["diffuse_map"]), key
    t0 = time.time()
    t = lazy.get().render_map(info["observer_pc"], int(g["nside"]), info["R_gc_to_cel"], r_res_pc=r_res, inner_factor=inner)
    cache.save("galaxy", key, diffuse_map=np.asarray(t.diffuse_map, np.float32))
    log(f"galaxy map nside {g['nside']} (r_res {r_res} pc, inner_factor {inner:.3g}): {time.time() - t0:.1f} s")
    return t, key


def sun_sources(cfg, grid):
    s = cfg["scenario"].get("sun") or {}
    if not s.get("enabled"):
        return sky.PointSources.empty(grid.n)
    d = frames.sun_celestial(s.get("ecliptic_longitude_deg", 0.0))
    return sky.PointSources(d[None], spectral.blackbody_irradiance(SOLAR_CONSTANT, SUN_TEFF_K, grid)[None], ["Sun"])


def build_toa(cfg, grid, cache, stop_after=None):
    lazy = LazyGalaxy(cfg, grid)
    star, info, k1 = stage_starfield(cfg, grid, cache, lazy)
    if stop_after == "starfield":
        return star, info, k1
    gal, k2 = stage_galaxy(cfg, grid, info, cache, lazy)
    toa = star if gal is None else star.merged(gal)
    sun = sun_sources(cfg, grid).concat(_mod("moon").moon_sources(cfg, grid))
    if len(sun):
        toa = sky.TOASky(grid, toa.points.concat(sun), toa.diffuse_map, toa.diffuse_fn, toa.meta)
    if gal is not None and cfg["galaxy"].get("exact_view"):
        r_res, inner = galaxy_params(cfg, info)
        fn = lazy.get().diffuse_fn(info["observer_pc"], info["R_gc_to_cel"], r_res_pc=r_res, inner_factor=inner)
        base = star.diffuse_map
        toa.diffuse_fn = lambda d: fn(d) + frames.sample_healpix(base, d)
    moon = cfg["scenario"].get("moon") or {}  # hashed only when enabled, so older caches stay valid
    return toa, info, config.stable_hash(k1, k2, cfg["scenario"].get("sun"), *([moon] if moon.get("enabled") else []))


# ----------------------------------------------------------------------------- stage 3
def _chunks(fn, d, n):
    outs = [fn(d[i:i + n]) for i in range(0, len(d), n)]
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def _upsample(a, shape):
    """Bilinear (h, w, c) -> (H, W, c) on pixel centres."""
    h, w = a.shape[:2]
    H, W = shape
    if (h, w) == (H, W):
        return a
    r = np.clip((np.arange(H) + 0.5) * h / H - 0.5, 0, h - 1)
    c = np.clip((np.arange(W) + 0.5) * w / W - 0.5, 0, w - 1)
    R, C = np.meshgrid(r, c, indexing="ij")
    return np.stack([map_coordinates(a[..., k], [R, C], order=1, mode="nearest") for k in range(a.shape[2])], axis=-1)


def sky_terms(atm, toa, R, dirs, chunk=262144):
    """atmosphere.sky on (N, 3) horizon dirs in chunks -> dict(T, inscatter, airglow)."""
    return _chunks(lambda d: {k: np.asarray(v, np.float32) for k, v in atm.sky(toa, R, d).items()
                              if k in ("T", "inscatter", "airglow")}, dirs, chunk)


def sky_image(atm, toa, R, view, vcfg, ctx, scfg):
    """(H, W, nbins) float32 sky radiance (no point sources) for every pixel of the view."""
    sub = max(1, int(scfg.get("subsample", 1)))
    vc = dict(vcfg)
    if vc.get("kind", "perspective") == "perspective":
        vc["azimuth_deg"], vc["pitch_deg"], _ = camera.resolve_pointing(vc, ctx)
    coarse = view if sub == 1 else camera.make_view(vc, ctx, scale=1.0 / sub)
    h, w = coarse.height, coarse.width
    t = sky_terms(atm, toa, R, coarse.dirs.reshape(-1, 3), scfg.get("chunk", 262144))
    shape = (view.height, view.width)
    T = _upsample(t["T"].reshape(h, w, -1), shape)
    add = _upsample((t["inscatter"] + t["airglow"]).reshape(h, w, -1), shape)
    diff = np.asarray(toa.diffuse(view.dirs.reshape(-1, 3) @ R), np.float32).reshape(T.shape)
    return (T * diff + add).astype(np.float32)


# ----------------------------------------------------------------------------- stage 5
def env_sky_fn(atm, toa, R, faint, nside, grid):
    """Callable horizon dirs -> downwelling sky radiance: atmosphere.sky tabulated on HEALPix,
    with the faint point sources (PointSources, horizon frame) added as radiance."""
    import healpy as hp
    d = frames.healpix_dirs(nside)
    t = sky_terms(atm, toa, R, d)
    tab = t["T"] * np.asarray(toa.diffuse(d @ R), np.float32) + t["inscatter"] + t["airglow"]
    if len(faint):
        pix = hp.vec2pix(nside, *faint.dirs.T)
        np.add.at(tab, pix, (faint.irr / hp.nside2pixarea(nside)).astype(np.float32))
    tab = tab.astype(np.float64)
    return lambda dirs: frames.sample_healpix(tab, np.asarray(dirs, float).reshape(-1, 3)), tab


def foreground_layers(cfg, grid, view, vname, atm, toa, R, pts, cache, key_base, cpus):
    fcfg, fr = cfg["foreground"], cfg["foreground_render"]
    key = config.stable_hash(key_base, _mod("foreground").resolve_scene(fcfg), fr, view.kind, view.width, view.height, view.dirs[::max(1, view.height // 4), ::max(1, view.width // 4)])
    z = cache.load(f"foreground_{vname}", key)
    if z is not None:
        return z
    t0 = time.time()
    lit = pts.irr @ spectral.cmf_bins(grid)[:, 1] > 0
    order = pts.brightest(len(pts), grid)
    order = order[lit[order]]
    k = int(fr.get("n_lights", 32))
    while True:
        lights, faint = pts.select(order[:k]), pts.select(order[k:])
        sky_fn, _ = env_sky_fn(atm, toa, R, faint, int(fr.get("env_nside", 64)), grid)
        try:
            out = _mod("foreground").render_foreground(fcfg, view, sky_fn, lights, grid,
                                                       spp=int(fr.get("spp", 64)), cpus=cpus)
            break
        except KeyError as e:
            # Mitsuba >= 3.6 merges identical nested plugins, so directional emitters created with
            # the same 'uniform' irradiance share one parameter and light_k.irradiance.value
            # (k >= 1) is absent from mi.traverse(). Fall back to one explicit light; the other
            # points enter the environment map through sky_fn.
            if k <= 1 or "irradiance" not in str(e):
                raise
            log(f"[{vname}] foreground KeyError {e} with {k} lights; retrying with 1 light")
            k = 1
    finfo = out.get("info")
    out = {k: np.asarray(out[k], np.float32) for k in ("radiance", "alpha", "depth", "normal") if k in out}
    if finfo is not None:
        out["info"] = np.array(_info_json(finfo))
    cache.save(f"foreground_{vname}", key, **out)
    log(f"[{vname}] foreground {fcfg.get('type')}: {len(lights)} lights, {time.time() - t0:.1f} s")
    return out


def composite(L_sky, fg, atm, toa, R, view, premultiplied=True):
    """a (L_fg T_ap + S_ap) + (1 - a) L_sky with aerial perspective along the depth map.
    premultiplied: fg radiance already carries the coverage a (foreground.py convention)."""
    a = np.clip(fg["alpha"], 0, 1)
    cov = a > 0
    out = L_sky.copy()
    if not cov.any():
        return out
    depth = np.where(np.isfinite(fg["depth"]), fg["depth"], 1e5)[cov]
    T, S = atm.aerial_perspective(view.dirs[cov], depth, toa, R)
    ac = a[cov, None]
    surf = fg["radiance"][cov] * (1.0 if premultiplied else ac)
    out[cov] = surf * np.asarray(T, np.float32) + ac * np.asarray(S, np.float32) + (1 - ac) * L_sky[cov]
    return out


# ----------------------------------------------------------------------------- statistics
def source_table(pts, grid, n=10):
    if not len(pts):
        return []
    idx = pts.brightest(n, grid)
    alt, az = frames.vec_to_altaz(pts.dirs[idx])
    lux = spectral.to_xyz(pts.irr[idx], grid)[:, 1] * np.clip(pts.dirs[idx, 2], 0, None)
    v = np.atleast_1d(spectral.vmag_of(pts.irr[idx], grid))
    names = pts.names if len(pts.names) == len(pts) else [""] * len(pts)
    return [{"name": names[i], "alt_deg": float(a), "az_deg": float(z), "V": float(m), "horizontal_lux": float(x)}
            for i, a, z, m, x in zip(idx, alt, az, v, lux)]


def scene_stats(atm, toa, R, grid, pts):
    z = np.array([[0.0, 0.0, 1.0]])
    t = sky_terms(atm, toa, R, z)
    Lz = t["T"] * toa.diffuse(z @ R) + t["inscatter"] + t["airglow"]
    e = np.asarray(atm.ground_irradiance(toa, R), float)
    up = np.clip(pts.dirs[:, 2], 0, None) if len(pts) else np.zeros(0)
    e_pts = (pts.irr * up[:, None]).sum(0) if len(pts) else np.zeros(grid.n)
    return {
        "zenith_luminance_cd_m2": float(spectral.luminance(Lz, grid)[0]),
        "zenith_scotopic_cd_m2": float(spectral.scotopic_luminance(Lz, grid)[0]),
        "zenith_V_mag_arcsec2": float(np.atleast_1d(_mod("atmosphere").v_mag_arcsec2(Lz, grid))[0]),
        "zenith_airglow_cd_m2": float(spectral.luminance(t["airglow"], grid)[0]),
        "ground_illuminance_lux": float(spectral.to_xyz(e, grid)[1]),
        "ground_scotopic_lux": float(spectral.scotopic_luminance(e, grid)),
        "point_sources_horizontal_lux": float(spectral.to_xyz(e_pts, grid)[1]),
        "n_point_sources_above_horizon": int((up > 0).sum()) if len(pts) else 0,
        "toa_point_sources": int(len(toa.points)),
        "brightest": source_table(pts, grid),
    }


def image_stats(L, view, mask, grid):
    y = spectral.luminance(L[mask], grid) if mask.any() else np.zeros(1)
    lv = np.maximum(y, 1e-12)
    return {"n_px": int(mask.sum()), "mean_cd_m2": float(y.mean()), "median_cd_m2": float(np.median(y)),
            "log_average_cd_m2": float(np.exp(np.log(lv).mean())), "p05_cd_m2": float(np.percentile(y, 5)),
            "p95_cd_m2": float(np.percentile(y, 95)), "max_cd_m2": float(y.max())}


# ----------------------------------------------------------------------------- driver
def default_cpus():
    for k in ("SLURM_CPUS_PER_TASK", "NUMBA_NUM_THREADS"):
        if os.environ.get(k, "").isdigit():
            return int(os.environ[k])
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def render(cfg, views=None, stop_after=None, force=(), use_cache=True, cpus=None, scale=1.0):
    """Run the pipeline for a resolved config dict (config.load_config); returns output paths."""
    cpus = cpus or default_cpus()
    grid = make_grid(cfg)
    out_dir = cfg["output_dir"]
    cache = Cache(os.path.join(out_dir, "cache"), use_cache, force)
    timing = {}
    t0 = time.time()
    toa, info, toa_key = build_toa(cfg, grid, cache, stop_after)
    timing["sources_s"] = time.time() - t0
    if stop_after in ("starfield", "galaxy"):
        return {"toa_key": toa_key}
    sc = cfg["scenario"]
    R = frames.celestial_to_horizon(sc["hour_angle_deg"], sc["lat_deg"])
    atm = _mod("atmosphere").Atmosphere(cfg["atmosphere"], grid)
    pts = atm.point_irradiance(toa, R)
    stats = scene_stats(atm, toa, R, grid, pts)
    stats["naked_eye_limit_V_zenith"] = float(camera.limiting_vmag(stats["zenith_luminance_cd_m2"], cfg["points"].get("visibility_factor", 1.0)))
    ctx ={"R_c2h": R, "R_gc_to_cel": info["R_gc_to_cel"], "observer_pc": info["observer_pc"]}
    key_base = config.stable_hash(toa_key, cfg["atmosphere"], sc, cfg["grid"])
    results = {"scene": stats, "views": {}}
    log(f"scene: zenith {stats['zenith_luminance_cd_m2']:.3g} cd/m^2 ({stats['zenith_V_mag_arcsec2']:.2f} V/arcsec^2), "
        f"ground {stats['ground_illuminance_lux']:.3g} lx, {stats['n_point_sources_above_horizon']} points above horizon")
    for vcfg in cfg["views"]:
        vname = vcfg["name"]
        if views and vname not in views:
            continue
        vc = dict(vcfg)
        for k in ("width", "height", "size"):
            if k in vc:
                vc[k] = max(4, int(round(vc[k] * scale)))
        tv = {}
        t1 = time.time()
        view = camera.make_view(vc, ctx)
        _, _, tgt = camera.resolve_pointing(vc, ctx) if vc.get("kind") == "perspective" else (0, 0, None)
        skey = config.stable_hash(key_base, vc, cfg["sky"])
        z = cache.load(f"sky_{vname}", skey)
        if z is not None:
            L_sky = z["L"]
        else:
            L_sky = sky_image(atm, toa, R, view, vc, ctx, cfg["sky"])
            cache.save(f"sky_{vname}", skey, L=L_sky)
        tv["sky_s"] = time.time() - t1
        if stop_after == "sky":
            continue
        t1 = time.time()
        L = L_sky.copy()
        names = np.array(pts.names if len(pts.names) == len(pts) else [""] * len(pts), object)
        is_disk = (names == "Sun") | (names == "Moon")
        vis = None
        if cfg["camera"].get("mode", "eye") == "eye" and cfg["points"].get("visibility", True):
            n_dep, vis = camera.deposit_points_eye(L, view, pts.dirs[~is_disk], pts.irr[~is_disk], cfg["points"], grid, L_sky)
        else:
            n_dep = camera.deposit_points(L, view, pts.dirs[~is_disk], pts.irr[~is_disk], cfg["points"], grid)
        for i in np.flatnonzero(is_disk):
            if not pts.irr[i].any():
                continue
            if names[i] == "Sun":
                n_dep += camera.deposit_disk(L, view, pts.dirs[i], pts.irr[i], SUN_RADIUS_DEG, cfg["points"], grid)
            else:
                sun_h = R @ frames.sun_celestial((sc.get("sun") or {}).get("ecliptic_longitude_deg", 0.0))
                n_dep += _mod("moon").deposit(L, view, pts.dirs[i], pts.irr[i], sun_h, cfg["points"], grid)
        tv["points_s"] = time.time() - t1
        if stop_after == "points":
            continue
        fg = None
        if cfg.get("foreground"):
            t1 = time.time()
            fg = foreground_layers(cfg, grid, view, vname, atm, toa, R, pts, cache, config.stable_hash(key_base, vc), cpus)
            L = composite(L, fg, atm, toa, R, view, cfg['foreground_render'].get('premultiplied', True))
            tv["foreground_s"] = time.time() - t1
        if stop_after == "foreground":
            continue
        t1 = time.time()
        alpha = fg["alpha"] if fg is not None else np.zeros((view.height, view.width), np.float32)
        sky_mask = view.valid & (alpha < 0.5) & (view.dirs[..., 2] > 0)
        # adaptation over rendered pixels: sky above the horizon and foreground surfaces (an
        # unrendered, black below-horizon band would otherwise drag the log-average down)
        seen = view.valid & ((alpha >= 0.5) | (view.dirs[..., 2] > 0))
        rgb8, cinfo, xyzs = camera.develop(L, view, grid, cfg["camera"], mask=seen, cpus=cpus)
        if cinfo["mode"] == "eye" and cfg["camera"].get("moon_inset", True):
            for i in np.flatnonzero(names == "Moon"):
                if pts.irr[i].any():
                    sun_h = R @ frames.sun_celestial((sc.get("sun") or {}).get("ecliptic_longitude_deg", 0.0))
                    rgb8 = _mod("moon").inset(rgb8, view, pts.dirs[i], sun_h, pts.irr[i], grid,
                                              None if fg is None else fg["alpha"],
                                              night_veil=cfg["camera"].get("moon_night_veil", 1.0))
        tv["camera_s"] = time.time() - t1
        vinfo = {"config": cfg["name"], "view": vc, "pointing_target_alt_az": tgt,
                 "pixel_deg": camera.view_pixel_deg(view), "camera": cinfo,
                 "sky_luminance": image_stats(L_sky, view, sky_mask, grid),
                 "frame_luminance": image_stats(L, view, view.valid, grid),
                 "foreground_fraction": float(alpha[view.valid].mean()), "points_deposited": n_dep,
                 "scene": stats, "timing_s": {**timing, **tv}, "cpus": cpus}
        if vis is not None:
            vis["naked_eye_limit_V_sky_median"] = float(camera.limiting_vmag(vinfo["sky_luminance"]["median_cd_m2"], vis["visibility_factor"]))
            vinfo["visibility"] = vis
        if cfg["camera"].get("annotate", True):
            lab = (f"{cfg['name']} / {vname}: La {cinfo.get('La_cd_m2', 0):.2g} cd/m2, photopic {cinfo.get('photopic_fraction', 1):.2f}"
                   if cinfo["mode"] == "eye" else
                   f"{cfg['name']} / {vname}: ISO {cfg['camera']['iso']:.0f} f/{cfg['camera']['f_number']:.1f} {cfg['camera']['exposure_s']:g}s")
            rgb8 = camera.annotate(rgb8, view, [lab])
        extra = {"valid": view.valid}
        if fg is not None:
            extra.update(alpha=fg["alpha"], depth=fg["depth"])
        o = cfg["output"]
        paths = camera.write_outputs(os.path.join(out_dir, vname), rgb8, xyzs, vinfo,
                                     L if o.get("spectral") else None, extra, o.get("png", True), o.get("npz", True))
        log(f"[{vname}] {view.width}x{view.height}: sky log-avg {vinfo['sky_luminance']['log_average_cd_m2']:.3g} cd/m^2, "
            f"{n_dep} points" + (f" ({vis['n_points_visible']:.0f} visible)" if vis else "") + ", " + ", ".join(f"{k} {v:.1f}" for k, v in tv.items()) + f" -> {paths[0]}")
        results["views"][vname] = {"paths": paths, "info": vinfo}
    with open(os.path.join(out_dir, "scene.json"), "w") as f:
        json.dump(camera._jsonable({"scene": stats, "starfield_info": {k: v for k, v in info.items()},
                                    "config": cfg}), f, indent=1, default=str)
    return results


def cache_state(cfg):
    """Stage -> cache file names present for this config (for `nightsky info`)."""
    root = os.path.join(cfg["output_dir"], "cache")
    out = {}
    for f in (sorted(os.listdir(root)) if os.path.isdir(root) else []):
        out.setdefault(f.rsplit("_", 1)[0], []).append(f)
    return out
