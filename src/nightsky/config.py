"""Render configuration: YAML file deep-merged onto DEFAULTS, plus `--set a.b.c=value` overrides.

Values given with --set are parsed as YAML (`--set camera.iso=6400`, `--set foreground=null`,
`--set views[0].width=640`). Section hashes (sha1 of the canonical JSON of a sub-tree) key the
stage caches in compose.py, so editing one section invalidates only the stages that read it.

Sections: name, output_dir, scenario, grid, starfield, galaxy, atmosphere, foreground, views,
points, camera, output. See configs/*.yaml for documented examples.
"""
import copy
import hashlib
import json
import os
import re

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DEFAULTS = {
    "name": None,  # default: config file stem
    "output_dir": None,  # default: renders/nightsky/<name>/
    "scenario": {
        "mode": "gc",  # gc | earth
        "lat_deg": 40.0,
        "hour_angle_deg": 180.0,  # hour angle of the equinox direction (celestial x)
        "observer_offset_dir": [1.0, 0.0, 0.0],
        "observer_distance_pc": 3.0,
        "roll_deg": 0.0,
        "sun": {"enabled": False, "ecliptic_longitude_deg": 0.0},
    },
    "grid": {"n_bins": 20, "lo": 380.0, "hi": 780.0},
    "starfield": {},  # passed to starfield.build_starfield; mode/offset/roll filled from scenario
    "galaxy": {
        "enabled": True,
        "preset": "milky_way",
        "overrides": {},
        "nside": 128,
        "use_gas": False,
        "gas_seed": 0,
        "gas_resolution": "default",
        "r_res_pc": None,  # None: starfield info["r_res_pc"] (else starfield cfg r_res_pc, else 0)
        "inner_factor": None,  # None: starfield info["inner_factor"] (else 1)
        "exact_view": False,  # use GalaxyModel.diffuse_fn for the visible background
        "pass_to_starfield": True,  # starfield uses the model for extinction / local density
    },
    "atmosphere": "earth_clear",
    "foreground": None,  # scene_cfg dict for foreground.render_foreground, or None
    "foreground_render": {"spp": 64, "n_lights": 32, "env_nside": 64, "premultiplied": True},
    "views": [{"name": "allsky", "kind": "allsky", "width": 1024, "height": 1024}],
    "points": {
        "psf_sigma_px": 0.6,  # minimum Gaussian core width (pixels)
        "seeing_fwhm_arcsec": 1.5,
        "airy_aperture_m": None,  # diffraction core lambda/D (eye: 0.007)
        "radius_px": 4,
        "min_irradiance_frac": 0.0,  # skip sources fainter than this fraction of the brightest
        "chunk": 20000,
        # naked-eye detection (camera.mode eye only): Schaefer (1990) threshold x visibility_factor
        "visibility": True,
        "visibility_factor": 1.0,
        "visibility_sigma_mag": 0.3,  # width of the psychometric function (mag)
        "unresolved_blur_deg": 0.3,  # undetected starlight spread as a diffuse glow
        "unresolved_min_px": 1.5,
    },
    "sky": {"subsample": 2},  # atmosphere.sky evaluated on a coarser pixel grid, bilinear upsampled
    "camera": {
        "mode": "eye",  # eye | camera
        # eye
        "adaptation_cd_m2": None,  # None: log-average of the mesopic scene luminance
        "key": 0.18,
        "adaptation_gamma": 0.75,  # partial adaptation: La_eff = La^g * l_ref^(1-g)
        "l_ref_cd_m2": 30.0,
        "mesopic": True,
        "mesopic_range_cd_m2": [0.005, 5.0],  # CIE 191:2010 mesopic range
        "glare": True,
        "age_years": 25.0,
        "pigmentation": 0.5,  # CIE 2002: 0 very dark, 0.5 brown, 1.0 light, 1.2 very light eyes
        "glare_max_deg": 60.0,
        # camera
        "iso": 3200.0,
        "f_number": 2.8,
        "exposure_s": 20.0,
        "bloom": 0.02,  # fraction of light in the lens bloom halo
        "bloom_sigma_deg": 0.15,
        "spikes": 0,  # number of diffraction spikes (0, 4, 6, 8)
        "spike_fraction": 0.004,
        # tone mapping
        "tone": None,  # None: reinhard for eye, filmic for camera; or "linear"
        "white": 6.0,  # extended Reinhard white point (display units)
        "annotate": True,
    },
    "output": {"png": True, "npz": True, "spectral": False},
}


def deep_merge(base, over):
    """Recursive dict merge (over wins); lists and scalars are replaced."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


_KEY = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def _path(key):
    return [int(i) if i else k for k, i in _KEY.findall(key)]


def set_value(cfg, key, value):
    """Set cfg[a][b][c] for key 'a.b.c' (list items as 'views[0].width'), creating dicts."""
    parts = _path(key)
    node = cfg
    for p, nxt in zip(parts[:-1], parts[1:]):
        if isinstance(p, int):
            node = node[p]
            continue
        if not isinstance(node.get(p), (dict, list)):
            node[p] = [] if isinstance(nxt, int) else {}
        node = node[p]
    last = parts[-1]
    if isinstance(last, str) and isinstance(node.get(last), dict) and isinstance(value, dict):
        node[last] = deep_merge(node[last], value)
    else:
        node[last] = value
    return cfg


def apply_sets(cfg, sets):
    for s in sets or []:
        if "=" not in s:
            raise ValueError(f"--set expects key=value, got {s!r}")
        k, v = s.split("=", 1)
        set_value(cfg, k.strip(), yaml.safe_load(v))
    return cfg


def load_config(path=None, sets=(), overrides=None):
    """Resolved config dict: DEFAULTS <- YAML file <- overrides dict <- --set strings."""
    user = {}
    if path:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULTS, user)
    cfg = deep_merge(cfg, overrides or {})
    apply_sets(cfg, sets)
    if not cfg.get("name"):
        cfg["name"] = os.path.splitext(os.path.basename(path))[0] if path else "untitled"
    if not cfg.get("output_dir"):
        cfg["output_dir"] = os.path.join(ROOT, "renders", "nightsky", cfg["name"])
    for i, v in enumerate(cfg["views"]):
        v.setdefault("name", f"{v.get('kind', 'view')}{i}")
    return cfg


def _canon(x):
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if hasattr(x, "tolist"):
        return _canon(x.tolist())
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(f"{x:.12g}")
    return x


def stable_hash(*objs, n=12):
    s = json.dumps([_canon(o) for o in objs], sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def section_hash(cfg, *sections):
    """Hash of the named top-level sections (e.g. section_hash(cfg, 'grid', 'starfield'))."""
    return stable_hash(*[[s, cfg.get(s)] for s in sections])


def dump(cfg):
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None)
