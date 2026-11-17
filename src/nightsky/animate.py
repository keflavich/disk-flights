"""Day-night time-lapse: frame list -> compose.render per hour angle -> re-developed PNGs -> MP4.

    python3 -m nightsky.animate ../configs/gc_arches_timelapse.yaml [--hours 0:24:0.25 | 12,18,19]
            [--frames N] [--set key=value ...] [--scale s] [--cpus n] [--fps f] [--views a,b]
            [--exposure smoothed,per-frame,fixed] [--fixed-la cd/m2] [--tau-light-min m]
            [--tau-dark-min m] [--no-overlay] [--no-video] [--no-render] [--no-resume]

Time axis: local solar time t (h, 12 = noon). H = 15 (t - 12) + RA_sun deg, with RA_sun the right
ascension of scenario.sun (frames.sun_celestial; 0 for ecliptic longitude 0, 180 for 180), so
24 h of t is 360 deg of H (H is not wrapped; frames.celestial_to_horizon is periodic in it).
`--hours a:b:s` steps by s (b excluded); `a:b` with --frames N gives N frames, b excluded when
b - a is a whole number of days (closed cycle), else included; a comma list gives explicit times.
A stop before the start, or a list that steps backwards, continues through midnight
(22:2:1 -> 22, 23, 24, 25 h; 23,1 -> 23, 25 h). Frames are stored under frames/h<t mod 24>/, so
25 h reuses a 1 h render. Default: the config's animation.hours (quote it in YAML: unquoted 6:30
is read as the base-60 number 390), else 0:24:0.25.

Shared work: one Cache under <output_dir>/cache holds the H-independent stages (starfield,
galaxy; optionally symlinked from animation.seed_cache). compose.build_toa and
atmosphere.Atmosphere are memoised in-process for the run (both are H-independent; the
Atmosphere keys its per-scene cache on R). Sky, points and foreground lighting depend on H and
run per frame in <output_dir>/frames/h<t>/ (that frame cache is deleted after the frame unless
animation.keep_frame_cache). Perspective views with azimuth/pitch "auto" are frozen at the
pointing for animation.pointing_time_h (a tripod time-lapse); animation.pointing: track keeps
compose's per-frame target tracking.

Exposure (camera.mode eye): compose's npz holds the post-glare linear XYZ and scotopic images;
the glare PSF is linear and independent of La, so each frame is re-developed from the npz by
camera.develop itself with to_xyzs returning the stored image, glare off and
camera.adaptation_cd_m2 = the chosen La (develop_xyzs).
  per-frame  La_i from compose (log-average mesopic luminance of the frame, JSON sidecar).
  smoothed   first-order adaptation in ln La: d ln La_a / dt = (ln La(t) - ln La_a) / tau, with
             tau = tau_light when the scene is brighter than the adapted state and tau_dark
             otherwise (defaults 1 and 10 min of scene time: cone light adaptation takes seconds
             to a minute, the cone plateau of dark adaptation ~10 min and full rod adaptation
             30-40 min, Hecht, Haig & Chase 1937). ln La(t) is linear between frames; the ODE
             is solved exactly for the linear ramp on sub-steps <= tau/10 (tau re-chosen per
             sub-step), so the result does not depend on the frame spacing. A closed 24 h cycle is
             integrated over two periods and the second kept (periodic steady state).
  fixed      one La for all frames (--fixed-la, default the geometric mean of the per-frame La).
camera.mode camera: fixed uses the configured ISO / f-number / time (camera.camera_exposure);
per-frame and smoothed set the exposure time so that L_sat = La / key (auto exposure), with La
the log-average Y of the frame's valid pixels. Each view is smoothed independently.
"""
import argparse
import contextlib
import copy
import glob
import json
import os
import shutil
import subprocess
import time
import types

import numpy as np

from . import camera, compose, config, frames

MODES = ("per-frame", "smoothed", "fixed")
ANIM_DEFAULTS = {"hours": "0:24:0.25", "fps": 24, "exposure": "smoothed", "tau_light_min": 1.0,
                 "tau_dark_min": 10.0, "fixed_la_cd_m2": None, "pointing": "fixed", "pointing_time_h": None,
                 "overlay": True, "seed_cache": None, "keep_frame_cache": False}
DEVELOP_ONLY_KEYS = ("key", "adaptation_cd_m2", "local_mesopic", "adaptation_gamma", "l_ref_cd_m2", "mesopic", "mesopic_range_cd_m2",
                     "tone", "white", "annotate", "iso", "f_number", "exposure_s")
log = compose.log


# ----------------------------------------------------------------------------- time axis
def unwrap_hours(t_h):
    """Times made increasing by adding 24 h after every backwards step (23, 1 -> 23, 25)."""
    t = np.asarray(t_h, float)
    return t + 24.0 * np.concatenate([[0], np.cumsum(np.diff(t) <= 0)]) if len(t) else t


def parse_hours(spec=None, n_frames=None, default="0:24:0.25"):
    """Local solar times (h, increasing) from 'a:b:s', 'a:b' (+ n_frames), 'a,b,c' or None (+ n_frames).
    A stop before the start wraps through midnight (22:2:1 -> 22, 23, 24, 25); so does a list."""
    spec = default if spec is None and n_frames is None else spec
    if spec is None:
        spec = "0:24"
    if isinstance(spec, (list, tuple, np.ndarray)):
        spec = ",".join(str(x) for x in spec)
    elif isinstance(spec, (int, float, np.number)) and abs(spec) >= 48:
        # YAML 1.1 reads unquoted 6:30 as 390 and 0:24:0.25 as 1440.25 (base 60)
        raise ValueError(f"hours {spec} looks like a YAML base-60 number; quote it (hours: \"0:24:0.25\") or use --hours")
    spec = str(spec).strip()
    if ":" not in spec:
        t = unwrap_hours([float(x) for x in spec.split(",") if x.strip()])
        t = t[:n_frames] if n_frames else t
    else:
        p = [float(x) for x in spec.split(":")]
        a, b = p[0], p[1] + (24.0 if p[1] < p[0] else 0.0)
        if len(p) == 3:
            if p[2] <= 0:
                raise ValueError(f"--hours step must be positive, got {p[2]}")
            t = a + p[2] * np.arange(max(int(np.ceil((b - a) / p[2] - 1e-6)), 0))
            t = t[:n_frames] if n_frames else t
        else:
            n = int(n_frames or 96)
            closed = abs((b - a) / 24.0 - round((b - a) / 24.0)) < 1e-9 and b != a
            t = a + (b - a) * np.arange(n) / (n if closed or n == 1 else n - 1)
    if not len(t):
        raise ValueError(f"empty frame list from --hours {spec!r}")
    return t


def frame_tag(t_h):
    """Frame directory name: solar time mod 24 h, so 1 h and 25 h share one render."""
    return f"h{round(float(t_h) % 24.0, 3) % 24.0:07.3f}"


def is_closed_cycle(t_h):
    """True when the next uniform step after the last time is the first time + 24 h."""
    t = np.asarray(t_h, float)
    if len(t) < 2:
        return False
    d = np.diff(t)
    return bool(np.allclose(d, d[0], rtol=1e-6, atol=1e-9) and abs(t[-1] + d[0] - t[0] - 24.0) < 1e-6)


def sun_ra_deg(cfg):
    s = cfg["scenario"].get("sun") or {}
    d = frames.sun_celestial(s.get("ecliptic_longitude_deg", 0.0))
    return float(np.degrees(np.arctan2(d[1], d[0])))


def solar_to_hour_angle(t_h, ra_sun_deg=0.0):
    return 15.0 * (np.asarray(t_h, float) - 12.0) + ra_sun_deg


def sun_altitude(cfg, H):
    s = cfg["scenario"].get("sun") or {}
    d = frames.celestial_to_horizon(H, cfg["scenario"]["lat_deg"]) @ frames.sun_celestial(s.get("ecliptic_longitude_deg", 0.0))
    return float(frames.vec_to_altaz(d)[0])


def clock(t_h):
    m = int(round((t_h % 24.0) * 60)) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


# ----------------------------------------------------------------------------- adaptation
def smooth_log_la(t_h, la, tau_light_min=1.0, tau_dark_min=10.0, periodic=None, substep=0.1):
    """Adapted luminance per frame (cd/m^2) from the instantaneous La (first-order model in ln La)."""
    t = np.asarray(t_h, float) * 60.0
    y = np.log(np.maximum(np.asarray(la, float), camera.ABS_THRESHOLD_CD_M2))
    if len(t) < 2:
        return np.exp(y)
    if np.any(np.diff(t) <= 0):
        raise ValueError("frame times must increase")
    periodic = is_closed_cycle(t_h) if periodic is None else periodic
    if periodic:
        t, y = np.concatenate([t, t + 1440.0]), np.concatenate([y, y])
    h = substep * min(tau_light_min, tau_dark_min)
    a, out = y[0], [y[0]]
    for i in range(1, len(t)):
        n = max(1, int(np.ceil((t[i] - t[i - 1]) / h)))
        d, k = (t[i] - t[i - 1]) / n, (y[i] - y[i - 1]) / (t[i] - t[i - 1])
        for j in range(n):
            y0 = y[i - 1] + j * d * k
            tau = tau_light_min if y0 + 0.5 * d * k > a else tau_dark_min
            a = y0 + d * k - k * tau + (a - y0 + k * tau) * np.exp(-d / tau)  # exact for a linear ramp
        out.append(a)
    out = np.array(out)
    return np.exp(out[len(out) // 2:] if periodic else out)


@contextlib.contextmanager
def _given_xyzs(xyzs):
    orig = camera.to_xyzs
    camera.to_xyzs = lambda img, grid: xyzs
    try:
        yield
    finally:
        camera.to_xyzs = orig


def develop_xyzs(xyzs, valid, ccfg, la=None, kind="perspective"):
    """uint8 sRGB from post-PSF XYZS (H, W, 4): camera.develop with to_xyzs returning xyzs, the
    glare / lens convolution off (already applied) and the adaptation fixed at la (eye), or an
    auto-exposure time t = 78 N^2 key / (0.65 S la) (camera; la None keeps the configured t)."""
    c = {**ccfg, "glare": False, "bloom": 0.0, "spikes": 0}
    if c.get("mode", "eye") == "eye":
        c["adaptation_cd_m2"] = float(la)
    elif la is not None:
        c["exposure_s"] = 78.0 * c["f_number"] ** 2 * c.get("key", 0.18) / (0.65 * c["iso"] * float(la))
    with _given_xyzs(np.array(xyzs, np.float32)):
        rgb, info, _ = camera.develop(None, types.SimpleNamespace(valid=valid, kind=kind), None, c, mask=valid)
    if c.get("mode", "eye") == "camera" and la is not None:
        info["La_cd_m2"] = float(la)
    return rgb, info


def load_frame_view(fdir, vname):
    with np.load(os.path.join(fdir, vname + ".npz")) as z:
        xyzs = np.concatenate([z["xyz"], z["scotopic"][..., None]], axis=-1)
        valid = z["valid"].astype(bool)
    with open(os.path.join(fdir, vname + ".json")) as f:
        info = json.load(f)
    return xyzs, valid, info


def frame_la(xyzs, valid, info):
    """La (cd/m^2) of a rendered frame: compose's eye adaptation, else the log-average Y of the
    valid pixels with Y > 0 (unrendered pixels below the horizon are exactly 0)."""
    la = info.get("camera", {}).get("La_cd_m2")
    return float(la) if la else camera.log_average(xyzs[..., 1], valid & (xyzs[..., 1] > 0))


def exposure_series(t_h, la, mode, acfg, ccfg):
    """Adaptation La per frame for one exposure mode (None = configured camera exposure)."""
    la = np.asarray(la, float)
    if mode == "per-frame":
        return la
    if mode == "smoothed":
        return smooth_log_la(t_h, la, acfg["tau_light_min"], acfg["tau_dark_min"])
    if mode == "fixed":
        if ccfg.get("mode", "eye") == "camera" and not acfg.get("fixed_la_cd_m2"):
            return np.full(len(la), np.nan)
        v = acfg.get("fixed_la_cd_m2") or float(np.exp(np.mean(np.log(np.maximum(la, camera.ABS_THRESHOLD_CD_M2)))))
        return np.full(len(la), max(float(v), camera.ABS_THRESHOLD_CD_M2))
    raise ValueError(f"exposure mode must be one of {MODES}, got {mode!r}")


# ----------------------------------------------------------------------------- shared state
@contextlib.contextmanager
def shared_state(shared_cache):
    """Memoise compose.build_toa (on the shared cache) and atmosphere.Atmosphere for the block."""
    atm_mod = compose._mod("atmosphere")
    orig_toa, orig_atm = compose.build_toa, atm_mod.Atmosphere
    memo, stats = {}, {"toa_builds": 0, "atm_builds": 0}

    def build_toa(cfg, grid, cache, stop_after=None):
        sc = {k: v for k, v in cfg["scenario"].items() if k != "hour_angle_deg"}
        k = ("toa", config.stable_hash(sc, cfg["grid"], cfg["starfield"], cfg["galaxy"]), stop_after)
        if k not in memo:
            t0 = time.time()
            memo[k] = orig_toa(cfg, grid, shared_cache, stop_after)
            stats["toa_builds"], stats["toa_s"] = stats["toa_builds"] + 1, stats.get("toa_s", 0.0) + time.time() - t0
        return memo[k]

    def atmosphere(preset="earth_clear", grid=None):
        k = ("atm", config.stable_hash(preset, None if grid is None else grid.wl))
        if k not in memo:
            t0 = time.time()
            memo[k] = orig_atm(preset, grid)
            stats["atm_builds"], stats["atm_s"] = stats["atm_builds"] + 1, stats.get("atm_s", 0.0) + time.time() - t0
            log(f"atmosphere {memo[k].name}: built once for all frames in {stats['atm_s']:.1f} s")
        return memo[k]

    compose.build_toa, atm_mod.Atmosphere = build_toa, atmosphere
    try:
        yield stats
    finally:
        compose.build_toa, atm_mod.Atmosphere = orig_toa, orig_atm


def seed_cache(src, dst, stages=("starfield", "galaxy")):
    """Symlink the H-independent stage files of another cache dir into dst (existing files kept)."""
    if not src or not os.path.isdir(src):
        return []
    os.makedirs(dst, exist_ok=True)
    done = []
    for st in stages:
        for p in glob.glob(os.path.join(src, f"{st}_*.npz")):
            q = os.path.join(dst, os.path.basename(p))
            if not os.path.lexists(q):
                os.symlink(os.path.abspath(p), q)
                done.append(q)
    return done


def freeze_pointing(cfg, info, t_ref_h):
    """Resolve 'auto' azimuth / pitch of perspective views at solar time t_ref_h (in place)."""
    H = float(solar_to_hour_angle(t_ref_h, sun_ra_deg(cfg)))
    ctx = {"R_c2h": frames.celestial_to_horizon(H, cfg["scenario"]["lat_deg"]),
           "R_gc_to_cel": info["R_gc_to_cel"], "observer_pc": info["observer_pc"]}
    for v in cfg["views"]:
        if v.get("kind", "perspective") == "perspective" and "auto" in (v.get("azimuth_deg"), v.get("pitch_deg")):
            az, pitch, tgt = camera.resolve_pointing(v, ctx)
            v["azimuth_deg"], v["pitch_deg"] = round(az, 3), round(pitch, 3)
            log(f"[{v['name']}] pointing frozen at {clock(t_ref_h)} (H {H:.1f}): az {az:.1f}, pitch {pitch:.1f}, target alt/az {tgt}")


def render_hash(cfg):
    c = {k: v for k, v in cfg.items() if k not in ("name", "output_dir", "animation")}
    c["camera"] = {k: v for k, v in cfg["camera"].items() if k not in DEVELOP_ONLY_KEYS}
    c["scenario"] = {k: v for k, v in cfg["scenario"].items() if k != "hour_angle_deg"}
    return config.stable_hash(c)


# ----------------------------------------------------------------------------- driver
def render_frames(cfg, times, vnames, scale=1.0, cpus=None, resume=True, keep_frame_cache=False, st=None):
    """compose.render for every solar time (inside shared_state); returns per-frame records."""
    root = cfg["output_dir"]
    rh, ra = render_hash(cfg), sun_ra_deg(cfg)
    recs, st = [], st or {}
    for i, t in enumerate(times):
        H = float(solar_to_hour_angle(t, ra))
        tag = frame_tag(t)
        fdir = os.path.join(root, "frames", tag)
        mpath = os.path.join(fdir, "frame.json")
        rec = {"index": i, "t_h": float(t), "clock": clock(t), "hour_angle_deg": H, "sun_alt_deg": sun_altitude(cfg, H),
               "dir": fdir, "scale": scale}
        if resume and os.path.exists(mpath):
            with open(mpath) as f:
                old = json.load(f)
            if old.get("render_hash") == rh and old.get("scale") == scale and all(
                    os.path.exists(os.path.join(fdir, v + ".npz")) for v in vnames):
                log(f"frame {i} {tag}: reusing {fdir}")
                recs.append({**old, **rec, "reused": True})
                continue
        fcfg = copy.deepcopy(cfg)
        fcfg["scenario"]["hour_angle_deg"] = H
        fcfg["output_dir"] = fdir
        fcfg["camera"]["annotate"] = False
        fcfg["output"]["npz"] = True
        t0 = time.time()
        res = compose.render(fcfg, views=vnames, cpus=cpus, scale=scale)
        rec["wall_s"] = time.time() - t0
        rec["views"] = {v: {"timing_s": r["info"]["timing_s"], "La_cd_m2": r["info"]["camera"].get("La_cd_m2"),
                            "photopic_fraction": r["info"]["camera"].get("photopic_fraction")}
                        for v, r in res["views"].items()}
        rec["scene"] = {k: res["scene"][k] for k in ("zenith_luminance_cd_m2", "ground_illuminance_lux")}
        rec["render_hash"], rec["shared"] = rh, dict(st)
        if not keep_frame_cache:
            shutil.rmtree(os.path.join(fdir, "cache"), ignore_errors=True)
        with open(mpath, "w") as f:
            json.dump(camera._jsonable(rec), f, indent=1)
        log(f"frame {i}/{len(times)} {clock(t)} H {H:.1f} sun {rec['sun_alt_deg']:+.1f}: {rec['wall_s']:.1f} s; La "
            + ", ".join(f"{v} {d['La_cd_m2'] or 0:.3g}" for v, d in rec["views"].items()))
        recs.append(rec)
    return recs


def find_ffmpeg():
    """(ffmpeg executable, codec args): the first of imageio_ffmpeg, $IMAGEIO_FFMPEG_EXE, PATH and
    /apps/ffmpeg/*/bin (HiPerGator) with libx264, else one with mpeg4; (None, None) if none."""
    cands = [os.environ.get("IMAGEIO_FFMPEG_EXE"), shutil.which("ffmpeg")] + sorted(glob.glob("/apps/ffmpeg/*/bin/ffmpeg"))[::-1]
    try:
        import imageio_ffmpeg
        cands.insert(0, imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    fallback = (None, None)
    for p in dict.fromkeys(c for c in cands if c and os.access(c, os.X_OK)):
        try:
            enc = subprocess.run([p, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        if " libx264 " in enc:
            return p, ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17"]
        if " mpeg4 " in enc and fallback[0] is None:
            fallback = (p, ["-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-q:v", "2"])
    return fallback


def stitch(pattern, out_base, fps, n):
    """MP4 from the numbered PNGs 0..n-1 via ffmpeg (H.264, else MPEG-4 part 2; padded to even
    width and height for yuv420p), else an animated GIF via Pillow."""
    exe, codec = find_ffmpeg()
    if exe:
        cmd = [exe, "-y", "-loglevel", "error", "-framerate", str(fps), "-start_number", "0", "-i", pattern,
               "-frames:v", str(n), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"] + codec + [out_base + ".mp4"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return out_base + ".mp4"
        log(f"ffmpeg {exe} failed ({r.returncode}): {r.stderr.strip()[:300]}; writing a GIF")
    from PIL import Image
    ims = [Image.open(pattern % k) for k in range(n)]
    ims[0].save(out_base + ".gif", save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
    return out_base + ".gif"


def overlay(rgb8, kind, lines):
    """Text lines at the lower left, the font scaled with the image (height / 40, shrunk to fit
    the width); compass letters on all-sky frames via camera.annotate."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(camera.annotate(rgb8, types.SimpleNamespace(kind=kind), []))
    dr, (w, h) = ImageDraw.Draw(im), im.size
    size = int(np.clip(h // 40, 8, 32))
    font = ImageFont.load_default(size=size)
    while size > 6 and max(dr.textlength(t, font=font) for t in lines) > w - 12:
        size -= 1
        font = ImageFont.load_default(size=size)
    for i, t in enumerate(lines):
        dr.text((6, h - 4 - (size + 2) * (len(lines) - i)), t, font=font, fill=(225, 225, 225), stroke_width=1, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def develop_frames(cfg, recs, vnames, modes, acfg, video=True):
    """Re-develop every frame for each exposure mode; numbered PNGs + one video per (view, mode)."""
    from PIL import Image
    root, ccfg = cfg["output_dir"], cfg["camera"]
    t = np.array([r["t_h"] for r in recs])
    summary = {}
    for v in vnames:
        la = np.array([(r.get("views") or {}).get(v, {}).get("La_cd_m2") or frame_la(*load_frame_view(r["dir"], v)) for r in recs])
        kind = next((c.get("kind", "perspective") for c in cfg["views"] if c["name"] == v), "perspective")
        series = {m: exposure_series(t, la, m, acfg, ccfg) for m in modes}
        odir = {m: os.path.join(root, f"{v}_{m}") for m in modes}
        for m, d in odir.items():
            os.makedirs(d, exist_ok=True)
            for p in glob.glob(os.path.join(d, f"{v}_{m}_*.png")):  # a longer earlier run's frames
                os.remove(p)
        for k, r in enumerate(recs):
            xyzs, valid, _ = load_frame_view(r["dir"], v)
            for m in modes:
                lk = series[m][k]
                rgb, ad = develop_xyzs(xyzs, valid, ccfg, None if np.isnan(lk) else lk, kind)
                if acfg.get("overlay", True):
                    lab = (f"La {ad['La_cd_m2']:.2g} cd/m2 ({m})" if "La_cd_m2" in ad else
                           f"ISO {ccfg['iso']:.0f} f/{ccfg['f_number']:.1f} {ccfg['exposure_s']:g}s ({m})")
                    rgb = overlay(rgb, kind, [f"{r['clock']} solar time, Sun {r['sun_alt_deg']:+.1f} deg", lab])
                Image.fromarray(rgb).save(os.path.join(odir[m], f"{v}_{m}_{k:04d}.png"))
        for m in modes:
            for p in (odir[m] + ".mp4", odir[m] + ".gif"):  # an earlier run's video no longer matches the PNGs
                if os.path.exists(p):
                    os.remove(p)
            vid = stitch(os.path.join(odir[m], f"{v}_{m}_%04d.png"), odir[m], acfg["fps"], len(recs)) if video and len(recs) > 1 else None
            summary[f"{v}/{m}"] = {"La_frame_cd_m2": la.tolist(), "La_used_cd_m2": [None if np.isnan(x) else float(x) for x in series[m]],
                                   "frames_dir": odir[m], "video": vid}
            log(f"[{v}] {m}: {len(recs)} frames -> {vid or odir[m]}")
    return summary


def run(cfg_path, hours=None, n_frames=None, sets=(), scale=1.0, cpus=None, fps=None, vnames=None, modes=None,
        fixed_la=None, tau_light=None, tau_dark=None, overlay=None, video=True, render=True, resume=True):
    cfg = config.load_config(cfg_path, sets)
    acfg = {**ANIM_DEFAULTS, **(cfg.get("animation") or {})}
    for k, v in (("fps", fps), ("fixed_la_cd_m2", fixed_la), ("tau_light_min", tau_light), ("tau_dark_min", tau_dark), ("overlay", overlay)):
        if v is not None:
            acfg[k] = v
    ex = acfg["exposure"]
    modes = modes or [str(m).strip() for m in (ex if isinstance(ex, (list, tuple)) else str(ex).split(","))]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise ValueError(f"unknown exposure mode(s) {bad}; choose from {MODES}")
    times = parse_hours(hours if hours is not None or n_frames else acfg["hours"], n_frames)
    known = [v["name"] for v in cfg["views"]]
    vnames = vnames or known
    if set(vnames) - set(known):
        raise ValueError(f"unknown view(s) {sorted(set(vnames) - set(known))}; the config has {known}")
    if acfg["pointing"] not in ("fixed", "track"):
        raise ValueError(f"animation.pointing must be fixed or track, got {acfg['pointing']!r}")
    root = cfg["output_dir"]
    os.makedirs(root, exist_ok=True)
    seeded = seed_cache(os.path.join(config.ROOT, acfg["seed_cache"]) if acfg["seed_cache"] else None, os.path.join(root, "cache"))
    if seeded:
        log(f"seeded {len(seeded)} cache file(s) from {acfg['seed_cache']}")
    log(f"{cfg['name']}: {len(times)} frames, solar time {clock(times[0])} .. {clock(times[-1])}, views {vnames}, "
        f"exposure {modes} -> {root}")
    t0 = time.time()
    with shared_state(compose.Cache(os.path.join(root, "cache"))) as st:
        if render and acfg["pointing"] == "fixed":
            _, info, _ = compose.build_toa(cfg, compose.make_grid(cfg), None)
            freeze_pointing(cfg, info, acfg["pointing_time_h"] if acfg["pointing_time_h"] is not None else float(times[0]))
        if render:
            recs = render_frames(cfg, times, vnames, scale, cpus, resume, acfg["keep_frame_cache"], st)
        else:
            recs = []
            for i, t in enumerate(times):
                fdir = os.path.join(root, "frames", frame_tag(t))
                with open(os.path.join(fdir, "frame.json")) as f:
                    recs.append({**json.load(f), "index": i, "t_h": float(t), "clock": clock(t), "dir": fdir})
    t_render = time.time() - t0
    summary = develop_frames(cfg, recs, vnames, modes, acfg, video)
    out = {"config": cfg["name"], "times_h": times.tolist(), "frames": recs, "exposure": summary, "animation": acfg,
           "render_s": t_render, "develop_s": time.time() - t0 - t_render, "views": [v for v in cfg["views"] if v["name"] in vnames]}
    with open(os.path.join(root, "animation.json"), "w") as f:
        json.dump(camera._jsonable(out), f, indent=1, default=str)
    log(f"done: render {t_render:.1f} s, develop {out['develop_s']:.1f} s -> {os.path.join(root, 'animation.json')}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m nightsky.animate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--hours", default=None, help="local solar time (h): 'a:b:s', 'a:b' with --frames, or 'a,b,c'")
    ap.add_argument("--frames", type=int, default=None, help="number of frames (with 'a:b' or alone for 0:24)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--cpus", type=int, default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--views", default=None, help="comma-separated view names")
    ap.add_argument("--exposure", default=None, help=f"comma-separated subset of {MODES}")
    ap.add_argument("--fixed-la", type=float, default=None, help="La (cd/m^2) for --exposure fixed")
    ap.add_argument("--tau-light-min", type=float, default=None)
    ap.add_argument("--tau-dark-min", type=float, default=None)
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-render", action="store_true", help="re-develop existing frames only")
    ap.add_argument("--no-resume", action="store_true", help="re-render frames that already exist")
    a = ap.parse_args(argv)
    if a.cpus:
        os.environ["NUMBA_NUM_THREADS"] = str(a.cpus)
    split = lambda s: [x.strip() for x in s.split(",") if x.strip()] if s else None
    run(a.config, a.hours, a.frames, a.set, a.scale, a.cpus, a.fps, split(a.views), split(a.exposure),
        a.fixed_la, a.tau_light_min, a.tau_dark_min,
        False if a.no_overlay else None, not a.no_video, not a.no_render, not a.no_resume)


if __name__ == "__main__":
    main()
