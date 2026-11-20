"""Real-colour views from inside / outside a grid protoplanetary disk.

  python3 -m diskorbit.render mc     <outdir> [--nphot 2e7]          scattered field (cached)
  python3 -m diskorbit.render stills <outdir> [--width 1920 ...]     milestone stills on the orbit
  python3 -m diskorbit.render sed    <outdir>                         distant-camera SED vs Hyperion
  python3 -m diskorbit.render orbit  <outdir> --frames 0:1800         linear movie frames
  python3 -m diskorbit.render develop <outdir>                        tone-map frames -> PNG/MP4
  python3 -m diskorbit.render orbit-ir <outdir>                       IR / mm frames (nu I_nu per band)
  python3 -m diskorbit.render develop-ir <outdir> [--sets nir,mir,fir] false-colour IR (+ 1.3 mm overlay)
  python3 -m diskorbit.render stars <outdir> --modes eye,auto          screen-blend a fixed-stretch star layer

Disk frame: z = disk axis, AU. Orbit: circular, radius a, inclination inc (deg), ascending
node on +x; phase phi = 0 at the ascending node (in the midplane), 90 deg at the top.
Camera: looks along the orbital velocity turned by `toward_star_deg` toward the star, horizon
(image x axis) parallel to the disk midplane.
"""
import argparse
import json
import os
import time

import numpy as np

from nightsky import spectral, camera
import views as vw

from .model import DiskModel, AU_M
from . import bands as bd
from .raymarch import render_rays, column_segment
from .kernels import node_interp3
from . import mcscat

DEFAULTS = dict(model="dIULECee", bands="vis", nbins=20, a_au=30.0, inc_deg=30.0, toward_star_deg=35.0,
                fov_deg=100.0, width=1920, height=1080, frac=0.2, dtau_max=1.0, tau_stop=30.0,
                # non-axisymmetric (planet) disk: PPDONet sigma npz, planet radius, 3D Hyperion run
                planet_sigma="", planet_rp=100.0, planet_rtin="", planet_rtout="", mc_nphi=32,
                planet_lsun=0.0, planet_temp=1200.0, planet_clump_h=0.0,
                # orbit phase speed: "surface" (slow at the lit surface layer, fast through the
                # midplane) or "uniform"
                orbit_speed="surface",
                camera_up="z",
                # intro before the orbit (frames): hold at intro_from_au on the ray from the star
                # through the orbit's start, fly in to a_au, then pan from the star to the orbit's
                # view; intro_from_au = 0 -> no intro
                intro_from_au=0.0, intro_hold=0, intro_fly=0, intro_pan=0,
                # intro_star: draw the star point during the intro (its glare hides the face-on
                # disk); intro_key: exposure during the intro, x the frame's 99.5th-percentile
                # luminance (the eye adaptation of a mostly dark sky would blow the disk out)
                intro_star=0, intro_key=0.03,
                # pattern azimuth: "fixed" -> planet at pattern_deg (inertial); "coorbit" -> the
                # planet leads the camera's orbital phase by pattern_deg (same period)
                pattern_mode="fixed", pattern_deg=0.0,
                movie_name="", key_L_au=75.0, key_dex=7.0)  # movie file prefix (default diskorbit_<model>); key half-width, stretch


def movie_prefix(p):
    return p.get("movie_name") or f"diskorbit_{p['model']}"
EYE = {"mode": "eye", "glare": True, "age_years": 25.0, "mesopic": True, "local_mesopic": True}


class Scene:
    def __init__(self, outdir, model="dIULECee", nbins=20, nphot=2e7, log=print, bands="vis", disk=None):
        """disk: a ready model object (e.g. a non-axisymmetric diskplanet model) instead of
        DiskModel(model); its .tag goes into the cached Monte Carlo file name."""
        if bands == "vis":
            self.grid, tag = spectral.default_grid(nbins), f"{nbins}b"
        elif bands == "ir":
            self.grid, tag = bd.ir_grid(), f"ir{len(bd.IR_BANDS_UM)}"
        else:
            raise ValueError(f"bands must be vis or ir, got {bands!r}")
        self.visible = bands == "vis"
        self.model = disk if disk is not None else DiskModel(model)
        model = self.model.name + self.model.tag + ("" if self.visible else self.model.t_tag)
        # visual extinction per unit column, for labels (tau_V)
        self.k_v = float(self.model.optics(spectral.default_grid(1, 545.0, 555.0))["k"][0])
        self.opt = self.model.optics(self.grid)
        self.tab = self.model.tables()
        self.lb = self.model.star_bins(self.grid)
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"mc_{model}_{tag}.npz")
        if not os.path.exists(path):
            log(f"scattered field -> {path}")
            res = mcscat.scattered_field(self.model, self.grid, n_phot=int(nphot), log=log, thermal=not self.visible)
            np.savez_compressed(path, wl=self.grid.wl, **res)
        self.mc = dict(np.load(path))
        if self.mc["J"].ndim == 3:  # cache from the axisymmetric code: add the phi axis, H_phi = 0
            for q in ("J", "HR", "HZ"):
                self.mc[q] = self.mc[q][:, None]
            self.mc["HP"] = np.zeros_like(self.mc["J"])
        # F*/(4 pi) at 1 AU (W m^-2 nm^-1 sr^-1 per unit phase function)
        self.lfac = self.lb / (4 * np.pi * AU_M ** 2) / (4 * np.pi)

    # The model may be non-axisymmetric (a planet's pattern) and rotate: pattern_angle (rad) is the
    # pattern's azimuth in the camera (inertial) frame. Positions and directions passed to the
    # methods below are inertial and are rotated into the model frame here.
    pattern_angle = 0.0

    def to_model(self, v):
        c, s = np.cos(self.pattern_angle), np.sin(self.pattern_angle)
        v = np.asarray(v, float)
        out = v.copy()
        out[..., 0] = c * v[..., 0] + s * v[..., 1]
        out[..., 1] = -s * v[..., 0] + c * v[..., 1]
        return out

    def planet_irradiance(self, cam, frac=0.05):
        """(unit direction to the planet (inertial), attenuated irradiance W m^-2 nm^-1 per bin,
        column) or None when the model has no planet source."""
        pl = getattr(self.model, "planet", None)
        if pl is None:
            return None
        cm = self.to_model(cam)
        pp = np.array([pl["x_au"], 0.0, 0.0])
        d = pp - cm
        dist = float(np.linalg.norm(d))
        PT, LD = self.model.pert
        n = column_segment(cm, pp, self.model.params, PT, LD, frac)
        lb = mcscat.blackbody_bins(self.grid, pl["lum_w"], pl["temp_k"])
        irr = lb / (4 * np.pi * (dist * AU_M) ** 2) * np.exp(-self.opt["k"] * n)
        c, s = np.cos(self.pattern_angle), np.sin(self.pattern_angle)
        u = d / dist
        return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1], u[2]]), irr, n

    def star_irradiance(self, cam):
        """Attenuated stellar irradiance at the camera, W m^-2 nm^-1 per bin."""
        t = self.tab
        cam = self.to_model(cam)
        r = float(np.linalg.norm(cam))
        n = node_interp3(t["ncol"], t["lnr"][0], t["lnr"][1] - t["lnr"][0], t["mu"][0], t["mu"][1] - t["mu"][0],
                         t["phi0"], t["dphi"], np.log(r), abs(cam[2]) / r, float(np.arctan2(cam[1], cam[0])))
        return self.lb / (4 * np.pi * (r * AU_M) ** 2) * np.exp(-self.opt["k"] * n), n

    def render(self, cam, dirs, frac=0.2, dtau_max=1.0, tau_stop=30.0):
        t, mc, o = self.tab, self.mc, self.opt
        cam = self.to_model(cam)
        flat = np.ascontiguousarray(self.to_model(dirs.reshape(-1, 3)), dtype=np.float64)
        PT, LD = self.model.pert
        img, col = render_rays(np.asarray(cam, float), flat, self.model.params, PT, LD, o["k"], o["albedo"], o["g"],
                               o["phase"], self.lfac, self.grid.wl,
                               t["lnr"][0], t["lnr"][1] - t["lnr"][0], t["mu"][0], t["mu"][1] - t["mu"][0],
                               float(t["phi0"]), float(t["dphi"]), t["ncol"], t["temp"],
                               mc["elnr"][0], mc["elnr"][1] - mc["elnr"][0], len(mc["emu"]) - 1,
                               mc["J"], mc["HR"], mc["HZ"], mc["HP"], frac, dtau_max, tau_stop)
        return img.reshape(dirs.shape[:-1] + (-1,)), col.reshape(dirs.shape[:-1])


_DISKS = {}


def make_scene(outdir, p, nphot=2e7, log=print, bands="vis"):
    """Scene for parameter dict p: the grid model, or the planet-perturbed model when
    p["planet_sigma"] is set (the model object is shared between scenes)."""
    disk = None
    if p.get("planet_sigma"):
        from diskplanet.model3d import PlanetDisk
        key = (p["model"], p["planet_sigma"], p["planet_rp"], p["planet_rtin"], p["planet_rtout"], p["mc_nphi"],
               p["planet_lsun"], p["planet_temp"], p["planet_clump_h"])
        if key not in _DISKS:
            _DISKS[key] = PlanetDisk(p["model"], p["planet_sigma"], p["planet_rp"], p["planet_rtin"] or None,
                                     p["planet_rtout"] or None, mc_nphi=p["mc_nphi"], planet_lsun=p["planet_lsun"],
                                     planet_temp=p["planet_temp"], clump_h=p["planet_clump_h"])
        disk = _DISKS[key]
    return Scene(outdir, p["model"], p["nbins"], nphot, log=log, bands=bands, disk=disk)


def pattern_angle(p, phi_deg):
    """Azimuth (rad, inertial) of the model pattern (planet) when the camera is at orbital phase
    phi_deg."""
    if p.get("pattern_mode", "fixed") == "coorbit":
        return float(np.radians(phi_deg + p["pattern_deg"]))
    return float(np.radians(p.get("pattern_deg", 0.0)))


def deposit_planet(sc, img, v, pos, ir=False):
    """Add the planet point source (if the model has one); returns (in frame, column)."""
    r = sc.planet_irradiance(pos)
    if r is None:
        return 0, float("nan")
    u, irr, n = r
    if ir:
        return deposit_ir(img, v, u[None], irr[None], sc.grid), float(n)
    return camera.deposit_points(img, v, u[None], irr[None], POINTS, sc.grid), float(n)


def look_basis(fwd, up_ref=(0.0, 0.0, 1.0)):
    f = np.asarray(fwd, float)
    f = f / np.linalg.norm(f)
    up_ref = np.asarray(up_ref, float)
    if abs(f @ up_ref) > 0.999:
        up_ref = np.array([1.0, 0.0, 0.0])
    rt = np.cross(f, up_ref)
    rt /= np.linalg.norm(rt)
    return f, rt, np.cross(rt, f)


def view_up(p, pos):
    """Reference "up" for the camera basis. "z": the disk axis (legacy; the roll flips by 180 deg
    when the view passes the disk axis, which steep orbits do); "radial": away from the star,
    which stays well defined on any orbit because the camera never looks along its radius."""
    if p.get("camera_up", "z") == "radial":
        return np.asarray(pos, float) / np.linalg.norm(pos)
    return np.array([0.0, 0.0, 1.0])


def make_view(fwd, width, height, fov_deg, up_ref=(0.0, 0.0, 1.0)):
    """nightsky View (perspective) whose directions are in the disk frame."""
    v = vw.View("perspective", width, height, azimuth_deg=0.0, pitch_deg=0.0, fov_deg=fov_deg)
    f, rt, up = look_basis(fwd, up_ref)
    d = v.dirs
    v.dirs = d[..., :1] * f + (-d[..., 1:2]) * rt + d[..., 2:3] * up
    return v


def orbit_state(phi_deg, a=30.0, inc=30.0):
    p, i = np.radians(phi_deg), np.radians(inc)
    pos = a * np.array([np.cos(p), np.sin(p) * np.cos(i), np.sin(p) * np.sin(i)])
    vel = np.array([-np.sin(p), np.cos(p) * np.cos(i), np.cos(p) * np.sin(i)])
    return pos, vel


def camera_forward(pos, vel, toward_star_deg):
    a = np.radians(toward_star_deg)
    f = np.cos(a) * vel - np.sin(a) * pos / np.linalg.norm(pos)
    return f / np.linalg.norm(f)


def _rotate(axis, ang, v):
    """Rodrigues rotation of v about the unit vector axis by ang (rad)."""
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * (axis @ v) * (1 - c)


def _smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def intro_length(p):
    if p.get("intro_from_au", 0.0) <= 0:
        return 0
    return int(p["intro_hold"]) + int(p["intro_fly"]) + int(p["intro_pan"])


class CameraTrack:
    """Camera pose at (fractional) movie frame t: the optional intro, then one orbit over the
    remaining frames (orbit_schedule). state(t) -> (pos, fwd, up, orbit phase deg)."""

    def __init__(self, p, n_frames):
        self.p = p
        self.n_intro = intro_length(p)
        self.n_orbit = n_frames - self.n_intro
        ph = np.degrees(np.unwrap(np.radians(orbit_schedule(self.n_orbit, p["inc_deg"], speed=p["orbit_speed"]))))
        self.ph = np.append(ph, 2 * ph[-1] - ph[-2])  # one extra step for sub-frame interpolation
        pos0, vel0 = orbit_state(self.ph[0], p["a_au"], p["inc_deg"])
        self.f0 = camera_forward(pos0, vel0, p["toward_star_deg"])
        self.up0 = look_basis(self.f0, view_up(p, pos0))[2]
        self.rhat0 = pos0 / np.linalg.norm(pos0)
        # pan: rotate the view from the star (-rhat0) to f0 about their common normal
        self.fs = -self.rhat0
        self.ang = float(np.arccos(np.clip(self.fs @ self.f0, -1, 1)))
        ax = np.cross(self.fs, self.f0)
        self.axis = ax / np.linalg.norm(ax)
        self.ups = _rotate(self.axis, -self.ang, self.up0)

    def phase(self, t):
        j = t - self.n_intro
        if j <= 0:
            return float(self.ph[0])
        return float(np.interp(j, np.arange(len(self.ph)), self.ph))

    def state(self, t):
        p = self.p
        if t >= self.n_intro:
            ph = self.phase(t)
            pos, vel = orbit_state(ph, p["a_au"], p["inc_deg"])
            fwd = camera_forward(pos, vel, p["toward_star_deg"])
            return pos, fwd, look_basis(fwd, view_up(p, pos))[2], ph
        hold, fly = int(p["intro_hold"]), int(p["intro_fly"])
        u_fly = (t - hold) / max(fly, 1)
        u_pan = (t - hold - fly) / max(int(p["intro_pan"]), 1)
        d = float(np.exp(np.log(p["intro_from_au"]) + (np.log(p["a_au"]) - np.log(p["intro_from_au"])) * _smooth(u_fly)))
        a = self.ang * _smooth(u_pan)
        return d * self.rhat0, _rotate(self.axis, a, self.fs), _rotate(self.axis, a, self.ups), float(self.ph[0])


def phi_for_height(zr, a=30.0, inc=30.0):
    """Orbit phase (deg, first quarter) where z / R = zr."""
    ph = np.linspace(0, 90, 90001)
    pos = np.array([orbit_state(p, a, inc)[0] for p in ph[::100]])
    zr_s = pos[:, 2] / np.hypot(pos[:, 0], pos[:, 1])
    return float(np.interp(zr, zr_s, ph[::100]))


def render_frame(scene, pos, fwd, p, star=True, up=None):
    v = make_view(fwd, p["width"], p["height"], p["fov_deg"], up_ref=view_up(p, pos) if up is None else up)
    t0 = time.time()
    img, col = scene.render(pos, v.dirs, p["frac"], p["dtau_max"], p["tau_stop"])
    secs = time.time() - t0
    irr, nstar = scene.star_irradiance(pos)
    nd = 0
    if star:
        nd = camera.deposit_points(img, v, (-pos / np.linalg.norm(pos))[None], irr[None],
                                   {"psf_sigma_px": 0.6, "radius_px": 4}, scene.grid)
        deposit_planet(scene, img, v, pos, ir=not scene.visible)
    lux = float(spectral.luminance(irr, scene.grid)) if scene.visible else float("nan")
    return v, img, col, {"render_s": secs, "star_lux": lux,
                         "star_in_frame": nd, "star_column": float(nstar)}


def develop_frame(img, view, grid, ccfg, cpus=8):
    rgb8, info, xyzs = camera.develop(img, view, grid, ccfg, cpus=cpus)
    return rgb8, info, xyzs


def save_png(path, rgb8):
    import imageio.v3 as iio
    iio.imwrite(path, rgb8)


# ----------------------------------------------------------------------------- commands
def cmd_stills(a, p, log=print):
    sc = make_scene(a.outdir, p, a.nphot, log=log, bands=p["bands"])
    cases = [("above", 90.0), ("surface", phi_for_height(0.167, p["a_au"], p["inc_deg"])),
             ("upper_fog", phi_for_height(0.15, p["a_au"], p["inc_deg"])),
             ("deep_fog", phi_for_height(0.13, p["a_au"], p["inc_deg"])), ("midplane", 0.0)]
    meta = {}
    for name, phi in cases:
        pos, vel = orbit_state(phi, p["a_au"], p["inc_deg"])
        fwd = camera_forward(pos, vel, p["toward_star_deg"])
        v, img, col, info = render_frame(sc, pos, fwd, p)
        zr = pos[2] / np.hypot(pos[0], pos[1])
        rgb8, dinfo, xyzs = develop_frame(img, v, sc.grid, EYE, cpus=a.cpus)
        save_png(f"{a.outdir}/still_{name}_eye.png", rgb8)
        ccam = {"mode": "eye", "glare": True, "mesopic": False, "adaptation_cd_m2": None}
        # diagnostic exposure: photopic adaptation to the frame median without the star
        y = spectral.to_xyz(img, sc.grid)[..., 1]
        ccam["adaptation_cd_m2"] = float(max(np.median(y), 1e-9))
        rgb8b, _, _ = develop_frame(img, v, sc.grid, ccam, cpus=a.cpus)
        save_png(f"{a.outdir}/still_{name}_auto.png", rgb8b)
        np.savez_compressed(f"{a.outdir}/still_{name}_linear.npz", xyzs=xyzs.astype(np.float32), column=col.astype(np.float32))
        meta[name] = {"phi_deg": phi, "pos_au": pos.tolist(), "z_over_R": float(zr), **info,
                      "La_cd_m2": dinfo.get("La_cd_m2"), "median_Y_cd_m2": float(np.median(y)),
                      "max_Y_cd_m2": float(y.max()), "median_tau_V_ray": float(np.median(col) * sc.k_v)}
        log(f"{name}: phi {phi:.2f} z/R {zr:.3f} render {info['render_s']:.1f} s  La {dinfo.get('La_cd_m2'):.3g} cd/m2  median Y {np.median(y):.3g}")
    with open(f"{a.outdir}/stills.json", "w") as f:
        json.dump({"params": p, "cases": meta}, f, indent=1)


def hyperion_sed(model):
    """(wav_um, nuLnu components (4 origins, n_view, n_ap, n_wav) in W, view angles deg)."""
    import h5py
    from .model import GRID_ROOT, C_UMS
    xx = model.name[:2].lower()
    with h5py.File(f"{GRID_ROOT}/{model.geom}/output/{xx}/{model.name}.rtout", "r") as f:
        ds = f["Peeled/group_00001/seds"]
        flux = ds[()][0] * 1e-7  # Stokes I; ergs/s -> W
        numin, numax = ds.attrs["numin"], ds.attrs["numax"]
    with h5py.File(f"{GRID_ROOT}/{model.geom}/input/{xx}/{model.name}.rtin", "r") as f:
        ang = np.array(f["Output/Peeled/group_00001/angles"]["theta"]) if "theta" in f["Output/Peeled/group_00001/angles"].dtype.names else np.array([a[0] for a in f["Output/Peeled/group_00001/angles"]])
    wav = np.logspace(np.log10(C_UMS / numin), np.log10(C_UMS / numax), flux.shape[-1] * 2 + 1)[1::2]
    return wav, flux, ang


def cmd_sed(a, p, log=print):
    """Distant-camera total flux per bin vs the model's own Hyperion SED (largest aperture)."""
    sc = make_scene(a.outdir, p, a.nphot, log=log, bands=p["bands"])
    wav, flux, ang = hyperion_sed(sc.model)
    D = 3000.0
    out = {}
    lam_um = sc.grid.wl / 1000.0
    for iv in a.views:
        inc = float(ang[iv])
        cam = D * np.array([np.sin(np.radians(inc)), 0.0, np.cos(np.radians(inc))])
        fov = 2 * np.degrees(np.arctan(1.05 * sc.model.rmax / D))
        # nested images (each level zooms 4x on the centre) resolve the bright inner rim:
        # level k contributes its pixels outside the central quarter covered by level k+1
        t0 = time.time()
        n = a.sed_px
        q0, q1 = 3 * n // 8, 5 * n // 8
        f_disk = 0.0
        for lev in range(a.sed_levels):
            v = make_view(-cam, n, n, 2 * np.degrees(np.arctan(np.tan(np.radians(fov / 2)) / 4 ** lev)), up_ref=(0.0, 0.0, 1.0))
            img, col = sc.render(cam, v.dirs, p["frac"], p["dtau_max"], p["tau_stop"])
            w = v.omega.copy()
            if lev < a.sed_levels - 1:
                w[q0:q1, q0:q1] = 0.0
            f_disk = f_disk + np.tensordot(w, img, axes=([0, 1], [0, 1]))  # W m^-2 nm^-1
        irr, _ = sc.star_irradiance(cam)
        # Hyperion: nuL_nu (W) -> F_lambda at distance D:  F_lam = nuL_nu / (4 pi d^2) / lambda
        d_m = D * AU_M
        tot = flux[:, iv, -1, :]  # (4 origins, n_wav)
        hyp = {k: np.interp(np.log(lam_um), np.log(wav[::-1]), tot[i][::-1]) / (4 * np.pi * d_m ** 2) / sc.grid.wl
               for i, k in enumerate(["source_emit", "dust_emit", "source_scat", "dust_scat"])}
        out[f"{inc:.1f}"] = {"lam_nm": sc.grid.wl.tolist(), "ours_star": irr.tolist(), "ours_disk": f_disk.tolist(),
                             **{f"hyp_{k}": v_.tolist() for k, v_ in hyp.items()}, "render_s": time.time() - t0}
        with np.errstate(divide="ignore", invalid="ignore"):
            r_star = irr / hyp["source_emit"]
        r_scat = f_disk / (hyp["source_scat"] + hyp["dust_scat"] + hyp["dust_emit"])
        band = lambda x: float(np.sum(np.asarray(x) * sc.grid.dwl))
        out[f"{inc:.1f}"]["band_ratio_star"] = band(irr) / band(hyp["source_emit"]) if band(hyp["source_emit"]) > 0 else float("nan")
        out[f"{inc:.1f}"]["band_ratio_scat"] = band(f_disk) / band(hyp["source_scat"] + hyp["dust_scat"] + hyp["dust_emit"])
        log(f"inc {inc:5.1f}: 380-780 nm band ratio ours/hyp star {out[f'{inc:.1f}']['band_ratio_star']:.3f} scattered {out[f'{inc:.1f}']['band_ratio_scat']:.3f}")
        log(f"inc {inc:5.1f}: star ours/hyp {np.median(r_star):.3f} [{r_star.min():.3f}-{r_star.max():.3f}]  "
            f"scattered+thermal ours/hyp {np.median(r_scat):.3f} [{r_scat.min():.3f}-{r_scat.max():.3f}]  ({time.time() - t0:.0f} s)")
        if not sc.visible:
            for b in range(sc.grid.n):
                log(f"   {sc.grid.wl[b] / 1000:8.2f} um  star {r_star[b]:.3f}  disk {r_scat[b]:.3f}  "
                    f"(hyp disk: scat {hyp['source_scat'][b] + hyp['dust_scat'][b]:.3g} emit {hyp['dust_emit'][b]:.3g}, ours {f_disk[b]:.3g} W m-2 nm-1)")
    with open(f"{a.outdir}/sed_check{'' if sc.visible else '_' + p['bands']}.json", "w") as f:
        json.dump(out, f, indent=1)


# ----------------------------------------------------------------------------- movie
POINTS = {"psf_sigma_px": 0.6, "radius_px": 4}


def z_over_r(phi_deg, inc=30.0):
    p, i = np.radians(phi_deg), np.radians(inc)
    return np.sin(p) * np.sin(i) / np.sqrt(np.cos(p) ** 2 + (np.sin(p) * np.cos(i)) ** 2)


def orbit_schedule(n_frames, inc=30.0, slow_zr=0.15, slow_width=0.03, slow_factor=4.0,
                   black_zr=0.09, fast_factor=4.0, high_zr=0.3, high_factor=2.0, phi0=90.0, speed="surface"):
    """Orbit phase (deg) per frame for one full orbit starting at phi0 (top of the orbit).
    speed "surface": the phase speed is ~ w(|z/R|): slowed by slow_factor around the illuminated
    surface layer (|z/R| ~ slow_zr), sped up by fast_factor through the dark midplane
    (|z/R| < black_zr) and by high_factor high above the disk (|z/R| > high_zr), where the view
    changes slowly. speed "uniform": constant phase speed (a Keplerian circular orbit)."""
    if speed == "uniform":
        return (phi0 + 360.0 * np.arange(n_frames) / n_frames) % 360.0
    ph = np.linspace(phi0, phi0 + 360.0, 72001)
    zr = np.abs(z_over_r(ph, inc))
    w = 1.0 - (1.0 - 1.0 / slow_factor) * np.exp(-0.5 * ((zr - slow_zr) / slow_width) ** 2)
    w *= 1.0 + (fast_factor - 1.0) / (1.0 + np.exp((zr - black_zr) / 0.01))
    w *= 1.0 + (high_factor - 1.0) / (1.0 + np.exp(-(zr - high_zr) / 0.03))
    s = np.concatenate([[0.0], np.cumsum(0.5 * (1 / w[1:] + 1 / w[:-1]) * np.diff(ph))])
    s /= s[-1]
    return np.interp(np.arange(n_frames) / n_frames, s, ph) % 360.0


def background_stars(grid, vmax=6.5):
    """Bright-star-catalogue directions (equatorial axes taken as the disk frame) and irradiance
    (n, nbins) W m^-2 nm^-1, as a stand-in background sky."""
    from nightsky import starfield
    dirs, irr, _, _, source = starfield.earth_stars({"bsc": {"vmax": vmax, "fetch": False}, "sun_ra_deg": 0.0}, grid)
    return dirs, irr, source


def to_bf16(x):
    """float32 -> bfloat16 bit patterns (uint16; round to nearest, 8-bit mantissa, rel. 0.4%)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def from_bf16(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def frame_path(outdir, i):
    return os.path.join(outdir, "frames", f"f{i:05d}")


def cmd_orbit(a, p, log=print):
    """Linear (pre-glare) XYZS + metadata per frame; existing frames are skipped (restartable)."""
    sc = make_scene(a.outdir, p, a.nphot, log=log, bands=p["bands"])
    os.makedirs(os.path.join(a.outdir, "frames"), exist_ok=True)
    track = CameraTrack(p, a.n_frames)
    bdirs, birr, src = background_stars(sc.grid)
    log(f"background stars: {len(bdirs)} from {src}")
    lo, hi = (int(x) if x else None for x in a.frames.split(":"))
    for i in range(a.n_frames)[slice(lo, hi)]:
        base = frame_path(a.outdir, i)
        if os.path.exists(base + ".json"):
            continue
        t0 = time.time()
        pos, fwd, up, ph = track.state(i)
        sc.pattern_angle = pattern_angle(p, ph)
        v, img, col, info = render_frame(sc, pos, fwd, p, star=False, up=up)
        bg = img.copy()
        # background stars, attenuated by the disk column toward each of them
        _, bcol = sc.render(pos, bdirs[:, None, :], p["frac"], p["dtau_max"], p["tau_stop"])
        b_att = birr * np.exp(-np.outer(bcol[:, 0], sc.opt["k"]))
        _, sinfo = camera.deposit_points_eye(img, v, bdirs, b_att, POINTS, sc.grid, bg)
        irr, _ = sc.star_irradiance(pos)
        info["star_in_frame"] = 0
        if i >= track.n_intro or p["intro_star"]:
            info["star_in_frame"] = camera.deposit_points(img, v, (-pos / np.linalg.norm(pos))[None], irr[None], POINTS, sc.grid)
        info["planet_in_frame"], info["planet_column"] = deposit_planet(sc, img, v, pos)
        xyzs = camera.to_xyzs(img, sc.grid).astype(np.float32)
        xyzs[~v.valid] = 0
        ad = camera.eye_adaptation(xyzs, v.valid, EYE)
        meta = {"frame": i, "phi_deg": ph, "pos_au": pos.tolist(), "fwd": fwd.tolist(), "up": up.tolist(), "z_over_R": float(pos[2] / np.hypot(pos[0], pos[1])),
                "La_cd_m2": ad["La_cd_m2"], "median_Y_cd_m2": float(np.median(xyzs[..., 1])),
                "p995_Y_cd_m2": float(np.percentile(xyzs[..., 1][v.valid], 99.5)),
                "bg_stars_in_frame": sinfo["n_points_in_frame"], "bg_stars_visible": sinfo["n_points_visible"],
                "star_tau_V": float(sc.star_irradiance(pos)[1] * sc.k_v), **info}
        with open(base + ".tmp.npz", "wb") as f:
            np.savez_compressed(f, xyzs_bf16=to_bf16(xyzs))
        os.replace(base + ".tmp.npz", base + ".npz")
        meta["frame_s"] = time.time() - t0
        with open(base + ".json", "w") as f:
            json.dump(meta, f)
        log(f"frame {i:5d} phi {ph:7.2f} z/R {meta['z_over_R']:+.3f} La {meta['La_cd_m2']:.3g} "
            f"star {info['star_in_frame']} bg {sinfo['n_points_visible']:.0f}/{sinfo['n_points_in_frame']} "
            f"render {info['render_s']:.1f} s total {meta['frame_s']:.1f} s")


# ----------------------------------------------------------------------------- IR / mm movie
def planck_nm(lam_nm, T):
    """B_lambda (W m^-2 nm^-1 sr^-1); lam_nm (..., n), T broadcastable."""
    x = 1.4387769e7 / (np.asarray(lam_nm, float) * np.asarray(T, float))
    with np.errstate(over="ignore"):
        return 2 * 6.62607015e-34 * 2.99792458e8 ** 2 / (lam_nm * 1e-9) ** 5 / np.expm1(np.minimum(x, 700.0)) * 1e-9


def background_stars_ir(grid, vmax=6.5):
    """BSC stars as blackbodies at their colour temperature, scaled to their visual irradiance."""
    from nightsky import starfield
    vis = spectral.default_grid(20)
    dirs, irr_v, _, teff, source = starfield.earth_stars({"bsc": {"vmax": vmax, "fetch": False}, "sun_ra_deg": 0.0}, vis)
    iv = vis.bin_index(550.0)
    teff = np.asarray(teff, float)[:, None]
    irr = irr_v[:, iv:iv + 1] * planck_nm(grid.wl[None, :], teff) / planck_nm(vis.wl[iv], teff)
    return dirs, irr, source


def deposit_ir(img, view, dirs, irr, grid):
    """Point sources onto img (H, W, nb) as radiance; like camera.deposit_points without the
    visual-luminance cut (which would drop every source outside 380-780 nm)."""
    row, col = view.project(np.asarray(dirs))
    ok = np.isfinite(row) & np.isfinite(col)
    if not ok.any():
        return 0
    row, col, irr = row[ok], col[ok], np.asarray(irr, float)[ok]
    H, W = view.height, view.width
    flat = img.reshape(H * W, -1)
    omega = view.omega.ravel()
    for s, idx, w, k in camera._stencils(view, row, col, POINTS, grid):
        e = irr[s]
        for b in range(flat.shape[1]):
            flat[:, b] += (np.bincount(idx, weights=w * e[k, b], minlength=H * W) / omega).astype(flat.dtype)
    return int(ok.sum())


def band_stats(img, valid):
    """Per band: log-mean of nu I_nu over pixels > 0, fraction > 0, median, 99th percentile."""
    out = []
    for b in range(img.shape[-1]):
        v = img[..., b][valid]
        pos = v[v > 0]
        out.append({"logmean": float(np.exp(np.mean(np.log(pos)))) if len(pos) else 0.0,
                    "frac_pos": float(len(pos) / max(len(v), 1)),
                    "p50": float(np.median(v)), "p99": float(np.percentile(v, 99))})
    return out


def cmd_orbit_ir(a, p, log=print):
    """IR / mm frames: nu I_nu (W m^-2 sr^-1) per band, bfloat16, with the star, background stars."""
    p = dict(p, bands="ir")
    sc = make_scene(a.outdir, p, a.nphot, log=log, bands="ir")
    fdir = os.path.join(a.outdir, "frames_ir")
    os.makedirs(fdir, exist_ok=True)
    track = CameraTrack(p, a.n_frames)
    bdirs, birr, src = background_stars_ir(sc.grid)
    log(f"bands (um): {[round(w / 1000, 2) for w in sc.grid.wl]}; background stars: {len(bdirs)} from {src}")
    lo, hi = (int(x) if x else None for x in a.frames.split(":"))
    with open(os.path.join(fdir, "bands.json"), "w") as f:
        json.dump({"wl_nm": sc.grid.wl.tolist(), "lo_nm": sc.grid.lo.tolist(), "hi_nm": sc.grid.hi.tolist(),
                   "units": "nu I_nu = lambda I_lambda, W m^-2 sr^-1", "params": p}, f, indent=1)
    for i in range(a.n_frames)[slice(lo, hi)]:
        base = os.path.join(fdir, f"f{i:05d}")
        if os.path.exists(base + ".json"):
            continue
        t0 = time.time()
        pos, fwd, up, ph = track.state(i)
        sc.pattern_angle = pattern_angle(p, ph)
        v, img, col, info = render_frame(sc, pos, fwd, p, star=False, up=up)
        _, bcol = sc.render(pos, bdirs[:, None, :], p["frac"], p["dtau_max"], p["tau_stop"])
        nbg = deposit_ir(img, v, bdirs, birr * np.exp(-np.outer(bcol[:, 0], sc.opt["k"])), sc.grid)
        irr, ncol = sc.star_irradiance(pos)
        info["star_in_frame"] = 0
        if i >= track.n_intro or p["intro_star"]:
            info["star_in_frame"] = deposit_ir(img, v, (-pos / np.linalg.norm(pos))[None], irr[None], sc.grid)
        info["planet_in_frame"], info["planet_column"] = deposit_planet(sc, img, v, pos, ir=True)
        nuinu = (img * sc.grid.wl).astype(np.float32)
        nuinu[~v.valid] = 0
        meta = {"frame": i, "phi_deg": ph, "pos_au": pos.tolist(), "fwd": fwd.tolist(), "up": up.tolist(), "z_over_R": float(pos[2] / np.hypot(pos[0], pos[1])),
                "star_tau_V": float(ncol * sc.k_v), "bg_stars_in_frame": nbg, "bands": band_stats(nuinu, v.valid), **info}
        with open(base + ".tmp.npz", "wb") as f:
            np.savez_compressed(f, nuinu_bf16=to_bf16(nuinu))
        os.replace(base + ".tmp.npz", base + ".npz")
        meta["frame_s"] = time.time() - t0
        with open(base + ".json", "w") as f:
            json.dump(meta, f)
        log(f"ir frame {i:5d} phi {ph:7.2f} z/R {meta['z_over_R']:+.3f} render {info['render_s']:.1f} s total {meta['frame_s']:.1f} s")


MM_RGB = np.array([0.1, 0.9, 1.0])  # overlay colour of the 1.3 mm emission (cyan)
MM_GAMMA = 1.5
KEY_IR = 2.0  # the exposure level (bright-pixel nu I_nu) maps here before the Reinhard curve


def _tone(x, white=6.0):
    """Reinhard on the channel mean, keeping channel ratios; linear in, linear [0, 1] out."""
    y = np.maximum(x.mean(axis=-1, keepdims=True), 1e-30)
    return np.clip(x * (y * (1 + y / white ** 2) / (1 + y)) / y, 0, 1)


def _develop_ir_one(args):
    i, fdir, outdir, idx, w, la, mm_idx, mm_w, mm_la, label, lines = args
    with np.load(os.path.join(fdir, f"f{i:05d}.npz")) as z:
        u = z["nuinu_bf16"]
        rgb = np.stack([from_bf16(u[..., j]) for j in idx[::-1]], axis=-1) * (np.asarray(w[::-1]) * KEY_IR / la)
        mm = from_bf16(u[..., mm_idx]) * (mm_w * KEY_IR / mm_la) if mm_idx is not None else None
    lin = _tone(rgb)
    if mm is not None:
        a = _tone(mm[..., None]) ** MM_GAMMA  # (H, W, 1) in [0, 1); gamma > 1 keeps the faint haze down
        lin = 1 - (1 - lin) * (1 - a * MM_RGB)  # screen blend
    rgb8 = (camera.srgb_encode(lin) * 255 + 0.5).astype(np.uint8)
    if label:
        rgb8 = camera.annotate(rgb8, types_ns(), lines)
    save_png(os.path.join(outdir, f"f{i:05d}.png"), rgb8)
    return i


def types_ns():
    import types
    return types.SimpleNamespace(kind="perspective")


def _exposure(lm, fps, tau_light_s, tau_dark_s, floor_rel=1e-3):
    """Time-smoothed log-mean brightness per frame (as the eye adaptation, in movie seconds),
    floored at floor_rel x the movie median so near-black frames are not amplified into noise."""
    from nightsky import animate
    lm = np.asarray(lm, float)
    med = np.median(lm[lm > 0]) if (lm > 0).any() else 1.0
    lm = np.maximum(lm, floor_rel * med)
    t_s = np.arange(len(lm)) / fps
    # smooth_log_la floors at the eye's absolute threshold: work in units of the median
    return animate.smooth_log_la(t_s / 3600.0, lm / med, tau_light_s / 60.0, tau_dark_s / 60.0, periodic=False) * med


def cmd_develop_ir(a, p, log=print):
    """False-colour IR movies (blue/green/red = short/mid/long band of each set), optionally with
    the 1.3 mm emission screen-blended on top in cyan. Channel balance is fixed per movie (each
    channel's median-over-frames 99th-percentile brightness brought to the same level); overall
    exposure follows the time-smoothed bright-pixel level like the eye adaptation. The mm layer
    has one fixed exposure for the whole movie (95th percentile over frames of the per-frame
    99th percentile), so it lights up only where the line of sight runs through the midplane."""
    from concurrent.futures import ProcessPoolExecutor
    from nightsky import animate
    fdir = os.path.join(a.outdir, "frames_ir")
    with open(os.path.join(fdir, "bands.json")) as f:
        wl_um = np.array(json.load(f)["wl_nm"]) / 1000.0
    metas = []
    for i in range(a.n_frames):
        with open(os.path.join(fdir, f"f{i:05d}.json")) as f:
            metas.append(json.load(f))
    p99 = np.array([[b["p99"] for b in m["bands"]] for m in metas])  # (frames, bands)
    gm = np.array([np.median(c[c > 0]) if (c > 0).any() else 1.0 for c in p99.T])
    mm_idx = int(np.argmin(np.abs(wl_um - bd.MM_BAND_UM)))
    mm_level = float(np.percentile(p99[:, mm_idx], 95)) / gm[mm_idx]
    for name in a.sets:
        idx = [int(np.argmin(np.abs(wl_um - u))) for u in bd.COLOUR_SETS[name]]
        w = 1.0 / gm[idx]
        la = _exposure((p99[:, idx] * w).mean(axis=1), a.fps, a.tau_light_s, a.tau_dark_s)
        mm_la = np.full(len(metas), mm_level)
        for with_mm in (False, True):
            tag = f"ir_{name}" + ("_mm" if with_mm else "")
            odir = os.path.join(a.outdir, f"movie_{tag}")
            os.makedirs(odir, exist_ok=True)
            bands_txt = "/".join(f"{u:g}" for u in bd.COLOUR_SETS[name]) + " um" + (" + 1.3 mm (cyan)" if with_mm else "")
            jobs = [(i, fdir, odir, idx, w, la[i], mm_idx if with_mm else None, 1.0 / gm[mm_idx], mm_la[i], a.label,
                     [bands_txt, f"r = {p['a_au']:.0f} AU   z/R = {m['z_over_R']:+.3f}"]) for i, m in enumerate(metas)]
            t0 = time.time()
            with ProcessPoolExecutor(a.cpus) as ex:
                for n, _ in enumerate(ex.map(_develop_ir_one, jobs, chunksize=4)):
                    if n % 300 == 0:
                        log(f"develop {tag}: {n}/{len(jobs)} ({time.time() - t0:.0f} s)")
            with open(os.path.join(odir, "exposure.json"), "w") as f:
                json.dump({"bands_um": list(bd.COLOUR_SETS[name]), "channel_weights": w.tolist(), "level": la.tolist(),
                           "mm_weight": float(1.0 / gm[mm_idx]), "mm_level": mm_la.tolist() if with_mm else None,
                           "fps": a.fps, "tau_light_s": a.tau_light_s, "tau_dark_s": a.tau_dark_s}, f)
            out = animate.stitch(os.path.join(odir, "f%05d.png"), os.path.join(a.outdir, f"{movie_prefix(p)}_{tag}"), a.fps, a.n_frames)
            log(f"wrote {out}")


def _develop_one(args):
    i, outdir, mode, la, p, label, lines = args
    from nightsky import animate
    v = vw.View("perspective", p["width"], p["height"], azimuth_deg=0.0, pitch_deg=0.0, fov_deg=p["fov_deg"])
    with np.load(frame_path(outdir, i) + ".npz") as z:
        xyzs = from_bf16(z["xyzs_bf16"])
    c = dict(EYE) if mode == "eye" else {"mode": "eye", "glare": True, "mesopic": False}
    c["adaptation_cd_m2"] = float(la)
    with animate._given_xyzs(xyzs):
        rgb8, _, _ = camera.develop(None, v, None, c, cpus=1)
    if label:
        rgb8 = camera.annotate(rgb8, v, lines)
    save_png(os.path.join(outdir, f"movie_{mode}", f"f{i:05d}.png"), rgb8)
    return i


def intro_exposure(p, metas, la):
    """Adaptation levels with the intro's frames exposed on their bright pixels
    (intro_key x 99.5th-percentile luminance), blended (in log) back to la over the pan."""
    la = np.array(la, float)
    n = intro_length(p)
    hf = int(p["intro_hold"]) + int(p["intro_fly"])
    for i in range(min(n, len(la))):
        w = 1.0 - _smooth((i - hf) / max(int(p["intro_pan"]), 1))
        bright = p["intro_key"] * metas[i]["p995_Y_cd_m2"]
        la[i] = np.exp((1 - w) * np.log(la[i]) + w * np.log(max(bright, 1e-12)))
    return la


def cmd_develop(a, p, log=print):
    """PNG frames + MP4. eye: adaptation smoothed in movie time (light / dark time constants in
    movie seconds); auto: photopic per-frame adaptation to the frame median (diagnostic)."""
    from concurrent.futures import ProcessPoolExecutor
    from nightsky import animate
    metas = []
    for i in range(a.n_frames):
        with open(frame_path(a.outdir, i) + ".json") as f:
            metas.append(json.load(f))
    t_s = np.arange(a.n_frames) / a.fps
    for mode in a.modes:
        if mode == "eye":
            la = intro_exposure(p, metas, [m["La_cd_m2"] for m in metas])
            la = animate.smooth_log_la(t_s / 3600.0, la, a.tau_light_s / 60.0, a.tau_dark_s / 60.0, periodic=False)
        else:
            la = intro_exposure(p, metas, [max(m["median_Y_cd_m2"], 1e-9) for m in metas])
        os.makedirs(os.path.join(a.outdir, f"movie_{mode}"), exist_ok=True)
        jobs = []
        for i, m in enumerate(metas):
            lines = [f"r = {p['a_au']:.0f} AU   z/R = {m['z_over_R']:+.3f}",
                     f"tau_V to star = {m['star_tau_V']:.3g}" if m["star_tau_V"] >= 1e-3 else "tau_V to star < 0.001"]
            jobs.append((i, a.outdir, mode, la[i], p, a.label, lines))
        t0 = time.time()
        with ProcessPoolExecutor(a.cpus) as ex:
            for n, _ in enumerate(ex.map(_develop_one, jobs, chunksize=4)):
                if n % 100 == 0:
                    log(f"develop {mode}: {n}/{len(jobs)} ({time.time() - t0:.0f} s)")
        with open(os.path.join(a.outdir, f"movie_{mode}", "adaptation.json"), "w") as f:
            json.dump({"La_cd_m2": [float(x) for x in la], "fps": a.fps, "tau_light_s": a.tau_light_s,
                       "tau_dark_s": a.tau_dark_s}, f)
        out = animate.stitch(os.path.join(a.outdir, f"movie_{mode}", "f%05d.png"),
                             os.path.join(a.outdir, f"{movie_prefix(p)}_{mode}"), a.fps, a.n_frames)
        log(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mc", "stills", "sed", "orbit", "develop", "orbit-ir", "develop-ir", "stars", "stars-ir"])
    ap.add_argument("--sets", type=lambda s: s.split(","), default=list(bd.COLOUR_SETS))
    ap.add_argument("--n-frames", type=int, default=1800)
    ap.add_argument("--frames", default=":", help="python slice of frame indices for orbit, e.g. 0:900")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--modes", type=lambda s: s.split(","), default=["eye"])
    ap.add_argument("--tau-light-s", type=float, default=0.5)
    ap.add_argument("--tau-dark-s", type=float, default=3.0)
    ap.add_argument("--label", action="store_true")
    ap.add_argument("outdir")
    ap.add_argument("--nphot", type=float, default=2e7)
    ap.add_argument("--cpus", type=int, default=int(os.environ.get("NUMBA_NUM_THREADS", "8")))
    ap.add_argument("--views", type=lambda s: [int(x) for x in s.split(",")], default=[1, 4, 6, 8])
    ap.add_argument("--sed-px", type=int, default=640)
    ap.add_argument("--sed-levels", type=int, default=6)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    p = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        p[k] = type(DEFAULTS[k])(v)
    if a.cmd == "mc":
        make_scene(a.outdir, p, a.nphot, bands=p["bands"])
    elif a.cmd == "stills":
        cmd_stills(a, p)
    elif a.cmd == "sed":
        cmd_sed(a, p)
    elif a.cmd == "orbit":
        cmd_orbit(a, p)
    elif a.cmd == "develop":
        cmd_develop(a, p)
    elif a.cmd == "orbit-ir":
        cmd_orbit_ir(a, p)
    elif a.cmd == "develop-ir":
        cmd_develop_ir(a, p)
    elif a.cmd == "stars":
        from . import starlayer
        starlayer.run(a.outdir, a.modes, p, a.n_frames, a.fps, a.nphot)
    elif a.cmd == "stars-ir":
        from . import starlayer
        starlayer.run_ir(a.outdir, a.sets, p, a.n_frames, a.fps, a.nphot)


if __name__ == "__main__":
    main()
