"""Background star layer, toned on its own with one fixed stretch and screen-blended over the
developed (display-space) disk frames, following the 2026-11-10 orbit flythroughs
(nsfgrants/2026/protostars/flythrough_movies_orbit_2026-11-10/code: make_realsky.star_layer,
frames.tonemap). The scene's exposure never touches the stars, so they stay visible and do not
flicker with the eye adaptation.

Per star: amplitude amp = Y / Y(V=6) (the visual flux relative to a V = 6 star, i.e.
10^(-0.4 (V - 6)) for the catalogue V), colour = linear sRGB of the star's spectrum after the
disk's extinction along the line of sight (so stars are dimmed and reddened by the disk),
normalised so the unattenuated colour has max channel 1. PSF (defined at 1600 px width, scaled
by width / 1600): 0.96 exp(-r^2 / 2 s^2) + 0.04 (1 + r^2 / rh^2)^-1.7, s = 1.15 px, rh = 3.2 px,
peak 1. Fixed hue-preserving asinh stretch on the brightest channel:
f = asinh(I / beta) / asinh(Lmax / beta), beta 0.3, Lmax 960, with highlights above
white_start blended toward white (up to `white`). 180-degree shutter: 5 sub-positions over half
a frame step. Composite: out = 1 - (1 - scene)(1 - stars) in display space.
"""
import json
import os
import time

import numpy as np
from numba import njit, prange

from nightsky import spectral, camera, starfield

BETA, LMAX, WHITE, WHITE_START = 0.3, 960.0, 0.4, 0.6
AMP_MIN = 3e-3
N_SUB = 5


@njit(parallel=True)
def _splat(H, W, rows, cols, rgb, sig, rh, rad):
    """Sum of peak-normalised PSFs (n stars, colour rgb (n, 3)) on an (H, W, 3) image. Stars are
    distributed over threads by image row band to avoid write races."""
    img = np.zeros((H, W, 3))
    nb = 64
    band = (H + nb - 1) // nb
    for t in prange(nb):
        r_lo, r_hi = t * band, min(H, (t + 1) * band)
        for i in range(rows.shape[0]):
            R = rad[i]
            r0, c0 = rows[i], cols[i]
            a = max(int(np.floor(r0)) - R, r_lo)
            b = min(int(np.floor(r0)) + R + 1, r_hi)
            if a >= b:
                continue
            for rr in range(a, b):
                for cc in range(max(int(np.floor(c0)) - R, 0), min(int(np.floor(c0)) + R + 1, W)):
                    d2 = (rr - r0) ** 2 + (cc - c0) ** 2
                    v = 0.96 * np.exp(-0.5 * d2 / (sig * sig)) + 0.04 * (1.0 + d2 / (rh * rh)) ** -1.7
                    for ch in range(3):
                        img[rr, cc, ch] += v * rgb[i, ch]
    return img


def tone(I):
    """Fixed hue-preserving asinh stretch with white highlights; linear (H, W, 3) -> [0, 1]."""
    m = I.max(axis=-1, keepdims=True)
    f = np.arcsinh(m / BETA) / np.arcsinh(LMAX / BETA)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(m > 0, I * (f / m), 0.0)
    t = np.clip((f - WHITE_START) / (1 - WHITE_START), 0, 1) * WHITE
    return np.clip(out * (1 - t) + f * t, 0, 1)


class StarLayer:
    def __init__(self, scene, width, height, fov_deg, vmax=6.5):
        self.sc, self.W, self.H, self.fov = scene, width, height, fov_deg
        grid = scene.grid
        self.dirs, self.irr, _, _, self.source = starfield.earth_stars({"bsc": {"vmax": vmax, "fetch": False}, "sun_ra_deg": 0.0}, grid)
        y_ref = spectral.to_xyz(starfield.vmag_irr_fast(np.array([6.0]), np.array([9600.0]), grid), grid)[0, 1]
        xyz0 = spectral.to_xyz(self.irr, grid)
        self.amp0 = xyz0[:, 1] / y_ref
        rgb0 = np.maximum(camera.xyz_to_rgb(xyz0), 0)
        self.norm = self.amp0 / np.maximum(rgb0.max(axis=1), 1e-300)  # rgb * norm: max channel = amp
        s = width / 1600.0
        self.sig, self.rh = max(0.75, 1.15 * s), max(1.6, 3.2 * s)
        self.rad_big, self.rad_small = int(np.ceil(40 * s)), int(np.ceil(10 * s))

    def colours(self, pos, p):
        """Extinguished linear-RGB star colours (n, 3) and amplitudes from camera position pos."""
        _, col = self.sc.render(pos, self.dirs[:, None, :], p["frac"], p["dtau_max"], p["tau_stop"])
        irr = self.irr * np.exp(-np.outer(col[:, 0], self.sc.opt["k"]))
        rgb = np.maximum(camera.xyz_to_rgb(spectral.to_xyz(irr, self.sc.grid)), 0) * self.norm[:, None]
        return rgb, rgb.max(axis=1)

    def layer(self, views, rgb, amp):
        """Linear star image averaged over the sub-frame views (motion blur)."""
        keep = amp >= AMP_MIN
        out = np.zeros((self.H, self.W, 3))
        for v in views:
            row, col = v.project(self.dirs[keep])
            ok = np.isfinite(row) & np.isfinite(col)
            if not ok.any():
                continue
            a = amp[keep][ok]
            rad = np.where(a > 30, self.rad_big, self.rad_small).astype(np.int64)
            out += _splat(self.H, self.W, row[ok], col[ok], rgb[keep][ok] / len(views), self.sig, self.rh, rad)
        return out


def run(outdir, modes, p, n_frames, fps, nphot=2e7, log=print):
    import imageio.v3 as iio
    from nightsky import animate
    from .render import make_scene, movie_prefix, pattern_angle, make_view, orbit_state, camera_forward, orbit_schedule, save_png
    sc = make_scene(outdir, p, nphot, log=log)
    sl = StarLayer(sc, p["width"], p["height"], p["fov_deg"])
    log(f"star layer: {len(sl.dirs)} stars from {sl.source}; psf sigma {sl.sig:.2f} px, halo {sl.rh:.2f} px")
    phis = np.unwrap(np.radians(orbit_schedule(n_frames, p["inc_deg"])))
    phis = np.degrees(np.append(phis, phis[-1] + (phis[-1] - phis[-2])))
    for m in modes:
        os.makedirs(os.path.join(outdir, f"movie_{m}_stars"), exist_ok=True)
    t0 = time.time()
    for i in range(n_frames):
        outs = [os.path.join(outdir, f"movie_{m}_stars", f"f{i:05d}.png") for m in modes]
        if all(os.path.exists(o) for o in outs):
            continue
        pos, vel = orbit_state(phis[i], p["a_au"], p["inc_deg"])
        sc.pattern_angle = pattern_angle(p, phis[i])
        rgb, amp = sl.colours(pos, p)
        views = []
        for k in range(N_SUB):
            ph = phis[i] + 0.5 * (phis[i + 1] - phis[i]) * k / (N_SUB - 1)
            ps, vs = orbit_state(ph, p["a_au"], p["inc_deg"])
            views.append(make_view(camera_forward(ps, vs, p["toward_star_deg"]), p["width"], p["height"], p["fov_deg"]))
        st = tone(sl.layer(views, rgb, amp))
        for m, o in zip(modes, outs):
            scene = iio.imread(os.path.join(outdir, f"movie_{m}", f"f{i:05d}.png")).astype(np.float64) / 255.0
            out = 1 - (1 - scene[..., :3]) * (1 - st)
            save_png(o, (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
        if i % 100 == 0:
            log(f"stars frame {i}/{n_frames} ({time.time() - t0:.0f} s); {int((amp >= AMP_MIN).sum())} stars above cut")
    for m in modes:
        out = animate.stitch(os.path.join(outdir, f"movie_{m}_stars", "f%05d.png"),
                             os.path.join(outdir, f"{movie_prefix(p)}_{m}_stars"), fps, n_frames)
        log(f"wrote {out}")


def ir_colours(teff, col, k_bands, wl_nm):
    """False-colour star colours for one IR set, as the 2026-11-10 near-IR flythroughs: per band
    (red, green, blue = long, mid, short) the star's blackbody relative to a 9600 K blackbody,
    brightest band 1, times the disk transmission exp(-k_band N) along the line of sight."""
    from .render import planck_nm
    t = np.asarray(teff, float)[:, None]
    hue = planck_nm(wl_nm[None, :], t) / planck_nm(wl_nm[None, :], 9600.0)
    hue /= hue.max(axis=1, keepdims=True)
    return (hue * np.exp(-np.outer(col, k_bands)))[:, ::-1]  # (n, 3) as R, G, B


def run_ir(outdir, sets, p, n_frames, fps, nphot=2e7, log=print):
    """Star layer for the develop-ir movies movie_ir_{set}[_mm] -> movie_ir_{set}[_mm]_stars."""
    import imageio.v3 as iio
    from nightsky import animate
    from . import bands as bd
    from .render import make_scene, movie_prefix, pattern_angle, make_view, orbit_state, camera_forward, orbit_schedule, save_png
    sc = make_scene(outdir, p, nphot, log=log, bands="ir")
    vis = spectral.default_grid(20)
    dirs, irr_v, _, teff, source = starfield.earth_stars({"bsc": {"vmax": 6.5, "fetch": False}, "sun_ra_deg": 0.0}, vis)
    y_ref = spectral.to_xyz(starfield.vmag_irr_fast(np.array([6.0]), np.array([9600.0]), vis), vis)[0, 1]
    amp0 = spectral.to_xyz(irr_v, vis)[:, 1] / y_ref
    s = p["width"] / 1600.0
    sig, rh, rad_big, rad_small = max(0.75, 1.15 * s), max(1.6, 3.2 * s), int(np.ceil(40 * s)), int(np.ceil(10 * s))
    idx = {name: [bd.band_index(sc.grid, u) for u in bd.COLOUR_SETS[name]] for name in sets}
    log(f"IR star layer: {len(dirs)} stars from {source}; sets {sets}")
    phis = np.unwrap(np.radians(orbit_schedule(n_frames, p["inc_deg"])))
    phis = np.degrees(np.append(phis, phis[-1] + (phis[-1] - phis[-2])))
    tags = {name: [f"ir_{name}", f"ir_{name}_mm"] for name in sets}
    for name in sets:
        for tg in tags[name]:
            os.makedirs(os.path.join(outdir, f"movie_{tg}_stars"), exist_ok=True)
    t0 = time.time()
    for i in range(n_frames):
        todo = [(name, tg) for name in sets for tg in tags[name]
                if not os.path.exists(os.path.join(outdir, f"movie_{tg}_stars", f"f{i:05d}.png"))]
        if not todo:
            continue
        pos, _ = orbit_state(phis[i], p["a_au"], p["inc_deg"])
        sc.pattern_angle = pattern_angle(p, phis[i])
        _, col = sc.render(pos, dirs[:, None, :], p["frac"], p["dtau_max"], p["tau_stop"])
        views = []
        for k in range(N_SUB):
            ph = phis[i] + 0.5 * (phis[i + 1] - phis[i]) * k / (N_SUB - 1)
            ps, vs = orbit_state(ph, p["a_au"], p["inc_deg"])
            views.append(make_view(camera_forward(ps, vs, p["toward_star_deg"]), p["width"], p["height"], p["fov_deg"]))
        proj = [v.project(dirs) for v in views]
        for name in sorted({n for n, _ in todo}):
            rgb = ir_colours(teff, col[:, 0], sc.opt["k"][idx[name]], sc.grid.wl[idx[name]]) * amp0[:, None]
            amp = rgb.max(axis=1)
            keep = amp >= AMP_MIN
            lin = np.zeros((p["height"], p["width"], 3))
            for row, cl in proj:
                ok = keep & np.isfinite(row) & np.isfinite(cl)
                if ok.any():
                    rad = np.where(amp[ok] > 30, rad_big, rad_small).astype(np.int64)
                    lin += _splat(p["height"], p["width"], row[ok], cl[ok], rgb[ok] / N_SUB, sig, rh, rad)
            st = tone(lin)
            for n2, tg in todo:
                if n2 != name:
                    continue
                scene = iio.imread(os.path.join(outdir, f"movie_{tg}", f"f{i:05d}.png")).astype(np.float64) / 255.0
                out = 1 - (1 - scene[..., :3]) * (1 - st)
                save_png(os.path.join(outdir, f"movie_{tg}_stars", f"f{i:05d}.png"), (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
        if i % 100 == 0:
            log(f"IR stars frame {i}/{n_frames} ({time.time() - t0:.0f} s)")
    for name in sets:
        for tg in tags[name]:
            out = animate.stitch(os.path.join(outdir, f"movie_{tg}_stars", "f%05d.png"),
                                 os.path.join(outdir, f"{movie_prefix(p)}_{tg}_stars"), fps, n_frames)
            log(f"wrote {out}")
