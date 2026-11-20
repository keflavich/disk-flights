"""Position key ("inset") for the disk-orbit movies, after the 2026-11-10 flythrough key.

A meridional slice of the model (grey log density) through the camera, with the disk outline
(+-2 scale heights), the arc the orbit covers in this slice, the camera, and its vertical field of
view. Because the disk is axisymmetric, the slice through the camera is exact for every frame. The
camera looks mostly along the orbit, i.e. out of the slice, so the field-of-view wedge shows the
true elevation above the disk plane of the top-centre (solid) and bottom-centre (dashed) rays,
drawn toward the side the camera faces (always inward, toward the star, for this camera). The only
text: the scale bar, r_orbit and theta (the camera's angle from the rotation axis).

The insets are written as RGBA PNGs (key/f%05d.png) and overlaid on the existing movie frame
sequences with ffmpeg, writing diskorbit_<model>_<tag>_key.mp4 next to the originals.

    python -m diskorbit.inset --nproc 16 [--tags eye eye_stars ...] [--n-frames 1800]
"""
import argparse
import glob
import os
import subprocess
import time

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .render import DEFAULTS, movie_prefix, pattern_angle, look_basis, CameraTrack  # noqa: E402

OUTDIR = "/orange/adamginsburg/gc/visualization/renders/diskorbit"
REF_W = 1600  # the flythrough key is 360 px wide on 1600 px frames; scale with frame width
L_AU = 75.0  # half-width of the slice
SCALE_AU = 50.0


class Inset:
    def __init__(self, p, size=None):
        from .model import DiskModel
        self.p = p
        self.s = p["width"] / REF_W
        self.size = size or int(round(360 * self.s))
        self.R = p["a_au"]
        self.vh = np.degrees(np.arctan(np.tan(np.radians(p["fov_deg"]) / 2) * p["height"] / p["width"]))
        m = self.m = DiskModel(p["model"])
        self.h0, self.r0, self.beta, self.rmax = m.h0, m.r0, m.beta, m.rmax
        self.L0 = float(p.get("key_L_au", L_AU))
        # non-axisymmetric (planet) model: the slice depends on the camera's azimuth in the
        # pattern frame; the colour stretch is fixed by the axisymmetric slice
        self.pert = None
        if p.get("planet_sigma"):
            from diskplanet.perturb import Perturbation
            self.pert = Perturbation.from_npz(p["planet_sigma"], p["planet_rp"], clump_h=p.get("planet_clump_h", 0.0))
        self._set_extent(self.L0)
        self.vmax = float(np.nanpercentile(self.logrho_axi, 99.5))
        self.vmin = self.vmax - float(p.get("key_dex", 7.0))
        self.logrho = self._mask(self.logrho_axi)
        # theta range covered by the orbit (inclined circle: |latitude| <= inc)
        self.th_lo, self.th_hi = 90.0 - p["inc_deg"], 90.0 + p["inc_deg"]

    def _set_extent(self, L, n=600):
        """Slice grid of half-width L au (the key zooms out while the camera is far away)."""
        self.L = L
        g = np.linspace(-L, L, n)
        self.X, self.Z = np.meshgrid(g, g[::-1])
        rho = self.m.density(np.abs(self.X), self.Z)
        self.logrho_axi = np.where(rho > 0, np.log10(np.maximum(rho, 1e-300)), np.nan)
        self.h_au = self.h0 * (np.maximum(np.abs(self.X), 1e-9) / self.r0) ** self.beta
        if hasattr(self, "vmin"):
            self.logrho = self._mask(self.logrho_axi)

    def _mask(self, lr):
        return np.where(lr < self.vmin, np.nan, lr)

    def slice_at(self, phi_model):
        """log10 density in the meridional plane at azimuth phi_model (x > 0 side) and
        phi_model + pi (x < 0 side)."""
        if self.pert is None:
            return self.logrho
        ph = np.where(self.X >= 0, phi_model, phi_model + np.pi)
        ph = np.angle(np.exp(1j * ph))
        f = self.pert.factor(np.abs(self.X), ph, self.Z, self.h_au)
        return self._mask(self.logrho_axi + np.log10(np.maximum(f, 1e-300)))

    def draw(self, pos, fwd, psi=0.0, up=None):
        """psi: pattern azimuth (rad, inertial), see render.pattern_angle; up: the camera's up
        vector (default: the z-axis reference of render.look_basis)."""
        s = self.s
        rcam = float(np.linalg.norm(pos))
        L = max(self.L0, 1.15 * rcam)
        if abs(L - self.L) > 1e-9 * L:
            self._set_extent(L)
        img_slice = self.slice_at(np.arctan2(pos[1], pos[0]) - psi)
        W_px, FOOT = self.size, int(round(34 * s))
        H_px = W_px + FOOT
        fig = plt.figure(figsize=(W_px / 100.0, H_px / 100.0), dpi=100)
        fig.patch.set_facecolor((0.02, 0.02, 0.04, 0.80))
        ax = fig.add_axes([0, FOOT / H_px, 1, W_px / H_px])
        L = self.L
        ax.set_facecolor((0, 0, 0, 0.9))
        ax.imshow(img_slice, extent=(-L, L, -L, L), cmap="gray", vmin=self.vmin, vmax=self.vmax,
                  interpolation="bilinear", alpha=0.85)
        # disk outline: +-2 scale heights
        Rd = np.linspace(0.05, min(self.rmax, 1.5 * L), 400)
        h2 = 2 * self.h0 * (Rd / self.r0) ** self.beta
        for sz in (1, -1):
            for sx in (1, -1):
                ax.plot(sx * Rd, sz * h2, color="#ffb14a", lw=1.3)
        # the arc of the orbit in this slice (camera radius fixed, polar angle within 90 +- inc)
        t = np.radians(np.linspace(self.th_lo, self.th_hi, 200))
        ax.plot(self.R * np.sin(t), self.R * np.cos(t), color="#ffe14a", lw=1.3, ls=(0, (4, 3)), alpha=0.9)
        # field of view: wedge whose edges have the true elevation (above the disk plane) of the
        # top- and bottom-centre rays, drawn on the side the camera looks toward (inward/outward)
        cR, cz = np.hypot(pos[0], pos[1]), pos[2]
        f, _, u = look_basis(fwd) if up is None else look_basis(fwd, up)
        a = np.radians(self.vh)
        on_axis = cR < 1e-6 * rcam
        # on the axis the slice plane is the one containing the view direction
        Rhat = np.array([f[0], f[1], 0.0]) if on_axis else np.array([pos[0], pos[1], 0.0])
        Rhat = Rhat / (np.linalg.norm(Rhat) or 1.0)
        side = np.sign(f @ Rhat) or 1.0
        W = 3 * L
        edges = []
        for d in (np.cos(a) * f + np.sin(a) * u, np.cos(a) * f - np.sin(a) * u):
            dz = float(np.clip(d[2], -1, 1))
            sd = (np.sign(d @ Rhat) or 1.0) if on_axis else side
            edges.append((cR + W * sd * np.sqrt(1 - dz * dz), cz + W * dz))
        (tR, tz), (bR, bz) = edges
        ax.fill([cR, tR, bR], [cz, tz, bz], color="#ffe14a", alpha=0.18, lw=0)
        ax.plot([cR, tR], [cz, tz], color="#ffe14a", lw=1.6)
        ax.plot([cR, bR], [cz, bz], color="#ffe14a", lw=0.8, ls=(0, (3, 2)))
        ax.plot([0], [0], marker="*", color="white", ms=9 * s, mec="k", mew=0.5)
        ax.plot([cR], [cz], marker="o", color="#ffe14a", ms=7 * s, mec="k", mew=0.8)
        th = np.degrees(np.arccos(cz / rcam))
        where = rf"$r_{{\rm orbit}}$ = {self.R:g} au" if rcam < 1.005 * self.R else f"d = {rcam:.0f} au"
        fig.text(0.04, 8 * s / H_px, rf"{where}      $\theta$ = {th:.1f}°",
                 va="bottom", color="white", fontsize=13 * s, family="DejaVu Sans")
        bar = max([b for b in (SCALE_AU, 100.0, 200.0, 500.0, 1000.0) if b <= max(0.4 * L, SCALE_AU)])
        x0, y0 = -0.94 * L, -0.92 * L
        ax.plot([x0, x0 + bar], [y0, y0], color="white", lw=2)
        ax.text(x0, y0 + 0.03 * L, f"{bar:g} au", ha="left", va="bottom", fontsize=11 * s, color="white")
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#888888")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba()).copy()
        plt.close(fig)
        return img, th


_W = {}


def _worker(args):
    import imageio.v3 as iio
    i, n_frames, keydir, p = args
    out = os.path.join(keydir, f"f{i:05d}.png")
    if os.path.exists(out):
        return i, None
    if "inset" not in _W:
        _W["inset"] = Inset(p)
        _W["track"] = CameraTrack(p, n_frames)
    pos, fwd, up, phi = _W["track"].state(i)
    img, th = _W["inset"].draw(pos, fwd, pattern_angle(p, phi), up)
    iio.imwrite(out, img)
    return i, th


def make_insets(outdir, p, n_frames, nproc, log=print):
    from multiprocessing import Pool
    keydir = os.path.join(outdir, "key")
    os.makedirs(keydir, exist_ok=True)
    t0 = time.time()
    with Pool(nproc) as pool:
        for k, _ in enumerate(pool.imap_unordered(_worker, [(i, n_frames, keydir, p) for i in range(n_frames)], chunksize=8)):
            if k % 300 == 0:
                log(f"inset {k}/{n_frames} ({time.time() - t0:.0f} s)")
    return keydir


def overlay(outdir, tag, keydir, p, n_frames, fps, margin=None):
    from nightsky.animate import find_ffmpeg
    exe, codec = find_ffmpeg()
    m = margin if margin is not None else int(round(16 * p["width"] / REF_W))
    out = os.path.join(outdir, f"{movie_prefix(p)}_{tag}_key.mp4")
    cmd = [exe, "-y", "-loglevel", "error",
           "-framerate", str(fps), "-start_number", "0", "-i", os.path.join(outdir, f"movie_{tag}", "f%05d.png"),
           "-framerate", str(fps), "-start_number", "0", "-i", os.path.join(keydir, "f%05d.png"),
           "-filter_complex", f"[0:v][1:v]overlay=W-w-{m}:H-h-{m}:format=auto,pad=ceil(iw/2)*2:ceil(ih/2)*2",
           "-frames:v", str(n_frames)] + codec + [out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {tag}: {r.stderr.strip()[:500]}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--tags", nargs="*", default=None, help="movie_<tag> frame dirs (default: all with frames)")
    ap.add_argument("--n-frames", type=int, default=1800)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--insets-only", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="parameter override key=value (as diskorbit.render)")
    a = ap.parse_args()
    p = dict(DEFAULTS)
    for kv in a.set:
        k, v = kv.split("=", 1)
        p[k] = type(DEFAULTS[k])(v)
    keydir = make_insets(a.outdir, p, a.n_frames, a.nproc)
    if a.insets_only:
        return
    tags = a.tags
    if tags is None:
        tags = sorted(os.path.basename(d)[len("movie_"):] for d in glob.glob(os.path.join(a.outdir, "movie_*"))
                      if os.path.isdir(d) and os.path.exists(os.path.join(d, f"f{a.n_frames - 1:05d}.png")))
    from multiprocessing import Pool
    with Pool(min(len(tags), max(1, a.nproc // 4))) as pool:
        for out in pool.starmap(overlay, [(a.outdir, t, keydir, p, a.n_frames, a.fps) for t in tags]):
            print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
