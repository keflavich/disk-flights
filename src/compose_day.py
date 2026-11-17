"""Compose the day-cycle frames from the rendered light groups.

Per frame: sample each group's cube faces into the chosen view, convert the
wavelength bins to XYZ (plus scotopic luminance), add the Sun disk and the
cluster's stars as analytic point sources with slant-path extinction, apply a
smoothed eye-adaptation exposure and a simple mesopic (Purkinje) shift, tone map,
and write PNGs.

    python compose_day.py --data ../renders/day/data --out ../renders/day/frames --views allsky north
"""
import argparse
import os

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import atmosphere as at
import bright_stars as bs
import cluster as cl
import companion as co
import photometry as ph
import render_day as rd
import views as vw

N_TAU_SHELLS = 96
ALLOW_MISSING = False
DENOISE_SIGMA = 2.0  # px at a 128-px cube face; scaled with face size
_MASKS = {}
TWILIGHT_DENOISE_FACTOR = 2.0
_SHELLS = {}


def slant_tau(elev_deg, wavelengths_nm):
    """Rayleigh optical depth from the observer to space along elevation(s); (n_el, n_wl). inf below horizon."""
    el = np.atleast_1d(np.asarray(elev_deg, float))
    out = np.full((len(el), len(wavelengths_nm)), np.inf)
    up = el > 0
    r0 = at.R_EARTH + 1.7
    mu = np.sin(np.radians(el[up]))[:, None]
    for j, wl in enumerate(wavelengths_nm):
        key = (round(float(wl), 3))
        if key not in _SHELLS:
            _SHELLS[key] = at.build_atmosphere_media(N_TAU_SHELLS, float(wl))
        sh = _SHELLS[key]
        r_out = np.array([s["r_out"] for s in sh])[None]
        r_in = np.maximum(np.array([s["r_in"] for s in sh]), r0)[None]
        sig = np.array([s["sigma_t"] for s in sh])[None]
        chord = lambda r: -r0 * mu + np.sqrt((r0 * mu) ** 2 + r * r - r0 * r0)
        out[up, j] = (sig * np.clip(chord(r_out) - chord(np.minimum(r_in, r_out)), 0, None)).sum(axis=1)
    return out


def spectral_matrix(wls, dwl):
    """(n_bins, 4): radiance-per-nm bins -> [X, Y (cd/m^2), Z, scotopic luminance in photopic-equivalent units]."""
    cmf = ph.cie_xyz_cmf(wls)
    vs = 1.01 * np.exp(-0.5 * ((wls - 505.0) / 40.0) ** 2)  # coarse V'(lambda) fit
    return np.column_stack([ph.LM_PER_W * dwl * cmf, 1700.0 * dwl * vs])


def sky_mask(size):
    """(6, S, S) bool: pixel direction above the horizon."""
    if size not in _MASKS:
        m = np.zeros((6, size, size), bool)
        t = np.tan(np.radians(vw.FACE_FOV_DEG) / 2)
        u = (np.arange(size) + 0.5) / size * 2 - 1
        X, Y = np.meshgrid(u, -u)
        for k in range(6):
            fwd, right, up = vw.face_basis(k)
            d = fwd + right * X[..., None] * t + up * Y[..., None] * t
            m[k] = d[..., 2] > 0
        _MASKS[size] = m
    return _MASKS[size]


def denoise_faces(faces, sigma_128=None):
    """Gaussian-smooth each cube face separately on the sky and ground side of the horizon.

    The sky and ground are smooth apart from the horizon edge itself, so the
    Monte Carlo noise can be filtered aggressively without softening the horizon.
    """
    sigma_128 = DENOISE_SIGMA if sigma_128 is None else sigma_128
    size = faces.shape[-1]
    sigma = sigma_128 * size / 128.0
    if sigma < 0.3:
        return faces
    m = sky_mask(size).astype(np.float64)
    out = np.empty(faces.shape, np.float32)
    ss = (0, 0, sigma, sigma)
    for msk in (m, 1.0 - m):
        w = gaussian_filter(np.broadcast_to(msk, faces.shape[1:]).copy(), sigma=ss[1:], mode="nearest")
        num = gaussian_filter(faces * msk[None], sigma=ss, mode="nearest")
        sm = num / np.maximum(w[None], 1e-6)
        out = np.where(msk[None] > 0.5, sm, out)
    return out


def sample_bins(faces, dirs, M, horizon_margin_deg=None, sigma_128=None):
    """faces (B, 6, S, S) -> (H, W, 4) via the spectral matrix M (B, 4)."""
    faces = denoise_faces(faces, sigma_128)
    if horizon_margin_deg is None:  # keep the sample point about one face pixel off the horizon
        horizon_margin_deg = max(0.6, 0.5 * vw.FACE_FOV_DEG / faces.shape[-1])
    f = np.transpose(faces, (1, 2, 3, 0))
    s = vw.sample_cube(f, dirs, horizon_margin_deg)
    return s @ M


def load_static(data_dir):
    d = {g: np.load(os.path.join(data_dir, f"static_{g}.npz")) for g in ("sky", "arches")}
    return d


def point_layer(view, dirs, spec_irr, wls, M, extinct=True):
    """Analytic point sources -> (H, W, 4) radiance layer. spec_irr: (n, B) top-of-atmosphere W/m^2/nm."""
    el = np.degrees(np.arcsin(np.clip(dirs[:, 2], -1, 1)))
    vis = el > 0.3
    layer = np.zeros((view.height, view.width, 4))
    if not vis.any():
        return layer
    e = spec_irr[vis]
    if extinct:
        e = e * np.exp(-slant_tau(el[vis], wls))
    flux = e @ M  # (n, 4)
    return vw.add_points(layer, view, dirs[vis], flux, sigma_px=1.0)


def frame_layers(k, view, static, data_dir, wls, M, star_layer, sun_cache):
    """Full (H, W, 4) linear radiance layer for frame k, no exposure applied."""
    H = rd.frame_hour_angle(k)
    alt = rd.sun_altitude_deg(H)
    img = static.copy()
    if alt >= rd.SUN_MIN_ALT_DEG:
        kk = k if H <= 0 else rd.N_FRAMES - k
        path = os.path.join(data_dir, f"sun_{kk:03d}.npz")
        if kk not in sun_cache:
            if ALLOW_MISSING and not os.path.exists(path):
                return img, H, alt
            sun_cache.clear()
            sun_cache[kk] = np.load(path)["faces"]
        dirs = view.dirs if H <= 0 else view.dirs * np.array([1.0, -1.0, 1.0])
        # twilight skies are very smooth but very noisy (few sunlit scattering paths):
        # smooth them over a wider angle
        sigma = DENOISE_SIGMA * (TWILIGHT_DENOISE_FACTOR if alt < rd.TWILIGHT_ALT_DEG else 1.0)
        img = img + sample_bins(sun_cache[kk], dirs, M, sigma_128=sigma)
    return img, H, alt


def sun_disk_layer(view, H, alt, wls, M):
    if alt <= 0.3:
        return 0.0
    e = np.array([[rd.sun_irradiance(w) for w in wls]])
    return point_layer(view, rd.sun_direction(H)[None], e, wls, M)


def log_average(img, valid):
    y = img[..., 1][valid]
    return float(np.exp(np.mean(np.log(np.maximum(y, 1e-7)))))


def adaptation(log_la, gamma, l_ref, smooth_frames):
    ls = gaussian_filter1d(log_la, smooth_frames, mode="wrap")
    la = np.exp(ls)
    la_eff = la ** gamma * l_ref ** (1 - gamma)
    m = np.clip((np.log10(la) - np.log10(0.005)) / (np.log10(5.0) - np.log10(0.005)), 0, 1)
    return 0.18 / la_eff, m


TINT_XYZ = np.array([0.933, 1.0, 1.40])  # blue-shifted white for scotopic vision


def to_srgb8(img4, scale, mesopic, white=6.0):
    xyz = img4[..., :3]
    xyz = mesopic * xyz + (1 - mesopic) * img4[..., 3:4] * TINT_XYZ
    rgb = ph.xyz_to_linear_srgb(xyz) * scale
    y = np.maximum(xyz[..., 1] * scale, 1e-12)
    yd = y * (1 + y / white ** 2) / (1 + y)  # extended Reinhard on luminance
    rgb = np.clip(rgb * (yd / y)[..., None], 0, 1)
    out = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return (out * 255 + 0.5).astype(np.uint8)


def annotate(img8, view, H, alt, label=None):
    from PIL import Image, ImageDraw
    im = Image.fromarray(img8)
    dr = ImageDraw.Draw(im)
    hours = (12.0 + H / 15.0) % 24
    dr.text((10, 8), f"{int(hours):02d}:{int(round((hours % 1) * 60)) % 60:02d} local solar time   Sun altitude {alt:+.0f} deg",
            fill=(235, 235, 235), stroke_width=1, stroke_fill=(0, 0, 0))
    if label:
        dr.text((10, im.size[1] - 22), label, fill=(235, 235, 235), stroke_width=1, stroke_fill=(0, 0, 0))
    if view.kind == "allsky":
        w, h = im.size
        for label, (x, y) in {"N": (w / 2 - 3, 4), "S": (w / 2 - 3, h - 14), "W": (w - 14, h / 2 - 5), "E": (6, h / 2 - 5)}.items():
            dr.text((x, y), label, fill=(220, 220, 130), stroke_width=1, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def make_view(name, scale):
    if name == "allsky":
        n = int(1080 * scale)
        return vw.View("allsky", n, n)
    if name == "north":
        return vw.View("perspective", int(1280 * scale), int(720 * scale), azimuth_deg=0.0, pitch_deg=22.0, fov_deg=110.0)
    raise ValueError(name)


def main():
    global ALLOW_MISSING, DENOISE_SIGMA
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../renders/day/data")
    ap.add_argument("--out", default="../renders/day/frames")
    ap.add_argument("--views", nargs="+", default=["allsky", "north"], choices=["allsky", "north"])
    ap.add_argument("--frames", default=f"0-{rd.N_FRAMES - 1}")
    ap.add_argument("--scale", type=float, default=1.0, help="output size multiplier")
    ap.add_argument("--gamma", type=float, default=0.75, help="eye adaptation exponent (1 = fully adapted)")
    ap.add_argument("--l-ref", type=float, default=30.0, help="cd/m^2 at which exposure is neutral")
    ap.add_argument("--smooth-frames", type=float, default=1.5)
    ap.add_argument("--no-stars", action="store_true", help="omit the Arches stars")
    ap.add_argument("--no-local", action="store_true", help="omit the local field stars")
    ap.add_argument("--earth", action="store_true",
                    help="Earth comparison: natural night-sky background and real bright stars instead of the GC sky")
    ap.add_argument("--earth-sky-sb", type=float, default=21.7, help="Earth night-sky V surface brightness, mag/arcsec^2")
    ap.add_argument("--earth-sky-temp", type=float, default=5000.0, help="spectral shape of the Earth sky glow (blackbody K)")
    ap.add_argument("--earth-vmax", type=float, default=2.0, help="include catalogue stars brighter than this V")
    ap.add_argument("--fixed-la", type=float, default=None,
                    help="use this adaptation luminance (cd/m^2) for every frame, e.g. to compare skies at equal exposure")
    ap.add_argument("--save-la", default=None, help="save the computed per-frame log-La array (.npy) and exit before rendering")
    ap.add_argument("--load-la", default=None, help="load a per-frame log-La array (.npy) from another render, e.g. to give Earth the GC's exposure schedule")
    ap.add_argument("--label", default=None, help="text drawn at the bottom-left of each frame")
    ap.add_argument("--local-mass", type=float, default=1000.0, help="total mass of local field stars, Msun")
    ap.add_argument("--local-seed", type=int, default=7)
    ap.add_argument("--no-companion", action="store_true", help="omit the 1000 au Sun-like companion")
    ap.add_argument("--companion-au", type=float, default=co.SEP_AU)
    ap.add_argument("--companion-incl", type=float, default=co.INCLINATION_DEG, help="inclination to the ecliptic, deg")
    ap.add_argument("--companion-node", type=float, default=co.NODE_DEG, help="longitude of ascending node, deg")
    ap.add_argument("--companion-phase", type=float, default=co.PHASE_DEG, help="orbital phase from the node, deg (circular orbit)")
    ap.add_argument("--denoise", type=float, default=DENOISE_SIGMA, help="smoothing sigma (px at 128-px faces); 0 = off")
    ap.add_argument("--allow-missing", action="store_true", help="treat missing Sun frames as zero (testing)")
    a = ap.parse_args()
    ALLOW_MISSING = a.allow_missing
    DENOISE_SIGMA = a.denoise

    static = load_static(a.data)
    earth = earth_irr = None
    if a.earth:
        # The sky group is linear per wavelength bin, so an Earth-like sky glow is the GC sky
        # (rendered for 17 mag/arcsec^2, 3500 K) rescaled bin by bin. No cluster light.
        w = static["sky"]["wavelengths"]
        ratio = (ph.radiance_from_surface_brightness(a.earth_sky_sb, w, a.earth_sky_temp)
                 / ph.radiance_from_surface_brightness(17.0, w, 3500.0))
        sky, arches = static["sky"], static["arches"]
        static = {"sky": {"faces": sky["faces"] * ratio[:, None, None, None], "wavelengths": w, "bin_width": sky["bin_width"]},
                  "arches": {"faces": np.zeros_like(arches["faces"]), "star_dirs": arches["star_dirs"], "star_irr": arches["star_irr"]}}
        a.no_stars, a.no_local = True, True
    wls, dwl = static["sky"]["wavelengths"], float(static["sky"]["bin_width"])
    M = spectral_matrix(wls, dwl)
    star_dirs, star_irr = static["arches"]["star_dirs"], static["arches"]["star_irr"]
    frames = rd.parse_frames(a.frames)
    os.makedirs(a.out, exist_ok=True)

    if a.earth:
        earth = bs.BrightStars(a.earth_vmax)
        earth_irr = earth.spectral_irradiance(wls)
        print(earth.summary(), flush=True)
    comp_dir = comp_irr = None
    if not a.no_companion:
        comp_dir = co.direction(a.companion_incl, a.companion_node, a.companion_phase)
        comp_irr = co.spectral_irradiance(wls, a.companion_au)
        print(co.summary(a.companion_au, a.companion_incl), flush=True)
    local = local_irr = None
    if not a.no_local:
        # fixed on the celestial sphere: rotated into the horizon frame every frame
        local = cl.LocalStars(total_mass=a.local_mass, seed=a.local_seed)
        local_irr = local.spectral_irradiance(wls)
        print(local.summary(), flush=True)

    # eye adaptation from a low-resolution all-sky pass over the whole day
    log_la = np.zeros(rd.N_FRAMES)
    if a.fixed_la:
        log_la[:] = np.log(a.fixed_la)
    elif a.load_la:
        log_la[:] = np.load(a.load_la)
    else:
        lowres = vw.View("allsky", 96, 96)
        st_low = sum(sample_bins(static[g]["faces"], lowres.dirs, M) for g in ("sky", "arches"))
        sun_cache = {}
        for k in range(rd.N_FRAMES):
            img, H, alt = frame_layers(k, lowres, st_low, a.data, wls, M, None, sun_cache)
            log_la[k] = np.log(log_average(img, lowres.valid))
    print("log-average luminance (cd/m^2) range:", np.exp(log_la.min()), np.exp(log_la.max()), flush=True)
    if a.save_la:
        np.save(a.save_la, log_la)
        print("wrote", a.save_la, flush=True)
        return
    scales, meso = adaptation(log_la, a.gamma, a.l_ref, a.smooth_frames)

    for vname in a.views:
        view = make_view(vname, a.scale)
        st = sum(sample_bins(static[g]["faces"], view.dirs, M) for g in ("sky", "arches"))
        if not a.no_stars:
            st = st + point_layer(view, star_dirs, star_irr, wls, M)
        sun_cache = {}
        for k in frames:
            img, H, alt = frame_layers(k, view, st, a.data, wls, M, None, sun_cache)
            if earth is not None:
                img = img + point_layer(view, earth.direction @ rd.celestial_to_horizon(H).T, earth_irr, wls, M)
            if comp_dir is not None:
                img = img + point_layer(view, (comp_dir @ rd.celestial_to_horizon(H).T)[None], comp_irr, wls, M)
            if local is not None:
                img = img + point_layer(view, local.direction @ rd.celestial_to_horizon(H).T, local_irr, wls, M)
            img = img + sun_disk_layer(view, H, alt, wls, M)
            out = to_srgb8(img, scales[k], meso[k])
            out = np.where(view.valid[..., None], out, 0).astype(np.uint8)
            out = annotate(out, view, H, alt, a.label)
            path = os.path.join(a.out, f"{vname}_{k:04d}.png")
            import imageio.v3 as iio
            iio.imwrite(path, out)
            print(f"[{vname}] frame {k} H={H:.1f} alt={alt:.1f} exposure x{scales[k]:.3g} photopic-fraction {meso[k]:.2f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
