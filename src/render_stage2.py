"""Stage 2 (see README.md build order): toy Arches.

One point-like source (V ~ -13.5, 35,000 K blackbody) plus a uniform orange sky
(17 mag/arcsec^2, V band), no Sun. Each light group is rendered alone per
wavelength bin (rendering is linear in the sources), stored in one .npz, and
recombined in numpy -- so re-weighting a group is a compose-only operation.

    python render_stage2.py --out ../renders/stage2_toy_arches/toy          # render + compose
    python render_stage2.py --out ../renders/stage2_toy_arches/toy --compose-only --sky-scale 0
"""
import argparse
import json
import os

import numpy as np

_LIBLLVM = os.path.join(os.path.dirname(__file__), "..", "third_party", "libllvm", "libLLVM-18.so")
if os.path.exists(_LIBLLVM) and "DRJIT_LIBLLVM_PATH" not in os.environ:
    os.environ["DRJIT_LIBLLVM_PATH"] = os.path.abspath(_LIBLLVM)

import photometry as ph  # noqa: E402

GROUPS = ("arches", "sky")


def group_kwargs(group, wl, a):
    """scene_dict emitter arguments for one light group at wavelength wl (nm)."""
    if group == "arches":
        return dict(
            solar_irradiance=0.0,
            extra_sources=[dict(
                elevation_deg=a.arches_el, azimuth_deg=a.arches_az,
                angular_radius_deg=a.arches_radius_deg,
                irradiance=float(ph.irradiance_from_vmag(a.vmag, wl, a.arches_temp)))])
    if group == "sky":
        return dict(
            solar_irradiance=0.0,
            sky_radiance=float(ph.radiance_from_surface_brightness(a.sky_sb, wl, a.sky_temp)))
    raise ValueError(group)


def render_groups(a):
    import mitsuba as mi
    mi.set_variant(a.variant)
    import atmosphere
    if "SLURM_CPUS_PER_TASK" in os.environ:
        import drjit as dr
        dr.set_thread_count(int(os.environ["SLURM_CPUS_PER_TASK"]))
        print(f"drjit threads: {dr.thread_count()}", flush=True)

    wls, dwl = ph.bin_centers(a.bins)
    cubes = {}
    for gi, group in enumerate(a.groups):
        cube = []
        for wl in wls:
            print(f"[{group}] bin {wl:.1f} nm", flush=True)
            scene = mi.load_dict(atmosphere.scene_dict(
                wavelength_nm=float(wl), n_shells=a.shells,
                resolution=(a.width, a.height), ground_albedo=a.albedo,
                camera_pitch_deg=a.pitch, fov_deg=a.fov, **group_kwargs(group, wl, a)))
            img = mi.render(scene, spp=a.spp, seed=1 + gi)
            cube.append(np.array(img)[..., 0])
        cubes[group] = np.stack(cube)
    np.savez(a.out + "_groups.npz", wavelengths=wls, bin_width=dwl,
             params=json.dumps(vars(a)), **cubes)
    return a.out + "_groups.npz"


def tonemap(xyz_lin_srgb, luminance, key=0.18, white_percentile=99.5):
    """Global photographic tone map on luminance (hue-preserving), then sRGB gamma."""
    log_avg = np.exp(np.mean(np.log(luminance + 1e-9)))
    s = key / log_avg
    ls = luminance * s
    lw = max(np.percentile(ls, white_percentile), 1e-9)
    ld = ls * (1 + ls / lw ** 2) / (1 + ls)
    rgb = xyz_lin_srgb * (ld / np.maximum(luminance, 1e-12))[..., None]
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def compose(npz_path, scales, out_png):
    d = np.load(npz_path)
    xyz = 0
    for group, scale in scales.items():
        if group in d.files and scale != 0:
            xyz = xyz + scale * ph.bins_to_xyz(d[group], d["wavelengths"], float(d["bin_width"]))
    lum = xyz[..., 1]
    print(f"luminance (cd/m^2): min {lum.min():.3e}  median {np.median(lum):.3e}  max {lum.max():.3e}")
    srgb = tonemap(ph.xyz_to_linear_srgb(xyz), lum)
    import imageio.v3 as iio
    iio.imwrite(out_png, (srgb * 255 + 0.5).astype(np.uint8))
    print(f"wrote {out_png}")
    return xyz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../renders/stage2_toy_arches/toy")
    ap.add_argument("--variant", default="llvm_ad_mono")
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=GROUPS)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--height", type=int, default=108)
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--shells", type=int, default=8)
    ap.add_argument("--albedo", type=float, default=0.3)
    ap.add_argument("--pitch", type=float, default=15.0, help="camera pitch, deg")
    ap.add_argument("--fov", type=float, default=100.0)
    ap.add_argument("--vmag", type=float, default=-13.5, help="apparent V of the Arches source")
    ap.add_argument("--arches-temp", type=float, default=35000.0)
    ap.add_argument("--arches-el", type=float, default=20.0)
    ap.add_argument("--arches-az", type=float, default=10.0)
    ap.add_argument("--arches-radius-deg", type=float, default=0.1)
    ap.add_argument("--sky-sb", type=float, default=17.0, help="sky V surface brightness, mag/arcsec^2")
    ap.add_argument("--sky-temp", type=float, default=3500.0, help="sky spectral shape (blackbody K)")
    ap.add_argument("--compose-only", action="store_true")
    ap.add_argument("--arches-scale", type=float, default=1.0)
    ap.add_argument("--sky-scale", type=float, default=1.0)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    npz = a.out + "_groups.npz"
    if not a.compose_only:
        npz = render_groups(a)
    tag = "" if (a.arches_scale, a.sky_scale) == (1.0, 1.0) else f"_a{a.arches_scale:g}_s{a.sky_scale:g}"
    compose(npz, {"arches": a.arches_scale, "sky": a.sky_scale}, a.out + tag + ".png")


if __name__ == "__main__":
    main()
