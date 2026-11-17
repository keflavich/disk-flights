"""Stage 3: sunrise-sunset animation, observer at +40 deg latitude on the equinox,
with a synthetic (IMF-sampled) cluster fixed at the celestial pole.

Rendering is linear in the sources, so light groups are rendered separately as
six cube faces per wavelength bin and recombined in compose_day.py:

  static  sky (uniform diffuse background) and cluster: both fixed in horizon
          coordinates, rendered once.
  sun     the Sun (delta light) for a set of frame indices. At the equinox the Sun's
          path is mirror-symmetric about noon, so only hour angles H <= 0 are rendered
          (compose mirrors y -> -y for H > 0). Frames with the Sun below -20 deg
          altitude are skipped (treated as zero).

    python render_day.py static --out ../renders/day/data
    python render_day.py sun --out ../renders/day/data --frames 26-72
"""
import argparse
import os

import numpy as np

_LIBLLVM = os.path.join(os.path.dirname(__file__), "..", "third_party", "libllvm", "libLLVM-18.so")
if os.path.exists(_LIBLLVM) and "DRJIT_LIBLLVM_PATH" not in os.environ:
    os.environ["DRJIT_LIBLLVM_PATH"] = os.path.abspath(_LIBLLVM)

import atmosphere as at  # noqa: E402
import cluster as cl  # noqa: E402
import photometry as ph  # noqa: E402

LAT_DEG = 40.0
N_FRAMES = int(os.environ.get("GC_N_FRAMES", 144))  # frames per solar day, H = -180 .. +180 deg
SUN_MIN_ALT_DEG = -20.0
TWILIGHT_ALT_DEG = 8.0  # below this the sky is smooth but very noisy: render small faces, many samples
SOLAR_CONSTANT = 1361.0
SUN_TEMP = 5772.0


def frame_hour_angle(k, n_frames=N_FRAMES):
    return -180.0 + 360.0 * k / n_frames


def sun_direction(hour_angle_deg, lat_deg=LAT_DEG):
    """Unit vector to the Sun, equinox (declination 0). Axes: x=North, y=West, z=Up."""
    h, phi = np.radians(hour_angle_deg), np.radians(lat_deg)
    return np.array([-np.cos(h) * np.sin(phi), np.sin(h), np.cos(h) * np.cos(phi)])


def celestial_to_horizon(hour_angle_deg, lat_deg=LAT_DEG):
    """Rotation matrix from celestial equatorial axes (x = equinox / Sun at the equinox,
    y = RA 6h, z = north celestial pole) to horizon axes (x=North, y=West, z=Up), when the
    equinox has hour angle `hour_angle_deg` (the Sun's hour angle at the equinox)."""
    phi = np.radians(lat_deg)
    ex = sun_direction(hour_angle_deg, lat_deg)
    ez = np.array([np.cos(phi), 0.0, np.sin(phi)])
    return np.column_stack([ex, np.cross(ez, ex), ez])


def sun_altitude_deg(hour_angle_deg, lat_deg=LAT_DEG):
    return float(np.degrees(np.arcsin(sun_direction(hour_angle_deg, lat_deg)[2])))


def sun_irradiance(wavelength_nm):
    return float(ph.irradiance_bb(SOLAR_CONSTANT, SUN_TEMP, wavelength_nm))


def make_cluster(a):
    center = at.direction_from_altaz(LAT_DEG, 0.0)  # "pole star": altitude = latitude, due north
    return cl.Cluster(center, total_mass=a.cluster_mass, alpha=a.imf_alpha,
                      distance_pc=a.distance_pc, seed=a.seed)


def parse_frames(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def setup(a):
    import mitsuba as mi
    mi.set_variant(a.variant)
    if "SLURM_CPUS_PER_TASK" in os.environ:
        import drjit as dr
        dr.set_thread_count(int(os.environ["SLURM_CPUS_PER_TASK"]))
        print(f"drjit threads: {dr.thread_count()}", flush=True)
    return mi


def render_cube(mi, a, wavelength_nm, seed, spp=None, face=None, **emitters):
    scene = mi.load_dict(at.scene_dict(
        wavelength_nm=float(wavelength_nm), n_shells=a.shells, cube_face_size=face or a.face,
        ground_albedo=a.albedo, sun_delta=True, **emitters))
    return np.stack([np.array(mi.render(scene, sensor=k, spp=spp or a.spp, seed=seed))[..., 0]
                     for k in range(6)]).astype(np.float32)


def run_static(a):
    mi = setup(a)
    wls, dwl = ph.bin_centers(a.bins)
    c = make_cluster(a)
    print(c.summary(), flush=True)
    dirs, irr, _ = c.resolved_and_rest(wls, k=a.stars)
    sky_rad = ph.radiance_from_surface_brightness(a.sky_sb, wls, a.sky_temp)
    cubes = {g: [] for g in a.groups}
    for b, wl in enumerate(wls):
        if "sky" in a.groups:
            print(f"[sky] bin {wl:.1f} nm", flush=True)
            cubes["sky"].append(render_cube(mi, a, wl, 1, solar_irradiance=0.0, sky_radiance=float(sky_rad[b])))
        if "arches" in a.groups:
            print(f"[arches] bin {wl:.1f} nm ({len(dirs)} sources)", flush=True)
            src = [dict(direction=dirs[i], irradiance=float(irr[i, b])) for i in range(len(dirs))]
            cubes["arches"].append(render_cube(mi, a, wl, 2, solar_irradiance=0.0, directional_sources=src))
    os.makedirs(a.out, exist_ok=True)
    star_irr = c.spectral_irradiance(wls)
    for g, cube in cubes.items():
        np.savez(os.path.join(a.out, f"static_{g}.npz"), faces=np.stack(cube), wavelengths=wls, bin_width=dwl,
                 star_dirs=c.direction, star_irr=star_irr, star_mass=c.mass, star_teff=c.teff)
        print("wrote", os.path.join(a.out, f"static_{g}.npz"), flush=True)


def run_sun(a):
    mi = setup(a)
    wls, dwl = ph.bin_centers(a.bins)
    os.makedirs(a.out, exist_ok=True)
    for k in parse_frames(a.frames):
        H = frame_hour_angle(k)
        alt = sun_altitude_deg(H)
        path = os.path.join(a.out, f"sun_{k:03d}.npz")
        if H > 0:
            print(f"frame {k}: H={H:.1f} > 0, mirrored from H<0 in compose; skipped", flush=True)
            continue
        if alt < SUN_MIN_ALT_DEG:
            print(f"frame {k}: sun altitude {alt:.1f} deg below {SUN_MIN_ALT_DEG}; skipped", flush=True)
            continue
        if os.path.exists(path):
            print(f"frame {k}: exists, skipping", flush=True)
            continue
        # twilight frames (Sun near/below the horizon) have very few sunlit scattering
        # paths. The sky is smooth, so trade resolution for samples at equal cost:
        # (face/twilight_face)^2 more samples per pixel, times twilight_spp_factor.
        face, spp = a.face, a.spp
        if alt < TWILIGHT_ALT_DEG:
            face = a.twilight_face
            spp = int(a.spp * a.twilight_spp_factor * (a.face / a.twilight_face) ** 2)
        sd = sun_direction(H)
        el, az = np.degrees(np.arcsin(sd[2])), np.degrees(np.arctan2(sd[1], sd[0]))
        cube = []
        for wl in wls:
            print(f"frame {k} (H={H:.1f}, alt={alt:.1f}) bin {wl:.1f} nm  face {face} spp {spp}", flush=True)
            cube.append(render_cube(mi, a, wl, 10 + k, spp=spp, face=face, solar_irradiance=sun_irradiance(wl),
                                    sun_elevation_deg=float(el), sun_azimuth_deg=float(az)))
        np.savez(path, faces=np.stack(cube), hour_angle=H, altitude=alt, wavelengths=wls, bin_width=dwl)
        print("wrote", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["static", "sun"])
    ap.add_argument("--out", default="../renders/day/data")
    ap.add_argument("--variant", default="llvm_ad_mono")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--face", type=int, default=128, help="cube face size, px")
    ap.add_argument("--spp", type=int, default=256)
    ap.add_argument("--twilight-spp-factor", type=int, default=4, help="extra spp multiplier for twilight frames")
    ap.add_argument("--twilight-face", type=int, default=32, help="cube face size for twilight frames")
    ap.add_argument("--shells", type=int, default=24)
    ap.add_argument("--albedo", type=float, default=0.3)
    ap.add_argument("--frames", default="26-72", help="sun mode: frame indices, e.g. 26-72 or 30,31")
    ap.add_argument("--groups", nargs="+", default=["sky", "arches"], choices=["sky", "arches"])
    ap.add_argument("--stars", type=int, default=300, help="individually resolved cluster sources in the render")
    ap.add_argument("--cluster-mass", type=float, default=1.5e4)
    ap.add_argument("--imf-alpha", type=float, default=1.85)
    ap.add_argument("--distance-pc", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sky-sb", type=float, default=17.0)
    ap.add_argument("--sky-temp", type=float, default=3500.0)
    a = ap.parse_args()
    (run_static if a.mode == "static" else run_sun)(a)


if __name__ == "__main__":
    main()
