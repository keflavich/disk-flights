"""Concentric-shell Rayleigh atmosphere scene builder for Mitsuba 3.

Coordinate convention (keeps float32 precision good near the camera):
camera at world origin, Earth's center at (0, 0, -R_EARTH).

Each shell boundary is a sphere. The medium between shell i and i+1 is
homogeneous, with a Rayleigh scattering coefficient set from the exponential
density profile evaluated at the shell's mid-altitude. This is the approach
described for the day-night-cycle GC rendering project (see README.md):
60-100 shells, twilight geometry comes out right automatically because it's
spherical, not a flat slab.
"""
import numpy as np

R_EARTH = 6.371e6  # m
ATM_HEIGHT = 1.0e5  # m, shells span [0, ATM_HEIGHT]
H_RAYLEIGH = 8.43e3  # m, kT/mg scale height at ~288 K
SIGMA_RAYLEIGH_550NM = 4.53e-27  # cm^2 per molecule at 550 nm (gives zenith tau ~0.097, Bodhaine+1999)
N_GROUND = 2.546e19  # molecules/cm^3 at STP (sea-level number density)


def rayleigh_sigma_t(altitude_m, wavelength_nm=550.0):
    """Rayleigh scattering coefficient (1/m) at given altitude and wavelength.

    Uses the standard sigma ~ lambda^-4 scaling from the 550nm reference cross
    section, times the exponential density profile.
    """
    density_ratio = np.exp(-altitude_m / H_RAYLEIGH)
    n_cm3 = N_GROUND * density_ratio
    sigma_cm2 = SIGMA_RAYLEIGH_550NM * (550.0 / wavelength_nm) ** 4
    sigma_t_per_cm = n_cm3 * sigma_cm2
    return sigma_t_per_cm * 100.0  # 1/cm -> 1/m


def shell_radii(n_shells=64, atm_height=ATM_HEIGHT, r_earth=R_EARTH):
    """Shell boundary radii from ground (r_earth) to top of atmosphere.

    Denser spacing near the ground where density (and gradient) is largest.
    """
    t = np.linspace(0, 1, n_shells + 1)
    altitudes = atm_height * (1 - (1 - t) ** 2)  # quadratic: dense near t=0 (ground)
    return r_earth + altitudes, altitudes


def build_atmosphere_media(n_shells=64, wavelength_nm=550.0):
    """Return list of (inner_radius, outer_radius, sigma_t) for each shell,
    evaluated at the shell's mid-altitude, plus the ground radius.
    """
    radii, altitudes = shell_radii(n_shells)
    shells = []
    for i in range(n_shells):
        r_in, r_out = radii[i], radii[i + 1]
        mid_alt = 0.5 * (altitudes[i] + altitudes[i + 1])
        sigma_t = rayleigh_sigma_t(mid_alt, wavelength_nm)
        shells.append(dict(r_in=r_in, r_out=r_out, sigma_t=sigma_t))
    return shells


def direction_from_altaz(elevation_deg, azimuth_deg):
    el, az = np.radians(elevation_deg), np.radians(azimuth_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def scene_dict(sun_elevation_deg=30.0, sun_azimuth_deg=0.0,
                wavelength_nm=550.0, n_shells=8,
                resolution=(512, 288), ground_albedo=0.3,
                solar_irradiance=1.0, extra_sources=(), sky_radiance=0.0,
                camera_pitch_deg=0.0, fov_deg=100.0,
                directional_sources=(), sun_delta=False, cube_face_size=None):
    """Build a Mitsuba 3 scene dict: ground + shelled Rayleigh atmosphere +
    emitters. Single wavelength bin (call once per bin and recombine).

    Emitters (each optional, so a "light group" can be rendered alone):
      solar_irradiance  Sun at (sun_elevation_deg, sun_azimuth_deg); 0 omits it
      extra_sources     iterable of dict(elevation_deg, azimuth_deg, irradiance,
                        angular_radius_deg) for small-disk sources (e.g. Arches)
      sky_radiance      uniform sky radiance from all directions; 0 omits it
      directional_sources  iterable of dict(direction=[x,y,z] unit vector towards the
                        source, irradiance); delta lights, invisible to the camera
      sun_delta         use a delta light for the Sun instead of a disk
    cube_face_size: if set, build six square perspective sensors (fov 92 deg, +x,+y,-x,
    -y,+z,-z; use mi.render(scene, sensor=k)) for resampling into any all-sky view.
    Irradiances are top-of-atmosphere, perpendicular to the source direction.
    """
    center = np.array([0.0, 0.0, -R_EARTH])
    shells = build_atmosphere_media(n_shells, wavelength_nm)

    sun_dir = direction_from_altaz(sun_elevation_deg, sun_azimuth_deg)
    pitch = np.radians(camera_pitch_deg)
    look = [np.cos(pitch), 0.0, np.sin(pitch)]

    def sensor_dict(to_world, fov, res):
        d = {
            "type": "perspective",
            "fov": fov,
            # default far_clip is 1e4 m, which truncates camera rays inside the
            # atmosphere (boundary is ~1e5 m away) so they never reach the sky
            "far_clip": 1e10,
            "to_world": to_world,
            "film": {
                "type": "hdrfilm",
                "width": res[0],
                "height": res[1],
                "pixel_format": "luminance",
                "rfilter": {"type": "gaussian"},
            },
            "sampler": {"type": "independent", "sample_count": 64},
        }
        if shells:
            # camera starts inside the innermost shell; without this the ray
            # starts in vacuum and never sees any medium. n_shells=0 (no
            # atmosphere) has no medium to reference -- vacuum is the default.
            d["medium"] = {"type": "ref", "id": "atm_medium_0"}
        return d

    eye = [0, 0, 1.7]  # eye height, m, above ground sphere apex
    scene = {
        "type": "scene",
        "integrator": {"type": "volpath", "max_depth": 12},
        **({f"sensor_{k}": sensor_dict(mi_lookat(eye, [eye[i] + f[i] for i in range(3)], u),
                                       CUBE_FOV_DEG, (cube_face_size, cube_face_size))
            for k, (f, u) in enumerate(CUBE_FACES)} if cube_face_size else
           {"sensor": sensor_dict(mi_lookat(eye, [look[0], look[1], 1.7 + look[2]], [0, 0, 1]),
                                  fov_deg, resolution)}),
        "ground": {
            "type": "sphere",
            "center": center.tolist(),
            "radius": R_EARTH,
            "bsdf": {"type": "diffuse", "reflectance": {"type": "uniform", "value": ground_albedo}},
        },
    }
    source_distance = 2 * ATM_HEIGHT + R_EARTH  # just outside the outermost shell
    if solar_irradiance > 0 and sun_delta:
        scene["sun"] = {"type": "directional", "direction": (-sun_dir).tolist(),
                        "irradiance": {"type": "uniform", "value": float(solar_irradiance)}}
    elif solar_irradiance > 0:
        scene["sun"] = sun_area_light(sun_dir, solar_irradiance, distance=source_distance)
    if directional_sources:
        wmax = max(src["irradiance"] for src in directional_sources)
        for k, src in enumerate(directional_sources):
            if src["irradiance"] <= 0:
                continue
            scene[f"dsrc_{k}"] = {
                "type": "directional",
                "direction": (-np.asarray(src["direction"], float)).tolist(),
                "irradiance": {"type": "uniform", "value": float(src["irradiance"])},
                # NEE picks emitters uniformly by default; weight by power instead
                "sampling_weight": float(src["irradiance"] / wmax),
            }
    for k, src in enumerate(extra_sources):
        if src["irradiance"] <= 0:
            continue
        scene[f"source_{k}"] = sun_area_light(
            direction_from_altaz(src["elevation_deg"], src["azimuth_deg"]),
            src["irradiance"], distance=source_distance,
            angular_radius_rad=np.radians(src["angular_radius_deg"]))
    if sky_radiance > 0:
        scene["sky"] = {"type": "constant", "radiance": {"type": "uniform", "value": float(sky_radiance)}}

    # innermost shell first (touches ground), then nested outward
    for i, sh in enumerate(shells):
        med_key = f"atm_medium_{i}"
        shape_key = f"atm_shell_{i}"
        scene[med_key] = {
            "type": "homogeneous",
            "sigma_t": {"type": "uniform", "value": sh["sigma_t"]},
            "albedo": {"type": "uniform", "value": 1.0},  # pure scattering, no absorption
            "phase": {"type": "rayleigh"},
        }
        scene[shape_key] = {
            "type": "sphere",
            "center": center.tolist(),
            "radius": sh["r_out"],
            "bsdf": {"type": "null"},
            "interior": {"type": "ref", "id": med_key},
            # crossing outward must enter the next shell's medium; a missing
            # exterior means vacuum, which silently drops every shell but the
            # innermost. The outermost shell has no exterior (vacuum).
            **({"exterior": {"type": "ref", "id": f"atm_medium_{i + 1}"}}
               if i + 1 < len(shells) else {}),
        }

    return scene


CUBE_FOV_DEG = 92.0
# (forward, up) per face; scene axes are x=North, y=West, z=Up
CUBE_FACES = [([1, 0, 0], [0, 0, 1]), ([0, 1, 0], [0, 0, 1]), ([-1, 0, 0], [0, 0, 1]),
              ([0, -1, 0], [0, 0, 1]), ([0, 0, 1], [1, 0, 0]), ([0, 0, -1], [1, 0, 0])]

SUN_ANGULAR_RADIUS_RAD = np.radians(0.2664)  # mean apparent solar radius from Earth


def sun_area_light(direction_unit, irradiance, distance, angular_radius_rad=SUN_ANGULAR_RADIUS_RAD):
    """Small emitting sphere standing in for the Sun (or any star).

    A finite-size emitter gives the Sun its real ~0.53 deg disk and penumbrae.
    Delta lights (point/directional) also work through media provided the
    sensor declares its initial medium; they are the cheaper option for
    unresolved stars.

    Radiance is distance-invariant for a fixed angular size, so `distance` can
    be any convenient placement value (chosen elsewhere to sit just outside
    the outermost atmosphere shell) without needing the true ~1 AU.
    """
    solid_angle = np.pi * angular_radius_rad ** 2  # small-angle disk approx
    radiance = irradiance / solid_angle
    radius = distance * np.tan(angular_radius_rad)
    center = (np.asarray(direction_unit) * distance).tolist()
    return {
        "type": "sphere",
        "center": center,
        "radius": radius,
        "emitter": {"type": "area", "radiance": {"type": "uniform", "value": float(radiance)}},
        "bsdf": {"type": "diffuse", "reflectance": {"type": "uniform", "value": 0.0}},
    }


def mi_lookat(origin, target, up):
    import mitsuba as mi
    return mi.ScalarTransform4f().look_at(origin=origin, target=target, up=up)
