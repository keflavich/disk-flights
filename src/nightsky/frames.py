"""Coordinate-frame conventions shared by every nightsky module.

HORIZON frame (observer's local): x = North, y = West, z = Up (right-handed). Same as the
Mitsuba scenes in atmosphere.py and views.py. Azimuth is measured from North through EAST
(standard astronomical convention), so az=90 deg is East = -y.

CELESTIAL frame (the observer planet's equatorial frame): x = vernal-equinox direction
(where the Sun sits on the equinox), y = RA 6h, z = north celestial pole. All sky sources
(TOASky, PointSources) are stored in this frame; rotate with celestial_to_horizon per time.

Time: one angle H = hour angle of the equinox direction (deg). On the equinox the Sun is at
celestial x, so H is also the Sun's hour angle and local solar time = 12 h + H / 15.

GALACTOCENTRIC frame (for 3D galaxy models): pc, Sgr A* at the origin, x points FROM Sgr A*
TOWARDS the Sun (the Sun is at x = +R0), y towards Galactic longitude l = 90 deg as seen from
the Sun, z towards the North Galactic Pole. These axes are LEFT-handed (x = -X_galactic);
dot products are unaffected, but cross products of components carry an extra minus sign.
So a source at (l, b, distance d from the Sun) sits at
    (R0 - d cos b cos l,  d cos b sin l,  d sin b).
"""
import numpy as np

R0_PC = 8178.0  # Sun - Sgr A* distance (GRAVITY 2019)


def altaz_to_vec(alt_deg, az_deg):
    """Horizon-frame unit vector(s); az from North through East."""
    a, z = np.radians(alt_deg), np.radians(az_deg)
    return np.stack(np.broadcast_arrays(np.cos(a) * np.cos(z), -np.cos(a) * np.sin(z), np.sin(a)), axis=-1)


def vec_to_altaz(v):
    v = np.asarray(v, float)
    alt = np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
    az = np.degrees(np.arctan2(-v[..., 1], v[..., 0])) % 360.0
    return alt, az


def equator_vec(hour_angle_deg, lat_deg):
    """Horizon-frame direction of a declination-0 point with the given hour angle."""
    h, phi = np.radians(hour_angle_deg), np.radians(lat_deg)
    return np.array([-np.cos(h) * np.sin(phi), np.sin(h), np.cos(h) * np.cos(phi)])


def celestial_to_horizon(hour_angle_deg, lat_deg):
    """3x3 R with v_horizon = R @ v_celestial (identical to render_day.celestial_to_horizon)."""
    phi = np.radians(lat_deg)
    ex = equator_vec(hour_angle_deg, lat_deg)
    ez = np.array([np.cos(phi), 0.0, np.sin(phi)])
    return np.column_stack([ex, np.cross(ez, ex), ez])


def sun_celestial(ecliptic_longitude_deg=0.0, obliquity_deg=23.44):
    """Sun direction in the celestial frame; longitude 0 = the equinox (celestial x)."""
    lam, eps = np.radians(ecliptic_longitude_deg), np.radians(obliquity_deg)
    return np.array([np.cos(lam), np.sin(lam) * np.cos(eps), np.sin(lam) * np.sin(eps)])


def lb_to_galactocentric(l_deg, b_deg, d_pc, r0_pc=R0_PC):
    l, b = np.radians(l_deg), np.radians(b_deg)
    return np.stack(np.broadcast_arrays(r0_pc - d_pc * np.cos(b) * np.cos(l),
                                        d_pc * np.cos(b) * np.sin(l), d_pc * np.sin(b)), axis=-1)


def galactocentric_to_celestial(observer_pc, pole_target_pc, reference_pc=(0.0, 0.0, 0.0), roll_deg=0.0):
    """3x3 R with v_celestial = R @ v_galactocentric (directions).

    The planet's north celestial pole points from the observer at `pole_target_pc` (e.g. the
    Arches). The equinox (celestial x) is the direction to `reference_pc` (default Sgr A*)
    projected onto the celestial equator, then rotated by roll_deg about the pole.
    """
    o = np.asarray(observer_pc, float)
    z = np.asarray(pole_target_pc, float) - o
    z /= np.linalg.norm(z)
    r = np.asarray(reference_pc, float) - o
    x = r - z * (r @ z)
    if np.linalg.norm(x) < 1e-9:
        x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    # the galactocentric axes are LEFT-handed (x towards the Sun = -X_galactic), so the
    # component cross product flips sign: this y completes a right-handed celestial frame
    y = np.cross(x, z)
    c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    x, y = c * x + s * y, -s * x + c * y
    return np.vstack([x, y, z])


def healpix_dirs(nside, nest=False):
    import healpy as hp
    return np.stack(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)), nest=nest), axis=-1)


def sample_healpix(m, dirs):
    """Bilinear HEALPix (RING) interpolation. m: (npix,) or (npix, n); dirs (..., 3)."""
    import healpy as hp
    nside = hp.npix2nside(m.shape[0])
    d = np.asarray(dirs, float).reshape(-1, 3)
    theta = np.arccos(np.clip(d[:, 2], -1, 1))
    phi = np.arctan2(d[:, 1], d[:, 0])
    pix, w = hp.get_interp_weights(nside, theta, phi)
    out = np.einsum("kn,kn...->n...", w, m[pix])
    return out.reshape(np.shape(dirs)[:-1] + m.shape[1:])
