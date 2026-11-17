"""A Sun-like binary companion at wide separation.

At 1000 au around 2 Msun total, the orbital period is ~22,000 yr (Kepler),
so over one day the companion is effectively fixed on the celestial sphere,
like the local field stars -- rotate with render_day.celestial_to_horizon
each frame.

Free parameters not specified by the "1000 au, 25 deg to the ecliptic" request:
the orbit is assumed circular, and the ascending node and orbital phase are
arbitrary choices (defaults below). Change via the CLI flags in compose_day.py.
"""
import numpy as np

import photometry as ph
import render_day as rd

SEP_AU = 1000.0
INCLINATION_DEG = 25.0
NODE_DEG = 0.0  # longitude of ascending node, measured from the equinox along the ecliptic
PHASE_DEG = 90.0  # position in the orbit, measured from the ascending node (circular orbit assumed)
OBLIQUITY_DEG = 23.44  # Earth's ecliptic obliquity, for ecliptic -> equatorial (celestial) conversion


def direction(inclination_deg=INCLINATION_DEG, node_deg=NODE_DEG, phase_deg=PHASE_DEG,
              obliquity_deg=OBLIQUITY_DEG):
    """Unit vector in the celestial equatorial frame (x=equinox, z=north celestial pole)."""
    i, om, th, eps = np.radians([inclination_deg, node_deg, phase_deg, obliquity_deg])
    # position in the orbital plane, then tilt by inclination about the ascending-node line
    x1, y1 = np.cos(th), np.sin(th)
    x2, y2, z2 = x1, y1 * np.cos(i), y1 * np.sin(i)
    # rotate the ascending node to its ecliptic longitude
    x_ecl = x2 * np.cos(om) - y2 * np.sin(om)
    y_ecl = x2 * np.sin(om) + y2 * np.cos(om)
    z_ecl = z2
    # ecliptic -> equatorial: rotate about the shared x-axis (the equinox) by the obliquity
    x_eq = x_ecl
    y_eq = y_ecl * np.cos(eps) - z_ecl * np.sin(eps)
    z_eq = y_ecl * np.sin(eps) + z_ecl * np.cos(eps)
    return np.array([x_eq, y_eq, z_eq])


def spectral_irradiance(wavelengths_nm, sep_au=SEP_AU):
    """(1, n_bins) W/m^2/nm, Sun-like spectrum scaled from 1 au by inverse-square law."""
    e1au = np.array([rd.sun_irradiance(w) for w in wavelengths_nm])
    return (e1au * (1.0 / sep_au) ** 2)[None, :]


def vmag(sep_au=SEP_AU):
    e = float(ph.irradiance_bb(1361.0, rd.SUN_TEMP, 550.0)) * (1.0 / sep_au) ** 2
    return -2.5 * np.log10(e / ph.F_V0_550NM)


def summary(sep_au=SEP_AU, inclination_deg=INCLINATION_DEG):
    return (f"Sun-like companion at {sep_au:.0f} au, i={inclination_deg:.0f} deg to the ecliptic: "
            f"V={vmag(sep_au):.2f}, period {(sep_au ** 3 / 2.0) ** 0.5:.0f} yr (circular, 2 Msun total)")
