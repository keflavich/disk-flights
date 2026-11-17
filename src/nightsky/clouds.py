"""Config-driven cloud VoxelFields placed around the observer (dark clouds, HII regions).

Each entry of cfg['galaxy']['extra_fields'] builds one fields.VoxelField in the GALACTOCENTRIC
frame (pc), centred on the observer plus `offset_pc`, sampled on a cube of side
2 * extent * radius_pc with `n_vox` voxels per side:

  {kind: extinction, radius_pc: 3.0, av_center: 20.0, profile: gauss, rv: 3.1, offset_pc: [0, 0, 0]}
      data = A_V per pc. av_center is the A_V from the cloud centre to infinity along any ray
      (the observer at the centre of a gauss cloud sees A_V = av_center in every direction).
  {kind: emission, radius_pc: 2.0, n_e: 300.0, t_e: 8000.0, template: hii_gc, profile: uniform}
      data = H-alpha emissivity per pc (case B, gas.halpha_coeff) for peak density n_e (cm^-3);
      spectra = lines + nebular continuum per unit H-alpha (gas.emission_spectrum).

Profiles in q = r / radius_pc: gauss exp(-q^2/2) (extent 4), uniform 1 for q < 1 with a
half-voxel soft edge (extent 1.1), shell exp(-(q - 1)^2 / (2 * 0.15^2)) (extent 1.6).
The ray marcher samples fields trilinearly, so n_vox ~ 64-96 resolves these smooth shapes.
"""
import numpy as np
from scipy.special import erf

from . import fields, frames, gas

EXTENT = {"gauss": 4.0, "uniform": 1.1, "shell": 1.6}


def _profile(q, profile, dq):
    if profile == "gauss":
        return np.exp(-0.5 * q * q)
    if profile == "uniform":
        return 0.5 * (1 - erf((q - 1) / max(dq, 1e-6)))
    if profile == "shell":
        return np.exp(-0.5 * ((q - 1) / 0.15) ** 2)
    raise ValueError(f"unknown cloud profile {profile!r}")


def _radial_integral(profile, dq):
    """Integral of the profile from q = 0 to infinity (path from the centre outward, units of radius)."""
    q = np.linspace(0, EXTENT[profile] * 1.5, 20001)
    return float(np.trapezoid(_profile(q, profile, dq), q))


def cloud_field(spec, grid, observer_pc):
    spec = dict(spec)
    kind, profile = spec.get("kind", "extinction"), spec.get("profile", "gauss")
    r, n = float(spec.get("radius_pc", 1.0)), int(spec.get("n_vox", 80))
    centre = np.asarray(observer_pc, float) + np.asarray(spec.get("offset_pc", (0.0, 0.0, 0.0)), float)
    half = EXTENT[profile] * r
    vox = 2 * half / (n - 1)
    ax = np.linspace(-half, half, n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    q = np.sqrt(X * X + Y * Y + Z * Z) / r
    shape = _profile(q, profile, 0.5 * vox / r)
    if kind == "extinction":
        peak = float(spec.get("av_center", 5.0)) / (r * _radial_integral(profile, 0.5 * vox / r))
        data, spectra = peak * shape, gas.extinction_spectrum(grid, spec.get("rv", 3.1))[None]
    elif kind == "emission":
        t_e = float(spec.get("t_e", 8000.0))
        data = gas.halpha_coeff(t_e) * float(spec.get("n_e", 100.0)) ** 2 * shape ** 2
        spectra = gas.emission_spectrum(grid, spec.get("template", "hii_gc"))[None]
    else:
        raise ValueError(f"extra_fields kind must be extinction or emission, got {kind!r}")
    return fields.VoxelField(centre - half, vox, data[..., None].astype(np.float32), spectra, kind,
                             spec.get("name", f"{kind}_{profile}_{r:g}pc"), meta=spec)


def extra_fields(specs, grid, observer_pc):
    return [cloud_field(s, grid, observer_pc) for s in (specs or [])]
