"""Shared spectral grid and colour/luminance conversions for the nightsky package.

Units: radiance W m^-2 sr^-1 nm^-1, irradiance W m^-2 nm^-1. Every per-bin array is a
bin AVERAGE over the bin (not the value at the bin centre), so sums of value * dwl are
band integrals and line emission can be deposited as flux / dwl into one bin.
"""
from dataclasses import dataclass

import numpy as np

import photometry as ph

LM_PER_W = 683.0  # photopic
LM_PER_W_SCOTOPIC = 1700.0

# CIE 1951 scotopic luminous efficiency V'(lambda), 380-780 nm in 10 nm steps
_VPRIME_WL = np.arange(380, 790, 10, dtype=float)
_VPRIME = np.array([
    5.890e-4, 2.209e-3, 9.290e-3, 3.484e-2, 9.660e-2, 1.998e-1, 3.281e-1, 4.550e-1, 5.670e-1,
    6.760e-1, 7.930e-1, 9.040e-1, 9.820e-1, 9.970e-1, 9.350e-1, 8.110e-1, 6.500e-1, 4.810e-1,
    3.288e-1, 2.076e-1, 1.212e-1, 6.550e-2, 3.315e-2, 1.593e-2, 7.370e-3, 3.335e-3, 1.497e-3,
    6.770e-4, 3.129e-4, 1.480e-4, 7.150e-5, 3.533e-5, 1.780e-5, 9.140e-6, 4.780e-6, 2.546e-6,
    1.379e-6, 7.600e-7, 4.250e-7, 2.410e-7, 1.390e-7])


@dataclass(frozen=True)
class SpectralGrid:
    edges: np.ndarray  # (n+1,) nm

    @property
    def wl(self):
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def dwl(self):
        return np.diff(self.edges)

    @property
    def n(self):
        return len(self.edges) - 1

    def bin_index(self, wavelength_nm):
        """Index of the bin containing wavelength_nm, or -1 if outside the grid."""
        i = int(np.searchsorted(self.edges, wavelength_nm, side="right")) - 1
        return i if 0 <= i < self.n else -1

    def subsample(self, per_bin=21):
        """(n, per_bin) wavelengths spanning each bin, for bin averages."""
        f = (np.arange(per_bin) + 0.5) / per_bin
        return self.edges[:-1, None] + f[None, :] * self.dwl[:, None]

    def bin_average(self, fn, per_bin=21):
        """Bin-average a function of wavelength (nm) that broadcasts; returns (..., n)."""
        sub = self.subsample(per_bin)  # (n, k)
        v = fn(sub)
        return np.asarray(v).mean(axis=-1)

    def line(self, wavelength_nm, flux):
        """Spectrum (..., n) of a line with integrated flux `flux` (W/m^2[/sr]) at wavelength_nm."""
        flux = np.asarray(flux, float)
        out = np.zeros(flux.shape + (self.n,))
        i = self.bin_index(wavelength_nm)
        if i >= 0:
            out[..., i] = flux / self.dwl[i]
        return out


def default_grid(n=20, lo=380.0, hi=780.0):
    return SpectralGrid(np.linspace(lo, hi, n + 1))


def cmf_bins(grid):
    """(n, 3) bin-averaged CIE 1931 2-deg x, y, z colour matching functions."""
    sub = grid.subsample()
    return ph.cie_xyz_cmf(sub).mean(axis=1)


def vprime_bins(grid):
    """(n,) bin-averaged scotopic V'(lambda) (log-linear interpolation of the CIE table)."""
    sub = grid.subsample()
    v = np.exp(np.interp(sub, _VPRIME_WL, np.log(_VPRIME), left=np.log(_VPRIME[0]) - 5, right=np.log(_VPRIME[-1]) - 5))
    return v.mean(axis=1)


def to_xyz(spec, grid):
    """(..., n) spectral radiance -> (..., 3) CIE XYZ with Y = luminance in cd/m^2.
    For spectral irradiance the result is illuminance-like (Y in lux)."""
    w = LM_PER_W * cmf_bins(grid) * grid.dwl[:, None]
    return np.asarray(spec) @ w


def luminance(spec, grid):
    return to_xyz(spec, grid)[..., 1]


def scotopic_luminance(spec, grid):
    """(..., n) spectral radiance -> scotopic luminance (scotopic cd/m^2)."""
    return np.asarray(spec) @ (LM_PER_W_SCOTOPIC * vprime_bins(grid) * grid.dwl)


def blackbody_irradiance(f_bol, temperature_k, grid):
    """(..., n) bin-averaged irradiance of blackbody sources with bolometric flux f_bol (W/m^2)."""
    f = np.asarray(f_bol, float)[..., None, None]
    t = np.asarray(temperature_k, float)[..., None, None]
    return ph.irradiance_bb(f, t, grid.subsample()).mean(axis=-1)


def vmag_irradiance(vmag, temperature_k, grid):
    """(..., n) irradiance of blackbody-shaped sources normalised to V magnitude (550 nm, Vega)."""
    v = np.asarray(vmag, float)[..., None, None]
    t = np.asarray(temperature_k, float)[..., None, None]
    shape = ph.planck_shape(grid.subsample(), t)
    return (ph.F_V0_550NM * 10 ** (-0.4 * v) * shape).mean(axis=-1)


def vmag_of(irr, grid):
    """Approximate V magnitude from spectral irradiance (..., n): flux at 550 nm, Vega zero point."""
    e550 = np.array([np.interp(550.0, grid.wl, row) for row in np.atleast_2d(irr)])
    out = -2.5 * np.log10(np.maximum(e550, 1e-300) / ph.F_V0_550NM)
    return out if np.ndim(irr) > 1 else float(out[0])


def surface_brightness_radiance(mag_arcsec2, temperature_k, grid):
    """(n,) radiance of a uniform surface brightness (V mag/arcsec^2) with a blackbody shape."""
    return vmag_irradiance(mag_arcsec2, temperature_k, grid) / ph.ARCSEC2_IN_SR


def radiance_to_mag_arcsec2(spec, grid):
    """V-band surface brightness (mag/arcsec^2) of spectral radiance (..., n), via 550 nm."""
    return vmag_of(np.asarray(spec) * ph.ARCSEC2_IN_SR, grid)
