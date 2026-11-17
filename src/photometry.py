"""Photometric helpers: magnitudes -> physical spectral flux, blackbody spectra,
and wavelength-bin -> sRGB conversion.

Units throughout: irradiance W m^-2 nm^-1, radiance W m^-2 sr^-1 nm^-1.
"""
import numpy as np

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23

# Vega-system V=0 monochromatic flux density at 550 nm: 3.63e-9 erg/s/cm^2/A
# = 3.63e-11 W/m^2/nm.
F_V0_550NM = 3.63e-11
ARCSEC2_IN_SR = (np.pi / 180 / 3600) ** 2
# Luminous efficacy at 555 nm (photopic) and V=0 illuminance
LM_PER_W = 683.0


def planck_shape(wavelength_nm, temperature_k):
    """B_lambda(T) normalized to 1 at 550 nm."""
    def b(lam_nm):
        lam = np.asarray(lam_nm) * 1e-9
        return 1.0 / (lam ** 5 * np.expm1(H * C / (lam * KB * temperature_k)))
    return b(wavelength_nm) / b(550.0)


def irradiance_from_vmag(vmag, wavelength_nm, temperature_k):
    """Irradiance (W/m^2/nm) of a blackbody-shaped source with apparent V mag."""
    return F_V0_550NM * 10 ** (-0.4 * vmag) * planck_shape(wavelength_nm, temperature_k)


def radiance_from_surface_brightness(mag_per_arcsec2, wavelength_nm, temperature_k):
    """Uniform-sky radiance (W/m^2/sr/nm) for a V-band surface brightness."""
    return irradiance_from_vmag(mag_per_arcsec2, wavelength_nm, temperature_k) / ARCSEC2_IN_SR


def bin_centers(n_bins, lo=400.0, hi=700.0):
    edges = np.linspace(lo, hi, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:]), (hi - lo) / n_bins


def _g(x, mu, s1, s2):
    return np.exp(-0.5 * ((x - mu) / np.where(x < mu, s1, s2)) ** 2)


def cie_xyz_cmf(wavelength_nm):
    """CIE 1931 2-degree CMFs, multi-lobe Gaussian fit (Wyman, Sloan & Shirley 2013)."""
    w = np.asarray(wavelength_nm, dtype=float)
    x = (1.056 * _g(w, 599.8, 37.9, 31.0) + 0.362 * _g(w, 442.0, 16.0, 26.7)
         - 0.065 * _g(w, 501.1, 20.4, 26.2))
    y = 0.821 * _g(w, 568.8, 46.9, 40.5) + 0.286 * _g(w, 530.9, 16.3, 31.1)
    z = 1.217 * _g(w, 437.0, 11.8, 36.0) + 0.681 * _g(w, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


XYZ_TO_SRGB = np.array([[3.2406, -1.5372, -0.4986],
                        [-0.9689, 1.8758, 0.0415],
                        [0.0557, -0.2040, 1.0570]])


def bins_to_xyz(spectral_cube, wavelengths_nm, bin_width_nm):
    """spectral_cube: (n_bins, H, W) radiance per nm -> XYZ (H, W, 3), with Y in cd/m^2."""
    cmf = cie_xyz_cmf(wavelengths_nm)  # (n_bins, 3)
    return LM_PER_W * bin_width_nm * np.einsum("bhw,bc->hwc", spectral_cube, cmf)


def xyz_to_linear_srgb(xyz):
    return np.einsum("ij,hwj->hwi", XYZ_TO_SRGB, xyz)


SIGMA_SB = 5.670374e-8


def irradiance_bb(f_bol, temperature_k, wavelength_nm):
    """Spectral irradiance (W/m^2/nm) of a blackbody-shaped source with bolometric flux f_bol (W/m^2).

    E_lambda = f_bol * pi * B_lambda(T) / (sigma T^4); broadcasts over f_bol / T.
    """
    lam = np.asarray(wavelength_nm, float) * 1e-9
    t = np.asarray(temperature_k, float)
    b = 2 * H * C ** 2 / lam ** 5 / np.expm1(H * C / (lam * KB * t))  # W/m^2/sr/m
    return f_bol * np.pi * b / (SIGMA_SB * t ** 4) * 1e-9
