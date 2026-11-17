"""Per-bin spectral albedos of natural surfaces for the foreground renderer.

Units: dimensionless hemispherical (Lambertian-equivalent) reflectance, returned as bin
AVERAGES on the caller's SpectralGrid (linear interpolation of tabulated knots, 380-780 nm).

The knot tables are smoothed shapes after published laboratory / field spectra, read
approximately (not digitised) from:
  - green grass, dry grass, conifer needles, deciduous leaves, bark, soil: USGS spectral
    library splib07 (Kokaly et al. 2017, USGS Data Series 1035; e.g. "Lawn_Grass GDS91",
    "Grass_dry.4+.6", "Fir_Tree IH91-2", "Aspen_Leaf-A", "Pinon_Bark", "Sandy loam");
    leaf shape (chlorophyll absorption at 450 / 670 nm, green bump at 550 nm, red edge at
    700-740 nm) consistent with PROSPECT (Jacquemoud & Baret 1990).
  - canopy-scale vegetation values are lower than single leaves (shadowing between
    leaves inside a crown; e.g. Asner 1998, Remote Sens. Environ. 64, 234), so the crown
    materials ("needles", "leaves") are scaled down from leaf spectra.
  - granite: ASTER/ECOSTRESS library (Baldridge et al. 2009) granite, grey ~0.2-0.3.
  - snow: fresh fine-grained snow, Wiscombe & Warren (1980), ~0.95 in the visible,
    slowly decreasing towards 780 nm.
  - water: diffuse body ("water-leaving") reflectance of clear inland water, a few per
    cent in blue-green and ~0 in the red (Mobley 1994); the Fresnel surface reflection is
    modelled separately by a dielectric BSDF in foreground.py.
"""
import numpy as np

_WL = np.array([380, 400, 450, 500, 530, 550, 570, 600, 650, 670, 690, 700, 710, 720, 740, 760, 780], float)

MATERIALS = {
    "grass":     [.040, .040, .045, .060, .095, .115, .100, .075, .055, .045, .060, .110, .190, .290, .410, .460, .475],
    "grass_dry": [.070, .080, .105, .140, .170, .185, .200, .230, .270, .280, .295, .300, .305, .310, .325, .335, .345],
    "needles":   [.025, .026, .028, .035, .055, .065, .058, .045, .032, .028, .040, .070, .120, .180, .250, .280, .290],
    "leaves":    [.030, .032, .036, .048, .085, .100, .088, .060, .040, .035, .055, .110, .200, .290, .380, .410, .420],
    "leaves_autumn": [.030, .032, .036, .050, .070, .095, .130, .190, .240, .245, .260, .290, .320, .350, .390, .410, .420],
    "bark":      [.040, .045, .055, .070, .080, .088, .095, .110, .135, .145, .155, .160, .165, .170, .180, .190, .200],
    "soil":      [.045, .050, .065, .085, .100, .110, .120, .135, .160, .170, .180, .185, .190, .195, .205, .210, .215],
    "rock":      [.180, .190, .210, .230, .240, .245, .250, .260, .270, .275, .280, .282, .285, .288, .292, .296, .300],
    "snow":      [.950, .955, .960, .960, .958, .957, .955, .950, .940, .935, .930, .928, .925, .922, .915, .910, .905],
    "water":     [.020, .025, .032, .035, .033, .030, .025, .015, .007, .005, .004, .004, .004, .003, .003, .003, .003],
    "lakebed":   [.030, .035, .045, .060, .068, .072, .076, .085, .100, .105, .110, .112, .114, .116, .120, .123, .126],
}
MATERIALS = {k: np.array(v) for k, v in MATERIALS.items()}

# role -> material, per albedo set; foreground.py looks materials up by role
ALBEDO_SETS = {
    "summer": dict(grass="grass", grass_dry="grass_dry", soil="soil", rock="rock", snow="snow",
                   needles="needles", leaves="leaves", bark="bark", water="water", lakebed="lakebed"),
    "autumn": dict(grass="grass_dry", grass_dry="grass_dry", soil="soil", rock="rock", snow="snow",
                   needles="needles", leaves="leaves_autumn", bark="bark", water="water", lakebed="lakebed"),
    "winter": dict(grass="snow", grass_dry="snow", soil="snow", rock="rock", snow="snow",
                   needles="needles", leaves="bark", bark="bark", water="water", lakebed="lakebed"),
}
# default terrain snowline (m) per albedo set, used when the scene leaves snowline_m unset:
# summer ~2.8 km (late-lying snow only, as in the Alps at 46 N; the default terrain.py ranges
# peak at ~2.2 km, so none), autumn 1.8 km (first snowfalls on the highest summits), winter
# down to the valley floor (open ground is already snow in that set; the snow role then also
# covers gentle rock, the forest texture stays dark)
SNOWLINE_M = dict(summer=2800.0, autumn=1800.0, winter=0.0)


def snowline(albedo_set):
    """Default snowline (m) of an albedo set name; a custom role->material dict counts as
    winter when it maps grass to snow, else as summer."""
    if isinstance(albedo_set, str):
        return SNOWLINE_M[albedo_set]
    return SNOWLINE_M["winter" if dict(albedo_set).get("grass") == "snow" else "summer"]


def reflectance(name, wavelength_nm):
    """Reflectance of a named material at wavelength(s) in nm (linear interpolation)."""
    return np.interp(wavelength_nm, _WL, MATERIALS[name])


def spectrum(name, grid, scale=1.0):
    """(nbins,) bin-averaged albedo of a named material, clipped to [0, 1]."""
    if np.ndim(name) == 0 and not isinstance(name, str):
        return np.full(grid.n, float(name))
    return np.clip(scale * grid.bin_average(lambda w: reflectance(name, w)), 0.0, 1.0)


def role_table(albedo_set, grid, roles, overrides=None):
    """(len(roles), nbins) albedo per role for an albedo set name or a role->material dict.

    overrides: role -> material name or constant albedo (float)."""
    s = dict(ALBEDO_SETS[albedo_set]) if isinstance(albedo_set, str) else dict(albedo_set)
    s.update(overrides or {})
    return np.stack([spectrum(s[r], grid) for r in roles])


if __name__ == "__main__":
    from . import spectral
    g = spectral.default_grid()
    for k in MATERIALS:
        a = spectrum(k, g)
        print(f"{k:14s} 450={a[g.bin_index(450)]:.3f} 550={a[g.bin_index(550)]:.3f} "
              f"650={a[g.bin_index(650)]:.3f} 750={a[g.bin_index(750)]:.3f}")
