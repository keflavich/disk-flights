"""Real bright stars (Yale Bright Star Catalogue, via VizieR V/50) for the Earth-sky comparison.

Coordinates are J2000 (no precession to the render date: ~0.36 deg by 2026). Directions are
returned in the render's celestial frame: x = the Sun at the September equinox (RA 12h),
y = RA 18h, z = north celestial pole, so render_day.celestial_to_horizon applies unchanged.
"""
import os

import numpy as np
from astropy.table import Table

import photometry as ph

CATALOG = os.path.join(os.path.dirname(__file__), "..", "data", "star_catalogs", "bsc5_vlt2.05.csv")


def fetch(path=CATALOG, vmax=2.05):
    from astroquery.vizier import Vizier
    v = Vizier(columns=["HR", "Name", "_RAJ2000", "_DEJ2000", "Vmag", "B-V", "SpType"],
               column_filters={"Vmag": f"<{vmax}"}, row_limit=-1)
    t = v.query_constraints(catalog="V/50/catalog")[0]
    t.sort("Vmag")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t.write(path, format="csv", overwrite=True)
    return t


def teff_from_bv(bv):
    """Ballesteros (2012) blackbody temperature from B-V."""
    return 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))


class BrightStars:
    def __init__(self, vmax=2.0, path=CATALOG):
        t = Table.read(path) if os.path.exists(path) else fetch(path)
        t = t[t["Vmag"] < vmax]
        ra = np.radians(np.asarray(t["_RAJ2000"], float)) - np.pi  # RA measured from the Sept equinox (12h)
        dec = np.radians(np.asarray(t["_DEJ2000"], float))
        self.direction = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=1)
        self.vmag = np.asarray(t["Vmag"], float)
        bv = np.ma.filled(t["B-V"], 0.6).astype(float)
        self.teff = teff_from_bv(bv)
        self.name = [str(n).strip() for n in t["Name"]]

    def spectral_irradiance(self, wavelengths_nm):
        """(n_stars, n_bins) W/m^2/nm: blackbody shape normalised to each star's V magnitude."""
        return np.stack([ph.irradiance_from_vmag(self.vmag, w, self.teff) for w in wavelengths_nm], axis=1)

    def summary(self):
        return (f"{len(self.vmag)} stars with V < {self.vmag.max():.2f}: brightest {self.name[int(np.argmin(self.vmag))]} "
                f"V={self.vmag.min():.2f}; Teff {self.teff.min():.0f}-{self.teff.max():.0f} K")
