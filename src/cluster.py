"""Synthetic young massive cluster: IMF-sampled stars, ZAMS-like L/Teff, Plummer
spatial profile, observed from a few pc away.

No stellar evolution, binaries, or interstellar extinction: this is a toy that
reproduces the right order of magnitude of flux and the right *distribution* of
which stars dominate (a handful of very luminous O/WR-like stars).
"""
import numpy as np

import photometry as ph

LSUN = 3.828e26  # W
PC = 3.0857e16  # m
SIGMA_SB = 5.670374e-8

# ZAMS-like mass (Msun) -> luminosity (Lsun), Teff (K)
_MASS = np.array([0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10, 20, 40, 60, 100, 120])
_LOGL = np.log10([0.011, 0.024, 0.04, 0.7, 16, 80, 250, 550, 5.6e3, 4.5e4, 2.4e5, 5.3e5, 1.1e6, 1.5e6])
_TEFF = np.array([3400, 3550, 3800, 5700, 9300, 12000, 14500, 16000, 25000, 35000, 42000, 46000, 49000, 50000.0])


def lum_teff(mass):
    lm = np.log10(mass)
    logL = np.interp(lm, np.log10(_MASS), _LOGL)
    teff = np.interp(lm, np.log10(_MASS), _TEFF)
    return 10 ** logL * LSUN, teff


def sample_imf(total_mass, m_min=1.0, m_max=120.0, alpha=1.85, rng=None):
    """Draw stars from dN/dM ~ M^-alpha until their total mass reaches total_mass."""
    rng = rng or np.random.default_rng(1)
    a1 = 1.0 - alpha
    masses = []
    tot = 0.0
    while tot < total_mass:
        u = rng.random(2000)
        m = (m_min ** a1 + u * (m_max ** a1 - m_min ** a1)) ** (1 / a1)
        c = tot + np.cumsum(m)
        keep = c <= total_mass
        masses.append(m[keep])
        tot = tot + m[keep].sum()
        if not keep.all():
            break
    return np.concatenate(masses)


def sample_kroupa(total_mass, m_min=0.3, m_max=4.0, rng=None):
    """Kroupa (2001) IMF, dN/dM ~ M^-1.3 below 0.5 Msun and M^-2.3 above, drawn until total_mass."""
    rng = rng or np.random.default_rng(1)
    grid = np.geomspace(m_min, m_max, 20001)
    w = np.where(grid < 0.5, grid ** -1.3, 0.5 * grid ** -2.3)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    masses, tot = [], 0.0
    while tot < total_mass:
        m = np.interp(rng.random(2000), cdf, grid)
        keep = tot + np.cumsum(m) <= total_mass
        masses.append(m[keep])
        tot += m[keep].sum()
        if not keep.all():
            break
    return np.concatenate(masses)


def plummer_positions(n, a_pc, rmax_pc, rng):
    """3D Plummer positions (pc), truncated at rmax_pc."""
    out = np.empty((0, 3))
    while len(out) < n:
        u = rng.random(2 * n)
        r = a_pc / np.sqrt(u ** (-2 / 3) - 1)
        r = r[r < rmax_pc]
        v = rng.normal(size=(len(r), 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        out = np.vstack([out, v * r[:, None]])
    return out[:n]


class _StarSet:
    """Shared spectral helpers; subclasses set mass, lum, teff, direction, dist_m, f_bol."""

    def spectral_irradiance(self, wavelengths_nm):
        """(n_stars, n_bins) blackbody irradiance, W/m^2/nm, bolometric-normalised."""
        wl = np.asarray(wavelengths_nm, float)
        cols = [ph.irradiance_bb(self.f_bol, self.teff, w) for w in wl]
        return np.stack(cols, axis=1)

    def brightest(self, k):
        """Indices of the k stars with largest ~550 nm irradiance."""
        e550 = ph.irradiance_bb(self.f_bol, self.teff, 550.0)
        return np.argsort(e550)[::-1][:k]


class Cluster(_StarSet):
    """Stars as seen from an observer at distance `distance_pc` from the centre."""

    def __init__(self, center_dir, total_mass=1.5e4, alpha=1.85, m_min=1.0, m_max=120.0,
                 a_pc=0.25, rmax_pc=1.5, distance_pc=3.0, seed=1):
        rng = np.random.default_rng(seed)
        self.mass = sample_imf(total_mass, m_min, m_max, alpha, rng)
        self.lum, self.teff = lum_teff(self.mass)
        pos = plummer_positions(len(self.mass), a_pc, rmax_pc, rng)
        center_dir = np.asarray(center_dir, float) / np.linalg.norm(center_dir)
        vec = center_dir * distance_pc + pos  # pc, observer at origin
        d = np.linalg.norm(vec, axis=1)
        self.direction = vec / d[:, None]
        self.dist_m = d * PC
        self.f_bol = self.lum / (4 * np.pi * self.dist_m ** 2)  # W/m^2

    def resolved_and_rest(self, wavelengths_nm, k=300):
        """Top-k stars as individual sources plus one aggregate source for the rest.

        Returns (directions (k+1,3), irradiance (k+1, n_bins), index array of the k).
        The aggregate sits at the flux-weighted mean direction of the unresolved stars
        so total flux is conserved.
        """
        idx = self.brightest(k)
        rest = np.setdiff1d(np.arange(len(self.mass)), idx)
        e = self.spectral_irradiance(wavelengths_nm)
        dirs = [self.direction[idx]]
        irr = [e[idx]]
        if len(rest):
            w = e[rest].sum(axis=1)
            d = (self.direction[rest] * w[:, None]).sum(axis=0)
            dirs.append((d / np.linalg.norm(d))[None])
            irr.append(e[rest].sum(axis=0)[None])
        return np.vstack(dirs), np.vstack(irr), idx

    def summary(self):
        e550 = ph.irradiance_bb(self.f_bol, self.teff, 550.0)
        vmag = -2.5 * np.log10(e550.sum() / ph.F_V0_550NM)
        top = np.sort(e550)[::-1]
        return (f"{len(self.mass)} stars, M={self.mass.sum():.3g} Msun, L={self.lum.sum() / LSUN:.3g} Lsun, "
                f"cluster V~{vmag:.2f}; brightest star = {100 * top[0] / top.sum():.1f}% of 550nm flux, "
                f"top 10 = {100 * top[:10].sum() / top.sum():.1f}%, top 300 = {100 * top[:300].sum() / top.sum():.1f}%")


AU = 1.495978707e11  # m


class LocalStars(_StarSet):
    """Field stars around the observer: Kroupa IMF, ZAMS L/Teff, uniform in a cubic box
    centred on the observer's own star, with any star closer than `min_dist_au` re-placed.

    Directions are in celestial equatorial axes (z = north celestial pole); rotate them
    into the horizon frame per frame with render_day.celestial_to_horizon.
    """

    def __init__(self, total_mass=1000.0, m_min=0.3, m_max=4.0, box_pc=1.0, min_dist_au=50.0, seed=7):
        rng = np.random.default_rng(seed)
        self.mass = sample_kroupa(total_mass, m_min, m_max, rng)
        self.lum, self.teff = lum_teff(self.mass)
        n = len(self.mass)
        pos = rng.uniform(-0.5, 0.5, size=(n, 3)) * box_pc * PC
        too_close = np.linalg.norm(pos, axis=1) < min_dist_au * AU
        self.n_rejected = 0
        while too_close.any():
            self.n_rejected += int(too_close.sum())
            pos[too_close] = rng.uniform(-0.5, 0.5, size=(int(too_close.sum()), 3)) * box_pc * PC
            too_close = np.linalg.norm(pos, axis=1) < min_dist_au * AU
        d = np.linalg.norm(pos, axis=1)
        self.direction = pos / d[:, None]
        self.dist_m = d
        self.f_bol = self.lum / (4 * np.pi * d ** 2)

    def summary(self):
        e550 = ph.irradiance_bb(self.f_bol, self.teff, 550.0)
        v = -2.5 * np.log10(e550 / ph.F_V0_550NM)
        return (f"{len(self.mass)} field stars, M={self.mass.sum():.4g} Msun, {self.n_rejected} positions rejected "
                f"(< min distance); nearest {self.dist_m.min() / AU:.3g} au; brightest V={v.min():.2f}; "
                f"total V={-2.5 * np.log10(e550.sum() / ph.F_V0_550NM):.2f}; {int((v < 0).sum())} stars brighter than V=0")
