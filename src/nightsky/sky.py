"""Top-of-atmosphere sky containers: the contract between source models (galaxy, starfield)
and everything downstream (atmosphere, foreground lighting, compositing).

All directions are in the CELESTIAL frame (see frames.py). Rotate to the horizon frame
with `.rotated(frames.celestial_to_horizon(H, lat))`.
"""
from dataclasses import dataclass, field

import numpy as np

from . import frames, spectral


@dataclass
class PointSources:
    dirs: np.ndarray  # (n, 3) unit vectors
    irr: np.ndarray  # (n, nbins) TOA spectral irradiance, W/m^2/nm
    names: list = field(default_factory=list)  # optional labels (len n, or empty)

    @classmethod
    def empty(cls, nbins):
        return cls(np.zeros((0, 3)), np.zeros((0, nbins)), [])

    def __len__(self):
        return len(self.dirs)

    def rotated(self, R):
        return PointSources(self.dirs @ np.asarray(R).T, self.irr, list(self.names))

    def concat(self, *others):
        ps = [self, *others]
        names = []
        for p in ps:
            names += list(p.names) if len(p.names) == len(p) else [""] * len(p)
        return PointSources(np.vstack([p.dirs for p in ps]), np.vstack([p.irr for p in ps]), names)

    def brightest(self, k, grid):
        """Indices of the k sources with the largest photopic illuminance."""
        y = spectral.to_xyz(self.irr, grid)[:, 1]
        return np.argsort(y)[::-1][:k]

    def select(self, idx):
        idx = np.asarray(idx)
        names = [self.names[i] for i in idx] if len(self.names) == len(self) else []
        return PointSources(self.dirs[idx], self.irr[idx], names)


@dataclass
class TOASky:
    """Light arriving at the top of the atmosphere, before any atmospheric effects.

    diffuse_map: (12 nside^2, nbins) RING-ordered HEALPix radiance (W/m^2/sr/nm) of everything
                 not in `points`; used as the scattering input and as a fallback renderer.
    diffuse_fn:  optional exact evaluator f(dirs_celestial (N, 3)) -> (N, nbins), e.g. a ray
                 marcher, used for the high-resolution visible sky background.
    """
    grid: spectral.SpectralGrid
    points: PointSources
    diffuse_map: np.ndarray
    diffuse_fn: object = None
    meta: dict = field(default_factory=dict)

    @property
    def nside(self):
        import healpy as hp
        return hp.npix2nside(self.diffuse_map.shape[0])

    def diffuse(self, dirs):
        dirs = np.asarray(dirs, float)
        if self.diffuse_fn is not None:
            return self.diffuse_fn(dirs.reshape(-1, 3)).reshape(dirs.shape[:-1] + (self.grid.n,))
        return frames.sample_healpix(self.diffuse_map, dirs)

    def merged(self, other):
        """Sum of two skies (maps resampled to the finer nside); exact evaluators are summed."""
        import healpy as hp
        a, b = self.diffuse_map, other.diffuse_map
        if a.shape[0] != b.shape[0]:
            ns = max(self.nside, other.nside)
            up = lambda m: np.stack([hp.ud_grade(m[:, i], ns) for i in range(m.shape[1])], axis=1)
            a = a if self.nside == ns else up(a)
            b = b if other.nside == ns else up(b)
        fa, fb = self.diffuse_fn, other.diffuse_fn
        fn = None
        if fa is not None or fb is not None:
            ga = fa if fa is not None else (lambda d, s=self: frames.sample_healpix(s.diffuse_map, d))
            gb = fb if fb is not None else (lambda d, s=other: frames.sample_healpix(s.diffuse_map, d))
            fn = lambda d: ga(d) + gb(d)
        return TOASky(self.grid, self.points.concat(other.points), a + b, fn, {**self.meta, **other.meta})

    def total_irradiance_on(self, normal, grid=None):
        """Spectral irradiance (nbins,) on a surface with unit `normal` (celestial frame), no atmosphere."""
        import healpy as hp
        n = np.asarray(normal, float)
        d = frames.healpix_dirs(self.nside)
        cos = np.clip(d @ n, 0, None)
        omega = 4 * np.pi / hp.nside2npix(self.nside)
        e = (self.diffuse_map * cos[:, None]).sum(0) * omega
        if len(self.points):
            e = e + (self.points.irr * np.clip(self.points.dirs @ n, 0, None)[:, None]).sum(0)
        return e
