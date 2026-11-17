"""Voxel fields: the contract between 3D source models (gas.py, galaxy.py clouds) and the
galaxy ray marcher.

A VoxelField is a regular grid in the GALACTOCENTRIC frame (pc, see frames.py) whose
`ncomp` channels each carry a fixed spectrum. Sampled with trilinear interpolation; zero
outside the grid.

kind == "emission":   sum_c data[..., c] * spectra[c]  = radiance added per pc of path
                      (W m^-2 sr^-1 nm^-1 pc^-1), before extinction.
kind == "extinction": sum_c data[..., c] * spectra[c]  = optical depth per pc (per bin).
                      Convention: data = A_V per pc, spectra[c] = tau_lambda / A_V for that
                      component's extinction curve (so tau_V = A_V / 1.0857).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VoxelField:
    origin_pc: np.ndarray  # (3,) galactocentric position of voxel (0,0,0)'s CENTRE
    voxel_pc: float  # cubic voxel edge length
    data: np.ndarray  # (nx, ny, nz, ncomp) float32
    spectra: np.ndarray  # (ncomp, nbins)
    kind: str = "emission"  # "emission" | "extinction"
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.data.shape[:3]

    @property
    def bounds_pc(self):
        """(2, 3) min / max galactocentric coordinates covered (voxel centres)."""
        lo = np.asarray(self.origin_pc, float)
        return np.stack([lo, lo + (np.array(self.shape) - 1) * self.voxel_pc])

    def sample(self, xyz_pc):
        """Trilinear sample (N, ncomp) at galactocentric points (N, 3); zero outside."""
        p = (np.asarray(xyz_pc, float).reshape(-1, 3) - np.asarray(self.origin_pc, float)) / self.voxel_pc
        n = np.array(self.shape)
        i0 = np.floor(p).astype(int)
        f = p - i0
        out = np.zeros((len(p), self.data.shape[3]))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    idx = i0 + np.array([dx, dy, dz])
                    ok = np.all((idx >= 0) & (idx < n), axis=1)
                    w = (np.where(dx, f[:, 0], 1 - f[:, 0]) * np.where(dy, f[:, 1], 1 - f[:, 1])
                         * np.where(dz, f[:, 2], 1 - f[:, 2]))
                    out[ok] += w[ok, None] * self.data[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        return out

    def spectral(self, xyz_pc):
        """(N, nbins) emissivity or opacity at the points."""
        return self.sample(xyz_pc) @ self.spectra

    def save(self, path):
        np.savez_compressed(path, origin_pc=self.origin_pc, voxel_pc=self.voxel_pc, data=self.data,
                            spectra=self.spectra, kind=self.kind, name=self.name)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z["origin_pc"], float(z["voxel_pc"]), z["data"], z["spectra"], str(z["kind"]), str(z["name"]))
