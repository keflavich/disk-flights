"""Planet-induced surface-density perturbation delta(R, phi) = Sigma / Sigma_init from a PPDONet
prediction (ppdo_predict.py), placed at planet radius r_p (au), planet at phi = 0.

The axisymmetric grid model is multiplied by delta at every height (the vertical structure and
scale height are kept): rho_3d(R, z, phi) = rho_axi(R, z) * delta(R, phi). delta is tapered to 1
inside FARGO3D's damping zones (r < 0.46 and r > 2.17 r_p, where the training runs were relaxed to
the initial condition) so the perturbation joins the unperturbed disk smoothly.

Optional vertical concentration of the planet's own overdensity (clump_h > 0): the planet's
excess over the local gap level, E = W(d / R_H) max(delta - delta_bg(R), 0) with delta_bg the
azimuthal median of delta at that radius and W a window of a few Hill radii around the planet, is
given a Gaussian vertical profile of scale clump_h * h(R) instead of the disk's h(R), with the
same surface density:

    rho_3d = rho_axi * [delta + E (f(z / h) - 1)],
    f(q) = exp(-q^2 (1 / clump_h^2 - 1) / 2) / clump_h,

so the circumplanetary material sits near the midplane, and above it the density drops to the
gap level (delta - E >= delta_bg). With clump_h = 0 the whole of delta applies at every height.
The gap and the spirals keep the disk's vertical profile.
"""
from dataclasses import dataclass

import numpy as np

DAMP_IN = (0.42, 0.55)    # taper ramps from 0 to 1 across this range (units of r_p)
DAMP_OUT = (2.15, 2.45)   # and from 1 back to 0 across this one


def _smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


@dataclass
class Perturbation:
    r: np.ndarray        # (ny,) cell centres in units of r_p
    phi: np.ndarray      # (nx,) cell centres, rad, [-pi, pi)
    log_delta: np.ndarray  # (ny, nx) natural log of Sigma / Sigma_init, tapered
    r_p: float           # au
    params: dict
    clump_h: float = 0.0   # clump scale height / disk scale height; 0: delta at every height
    clump_w: float = 2.0   # clump window radius, Hill radii

    @classmethod
    def from_npz(cls, path, r_p_au, clump_h=0.0, clump_w=2.0):
        d = np.load(path)
        r, phi = d["r"], d["phi"]
        ld = (d["log10_sigma"] - np.log10(r[:, None] ** -0.5)) * np.log(10.0)
        w = _smooth(r, *DAMP_IN) * (1 - _smooth(r, *DAMP_OUT))
        ld = ld * w[:, None]
        return cls(r, phi, ld, float(r_p_au), {k: float(d[k]) for k in ("alpha", "h0", "q")},
                   float(clump_h), float(clump_w))

    @property
    def r_hill(self):
        """au"""
        return self.r_p * (self.params["q"] / 3.0) ** (1.0 / 3.0)

    def _bg(self):
        """delta_bg on the PPDONet radii: azimuthal median of delta."""
        if not hasattr(self, "_bg_cache"):
            self._bg_cache = np.median(np.exp(self.log_delta), axis=1)
        return self._bg_cache

    def excess(self, R_au, phi, ld=None):
        """E: the planet's overdensity above the gap level, in units of the unperturbed density
        (0 everywhere when clump_h == 0)."""
        R_au, phi = np.broadcast_arrays(np.asarray(R_au, float), np.asarray(phi, float))
        if self.clump_h <= 0:
            return np.zeros(R_au.shape)
        ld = self(R_au, phi) if ld is None else ld
        bg = np.interp(R_au / self.r_p, self.r, self._bg(), left=1.0, right=1.0)
        d2 = R_au ** 2 + self.r_p ** 2 - 2 * R_au * self.r_p * np.cos(phi)
        w = np.exp(-(d2 / (self.clump_w * self.r_hill) ** 2) ** 2)
        return np.maximum(np.exp(ld) - bg, 0.0) * w

    def factor(self, R_au, phi, z_au, h_au):
        """rho_3d / rho_axi at (R, phi, z) for a disk of Gaussian scale height h (au)."""
        ld = self(R_au, phi)
        if self.clump_h <= 0:
            return np.exp(ld)
        e = self.excess(R_au, phi, ld)
        k = self.clump_h
        q = np.asarray(z_au, float) / h_au
        f = np.exp(-0.5 * q * q * (1.0 / (k * k) - 1.0)) / k
        return (np.exp(ld) - e) + e * f

    @property
    def lnr0(self):
        return float(np.log(self.r[0]))

    @property
    def dlnr(self):
        """PPDONet radii are linear in r; resample to uniform ln r for the kernels."""
        return float((np.log(self.r[-1]) - np.log(self.r[0])) / (len(self.r) - 1))

    def table(self, n_lnr=None, n_phi=None):
        """ln(delta - E) and the clump excess E on a uniform (ln(R / r_p), phi) table (the kernels
        add the two, which avoids cancellation where the clump sits above the gap level):
        (lnr0, dlnr, phi0, dphi, table (2, n_lnr, n_phi))."""
        n_lnr = n_lnr or len(self.r)
        n_phi = n_phi or len(self.phi)
        lnr = np.linspace(np.log(self.r[0]), np.log(self.r[-1]), n_lnr)
        phi = -np.pi + (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
        R, P = np.exp(lnr)[:, None] * self.r_p, phi[None, :]
        t = self(R, P)
        e = self.excess(R, P, t)
        t = np.stack([np.log(np.maximum(np.exp(t) - e, 1e-300)), e])
        return lnr[0], lnr[1] - lnr[0], phi[0], phi[1] - phi[0], t.astype(np.float32)

    def __call__(self, R_au, phi):
        """log delta at cylindrical radius R (au) and azimuth phi (rad); 0 outside the domain."""
        R_au, phi = np.broadcast_arrays(np.asarray(R_au, float), np.asarray(phi, float))
        x = R_au / self.r_p
        dr = self.r[1] - self.r[0]
        fi = (x - self.r[0]) / dr
        ok = (fi >= 0) & (fi <= len(self.r) - 1)
        i = np.clip(np.floor(fi).astype(int), 0, len(self.r) - 2)
        ti = np.clip(fi - i, 0, 1)
        nphi = len(self.phi)
        dp = 2 * np.pi / nphi
        fj = np.mod(phi - self.phi[0], 2 * np.pi) / dp
        j = np.floor(fj).astype(int) % nphi
        tj = fj - np.floor(fj)
        j1 = (j + 1) % nphi
        L = self.log_delta
        v = ((1 - ti) * ((1 - tj) * L[i, j] + tj * L[i, j1]) + ti * ((1 - tj) * L[i + 1, j] + tj * L[i + 1, j1]))
        return np.where(ok, v, 0.0)
