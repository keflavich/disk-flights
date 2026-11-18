"""Planet-induced surface-density perturbation delta(R, phi) = Sigma / Sigma_init from a PPDONet
prediction (ppdo_predict.py), placed at planet radius r_p (au), planet at phi = 0.

The axisymmetric grid model is multiplied by delta at every height (the vertical structure and
scale height are kept): rho_3d(R, z, phi) = rho_axi(R, z) * delta(R, phi). delta is tapered to 1
inside FARGO3D's damping zones (r < 0.46 and r > 2.17 r_p, where the training runs were relaxed to
the initial condition) so the perturbation joins the unperturbed disk smoothly.
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

    @classmethod
    def from_npz(cls, path, r_p_au):
        d = np.load(path)
        r, phi = d["r"], d["phi"]
        ld = (d["log10_sigma"] - np.log10(r[:, None] ** -0.5)) * np.log(10.0)
        w = _smooth(r, *DAMP_IN) * (1 - _smooth(r, *DAMP_OUT))
        ld = ld * w[:, None]
        return cls(r, phi, ld, float(r_p_au), {k: float(d[k]) for k in ("alpha", "h0", "q")})

    @property
    def lnr0(self):
        return float(np.log(self.r[0]))

    @property
    def dlnr(self):
        """PPDONet radii are linear in r; resample to uniform ln r for the kernels."""
        return float((np.log(self.r[-1]) - np.log(self.r[0])) / (len(self.r) - 1))

    def table(self, n_lnr=None, n_phi=None):
        """log delta on a uniform (ln(R / r_p), phi) table: (lnr0, dlnr, phi0, dphi, table)."""
        n_lnr = n_lnr or len(self.r)
        n_phi = n_phi or len(self.phi)
        lnr = np.linspace(np.log(self.r[0]), np.log(self.r[-1]), n_lnr)
        phi = -np.pi + (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
        t = self(np.exp(lnr)[:, None] * self.r_p, phi[None, :])
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
