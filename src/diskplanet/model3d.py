"""Non-axisymmetric (planet-perturbed) disk model for the diskorbit renderer.

rho(R, z, phi) = rho_axi(R, z) * delta(R, phi), with rho_axi the grid model's analytic disk
(diskorbit.model.DiskModel) and delta the PPDONet surface-density perturbation
(diskplanet.perturb). The planet sits at phi = 0 (x axis) at r_p; the pattern is fixed in the
model frame.

Tables (phi, ln r, mu):
  ncol  radial column from the star. A stellar ray is radial, so it keeps its azimuth and the
        column is a 1D integral per (phi, mu).
  temp  dust temperature from the 3D Hyperion run (diskplanet.hyperion3d), hemispheres averaged,
        bilinear in (mu, ln r) per Hyperion phi cell, linear periodic in phi. Inside t_blend
        (units of r_p; delta = 1 there) the axisymmetric model's temperature is used, which has
        far more photons per cell and the inner-rim treatment of DiskModel; the two are blended
        smoothly across t_blend. Cells the 3D run sampled with few photons (n_photons_ok ramp;
        the optically thick midplane, as the 3D run has no PDA) also take the axisymmetric value,
        and away from the planet (planet_zone, rad) the 3D values are smoothed over phi_smooth
        azimuthal cells. Without a 3D run the axisymmetric temperature is used throughout.
"""
import os

import h5py
import numpy as np
from numba import njit, prange
from scipy.interpolate import RegularGridInterpolator

from diskorbit.model import DiskModel, AU_CM
from diskorbit.kernels import density3

from .perturb import Perturbation


def _smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


@njit(parallel=True, cache=True)
def _column3(rf, mu, phis, P, PT, LD, keep):
    """Cumulative trapezoid of rho along radial rays: out[k, i, j] = int_0^{rf[i*keep]} rho dr
    at azimuth phis[k], |cos theta| = mu[j] (z > 0; the model is mirror-symmetric)."""
    nf = rf.shape[0]
    out = np.zeros((phis.shape[0], (nf - 1) // keep + 1, mu.shape[0]), np.float32)
    for k in prange(phis.shape[0]):
        cp, sp = np.cos(phis[k]), np.sin(phis[k])
        for j in range(mu.shape[0]):
            st = np.sqrt(max(0.0, 1.0 - mu[j] * mu[j]))
            acc = 0.0
            prev = density3(rf[0] * st * cp, rf[0] * st * sp, rf[0] * mu[j], P, PT, LD)
            for i in range(1, nf):
                r = rf[i]
                rho = density3(r * st * cp, r * st * sp, r * mu[j], P, PT, LD)
                acc += 0.5 * (rho + prev) * (r - rf[i - 1])
                prev = rho
                if i % keep == 0:
                    out[k, i // keep, j] = acc
    return out


class PlanetDisk(DiskModel):
    def __init__(self, name, sigma_npz=None, r_p=100.0, rtin3d=None, rtout3d=None, mc_nphi=32,
                 nphi_tab=256, n_lnr_pert=1024, n_phi_pert=512, t_blend=(0.30, 0.40),
                 planet_lsun=0.0, planet_temp=1200.0, phi_smooth=3, planet_zone=(0.25, 0.45),
                 n_photons_ok=(1.0, 5.0)):
        """sigma_npz None: delta = 1 everywhere (validation: must reproduce DiskModel).
        planet_lsun > 0: the planet is also an isotropic blackbody point source at (r_p, 0, 0)."""
        super().__init__(name)
        self.planet = None
        if planet_lsun > 0:
            self.planet = {"lum_w": planet_lsun * 3.828e26, "temp_k": float(planet_temp), "x_au": float(r_p),
                           "lsun": float(planet_lsun)}
        self.r_p = float(r_p)
        self.mc_nphi = int(mc_nphi)
        self.nphi_tab = int(nphi_tab)
        self.t_blend = t_blend
        self.phi_smooth, self.planet_zone, self.n_photons_ok = int(phi_smooth), planet_zone, n_photons_ok
        self.rtin3d, self.rtout3d = rtin3d, rtout3d
        if sigma_npz is not None:
            self.perturbation = Perturbation.from_npz(sigma_npz, r_p)
            lnr0, dlnr, phi0, dphi, ld = self.perturbation.table(n_lnr_pert, n_phi_pert)
            self._PT = np.array([lnr0 + np.log(self.r_p), dlnr, phi0, dphi, 1.0])
            self._LD = ld.astype(np.float64)
            stem = os.path.splitext(os.path.basename(sigma_npz))[0]
        else:
            self.perturbation = None
            self._PT = np.array([0.0, 1.0, 0.0, 1.0, 1.0])  # switched on, all zeros: exercises the 3D path
            self._LD = np.zeros((2, 2))
            stem = "flat"
        tsrc = os.path.splitext(os.path.basename(rtout3d))[0] if rtout3d else "Taxi"
        self.tag = f"_{stem}_j{self.mc_nphi}" + (f"_pl{planet_lsun:g}" if self.planet else "")
        self.t_tag = f"_{tsrc}"  # the temperature only enters the IR (thermal) Monte Carlo
        self._tables = None
        self._T3 = None
        if rtout3d is not None:
            self._load_3d()

    @property
    def pert(self):
        return self._PT, self._LD

    def density3d(self, x, y, z):
        """g cm^-3 at Cartesian (x, y, z) AU (numpy)."""
        x, y, z = np.broadcast_arrays(*(np.asarray(v, float) for v in (x, y, z)))
        R = np.hypot(x, y)
        d = self.density(R, z)
        if self.perturbation is not None:
            d = d * np.exp(self.perturbation(R, np.arctan2(y, x)))
        return d

    # ------------------------------------------------------------------ 3D Hyperion run
    def _load_3d(self):
        with h5py.File(self.rtin3d, "r") as f:
            g = f["Grid/Geometry"]
            self.r_w3 = np.array(g["walls_1"]["r"]) / AU_CM
            self.t_w3 = np.array(g["walls_2"]["t"])
            self.p_w3 = np.array(g["walls_3"]["p"])
            self.rho3 = np.array(f["Grid/Quantities/density"])[0]  # (nphi, nt, nr)
        with h5py.File(self.rtout3d, "r") as f:
            it = sorted(k for k in f if k.startswith("iteration_"))[-1]
            spe = np.array(f[it]["specific_energy"])[0]
            nph = None
            if "n_photons" in f[it]:
                nph = np.array(f[it]["n_photons"])
                nph = nph[0] if nph.ndim == 4 else nph  # (nphi, nt, nr); no dust axis
        self.iteration3d = it
        T = np.interp(spe, self._e_tab, self._t_tab)
        # light azimuthal smoothing (boxcar over phi_smooth cells) away from the planet, where the
        # perturbation varies slowly in phi
        pc = 0.5 * (self.p_w3[1:] + self.p_w3[:-1])
        dpl = np.abs(np.angle(np.exp(1j * pc)))  # angular distance from the planet (phi = 0)
        k = self.phi_smooth
        if k > 1:
            Ts = sum(np.roll(T, j, axis=0) for j in range(-(k // 2), k // 2 + 1)) / (2 * (k // 2) + 1)
            ws = _smooth(dpl, *self.planet_zone)[:, None, None]
            T = (1 - ws) * T + ws * Ts
        if not np.allclose(self.t_w3, np.pi - self.t_w3[::-1]):
            raise ValueError("3D theta walls are not mirror-symmetric")
        # axisymmetric temperature on the 3D grid's theta rows (the polar rows were merged)
        t_c3 = 0.5 * (self.t_w3[1:] + self.t_w3[:-1])
        t_c = 0.5 * (self.t_walls[1:] + self.t_walls[:-1])
        rows = np.array([int(np.argmin(np.abs(t_c - t))) for t in t_c3])
        T_axi = self.temp_grid[rows]
        r_c = np.sqrt(np.maximum(self.r_w3[:-1], 1e-12) * self.r_w3[1:])
        w = _smooth(r_c / self.r_p, *self.t_blend)[None, None, :]
        if nph is not None:
            # photon-starved cells (deep optically thick midplane; Hyperion ran without PDA) take
            # the axisymmetric model's (PDA) temperature
            w = w * _smooth(nph.astype(float), *self.n_photons_ok)
            self.n_photons3 = nph
        self._T3_raw = T
        self._T3 = (1 - w) * T_axi[None] + w * T  # (nphi, nt, nr)

    def cells(self):
        if self._T3 is None:
            return super().cells()
        return self.r_w3, self.t_w3, self.p_w3, self.rho3, self._T3

    # ------------------------------------------------------------------ (phi, ln r, mu) tables
    def tables(self, n_r=1000, n_mu=1500, r_out_fac=1.02, mu_top=None, tau_top=1e-6):
        if self._tables is not None:
            return self._tables
        # mu_top: above it the most opaque band's column to infinity (x max delta) < tau_top
        base0 = super().tables(n_r=400, n_mu=2000)
        k_max = self.optics_kmax()
        dmax = float(np.exp(self._LD.max()))
        ninf = base0["ncol"][0, -1, :] * dmax * k_max
        if mu_top is None:
            hi = np.nonzero(ninf > tau_top)[0]
            mu_top = float(min(1.0, base0["mu"][hi.max()] * 1.1 + 0.01)) if len(hi) else 1.0
        base = super().tables(n_r=n_r, n_mu=n_mu, r_out_fac=r_out_fac, mu_top=mu_top)
        lnr, mu = base["lnr"], base["mu"]
        n = self.nphi_tab
        dphi = 2 * np.pi / n
        phis = (np.arange(n) + 0.5) * dphi  # same convention as Hyperion cell centres
        rf = np.exp(np.linspace(lnr[0], lnr[-1], 4 * (n_r - 1) + 1))
        ncol = _column3(rf, mu, phis, self.params, self._PT, self._LD, 4)
        temp = np.repeat(base["temp"].astype(np.float32), n, axis=0) if self._T3 is None else self._temp_table(lnr, mu, base["temp"][0], n)
        self._tables = {"lnr": lnr, "mu": mu, "ncol": ncol, "temp": temp, "phi0": float(phis[0]), "dphi": dphi,
                        "mu_top": mu_top}
        return self._tables

    def optics_kmax(self):
        """Largest dust extinction per unit column (g cm^-3 AU) over the dust table."""
        return float(self.dust_chi.max() * AU_CM)

    def _temp_table(self, lnr, mu, T_axi_tab, n):
        """(n, n_r, n_mu) float32 from the blended 3D cell temperatures."""
        t_w, r_w, T = self.t_w3, self.r_w3, self._T3
        Tc = 0.5 * (T + T[:, ::-1])
        r_c = np.sqrt(np.maximum(r_w[:-1], 1e-12) * r_w[1:])
        first = int(np.searchsorted(r_w, self.rmin * (1 + 1e-9), side="right") - 1)
        Tc[:, :, :first] = Tc[:, :, first:first + 1]
        mu_c = np.abs(np.cos(0.5 * (t_w[:-1] + t_w[1:])))
        half = np.arange(len(mu_c) // 2)
        o = np.argsort(mu_c[half])
        mu_h = mu_c[half][o]
        lr_c = np.log(r_c[first:])
        mm, ll = np.meshgrid(np.clip(mu, mu_h[0], mu_h[-1]), np.clip(lnr, lr_c[0], lr_c[-1]), indexing="ij")
        pts = np.stack([mm, ll], axis=-1)
        nh = T.shape[0]
        per_h = np.empty((nh, len(lnr), len(mu)), np.float32)
        for j in range(nh):
            f = RegularGridInterpolator((mu_h, lr_c), Tc[j][half][o][:, first:], bounds_error=False, fill_value=None)
            per_h[j] = f(pts).T
        # Hyperion cell centres (j + 1/2) 2pi / nh -> table nodes (k + 1/2) 2pi / n, periodic linear
        fpos = (np.arange(n) + 0.5) * nh / n - 0.5
        j0 = np.floor(fpos).astype(int)
        t = (fpos - j0).astype(np.float32)[:, None, None]
        out = (1 - t) * per_h[j0 % nh] + t * per_h[(j0 + 1) % nh]
        return np.where(T_axi_tab[None] > 0, out, 0.0).astype(np.float32)
