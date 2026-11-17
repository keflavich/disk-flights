"""A Robitaille (2017) / Richardson et al. (2024) YSO grid model as renderer inputs.

Reads one model's Hyperion files from the grid tree
  {root}/{geom}/par/{xx}/{name}.par      disk parameters (mass = DUST mass, Msun)
  {root}/{geom}/input/{xx}/{name}.rtin   grid walls, discretised density, dust, stellar spectrum
  {root}/{geom}/output/{xx}/{name}.rtout specific energy (last iteration) -> dust temperature
(xx = first two characters of the name, lower case) and provides:
  * the analytic Hyperion flared-disk density (spherical inner and outer rims, as the grid's
    mysg setup uses), normalised like Hyperion does (to the discretised mass on the grid);
  * dust optical properties per SpectralGrid bin: mass extinction, albedo, phase function;
  * stellar luminosity per bin (the model photosphere spectrum in the rtin);
  * tables on a (ln r, mu = |cos theta|) grid: the radial column from the star and the dust
    temperature (both disk halves averaged, the model is mirror-symmetric).
Lengths in the renderer are AU, densities g cm^-3, columns g cm^-3 AU.
Only disk-only geometries (sp--s-i, sp--h-i, ...) are supported: envelopes are not modelled.
"""
import os
import numpy as np
import h5py

AU_CM = 1.495978707e13
AU_M = 1.495978707e11
MSUN_G = 1.98892e33  # Hyperion's msun
C_UMS = 2.99792458e14  # c in micron/s
GRID_ROOT = "/blue/adamginsburg/richardson.t/research/flux/grids-1.1"


def _read_par(path):
    out = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = (s.strip() for s in line.split("=", 1))
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def _cell_centres(r_w, t_w):
    """Hyperion SphericalPolarGrid cell centres and volumes (single phi cell)."""
    lr = np.log10(np.maximum(r_w, 1e-300))
    r = 10.0 ** (0.5 * (lr[:-1] + lr[1:]))
    if r_w[0] == 0:
        r[0] = r_w[1] / 2.0
    t = 0.5 * (t_w[:-1] + t_w[1:])
    vol = (np.diff(r_w ** 3) / 3.0)[None, :] * np.abs(np.diff(np.cos(t_w)))[:, None] * 2 * np.pi
    return r, t, vol


class DiskModel:
    def __init__(self, name, geom="sp--s-i", root=GRID_ROOT):
        xx = name[:2].lower()
        self.name, self.geom = name, geom
        self.par = _read_par(f"{root}/{geom}/par/{xx}/{name}.par")
        if any(k.startswith(("envelope", "cavity", "ambient")) for k in self.par):
            raise ValueError(f"{geom}/{name}: only disk-only models are supported")
        rtin = f"{root}/{geom}/input/{xx}/{name}.rtin"
        rtout = f"{root}/{geom}/output/{xx}/{name}.rtout"
        with h5py.File(rtin, "r") as f:
            g = f["Grid/Geometry"]
            r_w = np.array(g["walls_1"]["r"]) / AU_CM
            t_w = np.array(g["walls_2"]["t"])
            self.rho_grid = np.array(f["Grid/Quantities/density"])[0, 0]  # (n_theta, n_r)
            d = f["Dust/dust_001"]
            op = d["optical_properties"]
            self.dust_wav_um = C_UMS / np.array(op["nu"])
            self.dust_chi = np.array(op["chi"])  # cm^2 / g dust
            self.dust_albedo = np.array(op["albedo"])
            self.dust_p1 = np.array(op["P1"])  # (n_wav, n_mu)
            self.dust_mu = np.array(d["scattering_angles"]["mu"])
            mo = d["mean_opacities"]
            self._e_tab = np.array(mo["specific_energy"])
            self._t_tab = np.array(mo["temperature"])
            s = f["Sources/source_00001"]
            self.l_star_w = float(s.attrs["luminosity"]) * 1e-7
            self.r_star_au = float(s.attrs["r"]) / AU_CM
            sp = s["spectrum"]
            self.star_nu = np.array(sp["nu"])
            self.star_fnu = np.array(sp["fnu"])
        with h5py.File(rtout, "r") as f:
            it = sorted(k for k in f if k.startswith("iteration_"))[-1]
            spe = np.array(f[it]["specific_energy"])[0, 0]
        self.iteration = it
        self.r_walls, self.t_walls = r_w, t_w
        self.temp_raw = np.interp(spe, self._e_tab, self._t_tab)
        p = self.par
        self.rmin = float(r_w[1] if r_w[0] == 0 else r_w[0])
        self.rmax = float(r_w[-1])
        self.r0, self.h0 = 100.0, float(p["disk.h100"])
        self.beta, self.p = float(p["disk.beta"]), float(p["disk.p"])
        self.mass_msun = float(p["disk.mass"])
        # Hyperion normalisation: rho_0 = mass / sum(shape * volume) over the grid cells
        rc, tc, vol = _cell_centres(r_w * AU_CM, t_w)
        R, Z = rc[None, :] * np.sin(tc)[:, None], rc[None, :] * np.cos(tc)[:, None]
        shape = self._shape(R / AU_CM, Z / AU_CM, np.broadcast_to(rc[None, :] / AU_CM, R.shape))
        self.rho0 = self.mass_msun * MSUN_G / np.sum(shape * vol)
        nz = self.rho_grid > 0
        ratio = (self.rho0 * shape)[nz] / self.rho_grid[nz]
        # percentiles 0, 1, 50, 99, 100 (outliers sit far above the disk at negligible density)
        self.density_check = np.percentile(ratio, [0, 1, 50, 99, 100]).round(4).tolist()
        # Hyperion's very thin inner-rim cells (radial widths down to 1e-10 AU) are left at the
        # temperature floor although they hold dust: its photon packets evidently step over them,
        # so its effective rim is where the warm cells start. Use that radius as rmin (density
        # normalisation unchanged) and fill the few cold cells left beyond it.
        self.rmin_grid = self.rmin
        self.rmin = self._warm_rim_radius()
        self.temp_grid = self._fill_rim(self.temp_raw)
        self.params = np.array([self.rho0, self.r0, self.h0, self.beta, self.p, self.rmin, self.rmax])

    def _warm_rim_radius(self, t_floor=3.0, tau_v_min=1.0):
        """Smallest inner wall radius (AU) of the first warm (> t_floor) cell over the polar angles
        where the cold dusty skin in front of it is optically thick (tau_V > tau_v_min)."""
        dr = np.diff(self.r_walls) * AU_CM
        chi_v = np.exp(np.interp(np.log(0.55), np.log(self.dust_wav_um[::-1]), np.log(self.dust_chi[::-1])))
        best = np.inf
        for it in range(self.temp_raw.shape[0]):
            dusty = np.nonzero(self.rho_grid[it] > 0)[0]
            if len(dusty) == 0:
                continue
            warm = dusty[self.temp_raw[it, dusty] > t_floor]
            if len(warm) == 0:
                continue
            skin = slice(dusty[0], warm[0])
            if chi_v * np.dot(self.rho_grid[it, skin], dr[skin]) > tau_v_min:
                best = min(best, self.r_walls[warm[0]])
        return float(best) if np.isfinite(best) else self.rmin

    def _fill_rim(self, T, t_floor=3.0):
        """Cold (floor-temperature) dusty cells at r >= rmin in front of the first warm cell, at
        each polar angle, take that warm cell's temperature."""
        T = T.copy()
        self.rim_filled = np.zeros(T.shape[0], int)
        for it in range(T.shape[0]):
            dusty = np.nonzero((self.rho_grid[it] > 0) & (self.r_walls[:-1] >= self.rmin * (1 - 1e-9)))[0]
            if len(dusty) == 0:
                continue
            warm = dusty[T[it, dusty] > t_floor]
            if len(warm) == 0:
                continue
            cold = dusty[(dusty < warm[0])]
            T[it, cold] = T[it, warm[0]]
            self.rim_filled[it] = len(cold)
        return T

    def _shape(self, R, z, r):
        R = np.maximum(R, 1e-12)
        h = self.h0 * (R / self.r0) ** self.beta
        s = (self.r0 / R) ** (self.beta - self.p) * np.exp(-0.5 * (z / h) ** 2)
        return np.where((r >= self.rmin) & (r <= self.rmax), s, 0.0)

    def density(self, R, z):
        """g cm^-3 at cylindrical (R, z) in AU."""
        R, z = np.asarray(R, float), np.asarray(z, float)
        return self.rho0 * self._shape(R, z, np.hypot(R, z))

    def scale_height(self, R):
        return self.h0 * (np.asarray(R, float) / self.r0) ** self.beta

    # ------------------------------------------------------------------ per-bin optical data
    def optics(self, grid, n_phase=1024):
        """dict of per-bin arrays: k (chi * AU_CM, i.e. tau per (g cm^-3 AU)), albedo, g,
        phase (nb, n_phase) on cos(angle) = linspace(-1, 1, n_phase) normalised so that
        0.5 * int phase dmu = 1, and the inverse CDF icdf (nb, n_phase) for sampling mu."""
        lam = grid.wl / 1000.0
        o = np.argsort(self.dust_wav_um)
        lw = np.log(self.dust_wav_um[o])
        chi = np.exp(np.interp(np.log(lam), lw, np.log(self.dust_chi[o])))
        alb = np.interp(np.log(lam), lw, self.dust_albedo[o])
        mu = np.linspace(-1, 1, n_phase)
        nb = len(lam)
        phase = np.empty((nb, n_phase))
        icdf = np.empty((nb, n_phase))
        g = np.empty(nb)
        for b, x in enumerate(np.log(lam)):
            j = np.clip(np.searchsorted(lw, x), 1, len(lw) - 1)
            f = (x - lw[j - 1]) / (lw[j] - lw[j - 1])
            p1 = (1 - f) * self.dust_p1[o[j - 1]] + f * self.dust_p1[o[j]]
            ph = np.interp(mu, self.dust_mu, p1)
            ph /= 0.5 * np.trapezoid(ph, mu)
            phase[b] = ph
            g[b] = 0.5 * np.trapezoid(ph * mu, mu)
            cdf = np.concatenate([[0], np.cumsum(0.5 * (ph[1:] + ph[:-1]) * np.diff(mu))])
            cdf /= cdf[-1]
            icdf[b] = np.interp(np.linspace(0, 1, n_phase), cdf, mu)
        return {"k": chi * AU_CM, "chi": chi, "albedo": alb, "g": g, "phase": phase, "icdf": icdf}

    def star_bins(self, grid):
        """Stellar luminosity per bin, W nm^-1 (bin averages of L_lambda)."""
        nu, fnu = self.star_nu, self.star_fnu
        o = np.argsort(nu)
        nu, fnu = nu[o], fnu[o]
        lnu = fnu / np.trapezoid(fnu, nu) * self.l_star_w  # W / Hz
        lam_nm = C_UMS / nu * 1000.0
        llam = lnu * nu / lam_nm  # W / nm
        sub = grid.subsample()
        vals = np.interp(sub, lam_nm[::-1], llam[::-1])
        return vals.reshape(grid.n, -1).mean(axis=1)

    # ------------------------------------------------------------------ (ln r, mu) tables
    def tables(self, n_r=1500, n_mu=3000, r_out_fac=1.02):
        """Radial column from the star N(r, mu) = int_0^r rho dr' (g cm^-3 AU) and dust
        temperature T(r, mu), on ln r = linspace(ln rmin, ln(r_out_fac rmax), n_r) and
        mu = |cos theta| = linspace(0, 1, n_mu)."""
        lnr = np.linspace(np.log(self.rmin), np.log(self.rmax * r_out_fac), n_r)
        mu = np.linspace(0.0, 1.0, n_mu)
        r = np.exp(lnr)
        # column: integrate on a 4x finer radial grid (trapezoid in r), keep every 4th node
        rf = np.exp(np.linspace(lnr[0], lnr[-1], 4 * (n_r - 1) + 1))
        R = rf[None, :] * np.sqrt(1 - mu[:, None] ** 2)
        z = rf[None, :] * mu[:, None]
        rho = self.density(R, z)
        col = np.concatenate([np.zeros((n_mu, 1)), np.cumsum(0.5 * (rho[:, 1:] + rho[:, :-1]) * np.diff(rf)[None], axis=1)], axis=1)
        ncol = col[:, ::4].T.copy()  # (n_r, n_mu)
        # temperature: Hyperion cell values, mean of the two hemispheres, bilinear in
        # (ln r, |cos theta|) between cell centres (nearest-cell values give visible steps in the
        # thermal emission); cells inside rmin take the value of the first cell beyond it
        t_w, r_w = self.t_walls, self.r_walls
        if not np.allclose(t_w, np.pi - t_w[::-1]):
            raise ValueError("theta walls are not mirror-symmetric about the midplane")
        Tc = 0.5 * (self.temp_grid + self.temp_grid[::-1])
        r_c = np.sqrt(np.maximum(r_w[:-1], 1e-12) * r_w[1:])
        first = int(np.searchsorted(r_w, self.rmin * (1 + 1e-9), side="right") - 1)
        Tc[:, :first] = Tc[:, first:first + 1]
        mu_c = np.abs(np.cos(0.5 * (t_w[:-1] + t_w[1:])))
        half = np.arange(len(mu_c) // 2)  # upper-hemisphere rows (theta < pi / 2)
        o = np.argsort(mu_c[half])
        mu_h, T_h = mu_c[half][o], Tc[half][o]
        lr_c = np.log(r_c[first:])
        T_h = T_h[:, first:]
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((mu_h, lr_c), T_h, bounds_error=False, fill_value=None)
        mm, ll = np.meshgrid(np.clip(mu, mu_h[0], mu_h[-1]), np.clip(lnr, lr_c[0], lr_c[-1]), indexing="ij")
        T = f(np.stack([mm, ll], axis=-1))
        jt = np.clip(np.searchsorted(t_w, np.arccos(mu)) - 1, 0, len(t_w) - 2)
        jr = np.clip(np.searchsorted(r_w, np.maximum(r, self.rmin * (1 + 1e-9)), side="right") - 1, 0, len(r_w) - 2)
        T = np.where(self.rho_grid[jt][:, jr] > 0, T, 0.0).T.copy()
        return {"lnr": lnr, "mu": mu, "ncol": ncol, "temp": T}


def default_model():
    return DiskModel(os.environ.get("DISKORBIT_MODEL", "dIULECee"))


if __name__ == "__main__":
    import sys
    m = DiskModel(sys.argv[1] if len(sys.argv) > 1 else "dIULECee")
    print(m.name, m.iteration, "rmin", m.rmin, "rmax", m.rmax, "rho0", m.rho0)
    print("analytic / grid density at cell centres (min, max, n):", m.density_check)
    print("L*", m.l_star_w / 3.828e26, "Lsun  R*", m.r_star_au * AU_CM / 6.957e10, "Rsun")
