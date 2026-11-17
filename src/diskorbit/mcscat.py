"""Scattered-light radiation field of the disk: 2D (axisymmetric) Monte Carlo per wavelength bin.

Photon packets leave the star toward the disk (directions with optical depth to infinity
> 1e-8, weighted by that solid-angle fraction), are forced to interact (weight 1 - e^-tau_inf)
and from their first scattering on are followed with weight reduction by the albedo and
Russian roulette; scattering angles are drawn from the tabulated dust phase function. Path
lengths of the scattered packets are tallied on a (ln r, mu = |z|/r) grid, giving the
mean intensity J and the first moment H = (1/4pi) int I n dOmega (cylindrical-radial and
away-from-midplane components) of the SCATTERED light. Direct starlight is not tallied: the
renderer adds the single-scattered term exactly. The renderer's multiple-scattering source is
albedo * (J + 3 g H.n), the exact scattering integral for a radiance field with only the two
lowest angular moments.

Output units: W m^-2 nm^-1 sr^-1 (mean intensity per nm, bin average).
"""
import time

import numba
import numpy as np
from numba import njit, prange

from .kernels import density, geo_step, scatter_dir
from .model import AU_M


@njit(cache=True)
def _radial_tau_point(ncol, lnr0, dlnr, mu0, dmu, mu, target):
    """Radius (AU) along the radial ray at |cos theta| = mu where the column reaches target."""
    nr, nm = ncol.shape
    fy = (mu - mu0) / dmu
    j = min(max(int(fy), 0), nm - 2)
    t = min(max(fy - j, 0.0), 1.0)
    lo, hi = 0, nr - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        v = (1 - t) * ncol[mid, j] + t * ncol[mid, j + 1]
        if v < target:
            lo = mid
        else:
            hi = mid
    a = (1 - t) * ncol[lo, j] + t * ncol[lo, j + 1]
    b = (1 - t) * ncol[hi, j] + t * ncol[hi, j + 1]
    f = 0.0 if b <= a else (target - a) / (b - a)
    return np.exp(lnr0 + (lo + f) * dlnr)


@njit(inline="always")
def _propagate(x, y, z, ux, uy, uz, w, P, kb, alb, icdf, rout, elnr0, edlnr, nr_j, nmu_j,
               frac, w_min, tal):
    """Follow a packet from (x, y, z) in direction u with weight w until it escapes or is
    terminated, scattering with albedo weighting and Russian roulette, and tally w * path
    length (and its R / away-from-midplane projections) into tal[ln r cell, mu cell, 3]."""
    nic = icdf.shape[0]
    trem = -np.log(1.0 - np.random.random())
    for _step in range(2000000):
        R = np.sqrt(x * x + y * y)
        rr = np.sqrt(R * R + z * z)
        if rr > rout and x * ux + y * uy + z * uz >= 0:
            break
        s = geo_step(R, z, rr, P, frac)
        xm, ym, zm = x + 0.5 * s * ux, y + 0.5 * s * uy, z + 0.5 * s * uz
        Rm = np.sqrt(xm * xm + ym * ym)
        kr = kb * density(Rm, zm, P)
        hit = False
        if kr * s >= trem:
            s = trem / kr
            hit = True
            xm, ym, zm = x + 0.5 * s * ux, y + 0.5 * s * uy, z + 0.5 * s * uz
            Rm = np.sqrt(xm * xm + ym * ym)
        rm = np.sqrt(Rm * Rm + zm * zm)
        if rm > 0:
            fi = (np.log(rm) - elnr0) / edlnr
            if fi >= 0 and fi < nr_j:
                ia = int(fi)
                ib = min(int(abs(zm) / rm * nmu_j), nmu_j - 1)
                zs = 1.0 if zm >= 0 else -1.0
                uR = (xm * ux + ym * uy) / Rm if Rm > 0 else 0.0
                tal[ia, ib, 0] += w * s
                tal[ia, ib, 1] += w * s * uR
                tal[ia, ib, 2] += w * s * uz * zs
        x, y, z = x + s * ux, y + s * uy, z + s * uz
        if hit:
            w *= alb
            if w < w_min:
                if np.random.random() < 0.1:
                    w *= 10.0
                else:
                    break
            f = np.random.random() * (nic - 1)
            i = min(int(f), nic - 2)
            cost = icdf[i] + (f - i) * (icdf[i + 1] - icdf[i])
            ux, uy, uz = scatter_dir(ux, uy, uz, cost, 2 * np.pi * np.random.random())
            trem = -np.log(1.0 - np.random.random())
        else:
            trem -= kr * s


@njit(parallel=True)
def _mc_thermal_bin(n_phot, P, kb, alb, icdf, cdf, r_w3, ct_w, n_rc, elnr0, edlnr, nr_j, nmu_j,
                    frac, w_min, seed, n_chunks):
    """Dust thermal emission packets: cell drawn from cdf (flattened (theta, r) Hyperion cells,
    luminosity-weighted), position uniform in the cell volume, isotropic direction, weight 1.
    The whole path including the first flight is tallied (the renderer has no exact direct
    thermal term; its scattering source albedo * J must include the direct thermal field)."""
    nthr = numba.get_num_threads()
    tal = np.zeros((nthr, nr_j, nmu_j, 3))
    per = n_phot // n_chunks
    rout = P[6] * 1.02
    for c in prange(n_chunks):
        tid = numba.get_thread_id()
        np.random.seed(seed + 7919 * c)
        for _ in range(per):
            k = min(np.searchsorted(cdf, np.random.random()), cdf.shape[0] - 1)
            it, ir = k // n_rc, k % n_rc
            r = (r_w3[ir] + np.random.random() * (r_w3[ir + 1] - r_w3[ir])) ** (1.0 / 3.0)
            ct = ct_w[it] + np.random.random() * (ct_w[it + 1] - ct_w[it])
            st = np.sqrt(max(0.0, 1.0 - ct * ct))
            ph = 2 * np.pi * np.random.random()
            x, y, z = r * st * np.cos(ph), r * st * np.sin(ph), r * ct
            uz = 2 * np.random.random() - 1.0
            sp = np.sqrt(max(0.0, 1.0 - uz * uz))
            ph = 2 * np.pi * np.random.random()
            _propagate(x, y, z, sp * np.cos(ph), sp * np.sin(ph), uz, 1.0, P, kb, alb, icdf, rout,
                       elnr0, edlnr, nr_j, nmu_j, frac, w_min, tal[tid])
    return tal.sum(axis=0)


def thermal_luminosity(model, grid, opt):
    """(per-bin total dust emission W nm^-1, per-bin cell cdf (nb, n_cells)) from the Hyperion
    cells: L = 4 pi kappa_abs rho B_lambda(T) V."""
    from .model import AU_CM
    r_w, t_w = model.r_walls * AU_CM, model.t_walls
    vol = (np.diff(r_w ** 3) / 3.0)[None, :] * np.abs(np.diff(np.cos(t_w)))[:, None] * 2 * np.pi * 1e-6  # m^3
    rho, T = model.rho_grid, model.temp_grid
    hc_k_nm, two_hc2 = 1.4387769e7, 2 * 6.62607015e-34 * 2.99792458e8 ** 2
    L = np.zeros(grid.n)
    cdfs = []
    for b, lam in enumerate(grid.wl):
        x = np.where(T > 0, hc_k_nm / (lam * np.maximum(T, 1e-3)), np.inf)
        with np.errstate(over="ignore"):
            B = np.where(x < 600, two_hc2 / (lam * 1e-9) ** 5 / np.expm1(np.minimum(x, 600)) * 1e-9, 0.0)
        kabs = opt["chi"][b] * (1 - opt["albedo"][b]) * 100.0  # chi [cm^2/g] * rho [g/cm^3] is per cm; x100 -> per m
        lc = 4 * np.pi * kabs * rho * B * vol
        L[b] = lc.sum()
        cdf = np.cumsum(lc.ravel())
        cdfs.append(cdf / cdf[-1] if cdf[-1] > 0 else cdf)
    return L, np.array(cdfs)


@njit(parallel=True)
def _mc_bin(n_phot, P, kb, alb, icdf, ncol, lnr0, dlnr, mu0, dmu, mu_max,
            elnr0, edlnr, nr_j, nmu_j, frac, w_min, seed, n_chunks):
    nthr = numba.get_num_threads()
    tal = np.zeros((nthr, nr_j, nmu_j, 3))
    per = n_phot // n_chunks
    nic = icdf.shape[0]
    rout = P[6] * 1.02
    for c in prange(n_chunks):
        tid = numba.get_thread_id()
        np.random.seed(seed + 7919 * c)
        for _ in range(per):
            m0 = np.random.random() * mu_max
            sgn = 1.0 if np.random.random() < 0.5 else -1.0
            ph = 2 * np.pi * np.random.random()
            st = np.sqrt(1 - m0 * m0)
            ux, uy, uz = st * np.cos(ph), st * np.sin(ph), sgn * m0
            # optical depth to infinity along this radial ray (last table row)
            fy = (m0 - mu0) / dmu
            j = min(int(fy), ncol.shape[1] - 2)
            t = fy - j
            tinf = kb * ((1 - t) * ncol[-1, j] + t * ncol[-1, j + 1])
            if tinf < 1e-8:
                continue
            e = 1.0 - np.exp(-tinf)
            w = mu_max * e * alb
            tau = -np.log(1.0 - np.random.random() * e)
            r = _radial_tau_point(ncol, lnr0, dlnr, mu0, dmu, m0, tau / kb)
            x, y, z = r * ux, r * uy, r * uz
            f = np.random.random() * (nic - 1)
            i = min(int(f), nic - 2)
            cost = icdf[i] + (f - i) * (icdf[i + 1] - icdf[i])
            ux, uy, uz = scatter_dir(ux, uy, uz, cost, 2 * np.pi * np.random.random())
            _propagate(x, y, z, ux, uy, uz, w, P, kb, alb, icdf, rout, elnr0, edlnr, nr_j, nmu_j,
                       frac, w_min, tal[tid])
    return tal.sum(axis=0)


def scattered_field(model, grid, n_phot=2_000_000, nr_j=240, nmu_j=400, frac=0.2,
                    w_min=1e-3, seed=1, log=print, thermal=False):
    """Run the Monte Carlo for every bin of `grid`. Returns dict with J, HR, HZ
    (nb, nr_j, nmu_j) in W m^-2 nm^-1 sr^-1, the cell axes (lnr edges, mu edges), the number of
    tallied cells per bin and timing. thermal: also add the field of the dust's own thermal
    emission (direct + scattered), needed in the near/mid-IR where the hot inner rim emits."""
    opt = model.optics(grid)
    tab = model.tables()
    lnr, mu = tab["lnr"], tab["mu"]
    ncol = tab["ncol"]
    # largest |cos theta| with a non-negligible radial optical depth in the most opaque bin
    tinf = opt["k"].max() * ncol[-1]
    mu_max = float(min(1.0, mu[np.nonzero(tinf > 1e-8)[0].max()] + 2 * (mu[1] - mu[0])))
    elnr = np.linspace(lnr[0], lnr[-1], nr_j + 1)
    emu = np.linspace(0.0, 1.0, nmu_j + 1)
    r_e = np.exp(elnr)
    vol = 2 * 2 * np.pi / 3 * np.diff(r_e ** 3)[:, None] * np.diff(emu)[None, :]  # AU^3, both halves
    lb = model.star_bins(grid)
    out = np.zeros((3, grid.n, nr_j, nmu_j))
    if thermal:
        l_th, cdfs = thermal_luminosity(model, grid, opt)
        r_w3 = np.maximum(model.r_walls, 0.0) ** 3
        ct_w = np.cos(model.t_walls)
        n_rc = len(model.r_walls) - 1
    th_frac = np.zeros(grid.n)
    n_chunks = max(64, 4 * numba.get_num_threads())
    t0 = time.time()
    for b in range(grid.n):
        if opt["albedo"][b] < 1e-3:  # thermal-IR / mm bands: no scattered field
            log(f"  mc bin {b} ({grid.wl[b]:.0f} nm): albedo {opt['albedo'][b]:.1e}, skipped")
            continue
        tal = _mc_bin(n_phot, model.params, opt["k"][b], opt["albedo"][b], opt["icdf"][b], ncol,
                      lnr[0], lnr[1] - lnr[0], mu[0], mu[1] - mu[0], mu_max,
                      elnr[0], elnr[1] - elnr[0], nr_j, nmu_j, frac, w_min, seed + 1000 * b, n_chunks)
        n_eff = (n_phot // n_chunks) * n_chunks
        norm = lb[b] / n_eff / (4 * np.pi * vol) / AU_M ** 2
        for q in range(3):
            out[q, b] = tal[..., q] * norm
        if thermal and l_th[b] > 0:
            tt = _mc_thermal_bin(n_phot, model.params, opt["k"][b], opt["albedo"][b], opt["icdf"][b], cdfs[b],
                                 r_w3, ct_w, n_rc, elnr[0], elnr[1] - elnr[0], nr_j, nmu_j, frac, w_min,
                                 seed + 1000 * b + 500, n_chunks)
            tnorm = l_th[b] / n_eff / (4 * np.pi * vol) / AU_M ** 2
            jt = tt[..., 0] * tnorm
            th_frac[b] = jt.sum() / max((out[0, b] + jt).sum(), 1e-300)
            for q in range(3):
                out[q, b] += tt[..., q] * tnorm
            log(f"  mc bin {b}: thermal L {l_th[b]:.3g} W/nm (star {lb[b]:.3g}), thermal share of sum J {th_frac[b]:.2f}")
        log(f"  mc bin {b} ({grid.wl[b]:.0f} nm): {time.time() - t0:.0f} s, cells hit {int((tal[..., 0] > 0).sum())}")
    return {"J": out[0], "HR": out[1], "HZ": out[2], "elnr": elnr, "emu": emu, "mu_max": mu_max,
            "n_phot": n_phot, "thermal": thermal, "thermal_frac": th_frac, "seconds": time.time() - t0}


def main():
    import argparse
    from nightsky import spectral
    from .model import DiskModel
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--model", default="dIULECee")
    ap.add_argument("--nphot", type=float, default=2e6)
    ap.add_argument("--nbins", type=int, default=20)
    a = ap.parse_args()
    m = DiskModel(a.model)
    g = spectral.default_grid(a.nbins)
    res = scattered_field(m, g, n_phot=int(a.nphot))
    np.savez_compressed(a.out, wl=g.wl, model=m.name, **res)
    print("wrote", a.out, f"{res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
