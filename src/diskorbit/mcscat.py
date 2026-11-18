"""Scattered-light radiation field of the disk: Monte Carlo per wavelength bin.

Photon packets leave the star toward the disk (directions with optical depth to infinity
> 1e-8, weighted by that solid-angle fraction), are forced to interact (weight 1 - e^-tau_inf)
and from their first scattering on are followed with weight reduction by the albedo and
Russian roulette; scattering angles are drawn from the tabulated dust phase function. Path
lengths of the scattered packets are tallied on a (phi, ln r, mu = |z|/r) grid (a single phi
cell for axisymmetric models), giving the mean intensity J and the first moment
H = (1/4pi) int I n dOmega (cylindrical-radial, away-from-midplane and azimuthal components) of
the SCATTERED light. Direct starlight is not tallied: the renderer adds the single-scattered term
exactly. The renderer's multiple-scattering source is albedo * (J + 3 g H.n), the exact
scattering integral for a radiance field with only the two lowest angular moments.

Output units: W m^-2 nm^-1 sr^-1 (mean intensity per nm, bin average).
"""
import time

import numba
import numpy as np
from numba import njit, prange

from .kernels import density3, geo_step, scatter_dir, phi_weights
from .model import AU_M


@njit(inline="always")
def _col_at(ncol, i, j, t, k0, k1, tk):
    a = (1 - t) * ncol[k0, i, j] + t * ncol[k0, i, j + 1]
    if tk == 0.0:
        return a
    return (1 - tk) * a + tk * ((1 - t) * ncol[k1, i, j] + t * ncol[k1, i, j + 1])


@njit(cache=True)
def _radial_tau_point(ncol, lnr0, dlnr, mu0, dmu, tphi0, tdphi, mu, phi, target):
    """Radius (AU) along the radial ray (|cos theta| = mu, azimuth phi) where the column from the
    star reaches target. ncol: (nphi, n_r, n_mu)."""
    nr, nm = ncol.shape[1], ncol.shape[2]
    fy = (mu - mu0) / dmu
    j = min(max(int(fy), 0), nm - 2)
    t = min(max(fy - j, 0.0), 1.0)
    k0, k1, tk = phi_weights(tphi0, tdphi, ncol.shape[0], phi)
    lo, hi = 0, nr - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _col_at(ncol, mid, j, t, k0, k1, tk) < target:
            lo = mid
        else:
            hi = mid
    a = _col_at(ncol, lo, j, t, k0, k1, tk)
    b = _col_at(ncol, hi, j, t, k0, k1, tk)
    f = 0.0 if b <= a else (target - a) / (b - a)
    return np.exp(lnr0 + (lo + f) * dlnr)


@njit(inline="always")
def _tally(tal, x, y, z, ux, uy, uz, w, s, elnr0, edlnr, nr_j, nmu_j, nphi_j):
    R = np.sqrt(x * x + y * y)
    rm = np.sqrt(R * R + z * z)
    if rm <= 0:
        return
    fi = (np.log(rm) - elnr0) / edlnr
    if fi < 0 or fi >= nr_j:
        return
    ia = int(fi)
    ib = min(int(abs(z) / rm * nmu_j), nmu_j - 1)
    ic = 0
    if nphi_j > 1:
        ic = min(int((np.arctan2(y, x) + np.pi) / (2 * np.pi) * nphi_j), nphi_j - 1)
    zs = 1.0 if z >= 0 else -1.0
    uR, uP = 0.0, 0.0
    if R > 0:
        uR = (x * ux + y * uy) / R
        uP = (x * uy - y * ux) / R
    tal[ic, ia, ib, 0] += w * s
    tal[ic, ia, ib, 1] += w * s * uR
    tal[ic, ia, ib, 2] += w * s * uz * zs
    tal[ic, ia, ib, 3] += w * s * uP


@njit(inline="always")
def _propagate(x, y, z, ux, uy, uz, w, P, PT, LD, kb, alb, icdf, rout, elnr0, edlnr, nr_j, nmu_j,
               frac, w_min, tal):
    """Follow a packet from (x, y, z) in direction u with weight w until it escapes or is
    terminated, scattering with albedo weighting and Russian roulette, and tally w * path
    length (and its R / away-from-midplane / azimuthal projections) into
    tal[phi cell, ln r cell, mu cell, 4]."""
    nic = icdf.shape[0]
    nphi_j = tal.shape[0]
    trem = -np.log(1.0 - np.random.random())
    for _step in range(2000000):
        R = np.sqrt(x * x + y * y)
        rr = np.sqrt(R * R + z * z)
        if rr > rout and x * ux + y * uy + z * uz >= 0:
            break
        s = geo_step(R, z, rr, P, frac)
        xm, ym, zm = x + 0.5 * s * ux, y + 0.5 * s * uy, z + 0.5 * s * uz
        kr = kb * density3(xm, ym, zm, P, PT, LD)
        hit = False
        if kr * s >= trem:
            s = trem / kr
            hit = True
            xm, ym, zm = x + 0.5 * s * ux, y + 0.5 * s * uy, z + 0.5 * s * uz
        _tally(tal, xm, ym, zm, ux, uy, uz, w, s, elnr0, edlnr, nr_j, nmu_j, nphi_j)
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
def _mc_thermal_bin(n_phot, P, PT, LD, kb, alb, icdf, cdf, r_w3, ct_w, p_w, n_rc, n_tc,
                    elnr0, edlnr, nphi_j, nr_j, nmu_j, frac, w_min, seed, n_chunks):
    """Dust thermal emission packets: cell drawn from cdf (flattened (phi, theta, r) Hyperion
    cells, luminosity-weighted), position uniform in the cell volume, isotropic direction,
    weight 1. The whole path including the first flight is tallied (the renderer has no exact
    direct thermal term; its scattering source albedo * J must include the direct thermal
    field)."""
    nthr = numba.get_num_threads()
    tal = np.zeros((nthr, nphi_j, nr_j, nmu_j, 4))
    per = n_phot // n_chunks
    rout = P[6] * 1.02
    for c in prange(n_chunks):
        tid = numba.get_thread_id()
        np.random.seed(seed + 7919 * c)
        for _ in range(per):
            k = min(np.searchsorted(cdf, np.random.random()), cdf.shape[0] - 1)
            ip = k // (n_tc * n_rc)
            it = (k // n_rc) % n_tc
            ir = k % n_rc
            r = (r_w3[ir] + np.random.random() * (r_w3[ir + 1] - r_w3[ir])) ** (1.0 / 3.0)
            ct = ct_w[it] + np.random.random() * (ct_w[it + 1] - ct_w[it])
            st = np.sqrt(max(0.0, 1.0 - ct * ct))
            ph = p_w[ip] + np.random.random() * (p_w[ip + 1] - p_w[ip])
            x, y, z = r * st * np.cos(ph), r * st * np.sin(ph), r * ct
            uz = 2 * np.random.random() - 1.0
            sp = np.sqrt(max(0.0, 1.0 - uz * uz))
            ph = 2 * np.pi * np.random.random()
            _propagate(x, y, z, sp * np.cos(ph), sp * np.sin(ph), uz, 1.0, P, PT, LD, kb, alb, icdf, rout,
                       elnr0, edlnr, nr_j, nmu_j, frac, w_min, tal[tid])
    return tal.sum(axis=0)


@njit(parallel=True)
def _mc_point_bin(n_phot, x0, y0, z0, P, PT, LD, kb, alb, icdf, elnr0, edlnr, nphi_j, nr_j, nmu_j, frac,
                  w_min, seed, n_chunks):
    """Isotropic point source at (x0, y0, z0) (e.g. a planet), weight 1; the whole path including
    the direct flight is tallied (as for the thermal packets)."""
    nthr = numba.get_num_threads()
    tal = np.zeros((nthr, nphi_j, nr_j, nmu_j, 4))
    per = n_phot // n_chunks
    rout = P[6] * 1.02
    for c in prange(n_chunks):
        tid = numba.get_thread_id()
        np.random.seed(seed + 7919 * c)
        for _ in range(per):
            uz = 2 * np.random.random() - 1.0
            sp = np.sqrt(max(0.0, 1.0 - uz * uz))
            ph = 2 * np.pi * np.random.random()
            _propagate(x0, y0, z0, sp * np.cos(ph), sp * np.sin(ph), uz, 1.0, P, PT, LD, kb, alb, icdf, rout,
                       elnr0, edlnr, nr_j, nmu_j, frac, w_min, tal[tid])
    return tal.sum(axis=0)


def blackbody_bins(grid, lum_w, temp_k):
    """W nm^-1 per bin (bin averages) of a blackbody of total luminosity lum_w."""
    sub = grid.subsample()
    x = np.minimum(1.4387769e7 / (sub * temp_k), 600.0)
    b = 2 * 6.62607015e-34 * 2.99792458e8 ** 2 / (sub * 1e-9) ** 5 / np.expm1(x) * 1e-9  # W m^-2 nm^-1 sr^-1
    sigma_t4 = 5.670374419e-8 * temp_k ** 4
    return (lum_w * np.pi * b / sigma_t4).reshape(grid.n, -1).mean(axis=1)


def thermal_luminosity(model, grid, opt):
    """(per-bin total dust emission W nm^-1, per-bin cell cdf (nb, n_cells)) from the Hyperion
    cells: L = 4 pi kappa_abs rho B_lambda(T) V. Cells are (phi, theta, r); axisymmetric models
    have one phi cell spanning 2 pi."""
    from .model import AU_CM
    r_w, t_w, p_w, rho, T = model.cells()
    r_w = r_w * AU_CM
    vol = ((np.diff(p_w))[:, None, None] * np.abs(np.diff(np.cos(t_w)))[None, :, None]
           * (np.diff(r_w ** 3) / 3.0)[None, None, :] * 1e-6)  # m^3
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
def _mc_bin(n_phot, P, PT, LD, kb, alb, icdf, ncol, lnr0, dlnr, mu0, dmu, tphi0, tdphi, mu_max,
            elnr0, edlnr, nphi_j, nr_j, nmu_j, frac, w_min, seed, n_chunks):
    nthr = numba.get_num_threads()
    tal = np.zeros((nthr, nphi_j, nr_j, nmu_j, 4))
    per = n_phot // n_chunks
    nic = icdf.shape[0]
    rout = P[6] * 1.02
    nr = ncol.shape[1]
    for c in prange(n_chunks):
        tid = numba.get_thread_id()
        np.random.seed(seed + 7919 * c)
        for _ in range(per):
            m0 = np.random.random() * mu_max
            sgn = 1.0 if np.random.random() < 0.5 else -1.0
            ph = 2 * np.pi * np.random.random() - np.pi
            st = np.sqrt(1 - m0 * m0)
            ux, uy, uz = st * np.cos(ph), st * np.sin(ph), sgn * m0
            # optical depth to infinity along this radial ray (last table row)
            fy = (m0 - mu0) / dmu
            j = min(int(fy), ncol.shape[2] - 2)
            t = fy - j
            k0, k1, tk = phi_weights(tphi0, tdphi, ncol.shape[0], ph)
            tinf = kb * _col_at(ncol, nr - 1, j, t, k0, k1, tk)
            if tinf < 1e-8:
                continue
            e = 1.0 - np.exp(-tinf)
            w = mu_max * e * alb
            tau = -np.log(1.0 - np.random.random() * e)
            r = _radial_tau_point(ncol, lnr0, dlnr, mu0, dmu, tphi0, tdphi, m0, ph, tau / kb)
            x, y, z = r * ux, r * uy, r * uz
            f = np.random.random() * (nic - 1)
            i = min(int(f), nic - 2)
            cost = icdf[i] + (f - i) * (icdf[i + 1] - icdf[i])
            ux, uy, uz = scatter_dir(ux, uy, uz, cost, 2 * np.pi * np.random.random())
            _propagate(x, y, z, ux, uy, uz, w, P, PT, LD, kb, alb, icdf, rout, elnr0, edlnr, nr_j, nmu_j,
                       frac, w_min, tal[tid])
    return tal.sum(axis=0)


def scattered_field(model, grid, n_phot=2_000_000, nr_j=240, nmu_j=400, nphi_j=None, frac=0.2,
                    w_min=1e-3, seed=1, log=print, thermal=False, planet_frac=0.1):
    """Run the Monte Carlo for every bin of `grid`. Returns dict with J, HR, HZ, HP
    (nb, nphi_j, nr_j, nmu_j) in W m^-2 nm^-1 sr^-1, the cell axes (lnr edges, mu edges, nphi_j),
    the number of tallied cells per bin and timing. nphi_j defaults to the model's
    (model.mc_nphi; 1 for axisymmetric models). thermal: also add the field of the dust's own
    thermal emission (direct + scattered), needed in the near/mid-IR where the hot inner rim
    emits. A model with a .planet point source (dict lum_w, temp_k, x_au; on the x axis) adds its
    direct + scattered field with planet_frac * n_phot packets per bin."""
    opt = model.optics(grid)
    tab = model.tables()
    lnr, mu, ncol = tab["lnr"], tab["mu"], tab["ncol"]
    nphi_j = nphi_j or getattr(model, "mc_nphi", 1)
    PT, LD = model.pert
    # largest |cos theta| with a non-negligible radial optical depth in the most opaque bin
    tinf = opt["k"].max() * ncol[:, -1, :].max(axis=0)
    mu_max = float(min(1.0, mu[np.nonzero(tinf > 1e-8)[0].max()] + 2 * (mu[1] - mu[0])))
    elnr = np.linspace(lnr[0], lnr[-1], nr_j + 1)
    emu = np.linspace(0.0, 1.0, nmu_j + 1)
    r_e = np.exp(elnr)
    vol = (2 * 2 * np.pi / 3 * np.diff(r_e ** 3)[:, None] * np.diff(emu)[None, :]) / nphi_j  # AU^3, both halves
    lb = model.star_bins(grid)
    out = np.zeros((4, grid.n, nphi_j, nr_j, nmu_j), np.float32)
    if thermal:
        l_th, cdfs = thermal_luminosity(model, grid, opt)
        c_r, c_t, p_w = model.cells()[:3]
        r_w3 = np.maximum(c_r, 0.0) ** 3
        ct_w = np.cos(c_t)
        n_rc, n_tc = len(c_r) - 1, len(c_t) - 1
    th_frac = np.zeros(grid.n)
    n_chunks = max(64, 4 * numba.get_num_threads())
    t0 = time.time()
    tphi0, tdphi = tab["phi0"], tab["dphi"]
    for b in range(grid.n):
        if opt["albedo"][b] < 1e-3:  # thermal-IR / mm bands: no scattered field
            log(f"  mc bin {b} ({grid.wl[b]:.0f} nm): albedo {opt['albedo'][b]:.1e}, skipped")
            continue
        tal = _mc_bin(n_phot, model.params, PT, LD, opt["k"][b], opt["albedo"][b], opt["icdf"][b], ncol,
                      lnr[0], lnr[1] - lnr[0], mu[0], mu[1] - mu[0], tphi0, tdphi, mu_max,
                      elnr[0], elnr[1] - elnr[0], nphi_j, nr_j, nmu_j, frac, w_min, seed + 1000 * b, n_chunks)
        n_eff = (n_phot // n_chunks) * n_chunks
        norm = lb[b] / n_eff / (4 * np.pi * vol) / AU_M ** 2
        for q in range(4):
            out[q, b] = tal[..., q] * norm[None]
        if thermal and l_th[b] > 0:
            tt = _mc_thermal_bin(n_phot, model.params, PT, LD, opt["k"][b], opt["albedo"][b], opt["icdf"][b], cdfs[b],
                                 r_w3, ct_w, p_w, n_rc, n_tc, elnr[0], elnr[1] - elnr[0], nphi_j, nr_j, nmu_j,
                                 frac, w_min, seed + 1000 * b + 500, n_chunks)
            tnorm = l_th[b] / n_eff / (4 * np.pi * vol) / AU_M ** 2
            jt = tt[..., 0] * tnorm[None]
            th_frac[b] = jt.sum() / max((out[0, b] + jt).sum(), 1e-300)
            for q in range(4):
                out[q, b] += tt[..., q] * tnorm[None]
            log(f"  mc bin {b}: thermal L {l_th[b]:.3g} W/nm (star {lb[b]:.3g}), thermal share of sum J {th_frac[b]:.2f}")
        pl = getattr(model, "planet", None)
        if pl is not None:
            lp = blackbody_bins(grid, pl["lum_w"], pl["temp_k"])[b]
            n_p = max(int(n_phot * planet_frac), n_chunks)
            tp = _mc_point_bin(n_p, pl["x_au"], 0.0, 0.0, model.params, PT, LD, opt["k"][b], opt["albedo"][b],
                               opt["icdf"][b], elnr[0], elnr[1] - elnr[0], nphi_j, nr_j, nmu_j, frac, w_min,
                               seed + 1000 * b + 700, n_chunks)
            pnorm = lp / ((n_p // n_chunks) * n_chunks) / (4 * np.pi * vol) / AU_M ** 2
            for q in range(4):
                out[q, b] += tp[..., q] * pnorm[None]
        log(f"  mc bin {b} ({grid.wl[b]:.0f} nm): {time.time() - t0:.0f} s, cells hit {int((tal[..., 0] > 0).sum())}")
    return {"J": out[0], "HR": out[1], "HZ": out[2], "HP": out[3], "elnr": elnr, "emu": emu, "nphi_j": nphi_j,
            "mu_max": mu_max, "n_phot": n_phot, "thermal": thermal, "thermal_frac": th_frac,
            "seconds": time.time() - t0}


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
