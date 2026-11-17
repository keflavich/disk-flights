"""Validation of nightsky.gas against literature column densities and emission measures.

Line integrals use a self-contained numba ray marcher (trilinear sampling, step = half of the
finest voxel at the current point) over the gas VoxelFields only (no stars, no Galactic disk
foreground). Outputs:
  * Earth view (Sun at x = R0), |l| < 1.5 deg, |b| < 0.5 deg at 0.005 deg: H-alpha emission
    measure EM = int n_e^2 ds (pc cm^-6) and N(H2) = N_H / 2 with N_H = A_V 1.9e21 / 2 cm^-2.
  * GC observer (Arches + 3 pc along +x) all-sky maps in observer-centred galactic-like
    coordinates (lon 0 = -x, i.e. away from the Sun; lon 90 = +y; lat 90 = +z): H-alpha
    surface brightness (Rayleigh) without and with the gas' own extinction (tau at 656 nm from
    the extinction fields), and A_V to the edge of the model.
References for the printed checks: Lang+2001 (Arched Filaments EM 1e5-1e6), Lang+1997
(Sickle), Zhao+2009 (Sgr A West), Lis & Goldsmith 1990 / Schmiedeke+2016 (Sgr B2), Longmore+
2012 (Brick), Molinari+2011 / Battersby+2011 (CMZ columns), Dahmen+1998 / Henshaw+2023 (CMZ
mass), Figer+2002 (Arches Q).
Usage: python3 -m nightsky.validate_gas [--resolution default] [--seed 0] [--out DIR]
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba
import numpy as np
from numba.typed import List

from . import gas, spectral

OUT = "/orange/adamginsburg/gc/visualization/renders/nightsky/validation/gas"


def _pack(fields):
    lst = List()
    for f in fields:
        lst.append(np.ascontiguousarray(f.data, np.float32))
    o = np.array([f.origin_pc for f in fields], float).reshape(-1, 3)
    v = np.array([f.voxel_pc for f in fields], float)
    return lst, o, v


@numba.njit(cache=True, inline="always")
def _tri(d, o, v, px, py, pz, out):
    nx, ny, nz, nc = d.shape
    fx, fy, fz = (px - o[0]) / v, (py - o[1]) / v, (pz - o[2]) / v
    i, j, k = int(np.floor(fx)), int(np.floor(fy)), int(np.floor(fz))
    if i < 0 or j < 0 or k < 0 or i >= nx - 1 or j >= ny - 1 or k >= nz - 1:
        return False
    ax, ay, az = fx - i, fy - j, fz - k
    for c in range(nc):
        out[c] = ((d[i, j, k, c] * (1 - ax) + d[i + 1, j, k, c] * ax) * (1 - ay)
                  + (d[i, j + 1, k, c] * (1 - ax) + d[i + 1, j + 1, k, c] * ax) * ay) * (1 - az) + \
                 ((d[i, j, k + 1, c] * (1 - ax) + d[i + 1, j, k + 1, c] * ax) * (1 - ay)
                  + (d[i, j + 1, k + 1, c] * (1 - ax) + d[i + 1, j + 1, k + 1, c] * ax) * ay) * az
    return True


@numba.njit(parallel=True, cache=True)
def _march(org, dirs, t0, t1, idata, io, iv, iinvk, mdata, mo, mv, tau_ha_per_av, out):
    """out[:, 0] EM (pc cm^-6), 1 H-alpha (W m^-2 sr^-1), 2 attenuated H-alpha, 3 A_V."""
    nbi, nbm = len(idata), len(mdata)
    for r in numba.prange(dirs.shape[0]):
        buf = np.zeros(8)
        t = t0[r]
        em, ha, haa, av = 0.0, 0.0, 0.0, 0.0
        while t < t1[r]:
            st = 4.0
            px, py, pz = org[0] + t * dirs[r, 0], org[1] + t * dirs[r, 1], org[2] + t * dirs[r, 2]
            for b in range(nbi):
                d = idata[b]
                if (px >= io[b, 0] and py >= io[b, 1] and pz >= io[b, 2] and px <= io[b, 0] + (d.shape[0] - 1) * iv[b]
                        and py <= io[b, 1] + (d.shape[1] - 1) * iv[b] and pz <= io[b, 2] + (d.shape[2] - 1) * iv[b]):
                    st = min(st, 0.5 * iv[b])
            for b in range(nbm):
                d = mdata[b]
                if (px >= mo[b, 0] and py >= mo[b, 1] and pz >= mo[b, 2] and px <= mo[b, 0] + (d.shape[0] - 1) * mv[b]
                        and py <= mo[b, 1] + (d.shape[1] - 1) * mv[b] and pz <= mo[b, 2] + (d.shape[2] - 1) * mv[b]):
                    st = min(st, 0.5 * mv[b])
            st = min(st, t1[r] - t)
            tm = t + 0.5 * st
            px, py, pz = org[0] + tm * dirs[r, 0], org[1] + tm * dirs[r, 1], org[2] + tm * dirs[r, 2]
            j = 0.0
            for b in range(nbi):
                d = idata[b]
                if _tri(d, io[b], iv[b], px, py, pz, buf):
                    for c in range(d.shape[3]):
                        j += buf[c]
                        em += buf[c] * iinvk[b, c] * st
            a = 0.0
            for b in range(nbm):
                if _tri(mdata[b], mo[b], mv[b], px, py, pz, buf):
                    a += buf[0]
            ha += j * st
            haa += j * st * np.exp(-(av + 0.5 * a * st) * tau_ha_per_av)
            av += a * st
            t += st
        out[r, 0], out[r, 1], out[r, 2], out[r, 3] = em, ha, haa, av


def ray_box(org, dirs, lo, hi):
    d = np.where(np.abs(dirs) < 1e-12, 1e-12, dirs)
    ta, tb = (lo - org) / d, (hi - org) / d
    t0 = np.maximum(np.minimum(ta, tb).max(axis=1), 0.0)
    t1 = np.maximum(ta, tb).min(axis=1)
    return t0, np.maximum(t1, t0)


def integrate(ion, mol, org, dirs, grid, tmax=None):
    org = np.asarray(org, float)
    dirs = np.ascontiguousarray(dirs, float)
    lo = np.min([f.bounds_pc[0] for f in ion + mol], axis=0)
    hi = np.max([f.bounds_pc[1] for f in ion + mol], axis=0)
    t0, t1 = ray_box(np.broadcast_to(org, dirs.shape), dirs, lo, hi)
    if tmax is not None:
        t1 = np.minimum(t1, tmax)
        t0 = np.minimum(t0, t1)
    idata, io, iv = _pack(ion)
    mdata, mo, mv = _pack(mol)
    iinvk = np.zeros((len(ion), 8))
    for b, f in enumerate(ion):
        for c, te in enumerate(f.meta["T_e"]):
            iinvk[b, c] = 1.0 / gas.halpha_coeff(te)
    tau_ha = float(mol[0].spectra[0, grid.bin_index(656.28)])
    out = np.zeros((len(dirs), 4))
    _march(org, dirs, t0, t1, idata, io, iv, iinvk, mdata, mo, mv, tau_ha, out)
    return out


def region(m, L, B, l0, l1, b0, b1):
    sel = (L >= l0) & (L <= l1) & (B >= b0) & (B <= b1)
    return float(m[sel].max()), float(np.median(m[sel]))


def q_regions(ion):
    """Recombination rate Q = int n_e^2 alpha_B dV (s^-1) within spheres, and the total."""
    regs = dict(arches20=(gas.ARCHES, 20.0), sgra4=(gas.SGRA, 4.0), sgrb2_10=(gas.lbd(0.667, -0.036, 8.0), 10.0))
    out = {k: 0.0 for k in regs} | {"total": 0.0}
    for f in ion:
        ax = [f.origin_pc[a] + np.arange(f.shape[a]) * f.voxel_pc for a in range(3)]
        for c, te in enumerate(f.meta["T_e"]):
            q = f.data[..., c].astype(np.float64) / gas.halpha_coeff(te) * 2.59e-13 * (te / 1e4) ** -0.7 * f.voxel_pc ** 3 * gas.PC_CM ** 3
            out["total"] += float(q.sum())
            for k, (cen, r) in regs.items():
                r2 = ((ax[0] - cen[0]) ** 2)[:, None, None] + ((ax[1] - cen[1]) ** 2)[None, :, None] + ((ax[2] - cen[2]) ** 2)[None, None, :]
                out[k] += float(q[r2 < r * r].sum())
    return out


def mass(mol):
    return float(sum(f.data.sum() * f.voxel_pc ** 3 for f in mol) / gas.AV_PER_PC_PER_NH * gas.MSUN_PER_NH_PC3)


def lonlat_dirs(nlon, nlat):
    lon = np.radians(np.linspace(180, -180, nlon, endpoint=False) - 180.0 / nlon)
    lat = np.radians(np.linspace(90, -90, nlat, endpoint=False) - 90.0 / nlat)
    LO, LA = np.meshgrid(lon, lat)
    d = np.stack([-np.cos(LA) * np.cos(LO), np.cos(LA) * np.sin(LO), np.sin(LA)], -1)
    return d, np.degrees(LO), np.degrees(LA)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", default="default")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--nlon", type=int, default=720)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    grid = spectral.default_grid()
    t0 = time.time()
    if a.no_cache:
        ion, mol = gas.gc_ionized_gas(grid, a.seed, a.resolution), gas.gc_molecular_clouds(grid, a.seed, a.resolution)
    else:
        allf = gas.load_cached(grid, a.seed, a.resolution)
        ion = [f for f in allf if f.kind == "emission"]
        mol = [f for f in allf if f.kind == "extinction"]
    print(f"fields ready in {time.time() - t0:.1f}s: " + ", ".join(f"{f.name}{f.shape}" for f in ion + mol), flush=True)
    res = {}

    # ---- Earth view
    t0 = time.time()
    dl = 0.005
    l = np.arange(1.5, -1.5 - 1e-9, -dl)
    b = np.arange(-0.5, 0.5 + 1e-9, dl)
    L, B = np.meshgrid(l, b)
    lr, br = np.radians(L), np.radians(B)
    dirs = np.stack([-np.cos(br) * np.cos(lr), np.cos(br) * np.sin(lr), np.sin(br)], -1).reshape(-1, 3)
    o = integrate(ion, mol, [gas.R0, 0, 0], dirs, grid).reshape(L.shape + (4,))
    em, av = o[..., 0], o[..., 3]
    nh2 = av * gas.NH_PER_AV / gas.DUST_TO_GAS_GC / 2
    print(f"Earth view {L.shape} in {time.time() - t0:.1f}s", flush=True)
    lw = np.where(L > 180, L - 360, L)
    checks_em = dict(arched=((0.03, 0.17, 0.0, 0.14), "1e5-1e6 (Lang+2001)"),
                     sickle=((0.17, 0.21, -0.09, 0.0), "~1e5 (Lang+1997, order of magnitude)"),
                     sgrA_west=((-0.075, -0.035, -0.066, -0.026), "1e7-1e8 at arcsec resolution (Zhao+2009); lower at 18\""),
                     sgrB2=((0.64, 0.70, -0.07, 0.0), "1e7-1e9 in UCHII at arcsec resolution (De Pree+1998)"),
                     sgrB1=((0.46, 0.56, -0.10, 0.0), "1e4-1e5 (Mehringer+1992)"),
                     sgrC=((-0.62, -0.52, -0.14, -0.04), "~1e4-1e5 (Liszt 1995, order of magnitude)"))
    print("\nEarth view H-alpha emission measure (pc cm^-6), 0.005 deg = 18\" pixels: peak / median  [reference]")
    for k, (bx, ref) in checks_em.items():
        mx, md = region(em, lw, B, *bx)
        res[f"EM_{k}"] = (mx, md)
        print(f"  {k:10s} {mx:9.2e} / {md:9.2e}   [{ref}]")
    mx, md = region(em, lw, B, -1.0, 1.0, -0.2, 0.2)
    res["EM_cmz"] = (mx, md)
    print(f"  {'CMZ |l|<1':10s} {mx:9.2e} / {md:9.2e}   [diffuse ELD gas: EM ~1e3-1e4 (Mezger+1996)]")
    checks_nh = dict(sgrB2=((0.62, 0.72, -0.09, 0.02), "1e24-1e25 (Lis & Goldsmith 1990; Schmiedeke+2016)"),
                     brick=((0.23, 0.28, -0.01, 0.045), "~1e23 mean, peak few 1e23 (Longmore+2012)"),
                     dust_ridge=((0.32, 0.52, 0.0, 0.08), "few 1e22-1e23 (Walker+2018)"),
                     cloud_20kms=((-0.2, -0.06, -0.13, -0.03), "~1e23 (Herschel, Battersby+2011)"),
                     cloud_50kms=((-0.05, 0.01, -0.11, -0.03), "~1e23"),
                     sgrC=((-0.62, -0.48, -0.16, -0.05), "~1e23"),
                     sgrA_star=((-0.06, -0.05, -0.05, -0.04), "CMZ part of A_V towards Sgr A* << total A_V ~30"),
                     arches=((0.117, 0.127, 0.014, 0.024), "CMZ part of A_V towards the Arches << total A_V ~30"))
    print("\nEarth view N(H2) (cm^-2): peak / median  [reference]")
    for k, (bx, ref) in checks_nh.items():
        mx, md = region(nh2, lw, B, *bx)
        res[f"NH2_{k}"] = (mx, md)
        print(f"  {k:11s} {mx:9.2e} / {md:9.2e}  (A_V {md * 2 * gas.DUST_TO_GAS_GC / gas.NH_PER_AV:7.1f})   [{ref}]")
    mx, md = region(nh2, lw, B, -1.0, 1.0, -0.15, 0.15)
    res["NH2_cmz"] = (mx, md)
    print(f"  {'CMZ |l|<1':11s} {mx:9.2e} / {md:9.2e}   [typical 1e22-1e23 (Molinari+2011)]")
    for k, (ll, bb, p) in dict(SgrA_star=(-0.0558, -0.0462, gas.SGRA), Arches=(0.1217, 0.0188, gas.ARCHES)).items():
        dist = np.linalg.norm(p - [gas.R0, 0, 0])
        lr, br = np.radians(ll), np.radians(bb)
        v = np.array([[-np.cos(br) * np.cos(lr), np.cos(br) * np.sin(lr), np.sin(br)]])
        av_front = float(integrate(ion, mol, [gas.R0, 0, 0], v, grid, tmax=dist)[0, 3])
        res[f"AV_front_{k}"] = av_front
        print(f"  A_V of the model gas in front of {k} (Earth view): {av_front:.1f} mag   [observed total A_V ~30-45 incl. the "
              "Galactic disk foreground (A_Ks ~2.5-3, Nishiyama+2008; Fritz+2011), so the CMZ part is <~10]")
    mtot = mass(mol)
    res["mass_total"] = mtot
    print(f"\nmolecular + diffuse gas mass in the model: {mtot:.2e} Msun   [CMZ 2-6e7 (Dahmen+1998; Henshaw+2023)]")
    q = q_regions(ion)
    res["Q"] = q
    print(f"recombinations Q (s^-1): total {q['total']:.2e} [order 1e52 for the CMZ], within 20 pc of the Arches "
          f"{q['arches20']:.2e} [Arches Q ~4e51 + Quintuplet, Figer+2002], within 4 pc of Sgr A* {q['sgra4']:.2e} "
          f"[central cluster ~1e50-1e51], Sgr B2 10 pc {q['sgrb2_10']:.2e}")

    ext = [l[0], l[-1], b[0], b[-1]]
    fig, axs = plt.subplots(2, 1, figsize=(14, 9.5), constrained_layout=True)
    for ax, m, lab, vr in [(axs[0], em, "log10 EM [pc cm$^{-6}$]", (2, 7)), (axs[1], nh2, "log10 N(H$_2$) [cm$^{-2}$]", (21, 24.5))]:
        im = ax.imshow(np.log10(np.maximum(m, 1e-30)), origin="lower", extent=ext, cmap="inferno", vmin=vr[0], vmax=vr[1],
                       aspect="equal", interpolation="nearest")
        fig.colorbar(im, ax=ax, label=lab, shrink=0.8)
        ax.set_xlabel("l [deg]")
        ax.set_ylabel("b [deg]")
        for nm, (ll, bb) in dict(SgrB2=(0.667, -0.036), Brick=(0.253, 0.016), Arches=(0.1217, 0.0188),
                                 SgrA=(-0.0558, -0.0462), SgrC=(-0.57, -0.09), Sickle=(0.19, -0.04)).items():
            ax.annotate(nm, (ll, bb), xytext=(ll, bb + 0.25), color="cyan", fontsize=8, ha="center",
                        arrowprops=dict(arrowstyle="-", color="cyan", lw=0.5))
    axs[0].set_title("gas.py from Earth: H-alpha emission measure (no foreground extinction)")
    axs[1].set_title("gas.py from Earth: N(H2) = A_V 1.9e21 / (2 x 2) cm^-2")
    fig.savefig(os.path.join(a.out, "earth_em_nh2.png"), dpi=110)
    plt.close(fig)

    # ---- GC observer all-sky
    t0 = time.time()
    d, LO, LA = lonlat_dirs(a.nlon, a.nlon // 2)
    o = integrate(ion, mol, gas.OBSERVER, d.reshape(-1, 3), grid).reshape(LO.shape + (4,))
    ha_r, haa_r, av_s = o[..., 1] / gas.RAYLEIGH_HA, o[..., 2] / gas.RAYLEIGH_HA, o[..., 3]
    print(f"\nGC all-sky {LO.shape} in {time.time() - t0:.1f}s", flush=True)
    w = np.cos(np.radians(LA))
    wq = lambda m, p: float(np.interp(p, np.cumsum(w.ravel()[np.argsort(m.ravel())]) / w.sum(), np.sort(m.ravel())))
    res["allsky_Ha_R_p5_50_95_max"] = [wq(ha_r, 0.05), wq(ha_r, 0.5), wq(ha_r, 0.95), float(ha_r.max())]
    res["allsky_Ha_att_R_p5_50_95"] = [wq(haa_r, 0.05), wq(haa_r, 0.5), wq(haa_r, 0.95)]
    res["allsky_AV_p5_50_95"] = [wq(av_s, 0.05), wq(av_s, 0.5), wq(av_s, 0.95)]
    res["allsky_frac_AV_gt_1_3_10_30"] = [float((w * (av_s > x)).sum() / w.sum()) for x in (1, 3, 10, 30)]
    print("  H-alpha (R) unattenuated p5/p50/p95/max: " + " ".join(f"{v:.3g}" for v in res["allsky_Ha_R_p5_50_95_max"]))
    print("  H-alpha (R) self-attenuated p5/p50/p95:  " + " ".join(f"{v:.3g}" for v in res["allsky_Ha_att_R_p5_50_95"]))
    print("  A_V to model edge p5/p50/p95:            " + " ".join(f"{v:.3g}" for v in res["allsky_AV_p5_50_95"]))
    print("  sky fraction with A_V > 1, 3, 10, 30:     " + " ".join(f"{v:.3f}" for v in res["allsky_frac_AV_gt_1_3_10_30"]))
    print("  [references: an observer inside the Arches HII region sees ~1e3-1e4 R if the Arches' Lyc photons are absorbed"
          " within ~50 pc; 1 R ~ 2.4e-10 W m^-2 sr^-1 at H-alpha]")
    tgt = dict(SgrA=gas.SGRA, SgrB2=gas.lbd(0.667, -0.036, 8.0), Brick=gas.lbd(0.253, 0.016, gas._ring_point(0.253)),
               Quintuplet=gas.QUINTUPLET)
    for k, p in tgt.items():
        v = p - gas.OBSERVER
        dist = np.linalg.norm(v)
        r = integrate(ion, mol, gas.OBSERVER, (v / dist)[None], grid, tmax=dist)[0]
        res[f"AV_obs_to_{k}"] = float(r[3])
        print(f"  A_V from the observer to {k:10s} ({dist:6.1f} pc): {r[3]:8.2f} mag")
    d10, _, LA10 = lonlat_dirs(256, 128)
    r10 = integrate(ion, mol, gas.OBSERVER, d10.reshape(-1, 3), grid, tmax=10.0)[:, 3].reshape(LA10.shape)
    w10 = np.cos(np.radians(LA10))
    res["AV_within_10pc_mean_median"] = [float((r10 * w10).sum() / w10.sum()), float(np.median(r10))]
    loc = sum(f.sample(gas.OBSERVER[None])[0, 0] for f in mol)
    res["AV_per_pc_at_observer"] = float(loc)
    print(f"  A_V within 10 pc of the observer: sky mean {res['AV_within_10pc_mean_median'][0]:.2f}, median "
          f"{res['AV_within_10pc_mean_median'][1]:.2f} mag [target ~1]; A_V/pc at the observer {loc:.3f}")

    ext = [180, -180, -90, 90]
    fig, axs = plt.subplots(3, 1, figsize=(13, 18), constrained_layout=True)
    for ax, m, lab, vr, cm in [(axs[0], ha_r, "log10 H-alpha [R], no extinction", (2, 6.5), "magma"),
                               (axs[1], haa_r, "log10 H-alpha [R], with gas extinction", (2, 6.5), "magma"),
                               (axs[2], av_s, "log10 A_V [mag]", (-1, 3), "cividis")]:
        im = ax.imshow(np.log10(np.maximum(m, 1e-6)), extent=ext, cmap=cm, vmin=vr[0], vmax=vr[1], aspect="equal",
                       interpolation="nearest")
        fig.colorbar(im, ax=ax, label=lab, shrink=0.7)
        ax.set_xlabel("observer longitude [deg] (0 = away from the Sun, 90 = +y)")
        ax.set_ylabel("latitude [deg]")
        for nm, p in dict(Arches=gas.ARCHES, **tgt).items():
            v = p - gas.OBSERVER
            v = v / np.linalg.norm(v)
            ax.plot(np.degrees(np.arctan2(v[1], -v[0])), np.degrees(np.arcsin(v[2])), "c+", ms=8)
            ax.text(np.degrees(np.arctan2(v[1], -v[0])), np.degrees(np.arcsin(v[2])) + 3, nm, color="c", fontsize=8)
    axs[0].set_title("GC observer (Arches + 3 pc towards the Sun): gas only, equirectangular")
    fig.savefig(os.path.join(a.out, "gc_allsky_halpha_av.png"), dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for c in gas.ION_COMPONENTS:
        ax[0].step(grid.wl, gas.emission_spectrum(grid, c), where="mid", label=c)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("wavelength [nm]")
    ax[0].set_ylabel("spectrum per unit H-alpha [nm$^{-1}$]")
    ax[0].legend()
    ax[1].step(grid.wl, gas.extinction_spectrum(grid), where="mid")
    ax[1].set_xlabel("wavelength [nm]")
    ax[1].set_ylabel("tau / A_V (CCM89, R_V=3.1)")
    fig.savefig(os.path.join(a.out, "spectra.png"), dpi=100)
    plt.close(fig)
    np.savez_compressed(os.path.join(a.out, "maps.npz"), earth_em=em.astype(np.float32), earth_nh2=nh2.astype(np.float32),
                        l=l, b=b, gc_ha_R=ha_r.astype(np.float32), gc_ha_att_R=haa_r.astype(np.float32),
                        gc_av=av_s.astype(np.float32))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
