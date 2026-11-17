"""3D Milky Way + Galactic Center light and dust model with an observer-centred ray marcher.

Frames and units
----------------
Positions are GALACTOCENTRIC pc (frames.py: Sun at (R0, 0, 0), x from Sgr A* towards the Sun,
y towards l = 90 deg, z to the NGP). Sgr A* sits at its catalogue (l, b) = (359.9442, -0.0462)
deg at R0, i.e. at frames.lb_to_galactocentric(359.9442, -0.0462, R0) ~ (0, -8.0, -6.6) pc;
every component is centred on it. The model midplane is the plane through Sgr A* and the
point (R0, 0, 0), so an Earth observer at EARTH_OBSERVER_PC = (R0, 0, +20.8) sits 20.8 pc
above it (the Sun's height, Bennett & Bovy 2019; same position as starfield.EARTH_OBSERVER_PC).
Densities in Msun pc^-3; emissivity in W m^-2 sr^-1 nm^-1 per pc of path; opacity in optical
depth per pc; spectra are bin averages on the caller's SpectralGrid.

Stellar components (masses are defaults; overrides via keyword arguments)
-------------------------------------------------------------------------
* Nuclear star cluster: flattened Dehnen (1993) profile, rho ~ m^-gamma (1 + m/a)^(gamma-4),
  m^2 = R^2 + z^2/q^2, gamma = 1.8, a = 4.4 pc (projected R_eff ~ 4.2 pc), q = 0.71,
  M = 2.5e7 Msun (Schoedel et al. 2014; Neumayer, Seth & Boeker 2020).
* Nuclear stellar disk: two flattened stretched exponentials,
  rho = rho1 exp[-(a/R1)^n1] + rho2 exp[-(a/R2)^n2], a^2 = R^2 + z^2/q^2, q = 0.37,
  R1 = 5.06 pc, n1 = 0.72, R2 = 24.6 pc, n2 = 0.79, rho1/rho2 = 1.311 (Sormani et al. 2020
  deprojection of Gallego-Cano et al. 2020), normalised to M = 1.05e9 Msun (Sormani et al.
  2022; Launhardt, Zylka & Mezger 2002 give 1.4e9).
* Boxy/peanut bulge: Dwek et al. (1995) G2 bar, rho ~ exp(-r_s^2/2),
  r_s = [((x/x0)^2 + (y/y0)^2)^2 + (z/z0)^4]^(1/4), (x0, y0, z0) = (1.58, 0.62, 0.43) kpc,
  bar angle 27 deg with the near end at l > 0 (Wegg & Gerhard 2013), Gaussian cut-off beyond
  R = 2.8 kpc, M = 1.4e10 Msun.
* Long bar: exp(-sqrt((x/2.5 kpc)^2 + (y/0.6 kpc)^2)) exp(-|z|/180 pc), half-length 4.6 kpc,
  same angle, M = 5e9 Msun (Wegg, Gerhard & Portail 2015; bulge + bar ~1.8e10 Msun,
  Portail et al. 2017; Bland-Hawthorn & Gerhard 2016).
* Thin / thick disks: exp(-(R-R0)/Rd) exp(-|z|/h), Rd = 2.5 / 2.0 kpc, h = 320 / 900 pc,
  local midplane density 0.045 Msun pc^-3 (stars + remnants) with a thick/thin ratio of 0.05
  (Bland-Hawthorn & Gerhard 2016; McKee, Parravano & Hollenbach 2015), multiplied by an inner
  hole 1 - exp(-(R/1.5 kpc)^2) where the bar dominates.

Light: V-band luminosity density rho / (M/L_V), with M/L_V = 1.3 (thin disk, including its young
stars; Flynn et al. 2006 give 1.5 +- 0.2 for the solar cylinder), 3.0 (thick disk,
bulge, long bar), 2.0 (NSD, NSC; intermediate-age light, Nogueras-Lara et al. 2020). 1 Lsun_V
is the 550 nm monochromatic luminosity of an M_V = 4.83 source on the spectral.py Vega scale.
Spectral shapes are V-flux-weighted blackbody mixtures: old (4000/4800/6000 K = 0.45/0.35/0.20,
K-giant dominated), nuclear (4000/4800/6000/9000 K = 0.42/0.33/0.17/0.08) and thin disk
(4500/5800/8000 K = 0.35/0.40/0.25). j_lambda = L_lambda density / (4 pi).

Dust: A_V per pc = a0 exp(-(R-R0)/3 kpc) exp(-|z|/120 pc) f_in(R) f_LB + CMZ, a0 = 0.75 mag/kpc
(Drimmel & Spergel 2001; Marshall et al. 2006), a Local Bubble cavity around (R0, 0, 0),
f_LB = 1 - exp(-(d/120 pc)^4) (Lallement et al. 2019), f_in = d_in + (1-d_in)(1 - exp(-(R/2.5 kpc)^4))
the depletion inside the bar, and a Gaussian CMZ inter-cloud component (sigma_R = 120 pc,
sigma_z = 25 pc, 4 mag of A_V across its diameter). d_in is solved at construction so that
A_V(Sun -> Sgr A*) = 30 mag (A_Ks ~ 2.5 with the steep GC law of Nogueras-Lara et al. 2018).
Extinction curve: Cardelli, Clayton & Mathis (1989), R_V = 3.1, bin averaged;
tau_lambda = A_lambda / 1.0857.

Diffuse galactic light (optional, dgl=True; preset "milky_way_dgl"): single scattering by the
smooth dust only, albedo 0.6 and asymmetry g = 0.6 (Draine 2003, R_V = 3.1 dust in V), with the
Henyey-Greenstein phase function truncated to its dipole term, p = (1 + 3 g mu) / 4 pi. The source
is j_sc = albedo kappa_dust [J + 3 g <I d> . d_view] (clamped >= 0), with the mean intensity J and
first moment <I d> precomputed by the same marcher (stars + fields + dust, no scattering) on three
nested grids centred on Sgr A* (4 pc / HEALPix nside 4 within +-40 pc, 16 pc / nside 2 within
+-240 pc, 800 pc / nside 2 over the disk). Extinction fields scatter nothing (the coarse J grid
cannot follow the attenuated field inside dense clouds).

External VoxelFields (fields.py) are sampled trilinearly inside the marcher: kind "emission"
adds emissivity, kind "extinction" adds opacity (data = A_V per pc, spectra = tau / A_V).

Public API: GalaxyModel(grid, preset, fields, **overrides) with stellar_density, component_density,
luminosity_density_v, emissivity, opacity, march, render_map (optional pixel-area supersampling),
diffuse_fn, optical_depth, extinction_av; helpers sgr_a_star_pc, arches_pc, gc_observer_pc, ccm89,
sed_shape, pack_fields; constant EARTH_OBSERVER_PC.
CLI: python3 -m nightsky.galaxy --observer gc|earth --nside 256 [--supersample 2] --out map.npz [--set key=value].

Ray marcher: per direction, steps ds = max(ds_min, f_obs s), limited to f_c times the distance
to Sgr A* (resolves the NSC cusp), to step_vox voxels inside field bounds, and clipped at
field entry points and at r_res. Each step integrates exactly for constant j, kappa:
dI = j T (1 - exp(-kappa ds)) / kappa, T *= exp(-kappa ds). Default s_max = 25 kpc.
"""
import math
import time

import numpy as np
from numba import njit, prange

from . import frames, sky, spectral
import photometry as ph

PC_M = 3.0856775814913673e16
L550_SUN = 4 * np.pi * (10 * PC_M) ** 2 * ph.F_V0_550NM * 10 ** (-0.4 * 4.83)  # W/nm per Lsun_V
SGRA_LB = (359.9442, -0.0462)
ARCHES_LB = (0.1217, 0.0188)
EARTH_OBSERVER_PC = (frames.R0_PC, 0.0, 20.8)

COMPONENTS = ("nsc", "nsd", "bulge", "longbar", "thin", "thick")
NCOMP = len(COMPONENTS)

DEFAULTS = dict(
    R0=frames.R0_PC,
    # NSC
    M_nsc=2.5e7, nsc_a=4.4, nsc_gamma=1.8, nsc_q=0.71, nsc_mfloor=0.01,
    # NSD
    M_nsd=1.05e9, nsd_q=0.37, nsd_R1=5.06, nsd_n1=0.72, nsd_R2=24.6, nsd_n2=0.79, nsd_ratio=1.311,
    # bulge (G2) and long bar
    M_bulge=1.4e10, bar_angle_deg=27.0, bul_x0=1580.0, bul_y0=620.0, bul_z0=430.0,
    bul_Rend=2800.0, bul_hend=500.0,
    M_longbar=5.0e9, lb_x=2500.0, lb_y=600.0, lb_z=180.0, lb_L=4600.0, lb_hend=500.0,
    # disks
    rho_local=0.045, thick_ratio=0.05, thin_Rd=2500.0, thin_h=320.0, thick_Rd=2000.0,
    thick_h=900.0, disk_Rhole=1500.0,
    # mass-to-light (V)
    ML_nsc=2.0, ML_nsd=2.0, ML_bulge=3.0, ML_longbar=3.0, ML_thin=1.3, ML_thick=3.0,
    # dust
    dust=True, dust_scale=1.0, dust_a0=0.75e-3, bubble_R=120.0, dust_Rd=3000.0, dust_h=120.0, dust_Rhole=2500.0,
    dust_din=None, AV_sgra=30.0, cmz_AV=4.0, cmz_sR=120.0, cmz_sz=25.0, R_V=3.1,
    # spectral shapes: (temperatures, V-flux weights)
    sed_old=((4000.0, 4800.0, 6000.0), (0.45, 0.35, 0.20)),
    sed_nuclear=((4000.0, 4800.0, 6000.0, 9000.0), (0.42, 0.33, 0.17, 0.08)),
    sed_disk=((4500.0, 5800.0, 8000.0), (0.35, 0.40, 0.25)),
    # diffuse galactic light: (half_xy, half_z, voxel, healpix nside) grids centred on Sgr A*, finest first
    dgl=False, dgl_albedo=0.6, dgl_g=0.6,
    dgl_boxes=((40.0, 40.0, 4.0, 4), (240.0, 80.0, 16.0, 2), (16000.0, 1600.0, 800.0, 2)),
    # marcher
    s_max=25000.0, f_obs=0.03, f_c=0.1, ds_min=0.05, ds_core=0.005, ds_max=500.0, step_vox=0.7,
)
PRESETS = {"milky_way": {}, "no_dust": {"dust": False}, "milky_way_dgl": {"dgl": True}}

# parameter-array layout for the numba kernels
_P = ["nsc_rho0", "nsc_a", "nsc_gamma", "nsc_q", "nsc_mfloor",
      "nsd_rho1", "nsd_R1", "nsd_n1", "nsd_rho2", "nsd_R2", "nsd_n2", "nsd_q",
      "bul_rho0", "bul_x0", "bul_y0", "bul_z0", "bul_Rend", "bul_hend", "cos_bar", "sin_bar",
      "lb_rho0", "lb_x", "lb_y", "lb_z", "lb_L", "lb_hend",
      "thin_rho0", "thin_Rd", "thin_h", "thick_rho0", "thick_Rd", "thick_h", "disk_Rhole", "R0",
      "dust_a0", "dust_Rd", "dust_h", "dust_Rhole", "dust_din", "cmz_a0", "cmz_sR", "cmz_sz", "bubble_R"]
_I = {k: i for i, k in enumerate(_P)}
(I_NSC_RHO0, I_NSC_A, I_NSC_G, I_NSC_Q, I_NSC_MF, I_NSD_RHO1, I_NSD_R1, I_NSD_N1, I_NSD_RHO2,
 I_NSD_R2, I_NSD_N2, I_NSD_Q, I_BUL_RHO0, I_BUL_X0, I_BUL_Y0, I_BUL_Z0, I_BUL_REND, I_BUL_HEND,
 I_COSB, I_SINB, I_LB_RHO0, I_LB_X, I_LB_Y, I_LB_Z, I_LB_L, I_LB_HEND, I_THIN_RHO0, I_THIN_RD,
 I_THIN_H, I_THICK_RHO0, I_THICK_RD, I_THICK_H, I_DISK_RHOLE, I_R0, I_DUST_A0, I_DUST_RD,
 I_DUST_H, I_DUST_RHOLE, I_DUST_DIN, I_CMZ_A0, I_CMZ_SR, I_CMZ_SZ, I_BUBBLE_R) = range(len(_P))


@njit(cache=True, fastmath=True)
def _components(x, y, z, P, out):
    """Unit-free densities (Msun/pc^3) of the stellar components at galaxy-frame (x, y, z)."""
    R2 = x * x + y * y
    R = math.sqrt(R2)
    for k in range(NCOMP):
        out[k] = 0.0
    q = P[I_NSC_Q]
    m = math.sqrt(R2 + (z / q) ** 2)
    if m < 2000.0 and P[I_NSC_RHO0] > 0:
        m = max(m, P[I_NSC_MF])
        u = m / P[I_NSC_A]
        out[0] = P[I_NSC_RHO0] * u ** (-P[I_NSC_G]) * (1.0 + u) ** (P[I_NSC_G] - 4.0)
    a = math.sqrt(R2 + (z / P[I_NSD_Q]) ** 2)
    if a < 4000.0 and P[I_NSD_RHO2] > 0:
        out[1] = (P[I_NSD_RHO1] * math.exp(-(a / P[I_NSD_R1]) ** P[I_NSD_N1])
                  + P[I_NSD_RHO2] * math.exp(-(a / P[I_NSD_R2]) ** P[I_NSD_N2]))
    xb = x * P[I_COSB] + y * P[I_SINB]
    yb = -x * P[I_SINB] + y * P[I_COSB]
    if R < 8000.0:
        if P[I_BUL_RHO0] > 0:
            u2 = (xb / P[I_BUL_X0]) ** 2 + (yb / P[I_BUL_Y0]) ** 2
            rs2 = math.sqrt(u2 * u2 + (z / P[I_BUL_Z0]) ** 4)
            cut = 1.0
            if R > P[I_BUL_REND]:
                cut = math.exp(-((R - P[I_BUL_REND]) / P[I_BUL_HEND]) ** 2)
            out[2] = P[I_BUL_RHO0] * math.exp(-0.5 * rs2) * cut
        if P[I_LB_RHO0] > 0:
            v = math.sqrt((xb / P[I_LB_X]) ** 2 + (yb / P[I_LB_Y]) ** 2)
            cut = 1.0
            ax = abs(xb)
            if ax > P[I_LB_L]:
                cut = math.exp(-((ax - P[I_LB_L]) / P[I_LB_HEND]) ** 2)
            out[3] = P[I_LB_RHO0] * math.exp(-v - abs(z) / P[I_LB_Z]) * cut
    hole = 1.0 - math.exp(-R2 / P[I_DISK_RHOLE] ** 2)
    out[4] = P[I_THIN_RHO0] * math.exp(-(R - P[I_R0]) / P[I_THIN_RD] - abs(z) / P[I_THIN_H]) * hole
    out[5] = P[I_THICK_RHO0] * math.exp(-(R - P[I_R0]) / P[I_THICK_RD] - abs(z) / P[I_THICK_H]) * hole


@njit(cache=True, fastmath=True)
def _dust(x, y, z, P):
    """Diffuse dust A_V per pc at galaxy-frame (x, y, z)."""
    if P[I_DUST_A0] <= 0 and P[I_CMZ_A0] <= 0:
        return 0.0
    R2 = x * x + y * y
    R = math.sqrt(R2)
    f_in = P[I_DUST_DIN] + (1.0 - P[I_DUST_DIN]) * (1.0 - math.exp(-(R2 / P[I_DUST_RHOLE] ** 2) ** 2))
    a = P[I_DUST_A0] * math.exp(-(R - P[I_R0]) / P[I_DUST_RD] - abs(z) / P[I_DUST_H]) * f_in
    if P[I_BUBBLE_R] > 0:
        d2 = ((x - P[I_R0]) ** 2 + y * y + z * z) / P[I_BUBBLE_R] ** 2
        if d2 < 9.0:
            a *= 1.0 - math.exp(-d2 * d2)
    if R < 2000.0:
        a += P[I_CMZ_A0] * math.exp(-0.5 * R2 / P[I_CMZ_SR] ** 2 - 0.5 * (z / P[I_CMZ_SZ]) ** 2)
    return a


@njit(cache=True, fastmath=True)
def _to_gal(px, py, pz, Rg, c0):
    dx, dy, dz = px - c0[0], py - c0[1], pz - c0[2]
    return (Rg[0, 0] * dx + Rg[0, 1] * dy + Rg[0, 2] * dz,
            Rg[1, 0] * dx + Rg[1, 1] * dy + Rg[1, 2] * dz,
            Rg[2, 0] * dx + Rg[2, 1] * dy + Rg[2, 2] * dz)


@njit(cache=True, fastmath=True)
def _field_sample(fi, px, py, pz, fmeta, fdata, buf):
    """Trilinear sample of field fi into buf[:ncomp]; corners outside the grid count as zero."""
    ox, oy, oz, vox = fmeta[fi, 0], fmeta[fi, 1], fmeta[fi, 2], fmeta[fi, 3]
    nx, ny, nz, nc = int(fmeta[fi, 4]), int(fmeta[fi, 5]), int(fmeta[fi, 6]), int(fmeta[fi, 7])
    off = int(fmeta[fi, 8])
    for c in range(nc):
        buf[c] = 0.0
    gx, gy, gz = (px - ox) / vox, (py - oy) / vox, (pz - oz) / vox
    if gx <= -1.0 or gy <= -1.0 or gz <= -1.0 or gx >= nx or gy >= ny or gz >= nz:
        return False
    i0, j0, k0 = int(math.floor(gx)), int(math.floor(gy)), int(math.floor(gz))
    fx, fy, fz = gx - i0, gy - j0, gz - k0
    for di in range(2):
        i = i0 + di
        if i < 0 or i >= nx:
            continue
        wx = fx if di else 1.0 - fx
        for dj in range(2):
            j = j0 + dj
            if j < 0 or j >= ny:
                continue
            wy = fy if dj else 1.0 - fy
            for dk in range(2):
                k = k0 + dk
                if k < 0 or k >= nz:
                    continue
                w = wx * wy * (fz if dk else 1.0 - fz)
                base = off + ((i * ny + j) * nz + k) * nc
                for c in range(nc):
                    buf[c] += w * fdata[base + c]
    return True


@njit(cache=True, fastmath=True)
def _box_hit(o, d, lo, hi):
    t0, t1 = 0.0, 1e30
    for a in range(3):
        if abs(d[a]) < 1e-15:
            if o[a] < lo[a] or o[a] > hi[a]:
                return 1e30, -1e30
        else:
            ta, tb = (lo[a] - o[a]) / d[a], (hi[a] - o[a]) / d[a]
            if ta > tb:
                ta, tb = tb, ta
            t0, t1 = max(t0, ta), min(t1, tb)
    return t0, t1


@njit(parallel=True, cache=True, fastmath=True)
def _march(obs2, kdiv, dirs, smax, Rg, c0, P, E, C, r_res, inner, fmeta, fdata, fspec, ctrl, want_I, out_I, out_tau):
    """Ray march from obs2[i // kdiv] along dirs[i] to smax[i]; fills out_I (radiance) and/or out_tau.
    Fields of kind code 2 are DGL mean-intensity grids (channels J, Gx, Gy, Gz per bin, finest first)."""
    f_obs, f_c, ds_min, ds_core, ds_max, step_vox = ctrl[0], ctrl[1], ctrl[2], ctrl[3], ctrl[4], ctrl[5]
    albedo, g3 = ctrl[6], 3.0 * ctrl[7]
    N, nb, nf = dirs.shape[0], C.shape[0], fmeta.shape[0]
    maxc = 1
    for f in range(nf):
        maxc = max(maxc, int(fmeta[f, 7]))
    for i in prange(N):
        d = dirs[i]
        obs = obs2[i // kdiv]
        rho = np.empty(NCOMP)
        buf = np.empty(maxc)
        jb = np.empty(nb)
        kb = np.empty(nb)
        T = np.ones(nb)
        I = np.zeros(nb)
        tau = np.zeros(nb)
        tin = np.empty(nf)
        tout = np.empty(nf)
        lo = np.empty(3)
        hi = np.empty(3)
        for f in range(nf):
            for a in range(3):
                lo[a] = fmeta[f, a] - fmeta[f, 3]
                hi[a] = fmeta[f, a] + fmeta[f, 4 + a] * fmeta[f, 3]
            t0, t1 = _box_hit(obs, d, lo, hi)
            tin[f] = t0
            tout[f] = t1
        s = 0.0
        S = smax[i]
        while s < S:
            px, py, pz = obs[0] + s * d[0], obs[1] + s * d[1], obs[2] + s * d[2]
            rc = math.sqrt((px - c0[0]) ** 2 + (py - c0[1]) ** 2 + (pz - c0[2]) ** 2)
            ds = max(ds_min, f_obs * s)
            ds = min(min(ds, max(f_c * rc, ds_core)), ds_max)
            for f in range(nf):
                if tout[f] > tin[f] and fmeta[f, 10] < 1.5:
                    if s < tin[f]:
                        ds = min(ds, tin[f] - s + 1e-6)
                    elif s < tout[f]:
                        ds = min(ds, step_vox * fmeta[f, 3])
            if s < r_res and s + ds > r_res:
                ds = r_res - s + 1e-9
            if s + ds > S:
                ds = S - s
            sm = s + 0.5 * ds
            px, py, pz = obs[0] + sm * d[0], obs[1] + sm * d[1], obs[2] + sm * d[2]
            gx, gy, gz = _to_gal(px, py, pz, Rg, c0)
            av = _dust(gx, gy, gz, P)
            for b in range(nb):
                kb[b] = av * C[b]
                jb[b] = 0.0
            if want_I:
                _components(gx, gy, gz, P, rho)
                fac = inner if sm < r_res else 1.0
                for k in range(NCOMP):
                    r = rho[k] * fac
                    if r > 0:
                        for b in range(nb):
                            jb[b] += r * E[k, b]
            dgl_done = not want_I or av <= 0.0
            for f in range(nf):
                if tout[f] > tin[f] and sm >= tin[f] and sm <= tout[f]:
                    if fmeta[f, 10] > 1.5:
                        if dgl_done:
                            continue
                        vx = fmeta[f, 3]
                        if (px < fmeta[f, 0] or py < fmeta[f, 1] or pz < fmeta[f, 2]
                                or px > fmeta[f, 0] + (fmeta[f, 4] - 1) * vx or py > fmeta[f, 1] + (fmeta[f, 5] - 1) * vx
                                or pz > fmeta[f, 2] + (fmeta[f, 6] - 1) * vx):
                            continue
                        _field_sample(f, px, py, pz, fmeta, fdata, buf)
                        for b in range(nb):
                            src = buf[b] + g3 * (buf[nb + b] * d[0] + buf[2 * nb + b] * d[1] + buf[3 * nb + b] * d[2])
                            if src > 0.0:
                                jb[b] += albedo * av * C[b] * src
                        dgl_done = True
                        continue
                    emis = fmeta[f, 10] < 0.5
                    if emis and not want_I:
                        continue
                    if _field_sample(f, px, py, pz, fmeta, fdata, buf):
                        so = int(fmeta[f, 9])
                        for c in range(int(fmeta[f, 7])):
                            v = buf[c]
                            if v != 0.0:
                                if emis:
                                    for b in range(nb):
                                        jb[b] += v * fspec[so + c, b]
                                else:
                                    for b in range(nb):
                                        kb[b] += v * fspec[so + c, b]
            tmax = 0.0
            for b in range(nb):
                dt = kb[b] * ds
                if want_I:
                    if dt > 1e-6:
                        e = math.exp(-dt)
                        I[b] += jb[b] * T[b] * (1.0 - e) / kb[b]
                    else:
                        e = 1.0 - dt
                        I[b] += jb[b] * T[b] * ds * (1.0 - 0.5 * dt)
                    T[b] *= e
                    tmax = max(tmax, T[b])
                tau[b] += dt
            s += ds
            if want_I and tmax < 1e-9:
                break
        for b in range(nb):
            out_tau[i, b] = tau[b]
            if want_I:
                out_I[i, b] = I[b]


@njit(parallel=True, cache=True, fastmath=True)
def _point_eval(xyz, Rg, c0, P, E, C, fmeta, fdata, fspec, out_rho, out_j, out_k):
    N, nb, nf = xyz.shape[0], C.shape[0], fmeta.shape[0]
    maxc = 1
    for f in range(nf):
        maxc = max(maxc, int(fmeta[f, 7]))
    for i in prange(N):
        rho = np.empty(NCOMP)
        buf = np.empty(maxc)
        px, py, pz = xyz[i, 0], xyz[i, 1], xyz[i, 2]
        gx, gy, gz = _to_gal(px, py, pz, Rg, c0)
        _components(gx, gy, gz, P, rho)
        av = _dust(gx, gy, gz, P)
        for k in range(NCOMP):
            out_rho[i, k] = rho[k]
        for b in range(nb):
            out_k[i, b] = av * C[b]
            v = 0.0
            for k in range(NCOMP):
                v += rho[k] * E[k, b]
            out_j[i, b] = v
        for f in range(nf):
            if fmeta[f, 10] > 1.5:
                continue
            if _field_sample(f, px, py, pz, fmeta, fdata, buf):
                so = int(fmeta[f, 9])
                for c in range(int(fmeta[f, 7])):
                    for b in range(nb):
                        if fmeta[f, 10] < 0.5:
                            out_j[i, b] += buf[c] * fspec[so + c, b]
                        else:
                            out_k[i, b] += buf[c] * fspec[so + c, b]


def ccm89(wl_nm, R_V=3.1):
    """Cardelli, Clayton & Mathis (1989) A_lambda / A_V, optical + IR branches (0.3-3.3 um^-1)."""
    x = 1000.0 / np.asarray(wl_nm, float)
    y = x - 1.82
    a_o = 1 + y * (0.17699 + y * (-0.50447 + y * (-0.02427 + y * (0.72085 + y * (0.01979 + y * (-0.77530 + y * 0.32999))))))
    b_o = y * (1.41338 + y * (2.28305 + y * (1.07233 + y * (-5.38434 + y * (-0.62251 + y * (5.30260 + y * -2.09002))))))
    a_ir, b_ir = 0.574 * x ** 1.61, -0.527 * x ** 1.61
    ir = x < 1.1
    return np.where(ir, a_ir + b_ir / R_V, a_o + b_o / R_V)


def sed_shape(grid, temps, weights):
    """(nbins,) bin-averaged blackbody mixture normalised to 1 at 550 nm (weights = V-flux fractions)."""
    w = np.asarray(weights, float) / np.sum(weights)
    return sum(wi * grid.bin_average(lambda lam, t=t: ph.planck_shape(lam, t)) for t, wi in zip(temps, w))


def sgr_a_star_pc(r0=frames.R0_PC):
    return frames.lb_to_galactocentric(*SGRA_LB, r0)


def arches_pc(r0=frames.R0_PC):
    return frames.lb_to_galactocentric(*ARCHES_LB, r0)


def galactocentric_to_celestial_rh(observer_pc, pole_target_pc, reference_pc=(0.0, 0.0, 0.0), roll_deg=0.0):
    """Alias of frames.galactocentric_to_celestial (kept for older callers).

    frames.py now handles the left-handed galactocentric triad itself (det R = -1), so built
    for Earth from the NCP and the equinox it reproduces equatorial (RA, Dec).
    """
    return frames.galactocentric_to_celestial(observer_pc, pole_target_pc, reference_pc, roll_deg)


def gc_observer_pc(offset_pc=3.0, offset_dir=(1.0, 0.0, 0.0), r0=frames.R0_PC):
    d = np.asarray(offset_dir, float)
    return arches_pc(r0) + offset_pc * d / np.linalg.norm(d)


def pack_fields(fields, nbins):
    """Flatten VoxelFields into the (meta, data, spectra) arrays the numba kernels read."""
    meta = np.zeros((len(fields), 12))
    datas, specs, off, soff = [], [], 0, 0
    for i, f in enumerate(fields):
        if f.spectra.shape[1] != nbins:
            raise ValueError(f"field {f.name!r} has {f.spectra.shape[1]} bins, grid has {nbins}")
        if f.kind not in ("emission", "extinction"):
            raise ValueError(f"field {f.name!r}: unknown kind {f.kind!r}")
        d = np.ascontiguousarray(f.data, np.float32)
        nx, ny, nz, nc = d.shape
        meta[i] = [*np.asarray(f.origin_pc, float), f.voxel_pc, nx, ny, nz, nc, off, soff,
                   0.0 if f.kind == "emission" else 1.0, 0.0]
        datas.append(d.ravel())
        specs.append(np.asarray(f.spectra, float))
        off += d.size
        soff += nc
    data = np.concatenate(datas) if datas else np.zeros(1, np.float32)
    spec = np.vstack(specs) if specs else np.zeros((1, nbins))
    return meta, data, spec


class GalaxyModel:
    """Analytic Milky Way stellar + dust model with optional voxel fields (see module docstring).

    GalaxyModel(grid, preset="milky_way", fields=(), **overrides); overrides are DEFAULTS keys
    (e.g. M_nsd=0, dust_scale=0.5, dust=False, AV_sgra=25). `preset` may also be a dict.
    """

    def __init__(self, grid=None, preset="milky_way", fields=(), **overrides):
        self.grid = grid if grid is not None else spectral.default_grid()
        base = dict(PRESETS[preset]) if isinstance(preset, str) else dict(preset)
        unknown = set(base) | set(overrides)
        unknown -= set(DEFAULTS)
        if unknown:
            raise KeyError(f"unknown galaxy parameters: {sorted(unknown)}")
        self.params = {**DEFAULTS, **base, **overrides}
        self.preset = preset if isinstance(preset, str) else "custom"
        self.fields = list(fields)
        p = self.params
        self.c0 = sgr_a_star_pc(p["R0"])
        ex = np.array([p["R0"], 0.0, 0.0]) - self.c0
        self.R0_model = float(np.linalg.norm(ex))
        ex /= self.R0_model
        ez = np.array([0.0, 0.0, 1.0]) - ex[2] * ex
        ez /= np.linalg.norm(ez)
        self.Rg = np.vstack([ex, np.cross(ez, ex), ez])
        self._build_params()
        self._build_spectra()
        self.fmeta, self.fdata, self.fspec = pack_fields(self.fields, self.grid.n)
        self.ctrl = np.array([p["f_obs"], p["f_c"], p["ds_min"], p["ds_core"], p["ds_max"], p["step_vox"],
                              p["dgl_albedo"], p["dgl_g"]])
        self.fdata_all, self.dgl_meta = self.fdata, np.zeros((0, 12))
        self.info = {"masses": dict(self.masses), "M_total": sum(self.masses.values()),
                     "L_V": dict(self.lum_v), "L_V_total": sum(self.lum_v.values())}
        if p["dust"]:
            self._calibrate_dust()
        if p["dgl"] and p["dust"]:
            self._build_dgl()

    # ---------------------------------------------------------------- setup
    def _build_params(self):
        p = self.params
        P = np.zeros(len(_P))
        P[I_R0] = self.R0_model
        P[I_NSC_A], P[I_NSC_G], P[I_NSC_Q], P[I_NSC_MF] = p["nsc_a"], p["nsc_gamma"], p["nsc_q"], p["nsc_mfloor"]
        P[I_NSC_RHO0] = p["M_nsc"] * (3 - p["nsc_gamma"]) / (4 * np.pi * p["nsc_q"] * p["nsc_a"] ** 3)
        from scipy.special import gamma as G
        q, R1, n1, R2, n2 = p["nsd_q"], p["nsd_R1"], p["nsd_n1"], p["nsd_R2"], p["nsd_n2"]
        unit = 4 * np.pi * q * (p["nsd_ratio"] * R1 ** 3 * G(3 / n1) / n1 + R2 ** 3 * G(3 / n2) / n2)
        rho2 = p["M_nsd"] / unit
        P[I_NSD_RHO1], P[I_NSD_RHO2] = p["nsd_ratio"] * rho2, rho2
        P[I_NSD_R1], P[I_NSD_N1], P[I_NSD_R2], P[I_NSD_N2], P[I_NSD_Q] = R1, n1, R2, n2, q
        P[I_COSB], P[I_SINB] = np.cos(np.radians(p["bar_angle_deg"])), np.sin(np.radians(p["bar_angle_deg"]))
        for k in ("bul_x0", "bul_y0", "bul_z0", "bul_Rend", "bul_hend", "lb_x", "lb_y", "lb_z", "lb_L",
                  "lb_hend", "thin_Rd", "thin_h", "thick_Rd", "thick_h", "disk_Rhole"):
            P[_I[k]] = p[k]
        P[I_THIN_RHO0] = p["rho_local"] / (1 + p["thick_ratio"])
        P[I_THICK_RHO0] = p["rho_local"] * p["thick_ratio"] / (1 + p["thick_ratio"])
        P[I_BUL_RHO0], P[I_LB_RHO0] = 1.0, 1.0
        self.P = P
        P[I_BUL_RHO0] = p["M_bulge"] / self._grid_mass(2, (6000, 3000, 2500)) if p["M_bulge"] > 0 else 0.0
        P[I_LB_RHO0] = p["M_longbar"] / self._grid_mass(3, (7000, 4000, 2000)) if p["M_longbar"] > 0 else 0.0
        # dust (d_in solved later)
        sc = p["dust_scale"] if p["dust"] else 0.0
        P[I_DUST_A0] = p["dust_a0"] * sc
        for k in ("dust_Rd", "dust_h", "dust_Rhole", "cmz_sR", "cmz_sz", "bubble_R"):
            P[_I[k]] = p[k]
        P[I_DUST_DIN] = 1.0 if p["dust_din"] is None else p["dust_din"]
        P[I_CMZ_A0] = sc * p["cmz_AV"] / (p["cmz_sR"] * np.sqrt(2 * np.pi))
        self.masses = {"nsc": p["M_nsc"], "nsd": p["M_nsd"], "bulge": p["M_bulge"], "longbar": p["M_longbar"],
                       "thin": self._disk_mass(I_THIN_RHO0, I_THIN_RD, I_THIN_H),
                       "thick": self._disk_mass(I_THICK_RHO0, I_THICK_RD, I_THICK_H)}

    def _grid_mass(self, k, half, n=(240, 120, 100)):
        """Numerical mass of component k (galaxy-frame box of half-sizes `half`, midpoint rule)."""
        ax = [(np.arange(m) + 0.5) / m * 2 * h - h for m, h in zip(n, half)]
        dv = np.prod([2 * h / m for m, h in zip(n, half)])
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], -1)
        rho = self._eval_gal(pts)[0][:, k]
        return rho.sum() * dv

    def _disk_mass(self, i_rho, i_rd, i_h):
        R = np.linspace(0, 40000, 40001)[1:]
        P = self.P
        sig = 2 * P[i_h] * P[i_rho] * np.exp(-(R - P[I_R0]) / P[i_rd]) * (1 - np.exp(-R ** 2 / P[I_DISK_RHOLE] ** 2))
        return float(np.sum(2 * np.pi * R * sig) * (R[1] - R[0]))

    def _build_spectra(self):
        p, g = self.params, self.grid
        old, nuc, disk = (sed_shape(g, *p[k]) for k in ("sed_old", "sed_nuclear", "sed_disk"))
        shapes = {"nsc": nuc, "nsd": nuc, "bulge": old, "longbar": old, "thin": disk, "thick": old}
        self.shapes = shapes
        jfac = L550_SUN / (4 * np.pi * PC_M ** 2)  # W/m^2/sr/nm per pc per (Lsun_V/pc^3)
        self.E = np.array([shapes[c] * jfac / p["ML_" + c] for c in COMPONENTS])
        self.lum_v = {c: self.masses[c] / p["ML_" + c] for c in COMPONENTS}
        self.C = g.bin_average(lambda lam: ccm89(lam, p["R_V"])) / 1.0857

    def _eval_gal(self, pts_gal):
        """Component densities at galaxy-frame points (used for normalisation)."""
        xyz = pts_gal @ self.Rg + self.c0
        return self._eval(xyz)

    def _eval(self, xyz, fields=True):
        xyz = np.ascontiguousarray(np.asarray(xyz, float).reshape(-1, 3))
        n, nb = len(xyz), self.grid.n
        rho, j, k = np.zeros((n, NCOMP)), np.zeros((n, nb)), np.zeros((n, nb))
        E = getattr(self, "E", np.zeros((NCOMP, nb)))
        C = getattr(self, "C", np.zeros(nb))
        fm = getattr(self, "fmeta", np.zeros((0, 12))) if fields else np.zeros((0, 12))
        fd = getattr(self, "fdata", np.zeros(1, np.float32))
        fs = getattr(self, "fspec", np.zeros((1, nb)))
        _point_eval(xyz, self.Rg, self.c0, self.P, E, C, fm, fd, fs, rho, j, k)
        return rho, j, k

    def _calibrate_dust(self):
        """Solve the inner-disk depletion d_in so that A_V(Sun -> Sgr A*) = AV_sgra (dust_scale applies)."""
        p = self.params
        sun = np.array([p["R0"], 0.0, 0.0])
        if p["dust_din"] is None:
            av = []
            for d_in in (0.0, 1.0):
                self.P[I_DUST_DIN] = d_in
                av.append(self._av_between(sun, self.c0))
            target = p["AV_sgra"] * p["dust_scale"]
            d_in = (target - av[0]) / (av[1] - av[0]) if av[1] != av[0] else 1.0
            self.P[I_DUST_DIN] = float(np.clip(d_in, 0.0, 1.0))
            self.info["dust_din"] = float(d_in)
            if not 0.0 <= d_in <= 1.0:
                print(f"galaxy: AV_sgra={target:.1f} needs d_in={d_in:.2f}; clipped to [0, 1]")
        self.info["AV_sun_sgra"] = self._av_between(sun, self.c0)
        d_in, Rh = self.P[I_DUST_DIN], self.P[I_DUST_RHOLE]
        self.info["dust_local_mag_per_kpc"] = 1e3 * self.P[I_DUST_A0] * (d_in + (1 - d_in) * (1 - np.exp(-(self.R0_model / Rh) ** 4)))

    def _av_between(self, a, b):
        tau = self.optical_depth(a, np.atleast_2d(b), fields=False)[0]
        return float(np.interp(550.0, self.grid.wl, tau) / np.interp(550.0, self.grid.wl, self.C))

    def _fm(self):
        return np.vstack([self.dgl_meta, self.fmeta]) if len(self.dgl_meta) else self.fmeta

    def _build_dgl(self):
        """Mean intensity J and first moment G = <I d> on nested grids (single scattering source)."""
        t0 = time.time()
        nb, metas, datas = self.grid.n, [], []
        off = len(self.fdata)
        for half_xy, half_z, vox, ns in self.params["dgl_boxes"]:
            n = [max(int(round(2 * half_xy / vox)), 2)] * 2 + [max(int(round(2 * half_z / vox)), 2)]
            origin = self.c0 - 0.5 * vox * (np.array(n) - 1)  # no node on the NSC cusp
            ax = [origin[a] + vox * np.arange(n[a]) for a in range(3)]
            nodes = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
            dirs = frames.healpix_dirs(ns)
            K = len(dirs)
            I = np.zeros((len(nodes) * K, nb))
            tau = np.zeros_like(I)
            alld = np.ascontiguousarray(np.tile(dirs, (len(nodes), 1)))
            _march(np.ascontiguousarray(nodes), K, alld, np.full(len(alld), float(self.params["s_max"])), self.Rg,
                   self.c0, self.P, self.E, self.C, 0.0, 1.0, self.fmeta, self.fdata, self.fspec, self.ctrl, True, I, tau)
            I = I.reshape(len(nodes), K, nb)
            J = I.mean(axis=1)
            G = np.einsum("nkb,ka->nab", I, dirs) / K
            data = np.concatenate([J[:, None, :], G], axis=1).reshape(*n, 4 * nb).astype(np.float32)
            metas.append([*origin, vox, *n, 4 * nb, off, 0, 2.0, 0.0])
            datas.append(data.ravel())
            off += data.size
        self.dgl_meta = np.array(metas)
        self.fdata_all = np.concatenate([self.fdata] + datas)
        self.info["dgl_build_s"] = time.time() - t0

    # ------------------------------------------------------------- public API
    def stellar_density(self, xyz_pc):
        """(N,) total stellar mass density, Msun pc^-3."""
        return self._eval(xyz_pc, fields=False)[0].sum(axis=1)

    def component_density(self, xyz_pc):
        """(N, NCOMP) densities per component (order: COMPONENTS), Msun pc^-3."""
        return self._eval(xyz_pc, fields=False)[0]

    def luminosity_density_v(self, xyz_pc):
        """(N,) stellar V-band luminosity density, Lsun_V pc^-3 (rho / (M/L_V) summed over components)."""
        ml = np.array([self.params["ML_" + c] for c in COMPONENTS])
        return (self._eval(xyz_pc, fields=False)[0] / ml).sum(axis=1)

    def extinction_av(self, observer_pc, targets_pc, fields=True):
        """(N,) A_V (mag) from observer_pc to each target: tau at 550 nm / (tau / A_V of the CCM curve)."""
        tau = self.optical_depth(observer_pc, targets_pc, fields)
        return np.array([np.interp(550.0, self.grid.wl, t) for t in tau]) / np.interp(550.0, self.grid.wl, self.C)

    def emissivity(self, xyz_pc):
        """(N, nbins) W m^-2 sr^-1 nm^-1 per pc: stars + emission fields."""
        return self._eval(xyz_pc)[1]

    def opacity(self, xyz_pc):
        """(N, nbins) optical depth per pc: dust + extinction fields."""
        return self._eval(xyz_pc)[2]

    def march(self, observer_pc, dirs_gc, r_res_pc=0.0, inner_factor=1.0, s_max=None, chunk=1 << 20):
        """(N, nbins) radiance along galactocentric directions from observer_pc (zero for null or
        non-finite directions). The input array is not modified."""
        d = np.asarray(dirs_gc, float).reshape(-1, 3)
        n = np.linalg.norm(d, axis=1)
        ok = np.isfinite(n) & (n > 0)
        d = np.ascontiguousarray(np.where(ok[:, None], d / np.where(ok, n, 1.0)[:, None], [1.0, 0.0, 0.0]))
        obs = np.asarray(observer_pc, float)
        out = np.zeros((len(d), self.grid.n))
        tau = np.zeros((len(d), self.grid.n))
        S = float(self.params["s_max"] if s_max is None else s_max)
        for i in range(0, len(d), chunk):
            sl = slice(i, i + chunk)
            sm = np.full(len(d[sl]), S)
            _march(obs[None], len(d[sl]), d[sl], sm, self.Rg, self.c0, self.P, self.E, self.C, float(r_res_pc),
                   float(inner_factor), self._fm(), self.fdata_all, self.fspec, self.ctrl, True, out[sl], tau[sl])
        out[~ok] = 0.0
        return out

    def render_map(self, observer_pc, nside, R_gc_to_cel, r_res_pc=0.0, inner_factor=1.0, s_max=None,
                   attach_fn=False, verbose=False, supersample=1):
        """TOASky with a RING HEALPix diffuse map in the celestial frame (dirs_cel = R @ dirs_gc).

        supersample = k (power of 2) marches the k^2 NESTED children of every pixel at nside * k and
        stores their mean, i.e. a pixel-area average (k^2 times the cost); k = 1 samples pixel centres.
        """
        import healpy as hp
        t0 = time.time()
        ss = int(supersample)
        if ss < 1 or ss & (ss - 1):
            raise ValueError(f"supersample must be a power of 2, got {supersample}")
        R = np.asarray(R_gc_to_cel, float)
        dirs_cel = frames.healpix_dirs(nside) if ss == 1 else frames.healpix_dirs(nside * ss, nest=True)
        m = self.march(observer_pc, dirs_cel @ R, r_res_pc, inner_factor, s_max)  # row form of R.T @ d_cel
        if ss > 1:
            m = m.reshape(-1, ss * ss, self.grid.n).mean(axis=1)[hp.ring2nest(nside, np.arange(hp.nside2npix(nside)))]
        m = m.astype(np.float32)
        dt = time.time() - t0
        if verbose:
            print(f"render_map nside={nside} supersample={ss}: {dt:.1f} s", flush=True)
        meta = {"galaxy_preset": self.preset, "observer_pc": np.asarray(observer_pc, float).tolist(),
                "r_res_pc": r_res_pc, "inner_factor": inner_factor, "galaxy_runtime_s": dt, "galaxy_supersample": ss}
        fn = self.diffuse_fn(observer_pc, R, r_res_pc, inner_factor, s_max) if attach_fn else None
        return sky.TOASky(self.grid, sky.PointSources.empty(self.grid.n), m, fn, meta)

    def diffuse_fn(self, observer_pc, R_gc_to_cel, r_res_pc=0.0, inner_factor=1.0, s_max=None):
        """callable(dirs_cel (N, 3)) -> (N, nbins) radiance from the same marcher."""
        R = np.asarray(R_gc_to_cel, float)
        obs = np.asarray(observer_pc, float).copy()

        def fn(dirs_cel):
            d = np.asarray(dirs_cel, float)
            return self.march(obs, d.reshape(-1, 3) @ R, r_res_pc, inner_factor, s_max).reshape(d.shape[:-1] + (self.grid.n,))
        return fn

    def optical_depth(self, observer_pc, targets_pc, fields=True):
        """(N, nbins) optical depth from observer_pc to each target point (dust + extinction fields)."""
        obs = np.asarray(observer_pc, float)
        v = np.atleast_2d(np.asarray(targets_pc, float)) - obs
        dist = np.linalg.norm(v, axis=1)
        d = np.ascontiguousarray(v / np.maximum(dist, 1e-30)[:, None])
        nb = self.grid.n
        tau = np.zeros((len(d), nb))
        fm = self.fmeta if fields else np.zeros((0, 12))
        _march(obs[None], max(len(d), 1), d, np.ascontiguousarray(dist), self.Rg, self.c0, self.P, np.zeros((NCOMP, nb)),
               self.C, 0.0, 1.0, fm, self.fdata, self.fspec, self.ctrl, False, np.zeros((1, nb)), tau)
        return tau

    def describe(self):
        i = self.info
        lines = [f"GalaxyModel preset={self.preset}  M_total={i['M_total']:.3g} Msun  L_V={i['L_V_total']:.3g} Lsun"]
        lines += [f"  {c:8s} M={self.masses[c]:.3g}  L_V={self.lum_v[c]:.3g}" for c in COMPONENTS]
        if "AV_sun_sgra" in i:
            lines.append(f"  A_V(Sun->SgrA*)={i['AV_sun_sgra']:.2f} mag  d_in={self.P[I_DUST_DIN]:.3f}  "
                         f"local dust={i['dust_local_mag_per_kpc']:.2f} mag/kpc")
        lines.append(f"  fields: {[(f.name, f.kind, f.shape) for f in self.fields]}")
        return "\n".join(lines)


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Render a galaxy HEALPix map (galactic-like frame).")
    ap.add_argument("--observer", default="gc", help="gc | earth | x,y,z (galactocentric pc)")
    ap.add_argument("--nside", type=int, default=256)
    ap.add_argument("--preset", default="milky_way")
    ap.add_argument("--set", nargs="*", default=[], help="parameter overrides key=value")
    ap.add_argument("--supersample", type=int, default=1, help="pixel-area average over k^2 sub-pixels (power of 2)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    over = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        over[k] = v.lower() == "true" if v.lower() in ("true", "false") else float(v)
    obs = {"gc": gc_observer_pc(), "earth": np.array(EARTH_OBSERVER_PC)}.get(a.observer)
    if obs is None:
        obs = np.array([float(v) for v in a.observer.split(",")])
    m = GalaxyModel(spectral.default_grid(), a.preset, **over)
    print(m.describe())
    sk = m.render_map(obs, a.nside, np.diag([-1.0, 1.0, 1.0]), verbose=True, supersample=a.supersample)
    sb = spectral.radiance_to_mag_arcsec2(sk.diffuse_map, m.grid)
    print(f"V mag/arcsec2 min/median/max: {sb.min():.2f} / {np.median(sb):.2f} / {sb.max():.2f}")
    if a.out:
        np.savez_compressed(a.out, diffuse_map=sk.diffuse_map, edges=m.grid.edges, observer_pc=obs)
        print("wrote", a.out)


if __name__ == "__main__":
    _main()
