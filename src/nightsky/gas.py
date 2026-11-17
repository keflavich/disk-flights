"""Galactic Center ionized gas and molecular clouds as nested VoxelFields.

Frames and units
----------------
Galactocentric frame of frames.py (pc; x from Sgr A* towards the Sun, y towards l=90 deg,
z to the NGP). Every structure is placed at its catalogue (l, b) with an assumed depth
(pc, positive = nearer to the Sun than R0) through ``frames.lb_to_galactocentric``; note that
Sgr A* itself then sits at (0, -8.0, -6.6) pc because its (l, b) is not (0, 0).
Emission fields (kind "emission"): data[..., c] = H-alpha volume emissivity, W m^-2 sr^-1
per pc of path; spectra[c] = line + nebular continuum spectrum per unit H-alpha
(nm^-1, bin averages on the caller's SpectralGrid). Extinction fields (kind "extinction"):
data = A_V per pc, spectra = tau_lambda / A_V (CCM89, R_V = 3.1, bin averages).

Physics and parameter choices
-----------------------------
* H-alpha: case B, 4 pi j = 3.53e-25 (T/1e4 K)^-0.9 erg cm^3 s^-1 n_e n_p (Osterbrock &
  Ferland 2006; Draine 2011 eq. 14.8 fit), n_p = n_e. Densities quoted for ionized
  structures are rms values: n_e = n_rms exp(s d - s^2 v) with a unit-variance fBm d, so
  <n_e^2> = n_rms^2 and the emission measure EM = int n_e^2 ds matches radio-derived values.
* Spectral templates (all ratios relative to H-alpha; optical lines in the GC are not
  observable, so they are assumptions guided by the low excitation and ~2x solar metallicity
  seen in mid-IR/radio data, e.g. Simpson+2007, 2018; Lang+2001): "hii_gc" (Arched
  Filaments, Sickle, H regions, Sgr A East A-D, Sgr B1, Sgr C; T_e = 6500 K, Lang+2001),
  "hii_dense" (Sgr A West, Sgr B2; T_e = 7000 K, collisionally suppressed [SII]), "wim_gc"
  (diffuse ionized gas, T_e = 6000 K, [NII]/Ha ~ 0.9, [SII]/Ha ~ 0.4 as in the Galactic WIM,
  Haffner+1999; Madsen+2006). H I nebular continuum (free-free + free-bound in the Kramers
  approximation, Rybicki & Lightman 1979; two-photon spectrum of Nussbaumer & Schmutz 1984):
  1.7e-3 (6500 K) to 1.9e-3 (1e4 K) of the H-alpha flux per nm at 656 nm, i.e. EW(H-alpha)
  ~ 520-600 nm; He continua are omitted.
* Dust: A_V = N_H / 1.9e21 cm^-2 (Bohlin+1978) times DUST_TO_GAS_GC = 2 for the ~2x solar GC
  metallicity (Giannetti+2017 find a still lower gas-to-dust ratio at R_GC ~ 0); 1.4 m_H per H.
* Structures (positions from the cited papers; depths are uncertain and are assumptions):
  Arched Filaments (Lang+2001, 2002: 0.5-1 pc wide, ~20-30 pc long, n_e ~ 1e3-3e3,
  T_e ~ 6000-7000 K, ionised by the Arches from ~10 pc; here n_e,rms 540-960 cm^-3 per arc,
  set so that the 18" peak EM is ~1e6 pc cm^-6) as strand bundles on arcs around
  (l, b) = (0.105, -0.005) deg with molecular ridges (the -30/-20 km/s gas of Serabyn &
  Guesten 1987) on their outer side; the Sickle G0.18-0.04 around the Quintuplet and the Pistol
  (Lang+1997; Figer+1999); compact "H regions" (Zhao+1993) placed at random within their field;
  Sgr A West minispiral (Zhao+2009, 2010; n_e,rms 5-8e3 in sigma = 0.13-0.15 pc arms plus a
  600 cm^-3 diffuse core, set so that Q ~ 3e50 s^-1) as three schematic arms, the Western Arc on the inner edge of the
  CND; Sgr A East HII regions A-D (Mills+2011); Sgr B1 filaments
  (Mehringer+1992); Sgr B2 M/N/S compact HII and Sgr B2 DS (Meng+2019); Sgr C (Kendrew+2013);
  a CMZ warm ionized medium with n_e,rms ~ 10 cm^-3 and a 20 pc Gaussian scale height
  (Mezger+1996) plus the Arches-ionized
  inter-cloud gas (n_e,rms ~ 15 cm^-3 within ~15 pc, consistent with Q_Arches ~ 4e51 s^-1,
  Figer+2002). Molecular: Sgr B2 envelope + centre + cores (Lis & Goldsmith 1990; Schmiedeke+
  2016; total ~7e6 Msun), the Brick (1e5 Msun, r ~ 3 pc; Longmore+2012), dust-ridge clouds
  b-f (1e4-1e5 Msun; Lis+1999, Immer+2012, Walker+2018), 20 and 50 km/s clouds (Coil & Ho
  2000; Tsuboi+2009), CND torus (ring radius 2.2 pc,
  tube sigma 0.45 pc, inclination 60 deg, 5e4 Msun, sigma_ln 2 for its clumpiness; Christopher+
  2005; Requena-Torres+2012), Sgr C cloud,
  M0.11-0.08 and M0.20-0.033 (Tsuboi+1997; Handa+2006), the 1.3-deg complex (mass is an order-
  of-magnitude placeholder), a lumpy 1e7 Msun stream on a 90x60 pc ellipse through the dust
  ridge, Sgr B2 and Sgr C (geometry after Molinari+2011 / Kruijssen+2015, with the near side
  through the Brick ~57 pc in front of Sgr A*), a diffuse CMZ medium of n_H ~ 30 cm^-3
  (A_V ~ 1 mag per 10 pc, so nearby clouds appear as silhouettes), and procedural globules
  near the observer (no literature counterparts).
* Substructure: band-limited 3D Perlin fBm (lognormal turbulence, sigma_ln 1-1.6), ridged
  noise (sheets/wisps) and a domain warp; filaments are tubes around densely sampled curves.
  Sub-voxel tubes and blobs are broadened to >= 0.5 voxel, conserving mass (molecular) or
  int n_e^2 dV (ionized). The noise correlation length (16 pc) exceeds most structures, so
  each structure is renormalised after the noise is applied (two build passes): its mass
  (molecular) or int n_e^2 dV (ionized) over all boxes equals the noise-free value of the
  domain-warped structure. The warp itself changes int n_e^2 dV of strongly warped
  filaments (warp 0.6 pc vs sigma 0.3 pc: -35 % for sickle_0, +16 % for arched_3_0, +5 % for
  the CND mass); blobs are unaffected. The quoted n_e,rms values were tuned on the warped
  build.
* Nested grids (partition of unity): fine boxes around Sgr A* and Sgr B2 (0.1 pc) and around
  the Arches + observer (0.15 pc, 35x39x38 pc), a medium box (0.5 pc) over the inner CMZ and a
  coarse box (1.25 pc) over |y| < 260 pc, x in [-75, 85], |z| < 55 pc. Box k carries weight
  T_k prod_{j<k} (1 - T_j), with T_k a smoothstep taper to zero over a margin of 3-10 pc at
  its faces, so summing all fields counts the model once; trilinear interpolation of the
  tapers on the coarser grid leaves the sum of weights within 0.975-1.036 inside the taper
  shells at the default resolution (0.92-1.11 at "low"), and exactly 1 elsewhere.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field

import numba
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from . import frames, spectral
from .fields import VoxelField

R0 = frames.R0_PC
PC_CM = 3.0857e18
MSUN_G = 1.989e33
M_H_G = 1.6735e-24
MU_H = 1.4
NH_PER_AV = 1.9e21
DUST_TO_GAS_GC = 2.0
AV_PER_PC_PER_NH = PC_CM * DUST_TO_GAS_GC / NH_PER_AV  # A_V per pc for n_H = 1 cm^-3
MSUN_PER_NH_PC3 = MU_H * M_H_G * PC_CM ** 3 / MSUN_G  # Msun in 1 pc^3 of n_H = 1 cm^-3
RAYLEIGH_HA = 1e10 / (4 * np.pi) * 6.62607015e-34 * 2.99792458e8 / 656.28e-9  # W m^-2 sr^-1
CACHE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                          "renders", "nightsky", "cache", "gas"))


def lbd(l_deg, b_deg, depth_pc=0.0):
    """Galactocentric position of (l, b) at depth_pc nearer to the Sun than R0."""
    return np.asarray(frames.lb_to_galactocentric(l_deg, b_deg, R0 - np.asarray(depth_pc, float)), float)


ARCHES = lbd(0.1217, 0.0188)
QUINTUPLET = lbd(0.1604, -0.0591)
SGRA = lbd(359.9442, -0.0462)
OBSERVER = ARCHES + np.array([3.0, 0.0, 0.0])


def halpha_coeff(t_e):
    """H-alpha emissivity per pc of path per (n_e n_p / cm^-6): W m^-2 sr^-1 pc^-1 (case B)."""
    return 3.53e-25 * (t_e / 1e4) ** -0.9 / (4 * np.pi) * PC_CM * 1e-3


# ----------------------------------------------------------------------------- spectra
LINES = dict(Ha=656.28, Hb=486.13, Hg=434.05, Hd=410.17, NII6548=654.80, NII6584=658.35,
             SII6716=671.64, SII6731=673.08, OIII4959=495.89, OIII5007=500.68, OI6300=630.03,
             OI6363=636.38, HeI5876=587.56, ArIII7136=713.58, OII3727=372.7)


def _template(t_e, n2, s2a, s2b, o3_hb, o1, he_hb, ar3, o2_hb):
    hb = 1.0 / (2.86 * (t_e / 1e4) ** -0.1)
    return dict(T_e=t_e, lines=dict(
        Ha=1.0, Hb=hb, Hg=0.468 * hb, Hd=0.259 * hb, NII6584=n2, NII6548=n2 / 3.05, SII6716=s2a,
        SII6731=s2b, OIII5007=o3_hb * hb, OIII4959=o3_hb * hb / 2.98, OI6300=o1, OI6363=o1 / 3.1,
        HeI5876=he_hb * hb, ArIII7136=ar3, OII3727=o2_hb * hb))


TEMPLATES = {
    "hii_gc": _template(6500.0, 0.75, 0.13, 0.12, 0.30, 0.02, 0.05, 0.03, 1.5),
    "hii_dense": _template(7000.0, 0.50, 0.03, 0.05, 0.40, 0.01, 0.08, 0.04, 0.8),
    "wim_gc": _template(6000.0, 0.90, 0.24, 0.17, 0.10, 0.05, 0.02, 0.01, 2.5),
}
ION_COMPONENTS = ("hii_gc", "hii_dense", "wim_gc")


def nebular_continuum(lam_nm, t_e):
    """H I nebular continuum F_lambda per unit H-alpha flux (nm^-1).

    Free-free + free-bound in the Kramers approximation with unit Gaunt factors,
    4 pi j_nu = 6.8e-38 T^-1/2 [e^{-h nu/kT} + sum_n (2 chi_H / kT n^3) e^{-(h nu - chi_n)/kT}]
    (Rybicki & Lightman 1979 eq. 5.14b; Brown & Mathews 1970), plus two-photon 2s -> 1s decay
    with the Nussbaumer & Schmutz (1984) spectrum and alpha_eff(2s) = 0.32 alpha_B (Osterbrock &
    Ferland 2006, Table 4.11), all divided by 4 pi j(H-alpha) of halpha_coeff."""
    lam = np.asarray(lam_nm, float)
    kt = 8.617333e-5 * t_e
    e = 1239.842 / lam
    n = np.arange(1, 61)
    chi = 13.6057 / n ** 2
    de = e[..., None] - chi
    fb = np.where(de >= 0, 2 * 13.6057 / (kt * n ** 3) * np.exp(-np.maximum(de, 0) / kt), 0.0).sum(-1)
    g_nu = 6.8e-38 * t_e ** -0.5 * (np.exp(-e / kt) + fb)
    nu, nu12 = 2.99792458e17 / lam, 2.46607e15
    y = np.clip(nu / nu12, 0, 1)
    a2q = lambda yy: (lambda w: w * (1 - (4 * w) ** 0.8) + 0.88 * w ** 1.53 * (4 * w) ** 0.8)(yy * (1 - yy))
    yg = np.linspace(0, 1, 2001)
    psi = 2 * a2q(y) / np.trapezoid(a2q(yg), yg)
    g_nu = g_nu + 0.32 * 2.59e-13 * (t_e / 1e4) ** -0.7 * 6.62607e-27 * nu * psi / nu12
    return g_nu * 2.99792458e17 / lam ** 2 / (3.53e-25 * (t_e / 1e4) ** -0.9)


def emission_spectrum(grid, template):
    """(nbins,) spectrum per unit H-alpha flux (nm^-1): lines + nebular continuum."""
    t = TEMPLATES[template]
    s = np.zeros(grid.n)
    for k, r in t["lines"].items():
        s += grid.line(LINES[k], r)
    return s + grid.bin_average(lambda w: nebular_continuum(w, t["T_e"]))


def ccm89(wl_nm, rv=3.1):
    """A_lambda / A_V of Cardelli, Clayton & Mathis (1989), 0.125-3.3 um."""
    x = 1000.0 / np.asarray(wl_nm, float)
    a, b = np.zeros_like(x), np.zeros_like(x)
    ir, op, uv = x < 1.1, (x >= 1.1) & (x < 3.3), x >= 3.3
    a[ir], b[ir] = 0.574 * x[ir] ** 1.61, -0.527 * x[ir] ** 1.61
    y = x[op] - 1.82
    a[op] = np.polyval([0.32999, -0.77530, 0.01979, 0.72085, -0.02427, -0.50447, 0.17699, 1.0], y)
    b[op] = np.polyval([-2.09002, 5.30260, -0.62251, -5.38434, 1.07233, 2.28305, 1.41338, 0.0], y)
    xu = np.minimum(x[uv], 8.0)
    fa = np.where(xu >= 5.9, -0.04473 * (xu - 5.9) ** 2 - 0.009779 * (xu - 5.9) ** 3, 0.0)
    fb = np.where(xu >= 5.9, 0.2130 * (xu - 5.9) ** 2 + 0.1207 * (xu - 5.9) ** 3, 0.0)
    a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341) + fa
    b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263) + fb
    return a + b / rv


def extinction_spectrum(grid, rv=3.1):
    """(nbins,) tau_lambda / A_V, bin-averaged CCM89."""
    return grid.bin_average(lambda w: ccm89(w, rv)) / 1.0857


# ----------------------------------------------------------------------------- noise
_GRAD = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1],
                  [1, 0, -1], [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], np.float64)


@numba.njit(cache=True, inline="always")
def _hash(ix, iy, iz, seed):
    h = (ix * 374761393 + iy * 668265263 + iz * 1274126177 + seed * 974634721) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return h ^ (h >> 16)


@numba.njit(cache=True, inline="always")
def _gd(h, x, y, z):
    k = h % 12
    return _GRAD[k, 0] * x + _GRAD[k, 1] * y + _GRAD[k, 2] * z


@numba.njit(cache=True)
def _perlin(x, y, z, seed):
    xf, yf, zf = np.floor(x), np.floor(y), np.floor(z)
    ix, iy, iz = int(xf), int(yf), int(zf)
    fx, fy, fz = x - xf, y - yf, z - zf
    u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    w = fz * fz * fz * (fz * (fz * 6 - 15) + 10)
    n000 = _gd(_hash(ix, iy, iz, seed), fx, fy, fz)
    n100 = _gd(_hash(ix + 1, iy, iz, seed), fx - 1, fy, fz)
    n010 = _gd(_hash(ix, iy + 1, iz, seed), fx, fy - 1, fz)
    n110 = _gd(_hash(ix + 1, iy + 1, iz, seed), fx - 1, fy - 1, fz)
    n001 = _gd(_hash(ix, iy, iz + 1, seed), fx, fy, fz - 1)
    n101 = _gd(_hash(ix + 1, iy, iz + 1, seed), fx - 1, fy, fz - 1)
    n011 = _gd(_hash(ix, iy + 1, iz + 1, seed), fx, fy - 1, fz - 1)
    n111 = _gd(_hash(ix + 1, iy + 1, iz + 1, seed), fx - 1, fy - 1, fz - 1)
    x00 = n000 + u * (n100 - n000)
    x10 = n010 + u * (n110 - n010)
    x01 = n001 + u * (n101 - n001)
    x11 = n011 + u * (n111 - n011)
    y0 = x00 + v * (x10 - x00)
    y1 = x01 + v * (x11 - x01)
    return y0 + w * (y1 - y0)


@numba.njit(cache=True, inline="always")
def _fbm1(x, y, z, f0, amps, seed, ridged):
    s = 0.0
    f = f0
    for o in range(amps.shape[0]):
        a = amps[o]
        if a > 0:
            v = _perlin(x * f + 19.1 * o, y * f + 7.3 * o, z * f + 3.7 * o, seed + 101 * o)
            if ridged:
                v = 1.0 - abs(v)
                v = v * v
            s += a * v
        f *= 2.0
    return s


@numba.njit(parallel=True, cache=True)
def _fbm_grid(o0, o1, o2, vox, f0, amps, seed, ridged, out):
    nx, ny, nz = out.shape
    for i in numba.prange(nx):
        x = o0 + i * vox
        for j in range(ny):
            y = o1 + j * vox
            for k in range(nz):
                out[i, j, k] = _fbm1(x, y, o2 + k * vox, f0, amps, seed, ridged)


@numba.njit(parallel=True, cache=True)
def _fbm_pts(p, f0, amps, seed, ridged, out):
    for i in numba.prange(p.shape[0]):
        out[i] = _fbm1(p[i, 0], p[i, 1], p[i, 2], f0, amps, seed, ridged)


def octave_amps(voxel, base_wl=16.0, min_wl=0.4, gain=0.6):
    """fBm octave amplitudes (wavelengths base_wl / 2^k >= min_wl), band-limited to >= 4 voxels."""
    wl = base_wl / 2.0 ** np.arange(16)
    wl = wl[wl >= min_wl * 0.999]
    return gain ** np.arange(len(wl)) * np.clip((wl / voxel - 2.0) / 2.0, 0, 1)


class Noise:
    """Band-limited procedural fBm, consistent between grid levels (coordinate based)."""

    def __init__(self, seed, base_wl=16.0, min_wl=0.4, gain=0.6, ridged=False, nstat=200000):
        self.seed, self.base_wl, self.min_wl, self.gain, self.ridged = int(seed), base_wl, min_wl, gain, ridged
        rng = np.random.default_rng(12345)
        self._pstat = rng.uniform(-300, 300, (nstat, 3))
        self.mean_full, self.std_full = self.stats(min_wl / 4)

    def amps(self, voxel):
        return octave_amps(voxel, self.base_wl, self.min_wl, self.gain)

    def stats(self, voxel):
        out = np.empty(len(self._pstat))
        _fbm_pts(self._pstat, 1.0 / self.base_wl, self.amps(voxel), self.seed, self.ridged, out)
        return out.mean(), out.std()

    def grid(self, box):
        """Standardised field on a box: (fbm - mean_full) / std_full, plus its variance there."""
        out = np.empty(box.shape, np.float32)
        _fbm_grid(box.lo[0], box.lo[1], box.lo[2], box.voxel, 1.0 / self.base_wl, self.amps(box.voxel),
                  self.seed, self.ridged, out)
        m, s = self.stats(box.voxel)
        out -= self.mean_full
        out /= self.std_full
        return out, (s / self.std_full) ** 2, (m - self.mean_full) / self.std_full


# ----------------------------------------------------------------------------- boxes
@dataclass
class Box:
    name: str
    lo: np.ndarray
    hi: np.ndarray
    voxel: float
    margin: float

    def __post_init__(self):
        self.lo = np.asarray(self.lo, float)
        n = np.round((np.asarray(self.hi, float) - self.lo) / self.voxel).astype(int) + 1
        self.shape = tuple(int(v) for v in n)
        self.hi = self.lo + (n - 1) * self.voxel

    def axes(self):
        return [self.lo[a] + np.arange(self.shape[a]) * self.voxel for a in range(3)]

    def taper(self, axes):
        t = []
        for a in range(3):
            s = np.clip(np.minimum(axes[a] - self.lo[a], self.hi[a] - axes[a]) / self.margin, 0, 1)
            t.append(s * s * (3 - 2 * s))
        return (t[0][:, None, None] * t[1][None, :, None] * t[2][None, None, :]).astype(np.float32)

    def slice_for(self, lo, hi):
        i0 = np.maximum(np.ceil((np.asarray(lo) - self.lo) / self.voxel).astype(int), 0)
        i1 = np.minimum(np.floor((np.asarray(hi) - self.lo) / self.voxel).astype(int) + 1, self.shape)
        if np.any(i1 <= i0):
            return None
        return tuple(slice(int(a), int(b)) for a, b in zip(i0, i1))


RESOLUTIONS = {  # (Sgr A* / Sgr B2 boxes, Arches box, medium, coarse) voxel in pc
    "low": (0.2, 0.3, 1.0, 2.0),
    "default": (0.1, 0.15, 0.5, 1.25),
    "high": (0.07, 0.1, 0.35, 1.0),
}


def make_boxes(resolution="default"):
    vs, vf, vm, vc = RESOLUTIONS[resolution]
    a, s, b2 = ARCHES, SGRA, lbd(0.667, -0.036, 8.0)
    return [
        Box("sgra", s - 6.5, s + 6.5, vs, 3.5),
        Box("sgrb2", b2 - 7.0, b2 + 7.0, vs, 3.5),
        Box("arches", a + [-16, -22, -20], a + [19, 17, 18], vf, 3.0),
        Box("inner", [-60, -55, -35], [75, 120, 35], vm, 10.0),
        Box("cmz", [-75, -260, -55], [85, 260, 55], vc, 8.0),
    ]


# ----------------------------------------------------------------------------- structures
@dataclass
class Blob:
    """Ellipsoidal cloud / HII region. profile 'soft': 1/(1+q^2) exp(-(q/qt)^2); 'gauss';
    'shell': exp(-(q-shell)^2 / (2 thick^2)). q = ellipsoidal radius in units of rc."""
    comp: str
    center: np.ndarray
    rc: float
    n: float = 0.0
    mass: float = 0.0
    axes: tuple = (1.0, 1.0, 1.0)
    rot: np.ndarray = None
    profile: str = "soft"
    qt: float = 3.0
    shell: float = 0.0
    thick: float = 0.3
    sigma_ln: float = 1.0
    name: str = ""

    def __post_init__(self):
        self.center = np.asarray(self.center, float)
        self.rot = np.eye(3) if self.rot is None else np.asarray(self.rot, float)
        if self.mass > 0:
            self.n = self.mass / MSUN_PER_NH_PC3 / (self.volume_integral() * self.rc ** 3 * np.prod(self.axes))

    def rho(self, q):
        if self.profile == "soft":
            return np.exp(-(q / self.qt) ** 2) / (1 + q * q)
        if self.profile == "gauss":
            return np.exp(-0.5 * q * q)
        return np.exp(-0.5 * ((q - self.shell) / self.thick) ** 2)

    def extent(self):
        return {"soft": 2.3 * self.qt, "gauss": 4.0, "shell": self.shell + 4 * self.thick}[self.profile]

    def volume_integral(self, power=1):
        q = np.linspace(0, self.extent(), 4000)
        return np.trapezoid(4 * np.pi * q * q * self.rho(q) ** power, q)

    def bbox(self):
        r = self.rc * max(self.axes) * self.extent() + 1.0
        return self.center - r, self.center + r

    def evaluate(self, X, Y, Z, voxel, warp):
        rc_eff = np.sqrt(self.rc ** 2 + (0.5 * voxel) ** 2)
        scale = (self.rc / rc_eff) ** (1.5 if self.comp in ION_COMPONENTS else 3.0)
        d = [X - self.center[0], Y - self.center[1], Z - self.center[2]]
        q2 = 0.0
        for k in range(3):
            e = self.rot[:, k]
            q2 = q2 + ((e[0] * d[0] + e[1] * d[1] + e[2] * d[2]) / (rc_eff * self.axes[k])) ** 2
        return (self.n * scale) * self.rho(np.sqrt(q2)).astype(np.float32)


@dataclass
class Filament:
    """Gaussian tube of width sigma (pc) around a curve through control points `ctrl`.
    n: peak (molecular) or rms (ionized) density, scalar or per control point; along-curve
    lognormal modulation sigma_s on scale ell_s; `mass` (Msun) renormalises molecular tubes."""
    comp: str
    ctrl: np.ndarray
    sigma: float
    n: object = 0.0
    mass: float = 0.0
    closed: bool = False
    sigma_ln: float = 1.0
    sigma_s: float = 0.5
    ell_s: float = 3.0
    warp: float = 1.0
    seed: int = 0
    name: str = ""

    def __post_init__(self):
        c = np.asarray(self.ctrl, float)
        if self.closed:
            c = np.vstack([c, c[:1]])
        t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
        cs = CubicSpline(t, c, bc_type="periodic" if self.closed else "natural")
        npts = max(int(t[-1] / (self.sigma / 4)) + 2, 8)
        s = np.linspace(0, t[-1], npts)
        self.pts = cs(s)
        self.ds = np.gradient(s)
        nval = 1.0 if self.mass > 0 else self.n
        namp = np.interp(s, t, np.broadcast_to(np.asarray(nval, float), len(t))) if np.ndim(nval) else np.full(npts, float(nval))
        rng = np.random.default_rng(self.seed)
        nk = max(int(t[-1] / self.ell_s) + 4, 4)
        knots = rng.standard_normal(nk)
        if self.closed:
            knots[-1] = knots[0]
            g = CubicSpline(np.linspace(0, t[-1], nk), knots, bc_type="periodic")(s)
        else:
            g = CubicSpline(np.linspace(0, t[-1], nk), knots)(s)
        g = (g - g.mean()) / max(g.std(), 1e-6)
        ion = self.comp in ION_COMPONENTS
        self.amp = namp * np.exp(self.sigma_s * g - self.sigma_s ** 2 * (1.0 if ion else 0.5))
        if self.mass > 0:
            tot = np.sum(self.amp * 2 * np.pi * self.sigma ** 2 * self.ds)
            self.amp *= self.mass / MSUN_PER_NH_PC3 / tot
        self.tree = cKDTree(self.pts)

    def bbox(self):
        r = 4.5 * self.sigma + 3.0 * self.warp + 0.5
        return self.pts.min(0) - r, self.pts.max(0) + r

    def evaluate(self, X, Y, Z, voxel, warp):
        sig = np.sqrt(self.sigma ** 2 + (0.5 * voxel) ** 2)
        scale = (self.sigma / sig) ** (1.0 if self.comp in ION_COMPONENTS else 2.0)
        shp = np.broadcast_shapes(np.shape(X), np.shape(Y), np.shape(Z))
        p = np.empty(shp + (3,), np.float64)
        p[..., 0], p[..., 1], p[..., 2] = X, Y, Z
        if self.warp > 0:
            p += self.warp * warp
        d, i = self.tree.query(p.reshape(-1, 3), distance_upper_bound=4.5 * sig, workers=-1)
        ok = np.isfinite(d)
        out = np.zeros(len(d), np.float32)
        out[ok] = scale * self.amp[i[ok]] * np.exp(-0.5 * (d[ok] / sig) ** 2)
        return out.reshape(shp)


@dataclass
class Diffuse:
    comp: str
    fn: object
    lo: np.ndarray
    hi: np.ndarray
    sigma_ln: float = 1.0
    ridge: float = 0.0
    name: str = ""

    def bbox(self):
        return np.asarray(self.lo, float), np.asarray(self.hi, float)

    def evaluate(self, X, Y, Z, voxel, warp):
        return self.fn(X, Y, Z).astype(np.float32)


def _arc(center_lb, radius_deg, phi1, phi2, depth_fn, n=24, wobble=0.0, rng=None, ref=None):
    """Control points of an arc in the sky plane around center_lb at depths depth_fn(phi)."""
    phi = np.radians(np.linspace(phi1, phi2, n))
    r = radius_deg * (1 + (wobble * rng.standard_normal(3) @ np.sin(np.outer([1, 2, 3], phi) + rng.uniform(0, 6.3, (3, 1)))) if wobble else 1)
    l = center_lb[0] + r * np.cos(phi)
    b = center_lb[1] + r * np.sin(phi)
    return lbd(l, b, depth_fn(np.degrees(phi)))


def _rot_from_normal(nrm, major):
    nrm = np.asarray(nrm, float) / np.linalg.norm(nrm)
    e1 = np.asarray(major, float) - nrm * (nrm @ major)
    e1 /= np.linalg.norm(e1)
    return np.column_stack([e1, np.cross(nrm, e1), nrm])


def cnd_rotation(incl_deg=60.0, tilt_deg=20.0):
    """CND frame (columns: major axis, minor axis, normal); inclination to the line of sight
    incl_deg, major axis tilted tilt_deg from the Galactic plane (Jackson+1993; Christopher+2005)."""
    i, t = np.radians(incl_deg), np.radians(tilt_deg)
    return _rot_from_normal([np.cos(i), -np.sin(i) * np.sin(t), np.sin(i) * np.cos(t)], [0, np.cos(t), np.sin(t)])


def _ring_point(l_deg, near=True, ax=90.0, bx=60.0, y0=6.0):
    """Depth (pc, + = nearer) of longitude l on the stream ellipse x = bx cos t, y = y0 + ax sin t."""
    y = R0 * np.sin(np.radians(l_deg))
    st = np.clip((y - y0) / ax, -1, 1)
    return (1 if near else -1) * bx * np.sqrt(1 - st * st)


def ionized_structures(seed=0):
    rng = np.random.default_rng(seed + 1000)
    S = []
    arcs = [  # radius deg, phi range, depth d0 + d1 cos(phi) pc, n_rms, strands
        (0.060, 25, 100, -4.0, 2.0, 1300, 2), (0.072, 12, 95, -2.0, 3.0, 1600, 3),
        (0.085, 20, 92, 1.0, -2.0, 1300, 2), (0.090, 85, 165, 3.0, 2.0, 1500, 3),
        (0.100, 95, 172, 5.0, -3.0, 1300, 2), (0.112, 55, 140, -6.0, 2.0, 900, 2),
        (0.050, 110, 178, 2.0, 1.0, 900, 2)]
    c_af = (0.105, -0.005)
    S_mol = []
    for k, (rad, p1, p2, d0, d1, n, ns) in enumerate(arcs):
        dfn = lambda ph, d0=d0, d1=d1, k=k: d0 + d1 * np.cos(np.radians(ph)) + 0.8 * np.sin(np.radians(3 * ph) + k)
        for j in range(ns):
            off = (j - (ns - 1) / 2) * 0.0026 + rng.normal(0, 0.0008)
            ctrl = _arc(c_af, rad + off, p1 + rng.uniform(-8, 8), p2 + rng.uniform(-8, 8), dfn, wobble=0.04, rng=rng)
            S.append(Filament("hii_gc", ctrl, sigma=rng.uniform(0.18, 0.3), n=0.6 * n / np.sqrt(ns), sigma_ln=0.8,
                              sigma_s=0.6, ell_s=2.5, warp=0.6, seed=seed * 100 + k * 10 + j, name=f"arched_{k}_{j}"))
        ctrl = _arc(c_af, rad, p1, p2, dfn)
        S.append(Filament("hii_gc", ctrl, sigma=0.9, n=0.09 * n, sigma_ln=1.0, sigma_s=0.5, warp=0.8,
                          seed=seed * 100 + k * 10 + 9, name=f"arched_env_{k}"))
        ctrl = _arc(c_af, rad + 0.0065, p1 - 5, p2 + 5, lambda ph: dfn(ph) - 0.5)
        S_mol.append(Filament("mol", ctrl, sigma=0.55, n=rng.uniform(2.5e3, 6e3), sigma_ln=1.3, sigma_s=0.8, ell_s=2.0,
                              warp=0.7, seed=seed * 100 + k * 10 + 7, name=f"arched_ridge_{k}"))
    q = (0.1604, -0.0591)
    for j in range(3):
        ctrl = _arc(q, 0.043 + 0.003 * j, -35 + 6 * j, 72 - 5 * j, lambda ph: -1.5 + 0.02 * ph, wobble=0.05, rng=rng)
        S.append(Filament("hii_gc", ctrl, sigma=0.35, n=500 / np.sqrt(3), sigma_ln=0.8, sigma_s=0.6, warp=0.6,
                          seed=seed * 100 + 80 + j, name=f"sickle_{j}"))
    S.append(Blob("hii_gc", lbd(0.1525, -0.0500, 0.0), 0.55, n=900, profile="shell", shell=1.0, thick=0.35,
                  axes=(1.0, 1.0, 0.75), sigma_ln=0.8, name="pistol"))
    for j in range(6):
        S.append(Blob("hii_gc", lbd(rng.uniform(0.02, 0.075), rng.uniform(-0.07, -0.01), rng.uniform(-8, 8)),
                      rng.uniform(0.3, 0.7), n=rng.uniform(600, 1800), profile="gauss", sigma_ln=0.7, name=f"H{j + 1}"))
    for j, (l, b, r) in enumerate([(359.986, -0.067, 0.5), (359.980, -0.071, 0.4), (359.975, -0.076, 0.45),
                                   (359.970, -0.081, 0.6)]):
        S.append(Blob("hii_gc", lbd(l, b, -6.0), r, n=3000, profile="shell" if j in (0, 3) else "gauss", shell=1.0,
                      thick=0.4, sigma_ln=0.7, name="SgrAEast_" + "ABCD"[j]))
    s = SGRA
    u, w = np.array([0, 1.0, 0]), np.array([1.0, 0, 0])
    v = np.array([0, 0, 1.0])
    P = lambda pts: s + np.asarray(pts) @ np.vstack([u, v, w])  # rows (u=l, v=b, w=depth)
    S.append(Filament("hii_dense", P([(0.7, 1.9, -0.6), (0.5, 1.2, -0.3), (0.3, 0.6, 0.0), (0.12, 0.15, 0.1),
                                      (-0.15, -0.15, 0.1), (-0.5, -0.3, 0.0), (-0.9, -0.35, -0.2)]),
                      sigma=0.15, n=8e3, sigma_ln=0.8, sigma_s=0.5, ell_s=0.6, warp=0.15, seed=seed + 501, name="northern_arm"))
    S.append(Filament("hii_dense", P([(1.8, 0.5, 0.7), (1.3, 0.1, 0.5), (0.8, -0.2, 0.35), (0.35, -0.25, 0.2),
                                      (0.05, -0.1, 0.05)]),
                      sigma=0.15, n=5.5e3, sigma_ln=0.8, sigma_s=0.5, ell_s=0.6, warp=0.15, seed=seed + 502, name="eastern_arm"))
    Rc = cnd_rotation()
    ph = np.radians(np.linspace(150, 285, 16))
    S.append(Filament("hii_dense", s + (1.6 * np.column_stack([np.cos(ph), np.sin(ph), 0 * ph])) @ Rc.T,
                      sigma=0.13, n=5e3, sigma_ln=0.8, sigma_s=0.5, ell_s=0.8, warp=0.15, seed=seed + 503, name="western_arc"))
    S.append(Blob("hii_dense", s, 1.2, n=600, profile="gauss", axes=(1.0, 1.2, 0.9), sigma_ln=1.0, name="SgrA_West_diffuse"))
    b2 = lbd(0.667, -0.036, 8.0)
    S += [Blob("hii_dense", b2, 0.2, n=2e4, profile="gauss", sigma_ln=0.6, name="SgrB2_M_HII"),
          Blob("hii_dense", lbd(0.680, -0.028, 8.0), 0.15, n=2e4, profile="gauss", sigma_ln=0.6, name="SgrB2_N_HII"),
          Blob("hii_dense", lbd(0.663, -0.044, 8.0), 0.12, n=1e4, profile="gauss", sigma_ln=0.6, name="SgrB2_S_HII"),
          Blob("hii_dense", lbd(0.650, -0.070, 6.0), 1.2, n=400, profile="gauss", axes=(1, 1.3, 1), sigma_ln=0.9, name="SgrB2_DS")]
    b1 = lbd(0.51, -0.05, 20.0)
    for j in range(7):
        c = (0.51 + rng.uniform(-0.05, 0.05), -0.05 + rng.uniform(-0.04, 0.04))
        p0 = rng.uniform(0, 360)
        ctrl = _arc(c, rng.uniform(0.02, 0.04), p0, p0 + rng.uniform(70, 140), lambda ph, z=rng.uniform(-5, 5): 20.0 + z + 0 * ph,
                    wobble=0.08, rng=rng)
        S.append(Filament("hii_gc", ctrl, sigma=0.4, n=120, sigma_ln=0.9, sigma_s=0.6, warp=0.8, seed=seed + 600 + j,
                          name=f"SgrB1_fil{j}"))
    S.append(Blob("hii_gc", b1, 3.0, n=40, profile="gauss", axes=(1, 1.4, 1), sigma_ln=1.0, name="SgrB1_diffuse"))
    S.append(Blob("hii_gc", lbd(359.43, -0.09, -14.5), 2.5, n=140, profile="shell", shell=1.0, thick=0.25, sigma_ln=0.9,
                  name="SgrC_HII"))

    def wim(X, Y, Z):
        env = (np.exp(-0.5 * (Z / 20.0) ** 2) * np.exp(-(Y / 200.0) ** 4) * np.exp(-(X / 65.0) ** 4))
        r2 = (X - ARCHES[0]) ** 2 + (Y - ARCHES[1]) ** 2 + (Z - ARCHES[2]) ** 2
        return np.sqrt(100.0 * env + 225.0 * np.exp(-0.5 * r2 / 15.0 ** 2))
    S.append(Diffuse("wim_gc", wim, [-75, -260, -55], [85, 260, 55], sigma_ln=1.0, ridge=1.0, name="wim"))
    return S, S_mol


def molecular_structures(seed=0, local_av_per_10pc=1.0):
    rng = np.random.default_rng(seed + 2000)
    _, S = ionized_structures(seed)
    rot_yz = lambda deg: np.array([[1, 0, 0], [0, np.cos(np.radians(deg)), -np.sin(np.radians(deg))],
                                   [0, np.sin(np.radians(deg)), np.cos(np.radians(deg))]])
    b2 = lbd(0.667, -0.036, 8.0)
    S += [Blob("mol", b2 + [0, -1.0, -1.5], 7.0, mass=6e6, qt=3.0, axes=(1, 1.15, 0.85), sigma_ln=1.3, name="SgrB2_envelope"),
          Blob("mol", b2, 1.5, mass=8e5, qt=2.5, sigma_ln=1.0, name="SgrB2_centre"),
          Blob("mol", b2, 0.25, mass=5e4, profile="gauss", sigma_ln=0.5, name="SgrB2_M"),
          Blob("mol", lbd(0.680, -0.028, 8.0), 0.22, mass=3e4, profile="gauss", sigma_ln=0.5, name="SgrB2_N"),
          Blob("mol", lbd(0.663, -0.044, 8.0), 0.2, mass=1e4, profile="gauss", sigma_ln=0.5, name="SgrB2_S")]
    S.append(Blob("mol", lbd(0.253, 0.016, _ring_point(0.253)), 1.4, mass=1e5, qt=2.3, axes=(0.9, 1.7, 0.9),
                  rot=rot_yz(35), sigma_ln=1.2, name="Brick"))
    for nm, l, b, m, rc in [("b", 0.34, 0.06, 4e4, 1.2), ("c", 0.38, 0.04, 4e4, 1.3), ("d", 0.41, 0.03, 7e4, 1.5),
                            ("e", 0.48, 0.03, 1.5e5, 2.0), ("f", 0.50, 0.03, 8e4, 1.6)]:
        S.append(Blob("mol", lbd(l, b, _ring_point(l)), rc, mass=m, qt=2.3, axes=(1, 1.3, 0.9), sigma_ln=1.2,
                      name=f"dustridge_{nm}"))
    S += [Blob("mol", lbd(359.87, -0.08, 8.0), 2.0, mass=3e5, qt=2.5, axes=(1, 2.0, 0.9), rot=rot_yz(-10), sigma_ln=1.3,
               name="cloud_20kms"),
          Blob("mol", lbd(359.985, -0.07, -6.0), 2.0, mass=1.5e5, qt=2.5, axes=(1, 1.4, 1), rot=rot_yz(30), sigma_ln=1.3,
               name="cloud_50kms"),
          Blob("mol", lbd(359.45, -0.11, -14.5), 3.0, mass=2e5, qt=2.5, axes=(1, 1.2, 1.3), sigma_ln=1.3, name="SgrC_cloud"),
          Blob("mol", lbd(0.11, -0.08, 0.0), 2.0, mass=1e5, qt=2.3, axes=(1, 1.5, 0.9), rot=rot_yz(20), sigma_ln=1.4,
               name="M0.11-0.08"),
          Blob("mol", lbd(0.20, -0.033, -1.0), 1.5, mass=5e4, qt=2.3, axes=(1, 0.8, 1.4), sigma_ln=1.4, name="M0.20-0.033"),
          Blob("mol", lbd(1.3, 0.05, 0.0), 8.0, mass=1e6, qt=2.5, axes=(1, 1.5, 0.7), sigma_ln=1.6, name="complex_1.3deg")]
    Rc = cnd_rotation()
    ph = np.radians(np.arange(0, 360, 15))
    S.append(Filament("mol", SGRA + (2.2 * np.column_stack([np.cos(ph), np.sin(ph), 0 * ph])) @ Rc.T, sigma=0.45,
                      mass=5e4, closed=True, sigma_ln=2.0, sigma_s=0.8, ell_s=1.5, warp=0.4, seed=seed + 700, name="CND"))
    th = np.radians([-100, -8, 7, 19.5, 40, 80, 120, 170, 215])
    zt = np.array([-12.8, -13.5, -9.0, 2.3, 5.7, -5.0, 0.0, -4.0, -10.0])
    zs = CubicSpline(np.r_[th, th[0] + 2 * np.pi], np.r_[zt, zt[0]], bc_type="periodic")
    t = np.radians(np.arange(-100, 260, 3.0))
    ctrl = np.column_stack([60 * np.cos(t), 6 + 90 * np.sin(t), zs(t)])
    S.append(Filament("mol", ctrl, sigma=3.0, mass=1e7, closed=True, sigma_ln=1.4, sigma_s=0.9, ell_s=12.0, warp=2.5,
                      seed=seed + 800, name="CMZ_stream"))
    n_loc = 0.1 * local_av_per_10pc / AV_PER_PC_PER_NH

    def diffuse(X, Y, Z):
        return n_loc * np.exp(-0.5 * (Z / 20.0) ** 2) * np.exp(-(Y / 220.0) ** 4) * np.exp(-(X / 65.0) ** 4)
    S.append(Diffuse("mol", diffuse, [-75, -260, -55], [85, 260, 55], sigma_ln=1.0, name="diffuse_CMZ"))
    k = 0
    while k < 14:
        d = rng.normal(size=3)
        p = OBSERVER + d / np.linalg.norm(d) * rng.uniform(4, 25)
        if np.hypot(*(p - ARCHES)[1:]) < 2.5:
            continue  # keep the Earth -> observer -> Arches sight line clear
        S.append(Blob("mol", p, rng.uniform(0.25, 0.9), mass=10 ** rng.uniform(1.5, 2.7), qt=2.0,
                      axes=tuple(rng.uniform(0.6, 1.6, 3)), sigma_ln=1.0, name=f"globule_{k}"))
        k += 1
    return S


# ----------------------------------------------------------------------------- build
def _build_level(box, prev_boxes, structures, comps, noise, warpnoise, ridgenoise, verbose=True, norm=None, sums=None):
    """Deposit every structure into one nested box. With sums (array (nstruct, 2)) only accumulate
    the box-weighted integrals with and without the 3D noise (n_H dV or n_e^2 dV); with norm
    (nstruct,) multiply each structure's noisy contribution by norm[i]."""
    axes = box.axes()
    W = box.taper(axes)
    for pb in prev_boxes:
        W *= 1 - pb.taper(axes)
    data = None if sums is not None else np.zeros(box.shape + (len(comps),), np.float32)
    if W.max() <= 0:
        return data
    dnoise, var, _ = noise.grid(box)
    warp = None
    if any(isinstance(s, Filament) and s.warp > 0 for s in structures):
        wf = [Noise(warpnoise.seed + 17 * a, base_wl=6.0, min_wl=1.5, gain=0.5) for a in range(3)]
        warp = np.stack([w.grid(box)[0] for w in wf], axis=-1)
    rfield = None
    if any(isinstance(s, Diffuse) and s.ridge > 0 for s in structures):
        r, rv, rm = ridgenoise.grid(box)
        rfield = np.exp(1.2 * (r - rm) - 1.2 ** 2 * rv)
    kap = {c: halpha_coeff(TEMPLATES[c]["T_e"]) for c in comps if c in TEMPLATES}
    for i, s in enumerate(structures):
        lo, hi = s.bbox()
        sl = box.slice_for(lo, hi)
        if sl is None:
            continue
        w = W[sl]
        if w.max() <= 0:
            continue
        t0 = time.time()
        X, Y, Z = [axes[a][sl[a]].reshape([-1 if b == a else 1 for b in range(3)]) for a in range(3)]
        val = s.evaluate(X, Y, Z, box.voxel, None if warp is None else warp[sl])
        ion = s.comp in TEMPLATES
        lnm = np.exp(s.sigma_ln * dnoise[sl] - s.sigma_ln ** 2 * var * (1.0 if ion else 0.5))
        if ion:
            noisy = val * lnm
            if isinstance(s, Diffuse) and s.ridge > 0:
                noisy = noisy * rfield[sl]
            noisy, plain = noisy * noisy * w, val * val * w
        else:
            noisy, plain = val * lnm * w, val * w
        if sums is not None:
            sums[i] += (noisy.sum(dtype=np.float64) * box.voxel ** 3, plain.sum(dtype=np.float64) * box.voxel ** 3)
        else:
            f = norm[i] * (kap[s.comp] if ion else AV_PER_PC_PER_NH)
            data[sl + (comps.index(s.comp),)] += (f * noisy).astype(np.float32)
        if verbose and time.time() - t0 > 2:
            print(f"    {box.name}: {s.name} {time.time() - t0:.1f}s", flush=True)
    return data


def _build(grid, seed, resolution, kind, verbose=True, **kw):
    boxes = make_boxes(resolution)
    if kind == "emission":
        structures, _ = ionized_structures(seed)
        comps = list(ION_COMPONENTS)
        noise = Noise(seed * 7 + 1)
        spectra_all = np.stack([emission_spectrum(grid, c) for c in comps])
    else:
        structures = molecular_structures(seed, **kw)
        comps = ["mol"]
        noise = Noise(seed * 7 + 2)
        spectra_all = extinction_spectrum(grid)[None]
    warpnoise, ridgenoise = Noise(seed * 7 + 3, base_wl=6.0, min_wl=1.5, gain=0.5), Noise(seed * 7 + 4, base_wl=12.0, min_wl=0.4, gain=0.55, ridged=True)
    t0 = time.time()
    sums = np.zeros((len(structures), 2))
    for k, box in enumerate(boxes):
        _build_level(box, boxes[:k], structures, comps, noise, warpnoise, ridgenoise, verbose, sums=sums)
    norm = np.where(sums[:, 0] > 0, sums[:, 1] / np.maximum(sums[:, 0], 1e-300), 0.0)
    if verbose:
        print(f"  [{kind}] noise normalisation over {len(structures)} structures: factors {norm.min():.2f}-{norm.max():.2f}, "
              f"{time.time() - t0:.1f}s", flush=True)
    out = []
    for k, box in enumerate(boxes):
        t0 = time.time()
        data = _build_level(box, boxes[:k], structures, comps, noise, warpnoise, ridgenoise, verbose, norm=norm)
        keep = [c for c in range(len(comps)) if data[..., c].max() > 0]
        if not keep:
            continue
        tag = "ion" if kind == "emission" else "mol"
        vf = VoxelField(box.lo.copy(), float(box.voxel), np.ascontiguousarray(data[..., keep]), spectra_all[keep],
                        kind=kind, name=f"gas_{tag}_{box.name}",
                        meta=dict(components=[comps[c] for c in keep], level=box.name, seed=seed, resolution=resolution,
                                  T_e=[TEMPLATES[comps[c]]["T_e"] if comps[c] in TEMPLATES else None for c in keep]))
        out.append(vf)
        if verbose:
            print(f"  [{kind}] {box.name}: shape {box.shape} voxel {box.voxel} pc, comps {vf.meta['components']}, "
                  f"{data.nbytes / 1e6:.0f} MB, {time.time() - t0:.1f}s", flush=True)
    return out


_MEMO = {}


def _key(grid, seed, resolution, kind, kw):
    return (kind, int(seed), resolution, tuple(np.round(grid.edges, 6)), tuple(sorted(kw.items())))


def gc_ionized_gas(grid=None, seed=0, resolution="default", verbose=True):
    """Ionized gas of the inner CMZ as VoxelFields (kind 'emission', data = H-alpha W m^-2 sr^-1 pc^-1)."""
    grid = grid or spectral.default_grid()
    k = _key(grid, seed, resolution, "emission", {})
    if k not in _MEMO:
        _MEMO[k] = _build(grid, seed, resolution, "emission", verbose)
    return _MEMO[k]


def gc_molecular_clouds(grid=None, seed=0, resolution="default", verbose=True, local_av_per_10pc=1.0):
    """Molecular clouds + diffuse CMZ dust as VoxelFields (kind 'extinction', data = A_V per pc)."""
    grid = grid or spectral.default_grid()
    kw = dict(local_av_per_10pc=local_av_per_10pc)
    k = _key(grid, seed, resolution, "extinction", kw)
    if k not in _MEMO:
        _MEMO[k] = _build(grid, seed, resolution, "extinction", verbose, **kw)
    return _MEMO[k]


def _respectrum(vf, comps, grid):
    if vf.kind == "emission":
        return np.stack([emission_spectrum(grid, c) for c in comps])
    return extinction_spectrum(grid)[None]


def save_cache(fields, cache_dir=CACHE_DIR, tag="seed0_default"):
    os.makedirs(cache_dir, exist_ok=True)
    man_path = os.path.join(cache_dir, f"manifest_{tag}.json")
    man = []
    for vf in fields:
        path = os.path.join(cache_dir, f"{vf.name}_{tag}.npz")
        vf.save(path)
        man.append(dict(file=os.path.basename(path), name=vf.name, kind=vf.kind, meta=vf.meta))
    with open(man_path, "w") as f:
        json.dump(man, f, indent=1)
    return man_path


def load_cached(grid=None, seed=0, resolution="default", kinds=("emission", "extinction"), build=True,
                cache_dir=CACHE_DIR):
    """All cached gas VoxelFields (emission first), spectra re-binned onto `grid`.
    Builds and saves the cache if it is missing and build=True."""
    grid = grid or spectral.default_grid()
    tag = f"seed{seed}_{resolution}"
    man_path = os.path.join(cache_dir, f"manifest_{tag}.json")
    if not os.path.exists(man_path):
        if not build:
            raise FileNotFoundError(man_path)
        save_cache(gc_ionized_gas(grid, seed, resolution) + gc_molecular_clouds(grid, seed, resolution), cache_dir, tag)
    with open(man_path) as f:
        man = json.load(f)
    out = []
    for e in man:
        if e["kind"] not in kinds:
            continue
        vf = VoxelField.load(os.path.join(cache_dir, e["file"]))
        vf.meta = e["meta"]
        vf.spectra = _respectrum(vf, vf.meta["components"], grid)
        out.append(vf)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build and cache the GC gas VoxelFields.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resolution", default="default", choices=list(RESOLUTIONS))
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    a = ap.parse_args(argv)
    grid = spectral.default_grid()
    t0 = time.time()
    ion = gc_ionized_gas(grid, a.seed, a.resolution)
    t1 = time.time()
    mol = gc_molecular_clouds(grid, a.seed, a.resolution)
    t2 = time.time()
    path = save_cache(ion + mol, a.cache_dir, f"seed{a.seed}_{a.resolution}")
    print(f"ionized {t1 - t0:.1f}s, molecular {t2 - t1:.1f}s, saved {time.time() - t2:.1f}s -> {path}")


if __name__ == "__main__":
    main()
