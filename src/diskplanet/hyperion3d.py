"""Build a 3D Hyperion input (.rtin) for the planet-perturbed disk by copying an axisymmetric
model of the Robitaille/Richardson grid and rewriting its grid, so the dust, star and run settings
stay exactly those of the grid model.

    rho_3d(r, theta, phi) = rho_axi(r, theta) * delta(R = r sin(theta), phi)

The phi walls are uniform over [0, 2 pi) (Hyperion convention); delta's phi = 0 is the planet.
Peeled SEDs and raytracing are dropped (only the specific energy / temperature is needed). The
grid's Python package is not used (it does not import under numpy 2); the HDF5 is written with
h5py and run with the Fortran binary:

    python -m diskplanet.hyperion3d build --sigma sigma.npz --rp 100 --nphi 128 --out model.rtin
    mpirun -np 64 hyperion_sph_mpi model.rtin model.rtout
"""
import argparse
import glob
import hashlib
import os

import h5py
import numpy as np

from .perturb import Perturbation

GRID_ROOT = "/blue/adamginsburg/richardson.t/research/flux/grids-1.1"
AU_CM = 1.495978707e13
LSUN_ERG = 3.828e33


def grid_rtin(name, geom="sp--s-i", root=GRID_ROOT):
    return glob.glob(f"{root}/{geom}/input/*/{name}.rtin")[0]


def build(src, out, pert, nphi, n_photons=None, n_iter=None, n_last=0, rho_floor=1e-30, pad=2,
          planet=None, pda=True, h_scale=None):
    """planet: None or dict(r_au, lum_lsun, temp_k): an isotropic blackbody point source at
    (r_au, 0, 0). pda: Hyperion's partial diffusion approximation (slow, iterative, single rank
    for millions of cells). h_scale: callable R (au) -> disk scale height (au), needed when the
    perturbation has a vertically concentrated clump (pert.clump_h > 0)."""
    with h5py.File(src, "r") as hs, h5py.File(out, "w") as ho:
        for k, v in hs.attrs.items():
            ho.attrs[k] = v
        for g in ("Dust", "Sources"):
            hs.copy(g, ho, expand_external=True, expand_soft=True)
        if planet is not None:
            n_src = len(ho["Sources"])
            ps = ho["Sources"].create_group(f"source_{n_src + 1:05d}")
            ps.attrs["type"] = np.bytes_(b"point")
            ps.attrs["name"] = np.bytes_(b"planet")
            ps.attrs["luminosity"] = float(planet["lum_lsun"] * LSUN_ERG)
            ps.attrs["peeloff"] = np.bytes_(b"no")
            ps.attrs["x"], ps.attrs["y"], ps.attrs["z"] = float(planet["r_au"] * AU_CM), 0.0, 0.0
            ps.attrs["spectrum"] = np.bytes_(b"temperature")
            ps.attrs["temperature"] = float(planet["temp_k"])
        ho.attrs["pda"] = np.bytes_(b"yes" if pda else b"no")
        # output: specific energy only, no peeled SEDs, no raytracing
        go = ho.create_group("Output")
        for k, v in hs["Output"].attrs.items():
            go.attrs[k] = v
        go.create_group("Peeled")
        go.create_group("Binned")
        go.attrs["output_n_photons"] = np.bytes_(b"last")  # sampling quality per cell (model3d)
        ho.attrs["raytracing"] = np.bytes_(b"no")
        ho.attrs["n_last_photons"] = float(n_last)
        if n_photons is not None:
            ho.attrs["n_initial_photons"] = np.int64(n_photons)
        if n_iter is not None:
            ho.attrs["n_initial_iter"] = np.int64(n_iter)
        # grid
        r_w = hs["Grid/Geometry/walls_1"][...]
        t_w = hs["Grid/Geometry/walls_2"][...]
        rho = hs["Grid/Quantities/density"][0, 0]  # (nt, nr)
        # merge the empty polar rows into one cell per pole: keeps the cell count (and the
        # per-rank memory of the MPI run) down; those rows hold no dust in the grid model
        rows = np.nonzero(rho.max(axis=1) > rho_floor)[0]
        i0, i1 = max(int(rows[0]) - pad, 1), min(int(rows[-1]) + pad, len(t_w) - 3)
        t_w = np.concatenate([t_w[:1], t_w[i0:i1 + 2], t_w[-1:]])
        rho = np.concatenate([np.zeros((1, rho.shape[1]), rho.dtype), rho[i0:i1 + 1],
                              np.zeros((1, rho.shape[1]), rho.dtype)])
        p_w = np.zeros(nphi + 1, dtype=[("p", "<f8")])
        p_w["p"] = np.linspace(0.0, 2 * np.pi, nphi + 1)
        geo_hash = hashlib.md5(r_w.tobytes() + t_w.tobytes() + p_w.tobytes()).hexdigest().encode()
        gg = ho.create_group("Grid/Geometry")
        for k, v in hs["Grid/Geometry"].attrs.items():
            gg.attrs[k] = v
        gg.attrs["geometry"] = np.bytes_(geo_hash)
        for k, arr in (("walls_1", r_w), ("walls_2", t_w), ("walls_3", p_w)):
            d = gg.create_dataset(k, data=arr)
            for a, v in hs[f"Grid/Geometry/{k}"].attrs.items():
                d.attrs[a] = v
        rc = 0.5 * (r_w["r"][1:] + r_w["r"][:-1]) / AU_CM
        tc = 0.5 * (t_w["t"][1:] + t_w["t"][:-1])
        pc = 0.5 * (p_w["p"][1:] + p_w["p"][:-1])
        pc = np.where(pc > np.pi, pc - 2 * np.pi, pc)
        R = rc[None, :] * np.sin(tc)[:, None]  # (nt, nr)
        Z = rc[None, :] * np.cos(tc)[:, None]
        H = h_scale(np.maximum(R, 1e-9)) if pert.clump_h > 0 else None
        gq = ho.create_group("Grid/Quantities")
        for k, v in hs["Grid/Quantities"].attrs.items():
            gq.attrs[k] = v
        d = gq.create_dataset("density", shape=(1, nphi, len(tc), len(rc)), dtype="f4")
        for j, ph in enumerate(pc):
            fac = pert.factor(R, ph, Z, H) if pert.clump_h > 0 else np.exp(pert(R, ph))
            d[0, j] = (rho * fac).astype("f4")
        d.attrs["geometry"] = np.bytes_(geo_hash)
        m_axi = float(np.sum(rho * _vol(r_w["r"], t_w["t"])))
        m_3d = float(sum(np.sum(d[0, j] * _vol(r_w["r"], t_w["t"])) for j in range(nphi)) / nphi)
    return {"cells": nphi * len(tc) * len(rc), "n_theta": len(tc), "theta_band_deg": (float(np.degrees(t_w["t"][1])), float(np.degrees(t_w["t"][-2]))),
            "dust_mass_ratio_3d_to_axi": m_3d / m_axi}


def _vol(r_w, t_w):
    """Cell volume per unit phi-fraction (full 2 pi) for (nt, nr) cells."""
    dr3 = np.diff(r_w ** 3) / 3.0
    dmu = -np.diff(np.cos(t_w))
    return 2 * np.pi * dmu[:, None] * dr3[None, :]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--model", default="dIULECee")
    b.add_argument("--sigma", required=True, help="ppdo_predict npz")
    b.add_argument("--rp", type=float, default=100.0, help="planet radius (au)")
    b.add_argument("--nphi", type=int, default=128)
    b.add_argument("--photons", type=float, default=None, help="photons per initial iteration")
    b.add_argument("--iter", type=int, default=None)
    b.add_argument("--out", required=True)
    b.add_argument("--planet-lsun", type=float, default=0.0, help="planet point source luminosity (0: none)")
    b.add_argument("--planet-temp", type=float, default=1200.0)
    b.add_argument("--no-pda", action="store_true")
    b.add_argument("--clump-h", type=float, default=0.0,
                   help="planet clump scale height / disk scale height (0: delta at every height)")
    a = ap.parse_args()
    pert = Perturbation.from_npz(a.sigma, a.rp, clump_h=a.clump_h)
    h_scale = None
    if a.clump_h > 0:
        from diskorbit.model import DiskModel
        m = DiskModel(a.model)
        h_scale = lambda R: m.h0 * (R / m.r0) ** m.beta  # noqa: E731
    planet = dict(r_au=a.rp, lum_lsun=a.planet_lsun, temp_k=a.planet_temp) if a.planet_lsun > 0 else None
    info = build(grid_rtin(a.model), a.out, pert, a.nphi, None if a.photons is None else int(a.photons), a.iter,
                 planet=planet, pda=not a.no_pda, h_scale=h_scale)
    print(f"wrote {a.out}: {info}")


if __name__ == "__main__":
    main()
