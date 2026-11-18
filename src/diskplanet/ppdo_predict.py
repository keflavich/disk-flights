"""Predict the steady-state surface density of a disk with an embedded planet with PPDONet
(Mao, Dong et al. 2023; https://github.com/smao-astro/PPDONet, fork andizq/PPDONet), and save it as
a plain npz so the rest of the pipeline does not need jax.

Run in the `ppdonet` conda env (jax 0.4.30, dm-haiku 0.0.12); PPDONet itself (GPL-3) is not part of
this repository, point --ppdonet at a clone:

    conda run -n ppdonet python -m diskplanet.ppdo_predict --alpha 1e-3 --h0 0.051 --q 1e-3 \
        --out ../renders/diskplanet/sigma_q1e-3_a1e-3_h0.051.npz

Output (code units of the FARGO3D training runs: planet at r = 1, phi = 0, Sigma_0 = 1 at r = 1,
initial profile Sigma = r^-0.5, constant aspect ratio):
    r (ny,), phi (nx,) cell centres; log10_sigma (ny, nx); params.
Training ranges: alpha 3e-4 .. 0.1, h0 0.05 .. 0.1, q 5e-5 .. 2e-3, r 0.4 .. 2.5.
"""
import argparse
import os
import sys

import numpy as np

RANGES = {"ALPHA": (10 ** -3.52, 0.1), "ASPECTRATIO": (0.05, 0.1), "PLANETMASS": (10 ** -4.3, 10 ** -2.7)}


def predict(ppdonet, alpha, h0, q, ny=420, nx=1024, unknown="single_log_sigma"):
    sys.path.insert(0, ppdonet)
    import jax.numpy as jnp
    import onet_disk2D.run
    import onet_disk2D.grids

    run_dir = os.path.join(ppdonet, "trained_network", unknown)
    job = onet_disk2D.run.JOB(onet_disk2D.run.load_job_args(run_dir, "args.yml", "arg_groups.yml", "fargo_setups.yml"))
    job.load_model(run_dir)
    g = onet_disk2D.grids.Grids(ymin=0.4, ymax=2.5, xmin=-np.pi, xmax=np.pi, ny=ny, nx=nx)
    pars = {"ALPHA": alpha, "ASPECTRATIO": h0, "PLANETMASS": q}
    u = jnp.asarray([[pars[k] for k in sorted(pars)]])
    out = job.s_pred_fn(job.model.params, job.state, {"u_net": u, "y_net": g.coords_fargo_all["sigma"].reshape((-1, 2))})
    r = np.asarray(g.r_fargo_all["sigma"])
    phi = np.asarray(g.theta_fargo_all["sigma"])
    return r, phi, np.asarray(out).reshape(ny, nx)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ppdonet", default=os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "PPDONet"))
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--h0", type=float, required=True)
    ap.add_argument("--q", type=float, required=True)
    ap.add_argument("--ny", type=int, default=420)
    ap.add_argument("--nx", type=int, default=1024)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for k, v in (("ALPHA", a.alpha), ("ASPECTRATIO", a.h0), ("PLANETMASS", a.q)):
        lo, hi = RANGES[k]
        if not lo <= v <= hi:
            print(f"warning: {k} = {v:g} outside the training range [{lo:g}, {hi:g}]")
    r, phi, ls = predict(os.path.abspath(a.ppdonet), a.alpha, a.h0, a.q, a.ny, a.nx)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, r=r, phi=phi, log10_sigma=ls, alpha=a.alpha, h0=a.h0, q=a.q)
    ref = ls - np.log10(r[:, None] ** -0.5)
    print(f"wrote {a.out}: log10 sigma range {ls.min():.3f} .. {ls.max():.3f}; "
          f"log10(sigma / r^-0.5) at r = 2: {np.median(ref[np.argmin(abs(r - 2))]):+.3f}, "
          f"gap minimum (r = 1 +- 0.1, azimuthal median): {np.min(np.median(ref[abs(r - 1) < 0.1], axis=1)):+.3f}")


if __name__ == "__main__":
    main()
