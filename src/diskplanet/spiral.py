"""Add global spiral arms to a PPDONet surface density (ppdo_predict.py npz), e.g. to stand in for
arms driven by an outer companion or gravitational instability, keeping the planet's gap.

The arms are m-armed logarithmic spirals in log Sigma with a Gaussian radial envelope,

    log10 Sigma += amp * exp(-(r - rc)^2 / (2 sr^2)) * cos(m (phi - ln(r / r0) / tan(pitch))),

renormalised ring by ring so the azimuthally averaged Sigma(r) (and so the disk mass and the gap
profile) is unchanged. Radii are in units of the planet radius, as in the PPDONet grid, which ends
at 2.5 r_p; choose rc and sr so the arms have faded by then. The arms are tied to the PPDONet
frame, so in the movies they rotate rigidly with the planet's pattern.

    python -m diskplanet.spiral ../renders/diskplanet/sigma_q1e-3_a1e-3_h0.051.npz \
        ../renders/diskplanet/sigma_q1e-3_a1e-3_h0.051_sp2.npz --amp 0.25 --pitch 15 --rc 1.6 --sr 0.35
"""
import argparse

import numpy as np


def add_spirals(r, phi, log10_sigma, amp=0.25, pitch_deg=15.0, rc=1.6, sr=0.35, m=2, r0=1.2):
    R, P = np.meshgrid(r, phi, indexing="ij")
    arg = m * (P - np.log(R / r0) / np.tan(np.deg2rad(pitch_deg)))
    s = amp * np.exp(-0.5 * ((R - rc) / sr) ** 2) * np.cos(arg)
    sig0 = 10.0 ** log10_sigma
    sig1 = sig0 * 10.0 ** s
    return np.log10(sig1 * (sig0.mean(axis=1, keepdims=True) / sig1.mean(axis=1, keepdims=True)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--amp", type=float, default=0.25, help="cosine amplitude, dex")
    ap.add_argument("--pitch", type=float, default=15.0, help="pitch angle, deg")
    ap.add_argument("--rc", type=float, default=1.6, help="envelope centre, r_p")
    ap.add_argument("--sr", type=float, default=0.35, help="envelope Gaussian width, r_p")
    ap.add_argument("--m", type=int, default=2, help="number of arms")
    ap.add_argument("--r0", type=float, default=1.2, help="phase reference radius, r_p")
    a = ap.parse_args()
    d = dict(np.load(a.src))
    ls = add_spirals(d["r"], d["phi"], d["log10_sigma"], a.amp, a.pitch, a.rc, a.sr, a.m, a.r0)
    lin0, lin1 = 10.0 ** d["log10_sigma"], 10.0 ** ls
    d["log10_sigma"] = ls
    d.update(spiral_amp=a.amp, spiral_pitch=a.pitch, spiral_rc=a.rc, spiral_sr=a.sr, spiral_m=a.m, spiral_r0=a.r0)
    np.savez(a.out, **d)
    ring = np.max(np.abs(lin1.mean(axis=1) / lin0.mean(axis=1) - 1))
    nb = ls - np.median(ls, axis=1, keepdims=True)
    sel = (d["r"] > 1.25) & (d["r"] < 2.2)
    print(f"wrote {a.out}: max ring-mean change {ring:.1e}; arm contrast (p95 - p5, 1.25-2.2 r_p) "
          f"{np.percentile(nb[sel], 95) - np.percentile(nb[sel], 5):.2f} dex")


if __name__ == "__main__":
    main()
