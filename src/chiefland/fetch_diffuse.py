"""Integrated Gaia DR3 starlight (G >= 9.5) in nested HEALPix cells over the Chiefland frames.

Gaia source_id encodes the level-12 nested HEALPix index (source_id // 2^35), so a level-n cell
is a contiguous source_id range and GROUP BY GAIA_HEALPIX_INDEX(order, source_id) gives the cell
sums directly. Synchronous queries run at once (anonymous async jobs wait in a shared queue,
~2 min each) but return at most 2000 rows, so each query covers one chunk cell with
4^5 = 1024 output cells:
  main frame   : order 9 (6.9') in level-4 chunks
  teapot frame : order 10 (3.4') in level-5 chunks
(the render interpolates the cell map and smooths it to an eye-like resolution of several
arcmin). The archive answers HTTP 500 when an anonymous user runs too many jobs at once.
Per output cell: n, sum 10^(-0.4 G), sum 10^(-0.4 BP), sum 10^(-0.4 RP), and the counts with
BP and RP. Chunks are saved one file each (resumable).

Usage: python3 -m chiefland.fetch_diffuse <outdir> [nthreads]     (run from src/)
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import healpy as hp
import numpy as np
from astroquery.gaia import Gaia
from requests.exceptions import RequestException

from . import geometry

SID_PER_L12 = 2 ** 35
Q = """SELECT GAIA_HEALPIX_INDEX({order}, source_id) AS hpx, COUNT(*) AS n,
 SUM(POWER(10.0, -0.4*phot_g_mean_mag)) AS fg,
 SUM(POWER(10.0, -0.4*phot_bp_mean_mag)) AS fbp,
 SUM(POWER(10.0, -0.4*phot_rp_mean_mag)) AS frp,
 COUNT(phot_bp_mean_mag) AS nbp, COUNT(phot_rp_mean_mag) AS nrp
FROM gaiadr3.gaia_source_lite
WHERE source_id BETWEEN {a} AND {b} AND phot_g_mean_mag >= 9.5
GROUP BY hpx"""


def footprint(frame, chunk_order, min_alt=-4.0, step=8):
    """Nested level-`chunk_order` cells touched by the frame's pixels above min_alt, dilated by
    one ring of neighbours."""
    W, H, f, x0, y0 = geometry.FRAMES[frame]
    v = geometry.pixel_icrs_grid(frame, step=step)[::step, ::step].reshape(-1, 3)
    py, px = np.mgrid[0:H:step, 0:W:step]
    alt, _ = geometry.pix_to_altaz(px.ravel() + 0.5, py.ravel() + 0.5, f, x0, y0)
    v = v[alt >= min_alt]
    nside = 2 ** chunk_order
    cells = np.unique(hp.vec2pix(nside, v[:, 0], v[:, 1], v[:, 2], nest=True))
    nb = hp.get_all_neighbours(nside, cells, nest=True).ravel()
    return np.unique(np.concatenate([cells, nb[nb >= 0]]))


def fetch_chunk(outdir, order, chunk_order, cell, retries=4):
    path = Path(outdir) / f"o{order}_c{chunk_order}_{cell}.npz"
    if path.exists():
        return 0
    span = 4 ** (12 - chunk_order) * SID_PER_L12
    q = Q.format(order=order, a=cell * span, b=(cell + 1) * span - 1)
    for attempt in range(retries):
        try:
            tab = Gaia.launch_job(q).get_results()
            if len(tab) >= 2000:
                raise RuntimeError(f"chunk {cell}: sync row limit")
            break
        except (RequestException, ConnectionError, OSError, ValueError) as err:  # transient archive errors
            print(f"retry {cell} ({attempt}): {err}", flush=True)
            time.sleep(20 * (attempt + 1))
    else:
        raise RuntimeError(f"chunk {cell} failed")
    cols = {c: np.asarray(tab[c].filled(0) if hasattr(tab[c], "filled") else tab[c]) for c in tab.colnames}
    np.savez(path.with_suffix(".tmp.npz"), **cols)
    path.with_suffix(".tmp.npz").rename(path)
    return int(cols["n"].sum()) if len(tab) else 0


def main(outdir, nthreads=4):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(9, 4, int(c)) for c in footprint("main", 4)]
    jobs += [(10, 5, int(c)) for c in footprint("teapot", 5)]
    print(f"{len(jobs)} chunks", flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(nthreads) as ex:
        for n in ex.map(lambda j: fetch_chunk(out, *j), jobs):
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(jobs)} {time.time() - t0:.0f} s", flush=True)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
