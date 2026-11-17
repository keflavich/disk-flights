"""Gaia DR3 G < 9.5 over the Chiefland main-frame footprint with synchronous queries (the
anonymous async queue takes ~2 min per job). One query per level-3 HEALPix cell (a contiguous
source_id range), split into its four level-4 children when it reaches the 2000-row sync limit.
Writes gaia_bright.fits with the same columns as fetch_catalogs.py.

Usage: python3 -m chiefland.fetch_bright_sync <outdir> [nthreads]     (run from src/)
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from astropy.table import vstack
from astroquery.gaia import Gaia
from requests.exceptions import RequestException

from .fetch_catalogs import GAIA_Q
from .fetch_diffuse import SID_PER_L12, footprint

Q = GAIA_Q.replace("g.ra >= {a} AND g.ra < {b}", "g.source_id BETWEEN {a} AND {b}")
SYNC_ROWS = 2000


def fetch(level, cell, retries=5):
    span = 4 ** (12 - level) * SID_PER_L12
    q = Q.format(a=cell * span, b=(cell + 1) * span - 1)
    for attempt in range(retries):
        try:
            tab = Gaia.launch_job(q).get_results()
            break
        except (RequestException, OSError) as err:  # the archive answers HTTP 500 under load
            print(f"retry {level}/{cell} ({attempt}): {err}", flush=True)
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"cell {level}/{cell} failed")
    if len(tab) >= SYNC_ROWS:
        return vstack([fetch(level + 1, 4 * cell + i) for i in range(4)])
    return tab


def main(outdir, nthreads=4):
    cells = footprint("main", 3)
    t0 = time.time()
    with ThreadPoolExecutor(nthreads) as ex:
        parts = list(ex.map(lambda c: fetch(3, int(c)), cells))
    tab = vstack([t for t in parts if len(t)])
    tab.write(Path(outdir) / "gaia_bright.fits", overwrite=True)
    print("gaia_bright", len(tab), f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
