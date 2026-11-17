"""Download the point-source catalogues for the Chiefland sky (Gaia archive TAP; CDS is not
reachable from HiPerGator login nodes). The archive truncates results at 50000 rows, so the
queries are split into RA slices.

  hipparcos.fits   : public.hipparcos (original 1997 reduction: V, B-V, V-I), Vmag < 9.5
  gaia_bright.fits : gaiadr3.gaia_source G < 9.5 with the Hipparcos-2 best-neighbour id (if any)

Usage: python3 -m chiefland.fetch_catalogs <outdir>     (run from src/)
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from astropy.table import vstack
from astroquery.gaia import Gaia
from requests.exceptions import RequestException

MAX_ROWS = 50000

HIP_Q = """SELECT hip, ra, de, plx, pmra, pmde, vmag, b_v, v_i, hpmag
FROM public.hipparcos WHERE vmag < 9.5 AND ra >= {a} AND ra < {b}"""

GAIA_Q = """SELECT g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.parallax, g.phot_g_mean_mag,
       g.phot_bp_mean_mag, g.phot_rp_mean_mag, g.bp_rp, g.teff_gspphot,
       h.original_ext_source_id AS hip2, h.angular_distance AS hip2_sep
FROM gaiadr3.gaia_source AS g
LEFT OUTER JOIN gaiadr3.hipparcos2_best_neighbour AS h ON h.source_id = g.source_id
WHERE g.phot_g_mean_mag < 9.5 AND g.ra >= {a} AND g.ra < {b}"""


def _slice(q, a, b, retries=5):
    for attempt in range(retries):
        try:
            tab = Gaia.launch_job_async(q.format(a=a, b=b), dump_to_file=False).get_results()
            break
        except (RequestException, OSError) as err:  # the archive answers HTTP 500 under load
            print(f"retry {a} ({attempt}): {err}", flush=True)
            time.sleep(15 * (attempt + 1))
    else:
        raise RuntimeError(f"slice {a}-{b} failed")
    if len(tab) >= MAX_ROWS:
        raise RuntimeError(f"slice {a}-{b} hit the {MAX_ROWS}-row limit")
    return tab


def main(outdir):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    edges = list(range(0, 361, 10))
    for name, q in (("hipparcos", HIP_Q), ("gaia_bright", GAIA_Q)):
        with ThreadPoolExecutor(2) as ex:
            parts = list(ex.map(lambda ab: _slice(q, *ab), zip(edges[:-1], edges[1:])))
        tab = vstack(parts)
        tab.write(out / f"{name}.fits", overwrite=True)
        print(name, len(tab), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
