"""Season-dependent snowline: albedo set -> terrain.snowline_m, explicit override, cache key, snow cover.

    cd /orange/adamginsburg/gc/visualization && python3 -m pytest -q tests/test_nightsky_snowline.py
"""
import os
import sys

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from nightsky import albedo, foreground, terrain  # noqa: E402

N_THETA = 256  # coarse ground mesh (~130k vertices, ~1 s)


def _cover(aset, snowline_m=None, rmax=10e3):
    """Area-weighted fractions within rmax of (ground rendered with the 'snow' material, needles role)."""
    cfg = terrain.resolve({"n_theta": N_THETA, "albedo_set": aset, "snowline_m": snowline_m})
    g = terrain.ground_mesh(cfg)
    v, w = g["verts"], g["weights"].astype(float)
    r = np.hypot(v[:, 0], v[:, 1])
    m = (r > 0.3) & (r < rmax)
    aw = r[m] ** 2  # rings in geometric progression: vertex area ~ r^2
    white = [k for k, role in enumerate(terrain.ROLES) if albedo.ALBEDO_SETS[aset].get(role) == "snow"]
    return (np.average(w[m][:, white].sum(1), weights=aw),
            np.average(w[m, terrain.ROLES.index("needles")], weights=aw))


@pytest.mark.parametrize("aset", sorted(albedo.SNOWLINE_M))
def test_resolution_per_set(aset):
    want = albedo.SNOWLINE_M[aset]
    assert terrain.snowline({"albedo_set": aset}) == want
    assert terrain.resolve({"albedo_set": aset})["snowline_m"] == want
    cfg = foreground.resolve_scene({"type": "mountains", "seed": 1, "albedo_set": aset})
    assert cfg["snowline_m"] == want and cfg["mountain_height_m"] == terrain.DEFAULTS["mountain_height_m"]


def test_set_ordering_and_fallbacks():
    s = albedo.SNOWLINE_M
    assert s["summer"] > s["autumn"] > s["winter"]
    assert s["summer"] >= 2500  # no summer snow below the default ranges' peaks (~2.2 km)
    assert terrain.DEFAULTS["snowline_m"] is None
    assert foreground.resolve_scene(None)["snowline_m"] == s[foreground.DEFAULTS["albedo_set"]]
    assert terrain.resolve({})["snowline_m"] == s["summer"]
    assert albedo.snowline(dict(albedo.ALBEDO_SETS["winter"])) == s["winter"]
    assert albedo.snowline(dict(albedo.ALBEDO_SETS["autumn"])) == s["summer"]


def test_explicit_override_and_cache_key():
    sc = {"type": "mountains", "seed": 1, "albedo_set": "summer"}
    for v in (1300.0, 0.0, 99999.0):
        assert foreground.resolve_scene({**sc, "snowline_m": v})["snowline_m"] == v
    assert foreground.resolve_scene({**sc, "albedo_set": "winter", "snowline_m": 2500})["snowline_m"] == 2500
    key = lambda d: foreground._geometry_key(foreground.resolve_scene(d))
    k_sum, k_win = key(sc), key({**sc, "albedo_set": "winter"})
    assert len({k_sum, k_win, key({**sc, "snowline_m": 1300.0})}) == 3
    # the key holds the resolved value: explicit == default reuses the geometry
    assert key({**sc, "snowline_m": albedo.SNOWLINE_M["summer"]}) == k_sum
    assert key({**sc, "albedo_set": "winter", "snowline_m": albedo.SNOWLINE_M["winter"]}) == k_win
    # albedo_set itself is not a geometry key: sets with equal snowlines share the geometry
    assert key({**sc, "albedo_set": "autumn", "snowline_m": 500.0}) == key({**sc, "snowline_m": 500.0})


def test_snow_cover_by_season():
    sw, _ = _cover("summer")
    aw, _ = _cover("autumn")
    ww, wn = _cover("winter")
    old, _ = _cover("summer", 1300.0)  # the previous default
    print(f"snow (white) area fraction within 10 km: summer {sw:.4f} autumn {aw:.4f} winter {ww:.3f} "
          f"(winter needles {wn:.3f}); summer with snowline 1300 m {old:.3f}")
    assert sw < 0.01
    assert sw <= aw < 0.05
    assert ww > 0.4 and wn > 0.2  # winter: white open ground and gentle slopes, dark forest kept
    assert old > 0.03  # explicit override reaches the material map


def test_snow_mask_terms():
    """Snow needs height above the line and a gentle slope, and stays out of the forest texture."""
    x = np.array([20e3, 20e3, 20e3, 20e3])
    y = np.array([5e3, 5e3, 5e3, 5e3])
    z = np.array([3500.0, 3500.0, 100.0, 3500.0])
    nz = np.array([1.0, 0.3, 1.0, 1.0])
    cfg = terrain.resolve({"albedo_set": "summer"})
    w, _ = terrain.materials(x, y, z, nz, cfg)
    snow = w[:, terrain.ROLES.index("snow")]
    assert snow[0] > 0.9 and snow[1] < 0.05 and snow[2] == 0
    w2, _ = terrain.materials(x, y, z, nz, {**cfg, "snowline_m": 99999.0})
    assert np.all(w2[:, terrain.ROLES.index("snow")] == 0)
    np.testing.assert_allclose(w.sum(1), 1, atol=1e-5)
