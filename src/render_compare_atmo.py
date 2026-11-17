"""Single frame, cluster + wide companion only, rendered with and without the
Rayleigh atmosphere -- isolates what the atmosphere model adds (sky glow,
diffuse ground light) from the painted-in point sources.

Night side: Sun off, so the only illumination is the cluster and companion
themselves. Hour angle is chosen (or given) so the companion is above the
horizon; the cluster sits fixed at the pole (altitude = latitude) regardless.

    python3 render_compare_atmo.py --out ../renders/compare_atmo
"""
import argparse
import os

import numpy as np

_LIBLLVM = os.path.join(os.path.dirname(__file__), "..", "third_party", "libllvm", "libLLVM-18.so")
if os.path.exists(_LIBLLVM) and "DRJIT_LIBLLVM_PATH" not in os.environ:
    os.environ["DRJIT_LIBLLVM_PATH"] = os.path.abspath(_LIBLLVM)

import atmosphere as at  # noqa: E402
import companion as co  # noqa: E402
import compose_day as cd  # noqa: E402
import photometry as ph  # noqa: E402
import render_day as rd  # noqa: E402
import views as vw  # noqa: E402


def best_hour_angle():
    H = np.linspace(-180, 180, 361)
    alts = []
    for h in H:
        d = co.direction() @ rd.celestial_to_horizon(h).T
        alts.append(np.degrees(np.arcsin(np.clip(d[2], -1, 1))))
    return float(H[np.argmax(alts)])


def render_group(mi, a, n_shells, face, spp, directional_sources, wls):
    cubes = []
    for wl in wls:
        print(f"[shells={n_shells}] bin {wl:.1f} nm, face {face}, spp {spp}", flush=True)
        scene = mi.load_dict(at.scene_dict(
            wavelength_nm=float(wl), n_shells=n_shells, cube_face_size=face,
            ground_albedo=a.albedo, sun_delta=True, solar_irradiance=0.0,
            directional_sources=[dict(direction=s["direction"], irradiance=s["irradiance"][wls == wl][0])
                                  for s in directional_sources]))
        cubes.append(np.stack([np.array(mi.render(scene, sensor=k, spp=spp, seed=1))[..., 0]
                               for k in range(6)]).astype(np.float32))
    return np.stack(cubes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../renders/compare_atmo")
    ap.add_argument("--variant", default="llvm_ad_mono")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--face", type=int, default=160, help="cube face size, px (higher res than the day-cycle default 128)")
    ap.add_argument("--spp-atmo", type=int, default=8192)
    ap.add_argument("--spp-vacuum", type=int, default=1024)
    ap.add_argument("--shells", type=int, default=24)
    ap.add_argument("--albedo", type=float, default=0.3)
    ap.add_argument("--stars", type=int, default=300, help="individually resolved cluster sources fed to the renderer")
    ap.add_argument("--cluster-mass", type=float, default=1.5e4)
    ap.add_argument("--imf-alpha", type=float, default=1.85)
    ap.add_argument("--distance-pc", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hour-angle", type=float, default=None, help="deg; default picks the companion's highest altitude")
    ap.add_argument("--scale", type=float, default=1.5, help="output size multiplier on the 1080x1080 all-sky view")
    ap.add_argument("--l-ref", type=float, default=30.0)
    ap.add_argument("--gamma", type=float, default=0.75)
    a = ap.parse_args()
    mi = rd.setup(a)
    wls, dwl = ph.bin_centers(a.bins)
    M = cd.spectral_matrix(wls, dwl)

    c = rd.make_cluster(a)
    print(c.summary(), flush=True)
    H = a.hour_angle if a.hour_angle is not None else best_hour_angle()
    comp_cel = co.direction()
    comp_dir = comp_cel @ rd.celestial_to_horizon(H).T
    comp_alt = float(np.degrees(np.arcsin(np.clip(comp_dir[2], -1, 1))))
    print(f"hour angle {H:.1f} deg, companion altitude {comp_alt:.1f} deg", flush=True)
    print(co.summary(), flush=True)
    if comp_alt < 0:
        print("WARNING: companion below horizon at this hour angle", flush=True)

    res_dirs, res_irr, _ = c.resolved_and_rest(wls, k=a.stars)
    comp_irr_bins = co.spectral_irradiance(wls)[0]  # (n_bins,)
    directional_sources = (
        [dict(direction=res_dirs[i], irradiance=res_irr[i]) for i in range(len(res_dirs))]
        + [dict(direction=comp_dir, irradiance=comp_irr_bins)]
    )

    os.makedirs(a.out, exist_ok=True)
    faces_atmo = render_group(mi, a, a.shells, a.face, a.spp_atmo, directional_sources, wls)
    faces_vac = render_group(mi, a, 0, a.face, a.spp_vacuum, directional_sources, wls)
    np.savez(os.path.join(a.out, "faces.npz"), atmo=faces_atmo, vacuum=faces_vac, wavelengths=wls, bin_width=dwl,
             hour_angle=H)
    print("wrote", os.path.join(a.out, "faces.npz"), flush=True)

    # analytic point sources for the final image: the full star list (not just the
    # resolved subset fed to the renderer) plus the companion
    all_irr = c.spectral_irradiance(wls)
    dirs_all = np.vstack([c.direction, comp_dir[None]])
    irr_all = np.vstack([all_irr, comp_irr_bins[None]])

    view = vw.View("allsky", int(1080 * a.scale), int(1080 * a.scale))
    img_atmo = cd.sample_bins(faces_atmo, view.dirs, M) + cd.point_layer(view, dirs_all, irr_all, wls, M, extinct=True)
    img_vac = cd.sample_bins(faces_vac, view.dirs, M) + cd.point_layer(view, dirs_all, irr_all, wls, M, extinct=False)

    log_la = np.log(cd.log_average(img_atmo, view.valid))
    scale, meso = cd.adaptation(np.array([log_la, log_la]), a.gamma, a.l_ref, 0.01)
    scale, meso = float(scale[0]), float(meso[0])
    print(f"shared exposure: log-La(atmo)={np.exp(log_la):.4g} cd/m^2, scale x{scale:.3g}, photopic-fraction {meso:.2f}",
          flush=True)

    from PIL import Image, ImageDraw
    import imageio.v3 as iio
    for name, img in (("with_atmosphere", img_atmo), ("no_atmosphere", img_vac)):
        out8 = cd.to_srgb8(img, scale, meso)
        out8 = np.where(view.valid[..., None], out8, 0).astype(np.uint8)
        im = Image.fromarray(out8)
        ImageDraw.Draw(im).text((10, 8), name.replace("_", " "), fill=(235, 235, 235), stroke_width=1, stroke_fill=(0, 0, 0))
        path = os.path.join(a.out, f"{name}.png")
        iio.imwrite(path, np.asarray(im))
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
