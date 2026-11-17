"""Stage 1 (see README.md build order): Sun-only Earth sky, monochrome
wavelength-bin loop recombined to RGB. First pipeline smoke test before
swapping the Sun for the Arches cluster.

Usage:
    python render_stage1.py --elevation 20 --out renders/stage1_sun_only/test.png
    python render_stage1.py --variant cuda_ad_mono ...   # on a GPU node
"""
import argparse
import os
import numpy as np

# llvm_ad_* variants need libLLVM >= 15 (opaque-pointer support); the login
# node has none on the system path. A copy lives in third_party/ -- point
# drjit at it unless the caller already set DRJIT_LIBLLVM_PATH.
_LIBLLVM = os.path.join(os.path.dirname(__file__), "..", "third_party", "libllvm", "libLLVM-18.so")
if os.path.exists(_LIBLLVM) and "DRJIT_LIBLLVM_PATH" not in os.environ:
    os.environ["DRJIT_LIBLLVM_PATH"] = os.path.abspath(_LIBLLVM)


# 3 bins for the pipeline smoke test; production run should use ~16 bins
# spanning 380-780nm with proper CIE 1931 color-matching-function weights.
SMOKE_TEST_BINS_NM = [450.0, 550.0, 650.0]


def render_bin(atmosphere, wavelength_nm, elevation_deg, azimuth_deg, variant,
                resolution=(512, 288), spp=64):
    import mitsuba as mi
    mi.set_variant(variant)
    scene_d = atmosphere.scene_dict(
        sun_elevation_deg=elevation_deg,
        sun_azimuth_deg=azimuth_deg,
        wavelength_nm=wavelength_nm,
        resolution=resolution,
    )
    scene = mi.load_dict(scene_d)
    img = mi.render(scene, spp=spp)
    return np.array(img)[..., 0]  # luminance film -> single channel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elevation", type=float, default=20.0, help="Sun elevation, deg")
    ap.add_argument("--azimuth", type=float, default=0.0, help="Sun azimuth, deg")
    ap.add_argument("--variant", default="llvm_ad_mono",
                     help="llvm_ad_mono (vectorized CPU JIT, default) or "
                          "scalar_spectral (slow, no JIT needed) or "
                          "cuda_ad_mono (needs a GPU node)")
    ap.add_argument("--out", default="renders/stage1_sun_only/test.png")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--spp", type=int, default=64)
    args = ap.parse_args()

    import atmosphere  # local module

    channels = []
    for wl in SMOKE_TEST_BINS_NM:
        print(f"rendering bin {wl} nm ...", flush=True)
        channels.append(render_bin(
            atmosphere, wl, args.elevation, args.azimuth, args.variant,
            resolution=(args.width, args.height), spp=args.spp))
    # bins run 450 -> 650 nm (blue -> red); RGB order is the reverse.
    # Placeholder for proper CIE CMF weighting with ~16 bins.
    rgb = np.stack(channels[::-1], axis=-1)

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # raw linear radiance, for real photometry later (libRadtran cross-check,
    # mesopic tone mapping) -- PNG below is display-only.
    raw_path = os.path.splitext(args.out)[0] + "_raw.npy"
    np.save(raw_path, rgb)

    tonemap_and_save(rgb, args.out)
    print(f"wrote {args.out} and {raw_path}")


def tonemap_and_save(rgb, out_path, white_percentile=90):
    """Reinhard-style tone map: compresses the ground's direct-sun highlight
    without crushing the much dimmer singly-scattered-sky signal to zero,
    which is what a naive divide-by-global-max does.
    """
    key = np.percentile(rgb[rgb > 0], white_percentile) if np.any(rgb > 0) else 1.0
    key = max(key, 1e-12)
    mapped = rgb / (rgb + key)
    mapped = mapped / mapped.max() if mapped.max() > 0 else mapped
    mapped = np.clip(mapped, 0, 1) ** (1 / 2.2)

    import imageio.v3 as iio
    iio.imwrite(out_path, (mapped * 255).astype(np.uint8))


if __name__ == "__main__":
    main()
