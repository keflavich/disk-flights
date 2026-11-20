# disk-flights: physically based sky and disk visualizations

Spectral renderers for two families of scenes:

- **Disk flights** (`src/diskorbit/`): a camera on an orbit through a protoplanetary disk, rendered
  from a Hyperion radiative-transfer model (Robitaille/Richardson model grid) with our own numba
  ray marcher, optical (eye / auto exposure) and IR false-colour (+ 1.3 mm overlay) movies, a
  background star layer, and a position key. See "Disk flights" below.
- **Night skies** (`src/nightsky/`): Earth's sky and ground if the solar system sat a few pc from
  the Arches cluster near the Galactic Center (and Earth comparisons), plus the earlier Mitsuba
  day-cycle pipeline.

Data (`data/`), renders (`renders/`) and third-party binaries (`third_party/`) are not tracked.
Several scripts carry HiPerGator paths (`/orange/adamginsburg/...`, `/blue/...`) and SLURM
settings; adjust them for other systems.

License: BSD-3-Clause (see `LICENSE`).

## Disk flights (`src/diskorbit`)

    cd src
    python3 -m diskorbit.render stills|sed|orbit|develop|orbit-ir|develop-ir|stars|stars-ir ...
    python3 -m diskorbit.inset --nproc 16          # position key, overlaid on existing frames

- `model.py`: reads one Hyperion model of the grid (par / rtin / rtout: density, dust, specific
  energy -> temperature, star), analytic flared-disk density, warm-rim fix, (ln r, mu) tables.
- `raymarch.py`: camera rays; source function = attenuated single scattering + Monte Carlo
  multiple-scattering moments (`mcscat.py`) + thermal emission; returns the column so stars can
  be extinguished.
- `render.py`: scene, orbit schedule (slows near the disk surface), frame storage (bf16),
  development (eye adaptation or auto), IR band sets (`bands.py`).
- `starlayer.py`: background stars (BSC, V < 6.5) as a separate, fixed-stretch layer, disk-
  extinguished, screen-blended over the developed frames.
- `inset.py`: meridional density slice, orbit arc, camera and field-of-view key.
- Validation: distant-camera SEDs against Hyperion's peeled SEDs (optical star within 1.5 %,
  scattered light -8 % to +17 %; thermal 10-160 um and 1.3 mm within 2 %).

### Planet-gap disk (`src/diskplanet`)

A non-axisymmetric version of one grid model (dIULECee) with a q = 1e-3 planet at 100 au carving a
gap, from [PPDONet](https://github.com/andizq/PPDONet) (alpha = 1e-3, h0 = 0.051):

- `ppdo_predict.py`: PPDONet surface density (needs PPDONet + jax in a separate venv under
  `third_party/`, not tracked; PPDONet is GPL) -> `sigma_*.npz`.
- `perturb.py`: delta(R, phi) = Sigma / Sigma_0, tapered in PPDONet's damping zones;
  rho_3d = rho_axi(R, z) * delta(R, phi).
- `hyperion3d.py`: 3D spherical-polar Hyperion input from the model's 2D grid (128 phi cells),
  optional planet point source (1e-4 Lsun, 1200 K blackbody); run with `slurm/hyperion3d.sbatch`.
- `model3d.PlanetDisk`: the renderer's disk model with a (phi, ln r, mu) column/temperature
  table; 3D Hyperion temperature where it has enough photons, axisymmetric temperature otherwise.
- The renderer (`diskorbit`) takes the perturbation through `--set planet_sigma=...`; the scattering
  Monte Carlo is binned in phi (`mc_nphi`), and the pattern can co-rotate with the camera
  (`pattern_mode=coorbit`) or stay fixed.
- `slurm/diskplanet_movie.sh {coorbit|lap} {vis|ir|post}` submits frame-range jobs and runs the
  develop / star layer / key steps.

- `planet_clump_h` (`VARIANT=taper` in the driver): the planet's excess over the local gap level
  gets a vertical scale height of `planet_clump_h` x the disk's, at fixed surface density. With the
  default 0, delta(R, phi) multiplies the density at every height, so the planet's overdensity
  extends through the disk surface layers and shows as a vertical streak through the planet.
- `orbit_speed`: `surface` (default; slow at the lit surface layer, fast through the midplane) or
  `uniform`. The driver's `incl45` and `polar` movies fly 45 and 90 deg inclined circular orbits.
- `diskplanet/spiral.py` adds global m-armed logarithmic spiral arms (a stand-in for arms driven
  by an outer companion or gravitational instability, not by the planet) to a PPDONet surface
  density, keeping the ring-averaged Sigma(r). `VARIANT=spiral` in the driver = the taper setup
  plus two 15 deg pitch arms around 160 au with 0.25 dex cosine amplitude, rotating rigidly with
  the planet.
- `camera_up`: `z` (default) keeps the image horizon parallel to the midplane; the picture rolls
  by 180 deg in one frame whenever the view passes the disk axis (the `polar` movie does, twice).
  `radial` takes "up" as away from the star, which stays continuous on any orbit.
- Intro (`intro_from_au`, `intro_hold`, `intro_fly`, `intro_pan`, in frames): hold far out on the
  ray from the star through the orbit's start, fly in (log-eased) to `a_au`, then pan from the
  star to the orbit's view; the orbit uses the remaining frames (`diskorbit.render.CameraTrack`).
  During the intro the star point is left out (`intro_star=0`; its glare hides the face-on disk)
  and the eye exposure follows the bright pixels (`intro_key` x the 99.5th-percentile luminance),
  blending back to the normal adaptation over the pan. The driver's `polarfly` movie = `polar` with
  `camera_up=radial` and a 1 s hold at 1000 au, 10 s dive and 4 s pan (2250 frames).

## Night skies

## nightsky: configurable night-sky renderer (current pipeline)

`src/nightsky/` renders night skies from a YAML config with swappable parts: atmosphere
(none / Earth clear / observatory / hazy / Mars), sources (starfield populations, 3D galaxy and
GC gas model, Earth bright stars), foreground (mountains, meadow, forest, lakeshore, plane) and
an eye or camera model. The GC scenario puts the solar system 3 pc from the Arches cluster,
which sits at the north celestial pole (latitude +40, equinox).

    cd src
    python3 -m nightsky render ../configs/gc_arches.yaml [--scale 0.25] [--set key=value ...]
    python3 -m nightsky info ../configs/gc_arches.yaml
    python3 -m nightsky validate
    python3 -m nightsky.animate ../configs/gc_arches_timelapse.yaml   # hour-angle time-lapse
    python3 -m nightsky render ../configs/gc_arches.yaml --set name=gc_arches_moon --set scenario.moon.enabled=true

Scenario configs: gc_arches (flagship), gc_arches_forest, gc_noatm, gc_hazy, gc_mars,
gc_darkcloud (observer inside a dark cloud, `galaxy.extra_fields`), gc_hii_region (inside an
HII region), earth_darksky, earth_moonlit (`scenario.moon`), earth_darksky_camera, gc_legacy.

Common knobs (all settable with `--set`): `atmosphere=earth_hazy|mars|none`,
`foreground.type=mountains|meadow|forest|lakeshore`, `foreground.albedo_set=summer|autumn|winter`
(sets the snowline unless `foreground.snowline_m` is given), `camera.mode=eye|camera`,
`points.visibility=false` (draw every star; by default eye mode fades stars below the naked-eye
threshold for the local sky brightness into an unresolved glow), `starfield.arches.zams=true`
(unevolved Arches), `galaxy.extra_fields=[...]` (dark clouds, HII regions; see clouds.py).
Tests: `python3 -m pytest -q ../tests/` (58 pass, 1 skip).

Outputs go to `renders/nightsky/<config name>/` (`<view>.png`, linear `.npz`, `.json`
photometry, `scene.json`). The API contract is in `src/nightsky/INTERFACES.md`; state,
validation numbers and next steps are in `HANDOFF.md`. The sections below describe the
earlier Mitsuba volumetric pipeline (`src/render_day.py` and friends).

## Stack

| Job | Tool |
|---|---|
| Sun/Moon positions, sidereal time, alt-az | astropy |
| Arches/Quintuplet member lists | astroquery -> VizieR (Clark, Martins, Hosek) |
| Stellar spectra | PoWR, OSTAR2002, MIST/PARSEC; blackbody for first pass |
| Ionized-gas emissivities | Cloudy / pyCloudy |
| 3D dust RT -> all-sky map | own numpy/JAX ray marcher first, SKIRT later |
| Sky map handling | healpy |
| Atmosphere + landscape rendering | Mitsuba 3 (`mitsuba` pip package, installed) |
| Independent atmosphere check | libRadtran |
| Color/perception | colour-science, OpenEXR/imageio |
| Video | ffmpeg |

Mitsuba variants installed: scalar_rgb, scalar_spectral(_polarized),
llvm_ad_{rgb,mono,spectral}(_polarized), cuda_ad_{rgb,mono,spectral}(_polarized).
LLVM JIT unavailable on the login node (no libLLVM on path) — use
`scalar_spectral` for CPU dev/testing, `cuda_ad_mono` on a GPU node for
production wavelength-bin loops.

## Layout

- `src/atmosphere.py` -- concentric-shell Rayleigh atmosphere scene builder (Mitsuba 3)
- `src/photometry.py` -- magnitudes/blackbodies -> W/m^2/nm, CIE colour matching
- `src/cluster.py` -- IMF-sampled synthetic cluster (ZAMS-like L/Teff, Plummer profile)
- `src/views.py` -- cube-face resampling into all-sky/perspective views, point-source injection
- `src/render_stage1.py`, `src/render_stage2.py` -- Sun-only and toy-Arches tests
- `src/render_day.py`, `src/compose_day.py` -- the day-cycle animation (below)
- `slurm/` -- CPU batch scripts (account astronomy-dept, QOS astronomy-dept-b, hpg-default)
- `renders/` -- outputs; `data/` -- downloaded catalogs/spectra

## Day-cycle animation (Stage 3)

Observer at +40 deg latitude on the equinox; the synthetic cluster (1.5e4 Msun, IMF slope
1.85, 1-120 Msun, 3 pc away, V ~ -13.5) sits fixed at the celestial pole (altitude 40 deg,
due north). Because the cluster and the uniform diffuse sky (17 mag/arcsec^2, 3500 K
shape) are fixed in horizon coordinates, only the Sun group changes with time.

    python3 render_day.py static --out ../renders/day/data          # sky + cluster, once
    python3 render_day.py sun --frames 26-72 --out ../renders/day/data   # Sun frames, H <= 0
    python3 compose_day.py --data ../renders/day/data --out ../renders/day/frames
    (slurm/day.sbatch, slurm/compose.sbatch, slurm/video.sbatch wrap these)

- Each light group is rendered as six cube faces per wavelength bin (8 bins, 400-700 nm)
  and combined linearly in `compose_day.py`; the Sun and stars are delta lights.
- At the equinox the Sun's path is mirror-symmetric about noon, so only H <= 0 is rendered
  and compose mirrors y -> -y for the afternoon. Frames with the Sun below -20 deg
  altitude are treated as having no Sun contribution.
- The Sun disk and every star are painted in analytically with slant-path extinction
  (sources this small are far below the render resolution); the render supplies the
  smooth sky/ground illumination and scattered halos.
- Local field stars (`cluster.LocalStars`, on by default in `compose_day.py`): 1000 Msun drawn
  from a Kroupa (2001) IMF between 0.3 and 4 Msun (1288 stars, ZAMS L/Teff), uniform in a
  1x1x1 pc box centred on the Sun; positions closer than 50 au to the Sun are re-drawn (none
  in the default seed: nearest star 1.6e4 au). Total V ~ -10.9, brightest V ~ -7.8, 409 stars
  brighter than V = 0. They are fixed on the celestial sphere, so compose rotates them into
  the horizon frame every frame (`render_day.celestial_to_horizon`); the Arches sits at the
  pole and stays put. They are painted in post only: their light is ~10% of the Arches' and
  spread over the sky, so their effect on sky/ground illumination is neglected.
  Options: `--no-local`, `--local-mass`, `--local-seed`.
- Wide companion (`companion.py`, on by default): a Sun-like star at 1000 au, circular orbit
  inclined 25 deg to the ecliptic (V ~ -11.7, comparable to the full Moon). At 1000 au around
  2 Msun total the period is ~22,000 yr, so like the local field stars it is fixed on the
  celestial sphere and rotated into the horizon frame each frame -- not literally orbiting
  within the animation. The request specified separation and inclination only; the orbit
  shape (circular), ascending node and orbital phase are my choices, at
  `companion.NODE_DEG`/`PHASE_DEG` -- change via `--companion-node`/`--companion-phase`, or
  the whole thing with `--no-companion`, `--companion-au`, `--companion-incl`. Verified against
  two analytic checks (phase=0 lands exactly on the equinox direction; inclination=90 lands
  exactly on the ecliptic pole, computed independently from the obliquity).
- Earth comparison (`compose_day.py --earth`): same atmosphere, ground and Sun frames, but a natural
  dark-sky background (21.7 mag/arcsec^2, 5000 K shape; the GC sky group rescaled bin by bin, so no
  re-render) and the real stars with V < 2 from the Yale Bright Star Catalogue (`bright_stars.py`,
  VizieR V/50, J2000 coordinates, B-V -> blackbody Teff), rotated with the celestial sphere for the
  September equinox (Sun at RA 12h). `--fixed-la 0.036` puts any sky at the GC night exposure.
  No Moon, Milky Way, zodiacal light or airglow structure.
- Views: `allsky` (equidistant fisheye, N up, E left) and `north` (perspective window).
- Exposure follows a smoothed eye adaptation (exponent 0.75) and a simple mesopic blend
  towards a blue-shifted scotopic response at low luminance.

- Noise: single scattering by path tracing at tau ~ 0.1 leaves most pixels with no sunlit
  scattering event, so per-pixel noise is large (12% at 512 spp on a twilight preview frame).
  The sky is smooth, so (a) each cube face is Gaussian-smoothed separately on the sky and ground
  side of the horizon (no bleeding across it, verified), and (b) twilight frames (Sun altitude
  < 8 deg) are rendered on 32-px faces with 16384 spp instead of 128-px faces with 256 spp, at
  4x the cost of a daytime frame. Faces cross-fade over their 92-deg overlap. Preview-quality
  renders (32-px faces, 32-128 spp) show colour blotches shaped like the cube faces; that is noise.
- `GC_N_FRAMES=48` (env var) shrinks the day to 48 frames for cheap previews on the login node.

Limitations: Rayleigh scattering only (no aerosols, no ozone -- so twilight lacks the
Chappuis-band blue), constant ground albedo 0.3, flat featureless ground, no refraction,
no Moon, no stellar evolution or extinction in the cluster, mesopic model is a rough blend.

## Build order

1. **Sun only.** Render Earth's sky in Mitsuba, sunset/twilight, check against
   libRadtran and known zenith twilight brightness. (`src/render_stage1.py`)
2. **Toy Arches.** One directional light (V ~ -13.5, 35000 K blackbody) +
   uniform orange sky at ~17 mag/arcsec^2.
3. **Real star list.** Real Arches members from astroquery/VizieR, sampled
   line-of-sight depths — shadow fans appear here.
4. **Sky model.** Ray marcher first (numpy/JAX), SKIRT for multiple
   scattering in bright rims. This is where the research effort goes.
5. **Finishing.** Terrain, Moon (Hapke/Lommel-Seeliger + opposition surge),
   mesopic tone mapping (Kirk & O'Brien style), 24-hour animation.

## Known issues / findings

- **The sensor must declare its initial medium** (`"medium": {"type": "ref", ...}`
  on the `perspective` sensor) when the camera sits inside a medium boundary.
  Without it, camera rays start in vacuum and never interact with the
  atmosphere: the sky renders exactly 0, output is insensitive to `sigma_t`,
  and all wavelength bins come out identical. Delta lights (`point`,
  `directional`) work through media once this is set; they had appeared dead
  only because they are visible solely via medium interactions.
- **Perspective sensors default to `far_clip = 1e4` m**, which truncates camera
  rays inside the ~1e5 m-thick atmosphere; they then never reach the sky
  emitter (sky pixels showed only in-scattered light, 1-9% of the true
  radiance). `atmosphere.scene_dict` sets `far_clip = 1e10`.
- **Nested shell spheres need `exterior` = the next shell's medium.** With only
  `interior` set, a ray leaving shell i enters vacuum, so every shell except the
  innermost is silently dropped. Verified with an absorbing atmosphere against
  the analytic slant optical depth (1-2% agreement for 16 and 32 shells down to
  2 deg elevation). 48 shells at 10 deg elevation read 13% low and is
  unexplained; stay at <=32 shells until that is understood.
- Rayleigh constants give zenith tau = 0.216 / 0.097 / 0.050 at 450 / 550 /
  650 nm, matching the standard values.
- **Path-traced single scattering converges slowly at realistic Rayleigh
  optical depths** (zenith tau ~0.1-0.15), so expect to need spp in the
  thousands for a smooth sky. A dedicated single-scattering ray marcher for
  the full sky dome (build-order step 4) avoids this.

## Conventions

- World origin at the camera; Earth's center at `(0, 0, -R_EARTH)` — keeps
  float32 precision good near the camera regardless of scene scale.
- Monochrome wavelength-bin loop (~16 bins, 380-780nm) recombined to RGB in
  numpy, rather than RGB rendering — needed for line emission and very
  different source spectra (stars vs. ionized gas vs. dust continuum).
- Light groups: no native Mitsuba pass: render each source group alone with
  others zeroed, recombine in numpy. Makes per-source extinction reweighting
  (e.g. "what if Arches were more extincted") a numpy op, not a re-render.
