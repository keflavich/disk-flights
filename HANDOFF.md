# Handoff: nightsky (artificial night-sky renderer, GC flagship)

Last updated 2026-10-25. Nothing is committed to git yet (the repo has no commits; all files untracked).

## Goal

1. A photorealistic view of the sky from a planet whose star sits 3 pc from the Arches cluster
   (Arches = north celestial pole star, observer latitude +40 deg, equinox).
2. Reusable infrastructure for artificial night skies under different conditions: swappable
   atmosphere, simulated starfield, galaxy/gas model and a human-scale foreground (mountains,
   meadow, forest, lakeshore), all from one YAML config.
3. Later "what if" scenarios the user mentioned: solar system inside the Orion Nebula, inside a
   dark cloud, at the Galactic Center (done), plus a 24-h day-night animation.

## How to run

All commands from `src/`. Heavy runs need the worker (login node has 2 cores):

    cd /orange/adamginsburg/gc/visualization/src
    python3 -m nightsky render ../configs/gc_arches.yaml                  # full render
    python3 -m nightsky render ../configs/gc_arches.yaml --scale 0.25     # preview
    python3 -m nightsky render ../configs/gc_arches.yaml --set atmosphere=earth_hazy --set camera.mode=camera
    python3 -m nightsky info ../configs/gc_arches.yaml                    # resolved config + cache state
    python3 -m nightsky validate                                          # 12 photometric checks
    python3 -m pytest -q ../tests/                                        # 58 pass, 1 skip (2026-10-25)

Batch: `sbatch slurm/nightsky.sbatch configs/<name>.yaml` (astronomy-dept-b, hpg-default, 32 cpu;
this queue waited ~1 day in Oct 2026). Fast path: reserve an hpg-dev `sleep` job and run steps
with `slurm/onworker.sh` (see that file; set WORKER=<jobid>, CPUS, MEM, and PYTHONPATH=$PWD).
Outputs: `renders/nightsky/<config name>/<view>.{png,npz,json}` plus `scene.json`; stage caches
in `renders/nightsky/<config name>/cache/` (starfield, galaxy map, foreground keyed on config
hashes, so camera-only changes rerun in ~15 s).

## Architecture (src/nightsky/)

| Module | Role |
|---|---|
| `spectral.py`, `frames.py`, `sky.py`, `fields.py` | frozen contracts: SpectralGrid (20 bins 380-780 nm), frames, PointSources/TOASky, VoxelField |
| `INTERFACES.md` | API contract for every module + GC scenario definition |
| `atmosphere.py` | numba LUT atmosphere (Hillaire 2020 multiple scattering, ozone, aerosol, airglow); presets none, earth_clear, earth_observatory, earth_hazy, mars |
| `galaxy.py` | 3D Milky Way + GC stellar/dust model (NSC, NSD, bar, long bar, disks, dust, optional CMZ term, DGL) with a numba ray marcher -> HEALPix TOA map |
| `gas.py` | GC ionized gas (Arched Filaments, Sickle, Sgr A West, HII regions) and molecular clouds as VoxelFields; cached under renders/nightsky/cache/gas |
| `starfield.py`, `stellar_pops.py` | point sources: Arches, Quintuplet, NSC young stars, field WR/O, local old population within r_res, ZAMS box, 1000-au companion, Earth BSC (V<6.5) |
| `terrain.py`, `foreground.py`, `albedo.py` | procedural terrain + instanced trees, rendered per wavelength bin in Mitsuba 3 (llvm_ad_mono), lit by the sky env map + brightest point sources; returns premultiplied radiance, alpha, depth |
| `camera.py` | views (allsky, perspective, panorama), PSF deposit, eye model (mesopic adaptation, CIE glare, local mesopic for bright stars) or camera model (ISO/f/t, bloom), tone map |
| `compose.py`, `config.py`, `cli.py` | YAML -> stages -> composite -> PNG/npz/JSON with caching |
| `clouds.py` | config-driven extinction / emission VoxelFields around the observer (`galaxy.extra_fields`): dark cloud, HII region |
| `moon.py` | Moon (`scenario.moon`): ecliptic position from elongation, Krisciunas & Schaefer (1991) V(phase), reddened solar spectrum, Lommel-Seeliger disk with terminator |
| `animate.py` | hour-angle time-lapse over compose (see status) |
| `validate_*.py` | per-module validation scripts; outputs in renders/nightsky/validation/<module>/ |

Frames: horizon x=N, y=W, z=Up; celestial x=equinox, z=pole; galactocentric pc with x from
Sgr A* toward the Sun (LEFT-handed triad; `frames.galactocentric_to_celestial` handles it).

## Configs (configs/)

gc_arches (flagship, clear air, mountains), gc_arches_forest, gc_noatm (vacuum, no foreground),
gc_hazy (aerosol tau 0.4, meadow), gc_mars (Mars atmosphere, bare rock), gc_legacy (the earlier
toy setup: Arches + ZAMS box + companion), earth_darksky and earth_darksky_camera (Earth
comparison from a dark site, BSC V<6.5 + galaxy model Milky Way), earth_moonlit (earth_darksky
plus a waxing gibbous Moon), gc_darkcloud (observer inside a gauss cloud, A_V 4 from its centre),
gc_hii_region (observer inside a 2-pc HII region, n_e 300, EM 1.8e5 pc cm^-6), and the
time-lapse configs gc_arches_timelapse / earth_darksky_timelapse.
Any config can be varied on the command line, e.g. a full Moon over the GC:
`--set name=gc_arches_moon --set scenario.moon.enabled=true` (seed the new output dir's cache
by copying `starfield_*`/`galaxy_*` from renders/nightsky/gc_arches/cache to skip ~2 min).

## Key numbers (production renders, 2026-10-25)

- gc_arches (before the end-of-session changes; current numbers under Status at handoff):
  zenith 17.57 V/arcsec^2, ground 0.32 lx, eye La 0.04 cd/m^2 (mesopic, photopic
  fraction 0.3), 615k point sources above the horizon. Render 477 s on 24 CPUs (foreground
  ~190 s per view at 1920x1080, 64 spp x 20 bins).
- gc_noatm: TOA sky log-average 0.0024-0.0027 cd/m^2.
- earth_darksky: zenith 21.49 V/arcsec^2, ground 8.3e-4 lx (reference ranges 21.3-22.0 and
  5e-4 to 3e-3 lx).
- Moon: earth_moonlit (elongation 120, V -11.1, 30 deg up): zenith 20.40 V/arcsec^2, ground
  0.034 lx (moonless 21.49 and 8.3e-4 lx). gc_arches_moon (full Moon 22 deg up in the ESE next to
  the toward_nsc target): zenith 17.30 (moonless 17.57), ground 0.42 lx (moonless 0.32).
  The linear disk in the npz has the expected Lommel-Seeliger shading with the lit limb toward
  the Sun (peak ~1360 cd/m^2 after extinction and glare). The global eye adaptation clips the
  disk and its glare halo to white, so compose paints a "foveal inset" (moon.inset,
  camera.moon_inset, default true) with the disk re-toned at the Moon's own adaptation; the
  phase reads as a notch, but the disk is no brighter than the clipped halo on an 8-bit display.
  A local tone-mapping operator (Reinhard 2002 dodging, ~1 deg adaptation kernel) would fix
  this for the Moon and the bright-star halos together.
- Clouds (previews at scale 0.25): gc_darkcloud (A_V 4 to the cloud edge) zenith 20.37
  V/arcsec^2, ground 0.012 lx, 28.7k points above the horizon (615k without the cloud), fully
  scotopic; gc_hii_region (EM 1.8e5 pc cm^-6) zenith 17.10, ground 0.34 lx, faint purple cast.
- Dust budget: A_V(Sun -> Sgr A*) = 30.0 total (13.3 from gas fields + 16.7 smooth disk) when
  use_gas is on; the GC observer sees A_V ~ 310 toward Sgr A* through the CND model.

## Fixes made during integration (this session)

- frames.galactocentric_to_celestial mirrored the sky (E-W); fixed (det R = -1). Verified for
  Earth against astropy: Sgr A* RA 266.43 Dec -29.02, Vega 279.24/38.78, Betelgeuse 88.78/7.40.
  galaxy.galactocentric_to_celestial_rh is now an alias.
- Dust double counting with use_gas: compose.galaxy_model drops the smooth CMZ term and lowers
  the disk calibration by the gas A_V (`galaxy.gas_dust_budget`, default true).
- compose uses gas.load_cached; scene_stats V mag uses atmosphere.v_mag_arcsec2 (V filter).
- camera.develop: per-pixel local mesopic weight (`camera.local_mesopic`, default true).
- gc_mars: no trees, needles/bark albedo -> rock.
- New: clouds.py (extra_fields), moon.py (scenario.moon), tests test_nightsky_clouds.py (5) and
  test_nightsky_moon.py (3, now 4). Adding "extra_fields" to GALAXY_MODEL_KEYS changed every starfield
  cache key, so all configs recompute the starfield once (~2 min) after this session.

## End-of-session changes (14:00-15:30)

- starfield.arches default zams false: the Arches is evolved to 2.5 Myr with stellar_pops.evolve.
  Integrated M_V -11.78 (ZAMS: -10.85); top members M_V -8.3, Teff 34 kK, log L 6.5 (WNh-like,
  still labelled MS). Ground illuminance in gc_arches roughly doubles.
- Naked-eye visibility (camera.deposit_points_eye, compose eye mode, points.visibility default
  true; camera mode unchanged): Hecht/Schaefer (1990) threshold E_th(L_b) from the sky luminance
  at each star's pixel, detection probability p = erf in magnitudes (sigma 0.3 mag, p = 0.5 at
  m_lim, points.visibility_factor scales E_th). p*irr is deposited as a point; (1-p)*irr is
  blurred by points.unresolved_blur_deg (0.3 deg) and added as unresolved glow, so frame flux is
  conserved (1e-14 relative). GC sky: m_lim V ~4.1; ~6k of 88k stars in the toward_nsc frame
  stay visible. Sidecars gain a `visibility` block; scene stats gain naked_eye_limit_V_zenith.
  tests/test_nightsky_visibility.py.
- Season snowline (albedo.SNOWLINE_M: summer 2800 m, autumn 1800 m, winter 0 m; explicit
  foreground.snowline_m overrides; foreground.resolve_scene; geometry cache format 3; snow masked
  under forest). compose.foreground_layers now hashes resolve_scene(fcfg), so every foreground
  cache key changed once. tests/test_nightsky_snowline.py. Known approximation: thresholds
  compare against tangent-plane z, which includes the Earth-curvature drop (282 m at 60 km).
- Glare kernel taper (see Known gaps).
- Moon inset (moon.inset): the night side kept 0.8 x the glare veil and the lit side was capped
  near 0.85-1.0, so inside a saturated white halo the full Moon showed as a grey dot (earth_moonlit
  at 14:40). Now no disk pixel ends darker than the veil already there (camera.moon_night_veil,
  default 1.0); with a saturated halo the disk stays white and the phase is hidden. Showing the
  phase there needs a local tone operator (exposure varying with distance from the Moon, with
  its slope kept below the glare falloff to avoid a dark ring). Test: test_inset_never_darker_than_veil.
- Starfield cache key (compose.stage_starfield): it hashed the raw cfg["starfield"], so the
  evolved-Arches default in starfield.DEFAULTS did not invalidate old caches; gc_arches_moon at
  14:55 reused a 13:18 ZAMS starfield (zenith 17.30, ground 0.42 lx, darker than the moonless
  flagship). The key now hashes starfield.resolve_config(cfg) (cfg merged with DEFAULTS), so
  every config recomputes its starfield once (~30 s, plus the downstream galaxy, sky and
  foreground stages). The galaxy key (stage_galaxy) still hashes only the raw galaxy section:
  after editing galaxy.py presets or gas.py, run with --force galaxy (or clear the cache).
  Renders checked against this: gc_arches, gc_arches_forest and the v2 time-lapse built fresh
  starfields; the Earth configs do not use the Arches block.

## Status at handoff (2026-10-25, 15:30 EDT)

Done (all under /orange/adamginsburg/gc/visualization/renders/nightsky/):
- Flagship stills gc_arches/{toward_nsc,allsky}.png re-rendered at 14:33-14:37 with every
  end-of-session fix (evolved Arches, naked-eye visibility, season snowline, glare taper).
  The previous versions are kept as gc_arches/prev_zams_snow_{toward_nsc,allsky}.png.
  New numbers: zenith 16.88 V/arcsec^2 (was 17.57), ground 0.70 lx (was 0.32), eye La
  0.073-0.078 cd/m^2, photopic fraction 0.39; naked-eye limit V 4.1; toward_nsc 5.9k of 88k
  stars visible, allsky 66k of 616k. 224 s on 24 CPUs (foreground 70-76 s per view).
- Scenario variants re-rendered with the fixes (14:37-15:13):
  - earth_darksky: zenith 21.49 V/arcsec^2, ground 8.3e-4 lx; milky_way view 401 of 810 BSC
    stars visible, allsky 2617 of 4017.
  - earth_moonlit: zenith 20.40, ground 0.034 lx; moon view 76 of 769 visible, allsky 986 of
    4018; Moon inset fix applied (white disk in a white halo, no grey dot).
  - gc_arches_forest: same sky as gc_arches; toward_nsc 6266 of 94828 visible (eye La 0.0072,
    the dark canopy fills most of the frame and pulls the adaptation down), allsky 66329 of
    615835; 721 s on 24 CPUs (forest foreground 426 s for toward_nsc while the time-lapse ran).
  - gc_arches_moon (gc_arches.yaml --set name=gc_arches_moon --set scenario.moon.enabled=true),
    15:00 with the fixed starfield key: zenith 16.73 V/arcsec^2 (moonless 16.88), ground 0.80 lx
    (moonless 0.70; the Moon adds ~0.1 lx), eye La 0.09-0.11, toward_nsc 5326 of 88019 visible,
    allsky 64825 of 615836; 215 s on 24 CPUs. The Moon sits low in the ESE and also lands in
    the toward_nsc frame as the warm glowing disk (absent from gc_arches/toward_nsc). Its orange
    tint comes from the low-altitude path plus the inset tint; it may read as too saturated.
  - gc_hazy (15:05): zenith 16.04 V/arcsec^2, ground 0.60 lx; toward_nsc 2031 of 88018 visible,
    allsky 34817 of 615835; 200 s.
  - gc_noatm (15:08): TOA sky, zenith 19.36 V/arcsec^2, ground 0.79 lx; toward_nsc 23287 of
    87879 visible, allsky 184167 of 615315, panorama 181244; 76 s.
  - gc_mars (15:10): zenith 15.75 V/arcsec^2 (dusty Mars sky scatters the GC light), ground
    0.65 lx; toward_nsc 1562 of 88059 visible, allsky 29632 of 616026; 184 s.
  Contact sheets regenerated at 15:14 (gc_darkcloud and gc_hii_region panels are still pre-fix).
  Still from before the fixes (re-render: `python3 -m nightsky render ../configs/<name>.yaml`):
  gc_darkcloud, gc_hii_region, earth_darksky_timelapse.
- Contact sheets: contact_sheet_views.png (perspective views), contact_sheet_allsky.png.
  Script: renders/nightsky/validation/contact.py <renders/nightsky dir>.
- Time-lapse v1: gc_arches_timelapse/{toward_nsc,allsky}_{smoothed,per-frame}.mp4 (48 frames,
  00:00-23:30 local solar time every 30 min, 8 fps; 2449 s render on 40 CPUs, 47-50 s/frame).
  Frame strips: gc_arches_timelapse/strip_{toward_nsc,allsky}_smoothed.png. v1 predates the
  fixes and shows the square glare patch in daytime allsky frames.
- Time-lapse v2 (all fixes): gc_arches_timelapse_v2/{toward_nsc,allsky}_{smoothed,per-frame}.mp4,
  24 frames at 1-h steps (00:00-23:00 local solar time), 6 fps; render 1865 s on 40 CPUs
  (52-110 s/frame, slower while other renders shared the node). The 30-min run (48 frames) was
  stopped at 14:38 after 2 frames; its h000.500 frame sits in frames/ and is not in the 1-h movies.
  Frame strips: gc_arches_timelapse_v2/strip_{allsky,toward_nsc}_smoothed.png. The daytime
  allsky frames no longer show the square glare patch. At 1-h steps twilight is a single frame
  (05:00 sun -11.4, 06:00 sun 0). For smoother twilight rerun the same command with
  --hours 0:24:0.5 --fps 8: resume is per frame (frames/h<hours>), so the 24 hourly frames and
  h000.500 are reused and only the 23 missing half-hours render (~25 min on 40 CPUs, plus one
  starfield rebuild because of the cache-key fix).
- Tests: `python3 -m pytest -q ../tests/` -> 59 passed, 1 skipped (smoke, clouds, moon, animate,
  visibility, snowline).

Not done / next steps, in suggested order:
1. Decide the NSC question (see review table): the flagship toward_nsc view shows no NSC because
   the gas model puts A_V 310 on the line of sight. Add the per-structure A_V diagnostic to
   scene stats, then either retitle the view or move arched_ridge_6 / cloud_50kms and rebuild
   the gas cache. The Quintuplet is hidden the same way.
2. Near foreground for toward_nsc (hero trees at 15-25 m, lower horizon) and a finer far-ring
   mesh for the ranges.
3. Re-render gc_darkcloud, gc_hii_region and earth_darksky_timelapse
   with the new defaults (about 4 min each on 24 CPUs; gc_darkcloud and gc_hii_region also
   rebuild their diffuse maps). Then rerun validation/contact.py.
4. Star colour vs resolution (local mesopic weight at a fixed angular scale), zodiacal light,
   refraction, CIE 191 mesopic normalisation, acuity loss, clean PNGs without labels.
5. Orion Nebula scenario (recipe under Known gaps).
6. Commit: nothing is in git yet. .gitignore covers only third_party/, renders/**/*.npy,
   *.exr and bytecode; add renders/**/cache/, renders/**/*.npz, renders/**/frames/ and data/
   (or the large parts of it) before the first commit.

## Realism review (end of session)

A review workflow (4 critics: astrophysics, vision/eye model, atmosphere/photometry,
foreground; every issue checked by an adversarial verifier) raised 29 issues: 23 confirmed,
6 refuted. Full JSON (evidence, physics, verifier reasoning, corrected fixes):
`renders/nightsky/validation/realism_review.json`. Severity below is the verifier's.

| Sev. | Issue (lens) | Status |
|---|---|---|
| high | Quintuplet and Pistol star hidden by the gas model (astro). The verifier found the A_V figure and the blamed structure wrong but the hiding real | open: per-structure A_V diagnostic, then move the blocking cloud |
| medium | NSC hidden: A_V 310 to Sgr A* (astro). CND seen 3.3 deg from edge-on (122 mag noise-free, 310 with lognormal noise); arched_ridge_6 (5-9 pc from the observer) and cloud_50kms block the +-3 to +-10 pc offsets; diffuse NSC loses 3.5-5 mag over +-20 deg | open, model choice. observer_offset_dir cannot change the CND angle by more than ~7 deg; levers are arched_ridge_6 depth/density and cloud_50kms depth (needs a gas-cache rebuild). Per-structure A_V script: renders/nightsky/validation/per_struct.py |
| medium | Arches drawn at ZAMS (astro) | fixed: starfield.arches.zams default false (2.5 Myr, M_V,tot -11.78 vs -10.85) |
| medium | Stars below the naked-eye threshold drawn at visible contrast (astro + perception) | fixed: points.visibility (see below) |
| medium | Star colour depends on render resolution (perception) | open, documented in INTERFACES.md: compute the local mesopic weight from Y smoothed to a fixed angular scale |
| medium | toward_nsc has no near foreground; nearest ground ~100 m (foreground) | open: horizon_frac 0.3 or a lower pitch, hero trees at 15-25 m |
| medium | Distant ranges look soft and rounded (foreground) | open: finer far-ring mesh (n_theta 3-4k for r > 2 km), ridged noise |
| low | 1000 au companion at 2.3 deg altitude behind the mountains (astro) | open: choose companion phase so it sits 25-45 deg up |
| low | No per-component A_V/visibility diagnostics in sidecars (astro) | open |
| low | Earth Milky Way 0.5-0.8 mag faint (atmo; overstated) | open: calibrate against Leinert+1998 Table 35 |
| low | Atmosphere ground albedo fixed at 0.11 (atmo) | open: area-weighted foreground albedo into the atmosphere LUT |
| low | Mars foreground uses Earth albedos (atmo) | open: 'mars' albedo set |
| low | npz 'xyz' is post-glare (atmo) | open: save pre-glare 'xyz' and 'xyz_retina' |
| low | No zodiacal light (atmo) | open: Leinert Table 17 map in build_toa |
| low | No refraction (atmo) | open: Bennett formula in point_irradiance |
| low | No ciliary corona / lenticular halo on very bright stars (perception) | open: sparse flare kernel |
| low | No acuity loss at low La (perception) | open: blur background layer by 0.187/R(La) deg |
| low | Mesopic mix lacks the CIE 191 normalisation (perception) | open |
| low | Burned-in labels in PNGs (perception) | open: write clean `<view>.png` plus `<view>_label.png` |
| low | Auto-adaptation compresses GC/Earth contrast (perception) | open: comparison pair at fixed camera.adaptation_cd_m2 |
| low | No scotopic noise (perception) | open |
| low | Conifers read as black cut-outs (foreground) | open: vertex normals on tree PLYs |
| low | Summer albedo set with 1300 m snowline (foreground; refuted as stated, snow covers only crests) | fixed anyway: season snowline (albedo.SNOWLINE_M) |

Refuted: ionized-gas magenta hue, zenith diagnostics missing resolved stars, missing shadow
test (as framed), Arches as 32 hard lights giving discrete shadow edges, foreground denoiser
edge wrap.

## Known gaps / next steps

- Earth Milky Way is a smooth band: add a 3D dust map (Edenhofer+2024 or Green+2019) as an
  extinction VoxelField for patchy structure (Great Rift).
- NSC density slope (Dehnen gamma 1.8) may be steep; NSC resolved giants and WR line spectra
  are not modelled.
- Deep twilight (Sun -12 to -18) runs low against Mitsuba; no refraction.
- Foreground: no grass blades, low-poly flat-shaded trees (the faceting reads as CG in close-ups);
  Mitsuba rgb-packed bins would speed it up ~3x. gc_arches_forest toward_nsc: the trees stand
  close to the camera and fill ~80% of the frame, hiding the NSC; widen the clearing or raise
  pitch_deg. That view took 1620 s of foreground on 16 CPUs (3000 trees, 64 spp x 20 bins).
- Moon: no earthshine or local-starlight on the night side, no libration/albedo map (uniform disk
  with Lommel-Seeliger shading), no lunar parallax.
- Orion Nebula scenario: needs an Earth-position observer with a nearby young cluster. Recipe:
  starfield mode gc is tied to the GC geometry; add a generic `clusters: [{l, b, d_pc, mass, ...}]`
  starfield component (the Arches code path with a position argument) and an emission field via
  galaxy.extra_fields (n_e ~ 1e3-1e4 near the Trapezium) plus an extinction shell (the Veil).
- Interface notes: foreground returns premultiplied radiance + an 'info' key; INTERFACES.md
  not yet updated for that. Earth mode couples starfield.sun_ra_deg with
  scenario.sun.ecliptic_longitude_deg; nothing enforces it.
- animate.py (built and verified this session; 14 tests in tests/test_nightsky_animate.py):
  local solar time -> H = 15 (t - 12) + RA_sun; exposure modes smoothed (asymmetric eye
  adaptation, tau 1 min brightening / 10 min darkening), per-frame, fixed; resume by
  render_hash; --no-render re-develops from npz. Gaps: animate swaps compose.build_toa,
  atmosphere.Atmosphere and camera.to_xyzs in-process (add a compose session/state argument and
  a public camera.tone_xyzs to remove that); the foreground AOV pass and Mitsuba scene load are
  H-independent but recomputed per frame (split the scenario hash into H-dependent and
  H-independent parts); no automatic twilight densification (use --hours lists); each view
  adapts independently. Quote hours in YAML/--set ("0:24:0.25"): unquoted 6:30 is base-60 (390).
- Square bright patch around the Sun in daytime frames: fixed at the end of the session. The
  CIE glare kernel had square support with a hard edge at glare_max_deg (60 deg); against the Sun
  the truncated wing is a few % of the blue sky. camera.glare_kernel now uses circular support
  with a cos^2 taper over the outer 30%. Renders made before 14:30 on 2026-10-25 (including
  gc_arches_timelapse v1) still show it; npz 'xyz' is post-glare, so `animate --no-render`
  cannot remove it.
- The old Mitsuba volumetric pipeline (src/render_day.py etc., README) still has the
  unexplained 13% deficit at 48 shells.
