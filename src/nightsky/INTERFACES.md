# nightsky module interfaces (build contract, 2026-10-25)

Run everything from `/orange/adamginsburg/gc/visualization/src` (`python3 -m nightsky.<mod>`).
Shared, frozen contracts: `spectral.py` (SpectralGrid, colour), `frames.py` (horizon /
celestial / galactocentric frames), `sky.py` (PointSources, TOASky), `fields.py` (VoxelField).
Do not change these; if something is missing, write a helper in your own module and report it.

Units: spectral radiance W m^-2 sr^-1 nm^-1, irradiance W m^-2 nm^-1, bin AVERAGES on a
`SpectralGrid` passed in by the caller (default `spectral.default_grid()` = 20 bins,
380-780 nm). Never hard-code the bin count. Distances: pc for galaxy models, m for the
atmosphere/terrain. Horizon frame x=N, y=W, z=Up. All images are (H, W, nbins) float32
linear radiance; row 0 = top.

Views: `views.View(kind, width, height, **kw)` (existing module): kinds "allsky"
(equidistant fisheye, zenith centre, north up, east left, max_zenith_deg=95) and
"perspective" (azimuth_deg from N through E, pitch_deg, fov_deg = horizontal FOV).
Attributes `dirs (H,W,3)` horizon-frame unit vectors, `valid (H,W)`, `omega (H,W)` pixel
solid angle, `project(dirs) -> (row, col)`. camera.py may add a "panorama" kind
(cylindrical: azimuth across, elevation up) as a View subclass with the same attributes.

## Scenario (GC preset)
Observer (solar system) 3 pc from the Arches cluster centre. Arches at (l, b) =
(0.1217, 0.0188) deg, at the Sgr A* distance R0 = 8178 pc. Sgr A* at (359.9442, -0.0462)
deg. The planet's north celestial pole points at the Arches; observer latitude +40 deg;
equinox; time = hour angle H (deg) of the equinox (H=0 local noon, H=180 midnight).
Observer offset from the Arches: `observer_offset_dir` (galactocentric unit vector,
default (1,0,0) = on the Sun-facing side). Rotation into the celestial frame:
`frames.galactocentric_to_celestial(observer, arches, reference=sgr_a_star, roll_deg)`.
Existing flat-module models: `cluster.Cluster` (Arches: 1.5e4 Msun, 1-120 Msun,
alpha 1.85, Plummer a=0.25 pc, rmax 1.5 pc, seed 1), `cluster.LocalStars`
(1000 Msun, 0.3-4 Msun ZAMS in a 1 pc box, >50 au), `companion.py` (Sun-like star at
1000 au, inclination 25 deg).

## atmosphere.py  (owner: atmosphere agent)
```
Atmosphere(preset: str | dict, grid)          # presets: "none", "earth_clear",
                                              # "earth_hazy", "earth_observatory", "mars"
  .transmittance(view_dirs_h)      -> (N, nbins)   ground-to-space along each direction
                                                   (0 below the horizon)
  .sky(toa, R, view_dirs_h)        -> dict(T=(N,nbins), inscatter=(N,nbins), airglow=(N,nbins))
        observed sky radiance = T * toa.diffuse(R.T @ d) + inscatter + airglow
        inscatter includes single + multiple scattering of ALL toa light (points and
        diffuse map), and the ground bounce; R = frames.celestial_to_horizon(H, lat).
  .point_irradiance(toa, R)        -> PointSources in the HORIZON frame with irradiance
                                      multiplied by transmittance (below-horizon -> 0)
  .ground_irradiance(toa, R)       -> (nbins,) horizontal irradiance at the ground
  .aerial_perspective(view_dirs_h, dist_m, toa, R) -> (T (N,nbins), inscatter (N,nbins))
        for terrain at distance dist_m along view_dirs_h from an observer at the ground
  .preset (dict), .name
```

## galaxy.py  (owner: galaxy agent)
```
GalaxyModel(grid, preset="milky_way", fields=(), **overrides)
  .stellar_density(xyz_pc)          -> (N,) Msun pc^-3 (all stellar components)
  .emissivity(xyz_pc)               -> (N, nbins) W m^-2 sr^-1 nm^-1 per pc (stars + fields)
  .opacity(xyz_pc)                  -> (N, nbins) tau per pc (dust + extinction fields)
  .render_map(observer_pc, nside, R_gc_to_cel, r_res_pc=0, inner_factor=1.0, ...) -> TOASky
        HEALPix RING map in the CELESTIAL frame (dirs_cel = R @ dirs_gc), points empty.
        Stellar emissivity within r_res_pc of the observer is multiplied by inner_factor
        (the starfield resolves the bright stars there).
  .diffuse_fn(observer_pc, R_gc_to_cel, ...) -> callable(dirs_cel (N,3)) -> (N, nbins)
  .optical_depth(observer_pc, targets_pc (N,3)) -> (N, nbins) tau to each point source
```

## gas.py  (owner: gas agent)
```
gc_ionized_gas(grid, seed=0, resolution="default") -> list[VoxelField]   (kind "emission")
gc_molecular_clouds(grid, seed=0, resolution="default") -> list[VoxelField] (kind "extinction")
load_cached(grid, seed=0, resolution="default") -> both lists, cached under renders/nightsky/cache/gas
```

## clouds.py  (added after the build; config-driven fields around the observer)
```
extra_fields(specs, grid, observer_pc) -> list[VoxelField]
    specs = cfg["galaxy"]["extra_fields"], each
      {kind: extinction, profile: gauss|uniform|shell, radius_pc, av_center, rv, offset_pc, n_vox, name}
      {kind: emission,   profile: ..., radius_pc, n_e, t_e, template (gas.emission_spectrum), ...}
    av_center = A_V from the cloud centre to infinity. compose.galaxy_model appends these to the
    GalaxyModel fields (with or without use_gas), so they enter the diffuse map and the point
    source optical depths. Configs: gc_darkcloud.yaml, gc_hii_region.yaml.
```

## moon.py  (added after the build)
```
cfg["scenario"]["moon"] = {enabled, elongation_deg (180 full, 90 first quarter), ecliptic_latitude_deg, dmag}
moon_sources(cfg, grid) -> PointSources named "Moon" (celestial frame), V from Krisciunas &
    Schaefer (1991), solar spectrum x (lambda/550)^1.1. compose.build_toa appends it with the Sun,
    so atmosphere scattering and foreground lighting include it; the toa hash includes the moon
    block only when enabled.
deposit(img, view, d_h, irr, sun_dir_h, pcfg, grid) -> Lommel-Seeliger shaded disk with terminator
    (compose's point deposit calls it for the source named "Moon").
inset(rgb8, view, d, sun_dir, irr, grid, alpha, night_veil=1.0) -> 8-bit image with the disk re-toned
    for a Moon-adapted fovea (camera.moon_inset, default true). No disk pixel ends darker than
    night_veil (camera.moon_night_veil) x the glare veil already there, so the phase shows only where
    the halo is below white; with a saturated halo the disk stays white. The earlier fixed 0.8 put a
    grey dot inside a white halo. Showing the phase in that case needs a local tone operator.
```

## starfield.py  (owner: starfield agent)
```
build_starfield(cfg: dict, grid, galaxy=None) -> (TOASky with points only, info dict)
    cfg keys: mode ("gc" | "earth"), arches, quintuplet, nsc_young, local_old_population,
    local_zams_box (the earlier 1000 Msun box), companion, bsc (Earth V<6.5 catalogue),
    r_res_pc, observer_offset_dir, roll_deg, seed ...  info has "R_gc_to_cel",
    "observer_pc", "inner_factor" (fraction of the stellar V light within r_res_pc that
    stays diffuse, passed to galaxy.render_map), counts and magnitudes.
resolve_config(cfg) -> cfg merged with starfield.DEFAULTS; compose's starfield cache key hashes
    this, so editing a default invalidates cached starfields.
```

## foreground.py / terrain.py  (owner: foreground agent)
```
render_foreground(scene_cfg: dict, view: views.View, sky_fn, lights: PointSources (horizon,
                  ground irradiance), grid, spp=64, cpus=...) -> dict(
     radiance=(H,W,nbins) surfaces only, alpha=(H,W) coverage in [0,1],
     depth=(H,W) metres (inf where sky), normal=(H,W,3), info=dict)
    radiance is PREMULTIPLIED by coverage (compose.composite: L = a (L_fg T + S) + (1 - a) L_sky,
    T and S the aerial-perspective transmittance and in-scatter);
    cfg foreground_render.premultiplied selects the convention.
    sky_fn(dirs_h (N,3)) -> (N, nbins) downwelling sky radiance at the ground (lights excluded)
    scene_cfg: type ("mountains" | "meadow" | "forest" | "lakeshore" ...), seed, extent,
    tree density, albedo set, snowline_m ...
    snowline_m: None (default) = the albedo set's season value albedo.SNOWLINE_M (summer 2800,
    autumn 1800, winter 0 m; default ranges peak at ~2.2 km); a number in the config overrides it.
resolve_scene(scene_cfg) -> dict   scene_cfg merged with terrain.DEFAULTS + foreground.DEFAULTS,
    snowline_m resolved (terrain.resolve). The PLY geometry cache keys on this resolved config
    (format 3). compose.foreground_layers still hashes the RAW cfg["foreground"], so a cached
    foreground_<view> stage from before the snowline change is reused until compose hashes
    resolve_scene(fcfg) instead (or the stage is re-run with --force foreground_<view>).
```

## camera.py / compose.py / config.py / cli.py  (owner: compose agent)
```
python3 -m nightsky render configs/<name>.yaml [--set key=value ...] [--stage ...]
compose: TOASky -> atmosphere.sky -> sky image; foreground with aerial perspective;
point sources via PSF; camera (exposure: eye adaptation or fixed EV; mesopic; glare PSF;
tone map) -> PNG + linear npz. Stage caching under renders/nightsky/<config>/cache/.
camera eye mode: global mesopic adaptation plus a per-pixel photopic weight (camera.local_mesopic,
default true) so bright point sources keep their colour. Glare: CIE 2002 PSF out to
camera.glare_max_deg (60) with circular support and a cos^2 taper over the outer 30% (a hard
square edge showed as a bright box around the Sun); energy beyond the taper goes to the core.
Known limitation: that weight uses the pixel luminance after glare, and a point source's pixel
luminance scales as E / omega_px, so star colour depends on the render resolution (finer pixels ->
more photopic); --scale 0.25 previews show stars greyer than the full-resolution frame.
Naked-eye detection (eye mode only, points.visibility default true; camera mode deposits every
source unchanged): camera.deposit_points_eye compares each non-disk source's photopic
illuminance E (lux, spectral.to_xyz Y) with camera.threshold_illuminance(L_b), the Hecht/Schaefer
(1990, PASP 102, 212) threshold on the sky luminance L_b at its pixel (L_sky before points and
foreground), times points.visibility_factor. A fraction p = 0.5 (1 + erf(2.5 log10(E / E_th) /
(sqrt 2 visibility_sigma_mag))) goes into the normal PSF deposit; (1 - p) is deposited with the
same stencil into a flux image, blurred by points.unresolved_blur_deg (>= unresolved_min_px;
one pixel scale per view) with flux kept inside the valid mask, and added as radiance, so the frame
flux is unchanged. vinfo["visibility"]: visibility_factor, n_points_in_frame, n_points_visible (sum p),
n_points_p50, unresolved_blur_px, unresolved_lux, naked_eye_limit_V_star_bg_median,
naked_eye_limit_V_sky_median (only the first three and the last when no source lands in the frame);
scene["naked_eye_limit_V_zenith"]. points.* is not part of any cache key.
Earth mode: starfield.sun_ra_deg sets the RA of celestial +x (the Sun's RA); keep it consistent
with scenario.sun.ecliptic_longitude_deg (0 puts the Sun on +x). Nothing enforces this.
python3 -m nightsky.animate <config.yaml>: hour-angle time-lapse (see animate.py docstring).
```
