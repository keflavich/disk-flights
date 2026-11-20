#!/bin/bash
# Submit the planet-disk flythrough frames as frame-range jobs (restartable: finished frames are
# skipped), then develop + star layer + position key once all of them are done.
#
#   slurm/diskplanet_movie.sh coorbit vis 6      # 6 frame-range jobs for the optical frames
#   slurm/diskplanet_movie.sh lap ir 6           # IR frames (needs the 3D Hyperion temperature)
#   slurm/diskplanet_movie.sh coorbit post       # develop, stars, key (after vis and ir frames)
#   VARIANT=taper slurm/diskplanet_movie.sh ...  # planet clump confined to 0.5 disk scale heights
#   VARIANT=spiral slurm/diskplanet_movie.sh ... # taper + global spiral arms (diskplanet/spiral.py)
#
# Movies: coorbit = camera trails the planet by 30 deg of orbital phase (same period, pattern
# co-rotates); lap = planet frozen at azimuth 150 deg, the camera laps the gap once (both on a
# 15 deg inclined orbit); incl45 / polar = 45 / 90 deg inclined circular orbits past the frozen
# planet; polarfly = polar with the camera's up kept away from the star (no roll flip over the
# pole) and a 15 s intro: hold at 1000 au over the pole, dive to 100 au, pan from the star to the
# orbit view (450 extra frames; FRAMES=0:450 renders only those).
set -e
MOVIE=$1; WHAT=$2; NJOBS=${3:-6}
ROOT=/orange/adamginsburg/gc/visualization
# VARIANT: "" = delta(R, phi) at every height; taper = the planet's clump has 0.5 x the disk's
# scale height (diskplanet/perturb.py), with its own Hyperion run and output directories, and a
# uniform orbit speed (the default schedule rushes through the midplane, i.e. through the gap)
VARIANT=${VARIANT:-}
SFX=; HYP=planet_q1e-3_n128; CLUMP=0.0; SPEED=surface
if [ "$VARIANT" = taper ]; then SFX=_taper; HYP=planet_q1e-3_n128_cx0.5; CLUMP=0.5; SPEED=uniform; fi
# spiral = taper + two global log-spiral arms in the outer disk (diskplanet/spiral.py: 15 deg pitch,
# envelope 160 +- 35 au, 0.25 dex cosine amplitude, Sigma(r) unchanged), rigid with the planet
SIGMA=sigma_q1e-3_a1e-3_h0.051
if [ "$VARIANT" = spiral ]; then SFX=_spiral; SIGMA=${SIGMA}_sp2; HYP=planet_q1e-3_n128_sp2_cx0.5; CLUMP=0.5; SPEED=uniform; fi
OUT=$ROOT/renders/diskplanet/$MOVIE$SFX
MCDIR=$ROOT/renders/diskplanet/mc
NFR=1800
FLY=(--set camera_up=radial --set intro_from_au=1000 --set intro_hold=30 --set intro_fly=300 --set intro_pan=120)
COMMON=(--set a_au=100 --set inc_deg=15 --set planet_sigma=$ROOT/renders/diskplanet/$SIGMA.npz
        --set planet_lsun=1e-4 --set mc_nphi=32 --set key_L_au=125 --set key_dex=5
        --set planet_clump_h=$CLUMP --set orbit_speed=$SPEED)
# 3D Hyperion temperature: only the IR frames need it (optical thermal emission comes from the
# inner rim, which uses the axisymmetric temperature in either case)
TEMP3D=(--set planet_rtin=$ROOT/renders/diskplanet/hyperion/$HYP.rtin
        --set planet_rtout=$ROOT/renders/diskplanet/hyperion/$HYP.rtout)
case $MOVIE in
  coorbit) COMMON+=(--set pattern_mode=coorbit --set pattern_deg=30 --set movie_name=diskplanet_coorbit$SFX) ;;
  lap)     COMMON+=(--set pattern_mode=fixed --set pattern_deg=150 --set movie_name=diskplanet_lap$SFX) ;;
  # steeper circular orbits past the frozen planet (azimuth 190 deg: just after the descending
  # node at 180 deg, so the camera passes ~17 au from it); the 90 deg orbit crosses over the pole
  incl45)  COMMON+=(--set inc_deg=45 --set pattern_mode=fixed --set pattern_deg=190 --set movie_name=diskplanet_incl45$SFX) ;;
  polar)   COMMON+=(--set inc_deg=90 --set pattern_mode=fixed --set pattern_deg=190 --set movie_name=diskplanet_polar$SFX) ;;
  polarfly) NFR=2250
           COMMON+=(--set inc_deg=90 --set pattern_mode=fixed --set pattern_deg=190 --set movie_name=diskplanet_polarfly$SFX "${FLY[@]}") ;;
  *) echo "movie must be coorbit, lap, incl45, polar or polarfly"; exit 1 ;;
esac
mkdir -p "$OUT"
# Monte Carlo fields are shared between the movies (computed once in $MCDIR)
for f in "$MCDIR"/mc_*.npz; do [ -e "$f" ] && ln -sf "$f" "$OUT/"; done
SB=$ROOT/slurm/diskplanet_step.sbatch
case $WHAT in
  vis|ir)
    CMD=orbit; EXTRA=()
    if [ "$WHAT" = ir ]; then CMD=orbit-ir; EXTRA=("${TEMP3D[@]}"); fi
    F0=${FRAMES%%:*}; F1=${FRAMES##*:}; F0=${F0:-0}; F1=${F1:-$NFR}
    step=$(( (F1 - F0 + NJOBS - 1) / NJOBS ))
    for ((a = F0; a < F1; a += step)); do
      b=$(( a + step < F1 ? a + step : F1 ))
      sbatch --parsable --job-name="dp-$MOVIE$SFX-$WHAT-$a" --cpus-per-task=16 --time=12:00:00 \
        $SB $CMD "$OUT" --n-frames $NFR --frames $a:$b "${COMMON[@]}" "${EXTRA[@]}"
    done ;;
  post)
    cd $ROOT/src
    export NUMBA_NUM_THREADS=${CPUS:-2}
    python3 -m diskorbit.render develop "$OUT" --n-frames $NFR --modes eye,auto "${COMMON[@]}"
    python3 -m diskorbit.render stars "$OUT" --n-frames $NFR --modes eye,auto "${COMMON[@]}"
    python3 -m diskorbit.render develop-ir "$OUT" --n-frames $NFR "${COMMON[@]}" "${TEMP3D[@]}"
    python3 -m diskorbit.render stars-ir "$OUT" --n-frames $NFR "${COMMON[@]}" "${TEMP3D[@]}"
    python3 -m diskorbit.inset --outdir "$OUT" --n-frames $NFR --nproc ${CPUS:-2} "${COMMON[@]}" ;;
  *) echo "what must be vis, ir or post"; exit 1 ;;
esac
