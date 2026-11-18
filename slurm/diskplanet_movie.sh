#!/bin/bash
# Submit the planet-disk flythrough frames as frame-range jobs (restartable: finished frames are
# skipped), then develop + star layer + position key once all of them are done.
#
#   slurm/diskplanet_movie.sh coorbit vis 6      # 6 frame-range jobs for the optical frames
#   slurm/diskplanet_movie.sh lap ir 6           # IR frames (needs the 3D Hyperion temperature)
#   slurm/diskplanet_movie.sh coorbit post       # develop, stars, key (after vis and ir frames)
#
# Movies: coorbit = camera trails the planet by 30 deg of orbital phase (same period, pattern
# co-rotates); lap = planet frozen at azimuth 150 deg, the camera laps the gap once.
set -e
MOVIE=$1; WHAT=$2; NJOBS=${3:-6}
ROOT=/orange/adamginsburg/gc/visualization
OUT=$ROOT/renders/diskplanet/$MOVIE
MCDIR=$ROOT/renders/diskplanet/mc
NFR=1800
COMMON=(--set a_au=100 --set inc_deg=15 --set planet_sigma=$ROOT/renders/diskplanet/sigma_q1e-3_a1e-3_h0.051.npz
        --set planet_lsun=1e-4 --set mc_nphi=32 --set key_L_au=125 --set key_dex=5)
# 3D Hyperion temperature: only the IR frames need it (optical thermal emission comes from the
# inner rim, which uses the axisymmetric temperature in either case)
TEMP3D=(--set planet_rtin=$ROOT/renders/diskplanet/hyperion/planet_q1e-3_n128.rtin
        --set planet_rtout=$ROOT/renders/diskplanet/hyperion/planet_q1e-3_n128.rtout)
case $MOVIE in
  coorbit) COMMON+=(--set pattern_mode=coorbit --set pattern_deg=30 --set movie_name=diskplanet_coorbit) ;;
  lap)     COMMON+=(--set pattern_mode=fixed --set pattern_deg=150 --set movie_name=diskplanet_lap) ;;
  *) echo "movie must be coorbit or lap"; exit 1 ;;
esac
mkdir -p "$OUT"
# Monte Carlo fields are shared between the movies (computed once in $MCDIR)
for f in "$MCDIR"/mc_*.npz; do [ -e "$f" ] && ln -sf "$f" "$OUT/"; done
SB=$ROOT/slurm/diskplanet_step.sbatch
case $WHAT in
  vis|ir)
    CMD=orbit; EXTRA=()
    if [ "$WHAT" = ir ]; then CMD=orbit-ir; EXTRA=("${TEMP3D[@]}"); fi
    step=$(( (NFR + NJOBS - 1) / NJOBS ))
    for ((a = 0; a < NFR; a += step)); do
      b=$(( a + step < NFR ? a + step : NFR ))
      sbatch --parsable --job-name="dp-$MOVIE-$WHAT-$a" --cpus-per-task=16 --time=12:00:00 \
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
