#!/bin/bash
#SBATCH --job-name=gc-day
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --output=/orange/adamginsburg/gc/visualization/slurm/logs/%x_%A_%a.out

# Usage: sbatch --export=ALL,MODE=static,ARGS="--groups sky" slurm/day.sbatch
#        sbatch --array=0-7 --export=ALL,MODE=sun,FRAMES_PER_TASK=6,FIRST=26 slurm/day.sbatch
# CPU render with the llvm_ad_mono variant.
cd /orange/adamginsburg/gc/visualization/src
OUT="${OUT:-../renders/day/data}"

if [ "$MODE" = "sun" ]; then
    LO=$(( ${FIRST:-26} + ${SLURM_ARRAY_TASK_ID:-0} * ${FRAMES_PER_TASK:-6} ))
    HI=$(( LO + ${FRAMES_PER_TASK:-6} - 1 ))
    [ $HI -gt ${LAST:-72} ] && HI=${LAST:-72}
    python3 render_day.py sun --out "$OUT" --frames "$LO-$HI" $ARGS
else
    python3 render_day.py static --out "$OUT" $ARGS
fi
