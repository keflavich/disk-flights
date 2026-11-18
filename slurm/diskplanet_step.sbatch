#!/bin/bash
# One diskorbit.render step for the planet disk (src/diskplanet), e.g. the Monte Carlo field
# or a frame range:
#
#   sbatch slurm/diskplanet_step.sbatch mc ../renders/diskplanet/movie_x --nphot 1e8 --set ...
#
#SBATCH --job-name=diskplanet
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-dev
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/orange/adamginsburg/gc/visualization/slurm/logs/diskplanet_%j.log

set -e
cd /orange/adamginsburg/gc/visualization/src
export NUMBA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
echo "host $(hostname) cpus $SLURM_CPUS_PER_TASK start $(date)"
python3 -m diskorbit.render "$@" --cpus "$SLURM_CPUS_PER_TASK"
echo "done $(date)"
