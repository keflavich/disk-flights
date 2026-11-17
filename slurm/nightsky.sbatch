#!/bin/bash
#SBATCH --job-name=nightsky
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=/orange/adamginsburg/gc/visualization/slurm/logs/%x_%j.out

# Render one nightsky config:  sbatch slurm/nightsky.sbatch configs/gc_arches.yaml [--set k=v ...]
# (config path relative to the repository root, or absolute; extra arguments go to `render`)
set -euo pipefail
ROOT=/orange/adamginsburg/gc/visualization
CFG="${1:?usage: sbatch nightsky.sbatch <config.yaml> [render args]}"
shift
case "$CFG" in /*) ;; *) CFG="$ROOT/$CFG" ;; esac
export DRJIT_LIBLLVM_PATH="${DRJIT_LIBLLVM_PATH:-$ROOT/third_party/libllvm/libLLVM-18.so}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
cd "$ROOT/src"
python3 -m nightsky info "$CFG"
python3 -m nightsky render "$CFG" --cpus "${SLURM_CPUS_PER_TASK:-32}" "$@"
