#!/bin/bash
# 3D Hyperion temperature run for the planet-perturbed disk (src/diskplanet/hyperion3d.py builds
# the .rtin). Uses the Fortran binaries of the python312 env (its Python package does not import
# under numpy 2; the input is written with h5py).
#
#   sbatch --ntasks=64 --time=12:00:00 slurm/hyperion3d.sbatch model.rtin model.rtout
#
#SBATCH --job-name=hyperion3d
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-dev
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=1:00:00
#SBATCH --output=/orange/adamginsburg/gc/visualization/slurm/logs/hyperion3d_%j.log

set -e
export PATH=/orange/adamginsburg/miniconda3/envs/python312/bin:$PATH
# MPICH hydra starts ranks with srun, which refuses the inherited mixed memory variables
unset SLURM_MEM_PER_NODE SLURM_MEM_PER_GPU
echo "host $(hostname) ntasks $SLURM_NTASKS start $(date)"
mpirun -np "$SLURM_NTASKS" hyperion_sph_mpi -f "$1" "$2"
echo "done $(date)"
