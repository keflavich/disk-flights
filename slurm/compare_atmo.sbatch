#!/bin/bash
#SBATCH --job-name=gc-compare-atmo
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32gb
#SBATCH --time=04:00:00
#SBATCH --output=/orange/adamginsburg/gc/visualization/slurm/logs/%x_%A.out

# Usage: sbatch --export=ALL,ARGS="--face 160 --spp-atmo 8192" slurm/compare_atmo.sbatch
cd /orange/adamginsburg/gc/visualization/src
python3 render_compare_atmo.py --out "${OUT:-../renders/compare_atmo}" $ARGS
