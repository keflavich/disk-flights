#!/bin/bash
#SBATCH --job-name=gc-day-compose
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --output=/orange/adamginsburg/gc/visualization/slurm/logs/%x_%A_%a.out

# One array task per view (0 = allsky, 1 = north). Override data dir / extra args with --export.
VIEWS=(allsky north)
cd /orange/adamginsburg/gc/visualization/src
python3 compose_day.py --data "${DATA:-../renders/day/data}" --out "${FRAMES:-../renders/day/frames}" \
    --views "${VIEWS[$SLURM_ARRAY_TASK_ID]}" ${LABEL:+--label "$LABEL"} $ARGS
