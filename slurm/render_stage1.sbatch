#!/bin/bash
#SBATCH --job-name=gc-sky-stage1
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x_%j.out

module load cuda/12.8.1
cd /orange/adamginsburg/gc/visualization/src

python3 render_stage1.py \
    --variant cuda_ad_mono \
    --elevation "${ELEVATION:-20}" \
    --width 1024 --height 576 --spp 256 \
    --out "../renders/stage1_sun_only/el${ELEVATION:-20}.png"
