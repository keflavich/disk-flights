#!/bin/bash
#SBATCH --job-name=gc-toy-arches
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=/orange/adamginsburg/gc/visualization/slurm/logs/%x_%j.out

# CPU render using the llvm_ad_mono variant (no GPU needed).
# Override defaults via env, e.g.: sbatch --export=ALL,SPP=1024,WIDTH=640 slurm/render_stage2.sbatch
cd /orange/adamginsburg/gc/visualization/src

python3 render_stage2.py \
    --out "../renders/stage2_toy_arches/${TAG:-toy}" \
    --bins "${BINS:-12}" \
    --width "${WIDTH:-320}" --height "${HEIGHT:-180}" \
    --spp "${SPP:-512}" --shells "${SHELLS:-16}"
