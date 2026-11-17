#!/bin/bash
#SBATCH --job-name=gc-day-video
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --time=00:30:00
#SBATCH --output=/orange/adamginsburg/gc/visualization/slurm/logs/%x_%j.out

module load ffmpeg
FR="${FRAMES:-/orange/adamginsburg/gc/visualization/renders/day/frames}"
OUT="${VIDEOS:-/orange/adamginsburg/gc/visualization/renders/day/videos}"
mkdir -p "$OUT"
for v in allsky north; do
    ffmpeg -y -framerate "${FPS:-24}" -i "$FR/${v}_%04d.png" -c:v libx264 -pix_fmt yuv420p -crf 17 "$OUT/day_${v}.mp4"
done
ls -la "$OUT"
