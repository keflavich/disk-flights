#!/bin/bash
# Run a command on the reserved hpg-dev worker allocation (a long "sleep" job), so heavy
# steps don't wait in the queue. Clears SLURM_* inherited from the login shell's own
# interactive allocation (otherwise srun refuses: node-count conflict).
#
#   WORKER=<jobid> CPUS=16 MEM=32G slurm/onworker.sh python3 nightsky/whatever.py ...
#
# Runs in the current directory. Default WORKER is the session's reservation.
WORKER="${WORKER:-47658970}"
CPUS="${CPUS:-16}"
MEM="${MEM:-32G}"
for v in $(env | grep -o '^SLURM_[A-Z_]*'); do unset "$v"; done
export DRJIT_LIBLLVM_PATH="${DRJIT_LIBLLVM_PATH:-/orange/adamginsburg/gc/visualization/third_party/libllvm/libLLVM-18.so}"
export NUMBA_NUM_THREADS="$CPUS"
exec srun --jobid="$WORKER" --overlap --ntasks=1 --nodes=1 --cpus-per-task="$CPUS" --mem="$MEM" "$@"
