#!/bin/sh
cd "$(dirname "$0")"
mkdir -p logs
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
printf "%s\n" Mbp Plp1 Mobp Mog Cnp Sox10 Snap25 Malat1 | xargs -P 3 -L 1 sh -c 'python3 -u baseline_rerun.py run --gene $0 > logs/baselines_$0.log 2>&1'
python3 baseline_rerun.py summary > logs/baselines_summary.log 2>&1
