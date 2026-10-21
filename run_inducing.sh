#!/bin/sh
cd "$(dirname "$0")"
mkdir -p logs
export VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
{
for m in 800 200; do for g in Mbp Plp1 Snap25; do echo "$g $m"; done; done
} | xargs -P 2 -L 1 sh -c 'python3 -u inducing_sensitivity.py run --gene $0 --m $1 > logs/inducing_svgp_$0_m$1.log 2>&1'
python3 inducing_sensitivity.py summary > logs/inducing_summary.log 2>&1
