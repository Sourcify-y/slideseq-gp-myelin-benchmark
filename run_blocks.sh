#!/bin/sh
cd "$(dirname "$0")"
mkdir -p logs
export VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
{
for b in 150 10 20 80 300; do echo "Mbp $b"; done
for g in Plp1 Snap25; do for b in 150 10 20 80 300; do echo "$g $b"; done; done
} | xargs -P 2 -L 1 sh -c 'python3 -u matched_comparison.py svgp --gene $0 --n-sub 0 --blocks $1 > logs/blocks_svgp_$0_b$1.log 2>&1'
python3 matched_comparison.py summary > logs/blocks_summary.log 2>&1
