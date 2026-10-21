#!/bin/sh
cd "$(dirname "$0")"
mkdir -p logs
ALL="Mbp Plp1 Mobp Mog Cnp Sox10 Snap25 Malat1"

export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
printf "%s\n" "Mbp 6000" "Plp1 6000" "Snap25 6000" "Mbp 4000" "Plp1 4000" "Snap25 4000" | xargs -P 3 -L 1 sh -c 'python3 -u matched_comparison.py exact --gene $0 --n-sub $1 > logs/matched_exact_$0_n$1.log 2>&1'

export VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
{
for g in $ALL; do echo "$g 2000"; echo "$g 0"; done
for g in Mbp Plp1 Snap25; do echo "$g 4000"; echo "$g 6000"; done
} | xargs -P 2 -L 1 sh -c 'python3 -u matched_comparison.py svgp --gene $0 --n-sub $1 > logs/matched_svgp_$0_n$1.log 2>&1'

python3 matched_comparison.py summary > logs/matched_summary.log 2>&1
