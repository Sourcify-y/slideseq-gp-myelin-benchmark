#!/bin/sh
cd "$(dirname "$0")"
mkdir -p logs
printf "%s\n" Mbp Plp1 Mobp Mog Cnp Sox10 Snap25 Malat1 | xargs -P 3 -L 1 sh -c 'python3 -u uncertainty_scores.py run --gene $0 > logs/unc_$0.log 2>&1'
