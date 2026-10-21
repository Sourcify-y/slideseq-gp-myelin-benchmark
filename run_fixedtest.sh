#!/bin/sh
cd "$(dirname "$0")"
mkdir -p logs
export VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
{
for g in Mbp Plp1 Snap25; do for n in 1600 3200 4800 12800 25600; do echo "$g $n"; done; done
} | xargs -P 2 -L 1 sh -c 'python3 -u fixedtest_density.py run --gene $0 --n-train $1 > logs/fixedtest_svgp_$0_t$1.log 2>&1'
python3 fixedtest_density.py summary > logs/fixedtest_summary.log 2>&1
