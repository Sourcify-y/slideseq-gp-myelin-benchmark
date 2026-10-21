#!/bin/sh
cd "$(dirname "$0")"
mkdir -p logs
GENES="Mbp Plp1 Mobp Mog Cnp Sox10 Snap25 Malat1"

export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
echo $GENES | tr ' ' '\n' | xargs -P ${EXACT_WORKERS:-4} -I{} sh -c 'python3 -u permutation_tests.py exact --gene {} --n-perm 100 > logs/exact_{}.log 2>&1'

export VECLIB_MAXIMUM_THREADS=2 OMP_NUM_THREADS=2
echo $GENES | tr ' ' '\n' | xargs -P 2 -I{} sh -c 'python3 -u permutation_tests.py svgp --gene {} --n-perm 50 > logs/svgp_{}.log 2>&1'

python3 permutation_tests.py summary > logs/perm_summary.log 2>&1
