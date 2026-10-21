#!/bin/sh
cd "$(dirname "$0")"
mkdir -p logs
{
for g in Mbp Plp1 Snap25; do for b in 150 10 20 40 80 300; do echo "$g $b"; done; done
for g in Mobp Mog Cnp Sox10 Malat1; do echo "$g 40"; done
} | xargs -P 3 -L 1 sh -c 'python3 -u knn_ksweep.py run --gene $0 --blocks $1 > logs/knnk_$0_b$1.log 2>&1'
