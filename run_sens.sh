#!/bin/sh
cd "$(dirname "$0")"
mkdir -p logs
printf "%s\n" Sox10 Mbp Plp1 Mobp Mog Cnp Snap25 Malat1 | xargs -P 2 -L 1 sh -c 'python3 -u sensitivity_maps.py run --gene $0 > logs/sens_$0.log 2>&1'
