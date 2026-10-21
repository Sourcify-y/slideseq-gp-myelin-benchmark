#!/bin/sh
cd "$(dirname "$0")"
mkdir -p logs
{
echo "Mbp 150"; echo "Mbp 40"; echo "Plp1 150"; echo "Plp1 40"; echo "Snap25 40"; echo "Snap25 150"
} | xargs -P 3 -L 1 sh -c 'python3 -u null_control.py run --gene $0 --blocks $1 > logs/null_$0_b$1.log 2>&1'
