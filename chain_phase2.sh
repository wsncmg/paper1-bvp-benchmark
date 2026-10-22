#!/bin/bash
# chain_phase2.sh — 实验1(seeds展开)完成后，串跑实验2(敏感性) + 实验3(AirFi)。
cd /home/ubuntu/paper1_rerun || exit 1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY=/home/ubuntu/widarvenv/bin/python

echo "[phase2] waiting for seed extension..."
while [ ! -f logs/extend_done.flag ]; do sleep 60; done
echo "[phase2] seeds extension done at $(date)"

# 实验2：敏感性（单卡0，约25分钟）
CUDA_VISIBLE_DEVICES=0 $PY -u sensitivity_clean.py > logs/sensitivity_clean.log 2>&1 &
P2=$!

# 实验3：AirFi 全量（单卡1，约25分钟）
CUDA_VISIBLE_DEVICES=1 $PY -u run_airfi_adapted.py --out results/airfi.json \
    --seeds 0 1 2 3 4 > logs/airfi_full.log 2>&1 &
P3=$!

wait $P2 $P3
echo "[phase2] DONE at $(date)" > logs/phase2_done.flag