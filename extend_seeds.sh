#!/bin/bash
# extend_seeds.sh — 补跑 seeds 5-9 到已有结果文件（续跑会 skip 0-4）。
cd /home/ubuntu/paper1_rerun || exit 1
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
PY=/home/ubuntu/widarvenv/bin/python
SEEDS="5 6 7 8 9"
AXES="loc ori user"

# GPU0: ERM + EPCon (续跑 teachers_erm_epcon.json)
CUDA_VISIBLE_DEVICES=0 $PY -u run_teachers.py --methods erm epcon --axes $AXES \
    --seeds $SEEDS --out results/teachers_erm_epcon.json > logs/ext_erm_epcon.log 2>&1 &

# GPU1: SupCon (teachers_supcon.json) + GroupDRO (teachers_gpu1.json)
CUDA_VISIBLE_DEVICES=1 $PY -u run_teachers.py --methods supcon --axes $AXES \
    --seeds $SEEDS --out results/teachers_supcon.json > logs/ext_supcon.log 2>&1 &
G1=$!

# GroupDRO 在 gpu1 上串行排队（等 SupCon 完成再跑，避免同卡争抢）
CUDA_VISIBLE_DEVICES=1 bash -c "wait $G1; $PY -u run_teachers.py --methods groupdro --axes $AXES --seeds $SEEDS --out results/teachers_gpu1.json > logs/ext_groupdro.log 2>&1" &

wait
echo "extend_seeds DONE at $(date)" > logs/extend_done.flag