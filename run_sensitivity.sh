#!/bin/bash
cd /home/ubuntu/paper1_rerun || exit 1
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/widarvenv/bin/python -u sensitivity_clean.py > logs/sensitivity_clean.log 2>&1
echo "DONE rc=$?" > logs/sensitivity_done.flag