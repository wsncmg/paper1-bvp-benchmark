# Widar3.0 BVP Benchmark — Reproducible Results

Paper: How Evaluation Protocol, Not Method Choice, Determines Reported Accuracy in BVP-Based WiFi Gesture Recognition: A Controlled Benchmark

Authors: Ma Zhechao (一作), Zhao Zhibiao (二作/通讯) — 908377038@qq.com

Target: MDPI Sensors (JCR Q2)

## Contents

- lodo_lib.py — shared library: data loading, TeacherNet/StudentNet, losses, training
- run_teachers.py — LODO teacher runner (resumable via JSON state)
- run_students.py — student KD runner (scratch/kd/kd_epcon)
- run_latency.py — latency measurement with pinned threads + warm-up
- emit_tables.py — table generator from per-(seed, fold) results
- make_figures.py — generate Figures 1-4 from the same runs
- sensitivity_clean.py — hyperparameter sensitivity sweep
- run_airfi_adapted.py — AirFi mechanism adapted to BVP/GRU backbone
- results/*.json — per-(seed, fold) results for all experiments
- paper_tables_10seeds.md — 10-seed tables

## Reproducibility

Every number in the paper can be recomputed from results/*.json via emit_tables.py:
    python emit_tables.py --dir results --seeds 0 1 2 3 4 5 6 7 8 9 --out paper_tables.md

Environment: Python 3.8 + PyTorch 2.4.1+cu121 + scipy 1.10.1.

## License
MIT License.
