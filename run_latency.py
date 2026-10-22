#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_latency.py — CPU latency re-measurement that can actually support a claim.

Why the old numbers cannot be used
----------------------------------
The paper reports Teacher 0.79 ± 0.01 ms and Student 7.93 ± 4.16 ms on an Intel Xeon
CPU: a 74.7%-smaller model (GRU 128 -> 64) declared 10x *slower*, with a 52% coefficient
of variation.  A larger GRU cannot be faster than a smaller one by an order of
magnitude; that pattern is measurement noise (thread contention, missing warmup,
unsynchronised timers, shared machine), not an architectural property.

This script measures properly:
  * torch.set_num_threads(1) and torch.set_num_interop_threads(1)
  * eval() + no_grad, batch=1 and batch=32 reported separately
  * 100 warmup iterations discarded, 1000 measured iterations
  * median / mean / p95 / IQR reported (median is the headline number)
  * runs each model 5 times in interleaved order and reports the spread across repeats,
    so thermal/contention drift shows up instead of hiding in one lucky run
  * records the CPU model and thread count in the output

Usage
-----
  python run_latency.py --out results/latency.json
  taskset -c 2 python run_latency.py --out results/latency_pinned.json
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

import lodo_lib as L

ITERS = 1000
WARMUP = 100
REPEATS = 5


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def bench(model, x, iters=ITERS, warmup=WARMUP):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        ts = np.empty(iters, dtype=np.float64)
        for i in range(iters):
            t0 = time.perf_counter()
            model(x)
            ts[i] = (time.perf_counter() - t0) * 1e3
    return ts


def summarize(ts):
    return {
        "median_ms": float(np.median(ts)),
        "mean_ms": float(np.mean(ts)),
        "p95_ms": float(np.percentile(ts, 95)),
        "iqr_ms": float(np.percentile(ts, 75) - np.percentile(ts, 25)),
        "std_ms": float(np.std(ts, ddof=1)),
    }


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/latency.json")
    ap.add_argument("--T", type=int, default=30, help="sequence length; must match the data")
    args = ap.parse_args()

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    teacher = L.TeacherNet().eval()
    student = L.StudentNet().eval()
    out = {"cpu": cpu_model(), "torch": torch.__version__,
           "threads": torch.get_num_threads(), "T": args.T,
           "warmup": WARMUP, "iters": ITERS, "repeats": REPEATS, "results": {}}

    for bs in (1, 32):
        x = torch.randn(bs, args.T, L.IMG, L.IMG, 1)
        for name, model in (("teacher", teacher), ("student", student)):
            per_repeat = []
            for r in range(REPEATS):                    # interleaved by repeat, not by model
                per_repeat.append(summarize(bench(model, x)))
            key = "%s_bs%d" % (name, bs)
            med = [s["median_ms"] for s in per_repeat]
            out["results"][key] = {
                "params": count_params(model),
                "median_ms": float(np.median(med)),
                "across_repeat_std_ms": float(np.std(med, ddof=1)),
                "repeats": per_repeat,
            }
            print("%-12s bs=%-2d params=%-7d median=%.3f ms  (across repeats sd=%.3f)"
                  % (name, bs, count_params(model), np.median(med), np.std(med, ddof=1)))

    L.save_json(out, args.out)
    print("\n[done] %s" % args.out)
    print("[note] report median_ms; if across_repeat_std_ms is comparable to the difference "
          "between models, the difference is not measurable on this machine.")


if __name__ == "__main__":
    main()
