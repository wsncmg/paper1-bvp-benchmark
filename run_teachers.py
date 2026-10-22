#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_teachers.py — Table 2 re-run under the fixed LODO protocol.

Changes vs bvp_baselines_comprehensive.py:
  * no epoch is selected on the target test set (source-internal 10% validation)
  * user axis parsed from filename field 0, not field 4
  * session axis added
  * 'epcon' = prototype-augmented loss (single implementation); 'supcon' is a
    separate, explicit baseline so the EPCon-vs-SCL ablation the paper claims
    actually exists
  * GroupDRO groups by *location* with the standard exponentiated q update
  * IRM/DANN/CORAL use two location-based environments built from training data only
  * results stored per (method, axis, seed, fold) so paired tests are possible
  * std reported with ddof=1

Usage
-----
  python run_teachers.py --root /data/widar/bvp --out results/teachers.json
  CUDA_VISIBLE_DEVICES=0 python run_teachers.py --methods erm epcon --out results/t_gpu0.json
  # smoke test (1 fold, 1 seed):
  python run_teachers.py --seeds 0 --limit-folds 1 --methods erm --out /tmp/smoke.json
"""
import argparse
import json
import os

import numpy as np
import torch

import lodo_lib as L

METHODS = ["erm", "epcon", "supcon", "coral", "groupdro", "irm", "dann"]
AXES = ["loc", "ori", "user", "session", "id"]


def id_split(y, seed=0, frac=0.8):
    """Protocol A: stratified random 80/20 split of a single session pool.

    Kept only to quantify the in-domain / cross-domain gap under the *same*
    source-val epoch selection as every other row.  Never used for a main result.
    """
    rng = np.random.RandomState(1000 + seed)
    tr, te = [], []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        ci = ci[rng.permutation(len(ci))]
        k = int(round(frac * len(ci)))
        tr.append(ci[:k])
        te.append(ci[k:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(te))


def load_state(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/widar/bvp")
    ap.add_argument("--out", default="results/teachers.json")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--axes", nargs="+", default=AXES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--suffix", default="L0", help="'' loads every .mat; default keeps the old '*L0*' filter")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--lam-coral", type=float, default=1.0)
    ap.add_argument("--eta-dro", type=float, default=0.01)
    ap.add_argument("--lam-irm", type=float, default=0.1)
    ap.add_argument("--lam-dann", type=float, default=0.1)
    ap.add_argument("--limit-folds", type=int, default=0)
    ap.add_argument("--detail-method", default="epcon")
    args = ap.parse_args()

    print("[env] device=%s" % L.DEVICE)
    X, y, meta = L.load_dataset(args.root, suffix=args.suffix or None)

    cfg = dict(epochs=args.epochs, gamma=args.gamma, lam_coral=args.lam_coral,
               eta_dro=args.eta_dro, lam_irm=args.lam_irm, lam_dann=args.lam_dann)

    state = load_state(args.out)
    detail = state.get("_detail", {})

    for axis in args.axes:
        dom_axis = "loc" if axis == "id" else axis
        if axis == "id":
            base_folds = [id_split(y, args.seeds[0])]
        else:
            base_folds = L.lodo_splits(meta[axis])
        if args.limit_folds:
            base_folds = base_folds[: args.limit_folds]
        print("\n=== axis=%s folds=%d ===" % (axis, len(base_folds)))

        for method in args.methods:
            key = "%s|%s" % (method, axis)
            node = state.setdefault(method, {}).setdefault(axis, {"acc": {}, "fold_ids": []})
            node["fold_ids"] = (["random-80-20"] if axis == "id"
                                else [str(d) for d in np.unique(meta[axis])][: len(base_folds)])

            for seed in args.seeds:
                folds = [id_split(y, seed)] if axis == "id" else base_folds
                got = node["acc"].get(str(seed))
                if got is not None and len(got) == len(folds):
                    print("  [skip] %s seed=%d (already done)" % (key, seed))
                    continue
                accs = []
                for fi, (tr, te) in enumerate(folds):
                    tr2, va = L.source_val_split(y, tr, frac=0.10, seed=seed)
                    model = L.train_model(X, y, meta[dom_axis], tr2, va, seed, method=method, cfg=cfg)
                    a = L._eval_acc(model, X, y, te)
                    accs.append(a)
                    print("  %-9s %-4s seed=%d fold=%d/%d acc=%.4f"
                          % (method, axis, seed, fi + 1, len(folds), a))
                    if method == args.detail_method and seed == args.seeds[0]:
                        pred = L.predict(model, X, te)
                        d = detail.setdefault("%s|%s" % (method, axis), {"y": [], "p": []})
                        d["y"].extend(y[te].tolist())
                        d["p"].extend(pred.tolist())
                    node["acc"][str(seed)] = accs
                    state["_detail"] = detail
                    L.save_json(state, args.out)

            mat = np.array([node["acc"][str(s)] for s in args.seeds if str(s) in node["acc"]])
            if mat.size:
                print("  --> %s %s: %.2f ± %.2f (mean over %d seeds; fold-level %.2f)"
                      % (method, axis, mat.mean() * 100, np.std(mat.mean(1), ddof=1) * 100,
                         mat.shape[0], np.std(mat, ddof=1) * 100))

    L.save_json(state, args.out)
    print("\n[done] %s" % args.out)


if __name__ == "__main__":
    main()
