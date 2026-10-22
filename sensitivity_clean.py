# -*- coding: utf-8 -*-
"""sensitivity_clean.py — 干净协议下的超参敏感性（source-val 选 epoch，绝不碰 test）。

跑跨位置轴、单 seed，对 CORAL/DANN 的 lambda 与 SupCon/EPCon 的 gamma 各扫
{0.01, 0.1, 1.0}，产出每个点的 5-fold 平均 test acc，用于正文的超参报告。
"""
import json, os
import numpy as np
import lodo_lib as L

LAM = [0.01, 0.1, 1.0]
GAM = [0.01, 0.1, 1.0]

def main():
    X, y, meta = L.load_dataset("/data/widar/bvp", suffix="L0")
    folds = L.lodo_splits(meta["loc"])   # 5 folds
    results = {}

    # --- lambda sweep: CORAL, DANN
    for method in ("coral", "dann"):
        for lam in LAM:
            accs = []
            for tr, te in folds:
                tr2, va = L.source_val_split(y, tr, frac=0.10, seed=0)
                cfg = dict(L.DEFAULT_CFG, lam_coral=lam, lam_dann=lam, lam_irm=0.1, gamma=0.1)
                model = L.train_model(X, y, meta["loc"], tr2, va, 0, method=method, cfg=cfg)
                accs.append(L._eval_acc(model, X, y, te))
            key = "%s_lambda%.2f" % (method, lam)
            results[key] = {"mean": float(np.mean(accs)), "folds": [float(a) for a in accs]}
            print("%-22s mean=%.4f" % (key, np.mean(accs)), flush=True)

    # --- gamma sweep: SupCon, EPCon (lambda inactive)
    for method in ("supcon", "epcon"):
        for gam in GAM:
            accs = []
            for tr, te in folds:
                tr2, va = L.source_val_split(y, tr, frac=0.10, seed=0)
                cfg = dict(L.DEFAULT_CFG, gamma=gam)
                model = L.train_model(X, y, meta["loc"], tr2, va, 0, method=method, cfg=cfg)
                accs.append(L._eval_acc(model, X, y, te))
            key = "%s_gamma%.2f" % (method, gam)
            results[key] = {"mean": float(np.mean(accs)), "folds": [float(a) for a in accs]}
            print("%-22s mean=%.4f" % (key, np.mean(accs)), flush=True)

    json.dump(results, open("results/sensitivity_clean.json", "w"), indent=2)
    print("saved results/sensitivity_clean.json")

if __name__ == "__main__":
    main()