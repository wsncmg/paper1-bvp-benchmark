#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_students.py — Table 3 re-run under the fixed LODO protocol.

Changes vs bvp_student_baselines.py:
  * the teacher is an actual EPCon teacher (--teacher-mode epcon, gamma from --gamma);
    --teacher-mode erm reproduces the old (mislabelled) setting for reference
  * user axis parsed from filename field 0 (was field 4 = repetition index)
  * session axis exists, so the cross-session student cells have a real experiment
  * teacher/student/KD results stored per (axis, seed, fold) for paired testing
  * per-class accuracy and the confusion matrix are dumped from the *same* run as the
    aggregate numbers, so figures and tables can no longer disagree
  * optional --variants kd_epcon to answer "does the student also use EPCon?"

Usage
-----
  python run_students.py --root /data/widar/bvp --out results/students.json
  python run_students.py --axes loc session --variants scratch kd
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import lodo_lib as L


def train_teacher(X, y, dom, tr, seed, mode, cfg):
    tr2, va = L.source_val_split(y, tr, frac=0.10, seed=seed)
    method = "epcon" if mode == "epcon" else "erm"
    return L.train_model(X, y, dom, tr2, va, seed, method=method, cfg=cfg)


def train_student(X, y, dom, tr, seed, teacher, variant, cfg):
    tr2, va = L.source_val_split(y, tr, frac=0.10, seed=seed)
    cfg2 = dict(L.DEFAULT_CFG, **cfg)          # merge defaults: lr/tau/bs always defined
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = L.StudentNet().to(L.DEVICE)
    opt = torch.optim.RMSprop(model.parameters(), lr=cfg2["lr"])
    best_val, best_state = -1.0, None
    T, alpha = cfg2.get("T", 4.0), cfg2.get("alpha", 0.7)
    teacher.eval()

    for ep in range(cfg2["epochs"]):
        model.train()
        perm = np.random.permutation(tr2)
        for i in range(0, len(perm), cfg2["bs"]):
            idx = perm[i:i + cfg2["bs"]]
            xb = torch.from_numpy(X[idx]).to(L.DEVICE)
            yb = torch.from_numpy(y[idx]).to(L.DEVICE)
            opt.zero_grad()
            logits, feat = model(xb, return_feat=True)
            ce = F.cross_entropy(logits, yb)

            if variant == "scratch":
                loss = ce
            else:
                with torch.no_grad():
                    t_logits = teacher(xb)
                kd = F.kl_div(
                    F.log_softmax(logits / T, dim=1),
                    F.softmax(t_logits / T, dim=1),
                    reduction="batchmean") * (T * T)
                loss = alpha * kd + (1 - alpha) * ce
                if variant == "kd_epcon":
                    loss = loss + cfg2["gamma"] * L.epcon_loss(feat, yb, cfg2["tau"])

            loss.backward()
            opt.step()

        if len(va):
            v = L._eval_acc(model, X, y, va)
            if v > best_val:
                best_val, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/widar/bvp")
    ap.add_argument("--out", default="results/students.json")
    ap.add_argument("--axes", nargs="+", default=["loc", "ori", "user", "session"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--teacher-mode", default="epcon", choices=["epcon", "erm"])
    ap.add_argument("--variants", nargs="+", default=["scratch", "kd"],
                    choices=["scratch", "kd", "kd_epcon"])
    ap.add_argument("--suffix", default="L0")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--T", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--limit-folds", type=int, default=0)
    args = ap.parse_args()

    print("[env] device=%s teacher=%s gamma=%s" % (L.DEVICE, args.teacher_mode, args.gamma))
    X, y, meta = L.load_dataset(args.root, suffix=args.suffix or None)
    cfg = dict(L.DEFAULT_CFG, epochs=args.epochs, gamma=args.gamma, T=args.T, alpha=args.alpha)

    state = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            state = json.load(f)
    tag = "teacher_%s" % args.teacher_mode
    detail = state.get("_detail", {})

    for axis in args.axes:
        folds = L.lodo_splits(meta[axis])
        if args.limit_folds:
            folds = folds[: args.limit_folds]
        print("\n=== axis=%s folds=%d ===" % (axis, len(folds)))
        for seed in args.seeds:
            node = state.setdefault(axis, {})

            def _need(name):
                return name not in node or len(node[name].get(str(seed), [])) != len(folds)

            if not (_need(tag) or any(_need(v) for v in args.variants)):
                print("  [skip] axis=%s seed=%d already done" % (axis, seed))
                continue

            vals = {}
            for fi, (tr, te) in enumerate(folds):
                teacher = train_teacher(X, y, meta[axis], tr, seed, args.teacher_mode, cfg)
                rows = {tag: teacher}
                for variant in args.variants:
                    rows[variant] = train_student(X, y, meta[axis], tr, seed, teacher, variant, cfg)
                for name, m in rows.items():
                    a = L._eval_acc(m, X, y, te)
                    vals.setdefault(name, []).append(a)
                    print("  %-12s %-4s seed=%d fold=%d/%d acc=%.4f"
                          % (name, axis, seed, fi + 1, len(folds), a))
                    if seed == args.seeds[0]:
                        pred = L.predict(m, X, te)
                        d = detail.setdefault("%s|%s" % (name, axis), {"y": [], "p": []})
                        d["y"].extend(y[te].tolist())
                        d["p"].extend(pred.tolist())
                state["_detail"] = detail
                for name, v in vals.items():
                    node.setdefault(name, {})[str(seed)] = list(v)
                L.save_json(state, args.out)

            for name, v in vals.items():
                print("  --> %-12s %-4s seed=%d: %.2f (fold mean)"
                      % (name, axis, seed, float(np.mean(v)) * 100))

    L.save_json(state, args.out)
    print("\n[done] %s" % args.out)


if __name__ == "__main__":
    main()
