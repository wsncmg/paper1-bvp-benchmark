#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
emit_tables.py -- build the manuscript tables straight from results/*.json.

No number in the paper is typed by hand: every cell is computed here from per-(seed,
fold) records.  "+/-" is the sample std (ddof=1) over seeds; the fold-level std is
reported separately.  Epoch selection uses a 10% source-internal validation split.

Usage:  python emit_tables.py --dir results --out paper_tables.md
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

AXES = [("loc", "Cross-Location"), ("ori", "Cross-Orientation"),
        ("user", "Cross-User"), ("session", "Cross-Session")]
PRETTY = {"erm": "ERM", "supcon": "SupCon", "epcon": "EPCon", "coral": "EPCon+CORAL",
          "dann": "DANN", "groupdro": "GroupDRO", "irm": "IRM"}
ORDER = ["erm", "supcon", "epcon", "coral", "dann", "groupdro", "irm"]
STUDENTS = [("teacher_epcon", "Teacher (EPCon)"), ("teacher_erm", "Teacher (ERM)"),
            ("scratch", "Student (scratch)"), ("kd", "Student (KD)"),
            ("kd_epcon", "Student (KD+EPCon)")]
GESTURES = ["Push", "Pull", "Slide", "Up", "Down", "Push-Pull"]


def load_merged(pattern):
    out, files = {}, sorted(glob.glob(pattern))
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        for k, v in d.items():
            if k == "_detail":
                out.setdefault("_detail", {}).update(v)
            elif isinstance(v, dict):
                out.setdefault(k, {}).update(v)
    return out, files


def seed_matrix(node, seeds):
    """(n_complete_seeds, n_folds) accuracy matrix in percent, or None."""
    if not isinstance(node, dict) or not isinstance(node.get("acc"), dict):
        return None
    rows = [node["acc"][str(s)] for s in seeds if node["acc"].get(str(s))]
    if not rows:
        return None
    lens = [len(r) for r in rows]
    target = max(set(lens), key=lens.count)
    rows = [r for r in rows if len(r) == target]
    return np.array(rows, float) * 100.0


def cell(m):
    """(mean, std_over_seed_means, fold_level_std, shape) or None."""
    if m is None or m.size == 0:
        return None
    sd = float(np.std(m.mean(1), ddof=1)) if m.shape[0] > 1 else float("nan")
    return float(m.mean()), sd, float(np.std(m.ravel(), ddof=1)), m.shape


def fmt(c, with_std=True):
    if not c:
        return "-"
    if not with_std or not np.isfinite(c[1]):
        return "%.2f" % c[0]
    return "%.2f +/- %.2f" % (c[0], c[1])


def paired(a, b):
    av, bv = a.ravel(), b.ravel()
    t, pt = stats.ttest_rel(av, bv)
    try:
        pw = float(stats.wilcoxon(av, bv).pvalue)
    except Exception:
        pw = float("nan")
    return float(np.mean(av - bv)), float(t), float(pt), pw


def holm(ps):
    p = np.where(np.isfinite(ps), ps, 1.0)
    order = np.argsort(p)
    adj, prev = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = prev
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--out", default="paper_tables.md")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    T, tfiles = load_merged(os.path.join(args.dir, "teachers_*.json"))
    S, sfiles = load_merged(os.path.join(args.dir, "students_*.json"))
    md = ["# Manuscript tables (generated, do not hand-edit)", "",
          "Sources: `%s`" % ", ".join(os.path.basename(f) for f in tfiles + sfiles), "",
          "Convention: mean +/- sample std over seeds (ddof=1) of each seed's fold-mean "
          "accuracy; fold-level std separately. Epoch selection uses a 10% source-internal "
          "validation split; the target test split is evaluated once.", ""]

    md += ["## Coverage (completed seeds per cell, out of %d)" % len(args.seeds), "",
           "| Method | " + " | ".join(l for _, l in AXES) + " |",
           "|---|" + "---|" * len(AXES)]
    for m in ORDER:
        cells = []
        for ax, _ in AXES:
            mm = seed_matrix(T.get(m, {}).get(ax), args.seeds)
            cells.append("%d/%d" % (mm.shape[0], len(args.seeds)) if mm is not None else "-")
        md.append("| %s | %s |" % (PRETTY[m], " | ".join(cells)))

    md += ["", "## Table 1 -- Protocol A (in-domain) vs Protocol B (strict LODO)", "",
           "| Method | In-domain (80/20) | " + " | ".join(l for _, l in AXES) + " |",
           "|---|---|" + "---|" * len(AXES)]
    for m in ("erm", "epcon"):
        row = [fmt(cell(seed_matrix(T.get(m, {}).get("id"), args.seeds)))]
        for ax, _ in AXES:
            row.append(fmt(cell(seed_matrix(T.get(m, {}).get(ax), args.seeds))))
        md.append("| %s | %s |" % (PRETTY[m], " | ".join(row)))

    md += ["", "## Table 2 -- teacher domain generalisation", "",
           "| Method | " + " | ".join(l for _, l in AXES) + " |",
           "|---|" + "---|" * len(AXES)]
    for m in ORDER:
        row = [fmt(cell(seed_matrix(T.get(m, {}).get(ax), args.seeds))) for ax, _ in AXES]
        md.append("| %s | %s |" % (PRETTY[m], " | ".join(row)))

    md += ["", "_Fold-level std across all seed x fold cells:_", "",
           "| Method | " + " | ".join(l for _, l in AXES) + " |",
           "|---|" + "---|" * len(AXES)]
    for m in ORDER:
        row = []
        for ax, _ in AXES:
            c = cell(seed_matrix(T.get(m, {}).get(ax), args.seeds))
            row.append("%.2f" % c[2] if c else "-")
        md.append("| %s | %s |" % (PRETTY[m], " | ".join(row)))

    md += ["", "## Paired tests vs. ERM (matched seed x fold), Holm-corrected", ""]
    for ax, lbl in AXES:
        ref = seed_matrix(T.get("erm", {}).get(ax), args.seeds)
        if ref is None:
            continue
        rows, names = [], []
        for m in ORDER:
            if m == "erm":
                continue
            v = seed_matrix(T.get(m, {}).get(ax), args.seeds)
            if v is None or v.shape != ref.shape:
                continue
            rows.append(paired(v, ref))
            names.append(m)
        if not rows:
            continue
        arr = np.array(rows, float)
        at, aw = holm(arr[:, 2]), holm(arr[:, 3])
        md += ["", "**%s** (n=%d matched cells)" % (lbl, ref.size), "",
               "| Method | Delta vs ERM (pp) | t | p (t) | p (Holm) | p (Wilcoxon) | p (Holm) |",
               "|---|---|---|---|---|---|---|"]
        for i, m in enumerate(names):
            md.append("| %s | %+.2f | %.3f | %.4f | %.4f | %.4f | %.4f |"
                      % (PRETTY[m], arr[i, 0], arr[i, 1], arr[i, 2], at[i], arr[i, 3], aw[i]))

    if S:
        md += ["", "## Table 3 -- teacher vs. student", "",
               "| Model | " + " | ".join(l for _, l in AXES) + " |",
               "|---|" + "---|" * len(AXES)]
        for name, pretty in STUDENTS:
            row = []
            for ax, _ in AXES:
                # run_students.py stores {axis: {name: {seed: [...]}}} -- try both layouts.
                node = S.get(ax, {}).get(name)
                if not isinstance(node, dict):
                    node = S.get(name, {}).get(ax)
                rows = [node[str(s)] for s in args.seeds
                        if isinstance(node, dict) and node.get(str(s))]
                if not rows:
                    row.append("-")
                    continue
                row.append(fmt(cell(np.array(rows, float) * 100.0)))
            md.append("| %s | %s |" % (pretty, " | ".join(row)))

        det = S.get("_detail", {})
        keys = sorted(k for k in det if k.endswith("|loc"))
        if len(keys) >= 2:
            tk = next((k for k in keys if k.startswith("epcon") or "teacher" in k), keys[0])
            sk = next((k for k in keys if k.startswith("kd")), keys[1])
            md += ["", "## Per-class accuracy (cross-location, pooled over folds, same runs)", "",
                   "| Class | Gesture | %s | %s | Delta pp |" % (tk, sk),
                   "|---|---|---|---|---|"]
            acc, overall = {}, {}
            for k in (tk, sk):
                y = np.array(det[k]["y"])
                p = np.array(det[k]["p"])
                acc[k] = [100.0 * np.mean(p[y == c] == c) if (y == c).any() else float("nan")
                          for c in range(len(GESTURES))]
                overall[k] = 100.0 * np.mean(p == y)
            for c, g in enumerate(GESTURES):
                md.append("| %d | %s | %.2f | %.2f | %+.2f |"
                          % (c, g, acc[tk][c], acc[sk][c], acc[sk][c] - acc[tk][c]))
            md.append("| - | **overall** | %.2f | %.2f | %+.2f |"
                      % (overall[tk], overall[sk], overall[sk] - overall[tk]))
            md += ["", "_The unweighted mean of the per-class rows must match the overall row "
                   "when class supports are balanced; a gap means the per-class and aggregate "
                   "numbers came from different runs._"]

    with open(args.out, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\n[saved] %s" % args.out)


if __name__ == "__main__":
    main()
