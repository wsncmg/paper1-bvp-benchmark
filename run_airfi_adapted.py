# -*- coding: utf-8 -*-
"""run_airfi_adapted.py — 把 AirFi 的 domain-invariance 机制适配到 BVP backbone。

诚实定性：AirFi 原文处理原始 CSI 并做环境不变的表示学习；这里复现其两个核心机制
(1) 跨环境（location）特征分布对齐  (2) 特征增广下的分类一致性，
作用在我们的 GRU/BVP 特征上。不是原代码复现，是"adapted AirFi机制"。
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lodo_lib as L


def consistency_loss(logits, feat, trainer, yb, noise=0.01, p_drop=0.1):
    """对 GRU 特征做加噪 + 随机置零，要求 logits 不变。"""
    z = feat + noise * torch.randn_like(feat)
    mask = (torch.rand_like(feat) > p_drop).float()
    z = z * mask
    logits2 = trainer.classify_from_feat(z)
    return F.kl_div(F.log_softmax(logits2, 1), F.softmax(logits.detach(), 1),
                    reduction="batchmean")


class AirFiModel(L.TeacherNet):
    """教师网络 + 一个从特征到 logits 的分类头，用于增广一致性。"""
    def classify_from_feat(self, z):
        return self.fc(self.dropout(z))


def train_airfi(X, y, dom, train_idx, val_idx, seed=0, epochs=30, bs=32, lr=1e-3,
                lam_align=0.1, lam_cons=0.1):
    torch.manual_seed(seed); np.random.seed(seed)
    model = AirFiModel().to(L.DEVICE)
    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    best_val, best_state = -1.0, None
    domains = np.unique(dom[train_idx])
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(train_idx)
        for i in range(0, len(perm), bs):
            idx = perm[i:i+bs]
            xb = torch.from_numpy(X[idx]).to(L.DEVICE)
            yb = torch.from_numpy(y[idx]).to(L.DEVICE)
            opt.zero_grad()
            logits, feat = model(xb, return_feat=True)
            loss = F.cross_entropy(logits, yb)
            # (1) 跨环境对齐：对齐批内两个 location 组的二阶统计
            envs = L._env_by_domain(dom, idx)
            if len(envs) >= 2:
                a, b = envs[0], envs[1]
                pos = {v: k for k, v in enumerate(idx)}
                ia = torch.as_tensor([pos[v] for v in a], device=L.DEVICE)
                ib = torch.as_tensor([pos[v] for v in b], device=L.DEVICE)
                loss = loss + lam_align * L.coral_loss(feat[ia], feat[ib])
            # (2) 特征增广一致性
            loss = loss + lam_cons * consistency_loss(logits, feat, model, yb)
            loss.backward(); opt.step()
        if len(val_idx):
            v = L._eval_acc(model, X, y, val_idx)
            if v > best_val:
                best_val, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/airfi.json")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--lam-align", type=float, default=0.1)
    ap.add_argument("--lam-cons", type=float, default=0.1)
    args = ap.parse_args()

    X, y, meta = L.load_dataset("/data/widar/bvp", suffix="L0")
    folds = L.lodo_splits(meta["loc"])

    state = {}
    if os.path.exists(args.out):
        state = json.load(open(args.out))
    node = state.setdefault("airfi", {}).setdefault("loc", {})

    for seed in args.seeds:
        if str(seed) in node and len(node[str(seed)]) == len(folds):
            print("[skip] seed=%d" % seed); continue
        accs = []
        for fi, (tr, te) in enumerate(folds):
            tr2, va = L.source_val_split(y, tr, frac=0.10, seed=seed)
            model = train_airfi(X, y, meta["loc"], tr2, va, seed=seed,
                                lam_align=args.lam_align, lam_cons=args.lam_cons)
            a = L._eval_acc(model, X, y, te)
            accs.append(a)
            print("  airfi loc seed=%d fold=%d/%d acc=%.4f" % (seed, fi+1, len(folds), a), flush=True)
        node[str(seed)] = accs
        json.dump(state, open(args.out, "w"), indent=2)

    means = [np.mean(node[str(s)]) for s in args.seeds if str(s) in node]
    print("airfi loc mean over %d seeds: %.4f" % (len(means), np.mean(means)))
    json.dump(state, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()