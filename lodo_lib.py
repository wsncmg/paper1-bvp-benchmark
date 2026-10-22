#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lodo_lib.py — shared protocol/benchmark library for the corrected LODO evaluation.

Why this file exists
--------------------
The three delivered scripts (bvp_baselines_comprehensive.py, bvp_student_baselines.py,
bvp_epcon_gamma1.py) contain five protocol defects that invalidate the reported numbers:

  F1  bvp_student_baselines.py:92 and bvp_epcon_gamma1.py:63 read the user id from
      filename field 4 (a repetition index), so their "cross-user" axis is actually
      "leave-one-repetition-out".  Fixed here: user is filename field 0.
  F2  Only the comprehensive script builds a session axis; the student script has no
      session domain at all, so its cross-session numbers have no experiment behind
      them.  Fixed here: session is a first-class LODO axis everywhere.
  F3  bvp_baselines_comprehensive.py:183 returns
      `best = max(best, test_acc)` — the epoch is selected on the *target test set*.
      Fixed here: epoch selection uses a source-internal validation split (10% of the
      training domains, stratified by class); the test split is touched exactly once.
  F4  The "EPCon γ=0.1" main result uses a prototype-free SupCon
      (bvp_baselines_comprehensive.py:57-67) while "EPCon γ=1.0" uses a
      prototype-augmented loss (bvp_epcon_gamma1.py:82-121), so the γ comparison is
      confounded with the loss implementation.  Fixed here: one implementation per
      method, and both are available as explicit, separate baselines:
          supcon  — prototype-free supervised contrastive (Khosla et al.)
          epcon   — supervised contrastive + batch-mean class prototypes
  F5  Results are stored as per-seed fold means only, which forbids paired testing.
      Fixed here: every result is written as results[method][axis][seed][fold].

Architecture, optimiser, epochs, batch size and γ/λ/T/α defaults are kept identical to
the originals so that new numbers stay directly comparable to the old ones.
"""
import glob
import json
import os
from copy import deepcopy

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 20
N_CLASSES = 6
DEFAULT_SESSIONS = ("20181109-VS", "20181130-VS")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- data
def _normalize(d):
    """Per-sample min-max over the (T, H, W) axes — unchanged from the originals."""
    dmax, dmin = d.max(axis=(0, 1), keepdims=True), d.min(axis=(0, 1), keepdims=True)
    denom = dmax - dmin
    return d if np.any(denom == 0) else (d - dmin) / denom


def parse_name(path):
    """Parse Widar3.0 BVP filenames: userN-gesture-location-orientation-...-L0.mat

    Returns None for anything unexpected instead of silently mislabelling it.
    """
    stem = os.path.basename(path)[:-4]
    parts = stem.split("-")
    if len(parts) < 4:
        return None
    user = parts[0]
    if not user.lower().startswith("user"):
        return None          # guards against the parts[4] mistake: field 0 must be a user
    try:
        g = int(parts[1]) - 1
        loc = int(parts[2])
        ori = int(parts[3])
    except ValueError:
        return None
    if not (0 <= g < N_CLASSES):
        return None
    return {"user": user, "gesture": g, "loc": loc, "ori": ori}


def load_dataset(root, sessions=DEFAULT_SESSIONS, suffix=None, verbose=True):
    """Load all BVP samples into one padded array.

    suffix: keep the original '*L0*' file filter when reproducing the old N=14246;
            pass None to load every .mat under the session directory.
    """
    Xs, ys = [], []
    axes = {k: [] for k in ("loc", "ori", "user", "session")}
    for si, sess in enumerate(sessions):
        sd = os.path.join(root, sess)
        pattern = "**/*.mat"
        for f in sorted(glob.glob(os.path.join(sd, pattern), recursive=True)):
            if suffix and suffix not in os.path.basename(f):
                continue
            m = parse_name(f)
            if m is None:
                continue
            try:
                bvp = sio.loadmat(f)["velocity_spectrum_ro"]
            except Exception:
                continue
            if bvp.ndim != 3:
                continue
            Xs.append(_normalize(bvp).astype(np.float32))
            ys.append(m["gesture"])
            axes["loc"].append(m["loc"])
            axes["ori"].append(m["ori"])
            axes["user"].append(m["user"])
            axes["session"].append(si)

    if not Xs:
        raise RuntimeError("no usable samples found under %s" % root)

    T = max(x.shape[2] for x in Xs)
    X = np.zeros((len(Xs), T, IMG, IMG, 1), dtype=np.float32)
    for i, x in enumerate(Xs):
        X[i, : x.shape[2]] = x.transpose(2, 0, 1)[..., None]

    y = np.asarray(ys, dtype=np.int64)
    meta = {k: np.asarray(v) for k, v in axes.items()}
    if verbose:
        print("[data] N=%d T=%d classes=%s locs=%s oris=%s users=%d sessions=%s"
              % (len(ys), T, np.unique(y).tolist(), np.unique(meta["loc"]).tolist(),
                 np.unique(meta["ori"]).tolist(), len(np.unique(meta["user"])),
                 np.unique(meta["session"]).tolist()))
        print("[data] X footprint = %.1f MB" % (X.nbytes / 1e6))
    return X, y, meta


def lodo_splits(labels):
    """Leave-one-domain-out folds. Deterministic and target-disjoint by construction."""
    labels = np.asarray(labels)
    return [(np.where(labels != d)[0], np.where(labels == d)[0])
            for d in np.unique(labels)]


def source_val_split(y, train_idx, frac=0.10, seed=0):
    """Stratified 90/10 split *inside the training domains* — never the target."""
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for c in np.unique(y[train_idx]):
        ci = train_idx[y[train_idx] == c]
        ci = ci[rng.permutation(len(ci))]
        k = min(max(1, int(round(frac * len(ci)))), len(ci) - 1) if len(ci) > 1 else 0
        if k:
            va.append(ci[:k])
            tr.append(ci[k:])
        else:
            tr.append(ci)
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(va))


# ------------------------------------------------------------------------- models
class TeacherNet(nn.Module):
    """Identical to the delivered TeacherModel (182,310 parameters)."""

    def __init__(self, num_classes=N_CLASSES, dropout=0.5):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 10 * 10, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=64, hidden_size=128, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, return_feat=False):
        N, T, H, W, C = x.shape
        f = self.conv_net(x.reshape(N * T, C, H, W)).reshape(N, T, -1)
        out, _ = self.gru(f)
        out = self.dropout(out[:, -1, :])
        logits = self.fc(out)
        return (logits, out) if return_feat else logits


class StudentNet(nn.Module):
    """Identical to the delivered StudentModel (46,102 parameters)."""

    def __init__(self, num_classes=N_CLASSES, dropout=0.5):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 10 * 10, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32), nn.ReLU(),
        )
        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x, return_feat=False):
        N, T, H, W, C = x.shape
        f = self.conv_net(x.reshape(N * T, C, H, W)).reshape(N, T, -1)
        out, _ = self.gru(f)
        out = self.dropout(out[:, -1, :])
        logits = self.fc(out)
        return (logits, out) if return_feat else logits


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


# -------------------------------------------------------------------------- losses
def supcon_loss(feat, labels, tau=0.07):
    """Prototype-free supervised contrastive loss (matches the old 'epcon_only')."""
    z = F.normalize(feat, dim=1)
    sim = z @ z.t() / tau
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels.view(-1, 1) == labels.view(1, -1)) & ~eye
    denom = pos.sum(1).clamp(min=1)
    return -((logprob * pos).sum(1) / denom).mean()


def epcon_loss(feat, labels, tau=0.07):
    """Supervised contrastive loss + batch-mean class prototypes as extra positives.

    This is the loss the paper's Eq. (EPCon) describes and the one the γ=1.0 run used;
    it is now the single implementation used for every 'epcon' result.  Prototypes are
    recomputed per batch (no EMA memory bank) — stated explicitly in the paper.
    """
    z = F.normalize(feat, dim=1)
    uniq = torch.unique(labels)
    protos = F.normalize(torch.stack([z[labels == c].mean(0) for c in uniq]), dim=1)
    allz = torch.cat([z, protos], 0)
    ally = torch.cat([labels, uniq], 0)
    sim = allz @ allz.t() / tau
    n = allz.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (ally.view(1, -1) == ally.view(-1, 1)) & ~eye
    pos = pos[: z.size(0)]
    denom = pos.sum(1).clamp(min=1)
    return -((logprob[: z.size(0)] * pos).sum(1) / denom).mean()


def coral_loss(a, b):
    """Deep CORAL on the covariance only, as written in the paper's Eq. (L_CORAL)."""
    if a.size(0) < 2 or b.size(0) < 2:
        return a.new_zeros(())
    d = a.size(1)
    ca = a - a.mean(0, keepdim=True)
    cb = b - b.mean(0, keepdim=True)
    cov_a = ca.t() @ ca / (a.size(0) - 1)
    cov_b = cb.t() @ cb / (b.size(0) - 1)
    return ((cov_a - cov_b) ** 2).sum() / (4 * d * d)


def irm_penalty(logits, y):
    """IRMv1 penalty on one environment."""
    scale = torch.ones(1, device=logits.device, requires_grad=True)
    loss = F.cross_entropy(logits * scale, y)
    g = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return g.pow(2).sum()


# ----------------------------------------------------------------------- training
DEFAULT_CFG = dict(epochs=30, bs=32, lr=1e-3, tau=0.07, gamma=0.1, lam_coral=1.0,
                   eta_dro=0.01, lam_dann=0.1, lam_irm=0.1, val_frac=0.10)


def _env_by_domain(dom, idx, n_env=2):
    """Split training indices into n_env groups by a domain label (for CORAL/DANN/IRM)."""
    vals = np.unique(dom[idx])
    if len(vals) < n_env:
        return [idx] if len(vals) == 1 else [idx[dom[idx] != v] for v in vals[:n_env]]
    chunks = np.array_split(np.sort(vals), n_env)
    return [idx[np.isin(dom[idx], c)] for c in chunks if len(c)]


def _eval_acc(model, X, y, idx, bs=256):
    if len(idx) == 0:
        return float("nan")
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb = torch.from_numpy(X[b]).to(DEVICE)
            correct += (model(xb).argmax(1).cpu().numpy() == y[b]).sum()
    return correct / len(idx)


def train_model(X, y, dom, train_idx, val_idx, seed, method="erm", cfg=None, **kw):
    """Train one model.  Epoch chosen on val_idx only — val_idx is source-internal."""
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = TeacherNet().to(DEVICE)
    params = list(model.parameters())
    dom_head = None
    if method == "dann":
        dom_head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 2)).to(DEVICE)
        params += list(dom_head.parameters())
    opt = torch.optim.RMSprop(params, lr=cfg["lr"])
    domains = np.unique(dom[train_idx])
    q = torch.ones(len(domains), device=DEVICE) / max(len(domains), 1)
    best_val, best_state = -1.0, None
    steps = max(1, int(np.ceil(len(train_idx) / cfg["bs"])))

    for ep in range(cfg["epochs"]):
        model.train()
        perm = np.random.permutation(train_idx)
        for step, i in enumerate(range(0, len(perm), cfg["bs"])):
            idx = perm[i:i + cfg["bs"]]
            xb = torch.from_numpy(X[idx]).to(DEVICE)
            yb = torch.from_numpy(y[idx]).to(DEVICE)
            opt.zero_grad()

            if method == "erm":
                loss = F.cross_entropy(model(xb), yb)

            elif method in ("epcon", "supcon"):
                logits, feat = model(xb, return_feat=True)
                c = epcon_loss if method == "epcon" else supcon_loss
                loss = F.cross_entropy(logits, yb) + cfg["gamma"] * c(feat, yb, cfg["tau"])

            elif method == "coral":
                logits, feat = model(xb, return_feat=True)
                loss = F.cross_entropy(logits, yb)
                envs = _env_by_domain(dom, idx)
                if len(envs) >= 2:
                    a, b = envs[0], envs[1]
                    pos = {v: k for k, v in enumerate(idx)}
                    ia = torch.as_tensor([pos[v] for v in a], device=DEVICE)
                    ib = torch.as_tensor([pos[v] for v in b], device=DEVICE)
                    loss = loss + cfg["lam_coral"] * coral_loss(feat[ia], feat[ib])

            elif method == "groupdro":
                logits = model(xb)
                per = F.cross_entropy(logits, yb, reduction="none")
                gl = []
                for gi, d in enumerate(domains):
                    m = dom[idx] == d
                    if m.any():
                        gl.append((gi, per[torch.from_numpy(m).to(DEVICE)].mean()))
                if gl:
                    idxs = torch.as_tensor([g[0] for g in gl], device=DEVICE)
                    losses = torch.stack([g[1] for g in gl])
                    q[idxs] = q[idxs] * torch.exp(cfg["eta_dro"] * losses.detach())
                    q = q / q.sum()
                    loss = (q[idxs] * losses).sum()
                else:
                    loss = per.mean()

            elif method == "irm":
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                envs = _env_by_domain(dom, idx)
                for e in envs[:2]:
                    pos = {v: k for k, v in enumerate(idx)}
                    ie = torch.as_tensor([pos[v] for v in e], device=DEVICE)
                    loss = loss + cfg["lam_irm"] * irm_penalty(logits[ie], yb[ie])

            elif method == "dann":
                logits, feat = model(xb, return_feat=True)
                loss = F.cross_entropy(logits, yb)
                envs = _env_by_domain(dom, idx)
                if len(envs) >= 2:
                    pos = {v: k for k, v in enumerate(idx)}
                    eid = np.zeros(len(idx), dtype=np.int64)
                    for k, e in enumerate(envs[:2]):
                        eid[[pos[v] for v in e]] = k
                    grl = cfg["lam_dann"] * (2.0 / (1.0 + np.exp(-10.0 * (ep * steps + step) / (cfg["epochs"] * steps))) - 1.0)
                    dom_logits = dom_head(GradReverse.apply(feat, grl))
                    loss = loss + F.cross_entropy(dom_logits, torch.from_numpy(eid).to(DEVICE))
            else:
                raise ValueError("unknown method %r" % method)

            loss.backward()
            opt.step()

        if len(val_idx):
            v = _eval_acc(model, X, y, val_idx)
            if v > best_val:
                best_val, best_state = v, deepcopy(model.state_dict())
        if kw.get("verbose"):
            print("    ep%02d loss=%.4f val=%.4f" % (ep + 1, float(loss), best_val))

    if best_state is not None:          # no validation split -> keep the final epoch
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model, X, idx, bs=256):
    outs = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            outs.append(model(torch.from_numpy(X[b]).to(DEVICE)).argmax(1).cpu().numpy())
    return np.concatenate(outs) if outs else np.array([], dtype=np.int64)


def per_class_and_confusion(y_true, y_pred, n_classes=N_CLASSES):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    support = cm.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class = np.where(support > 0, cm.diagonal() / support, np.nan)
    return cm, per_class, support


# ----------------------------------------------------------------------- reporting
def summarize(vals):
    """mean ± sample std (ddof=1) over seeds — the paper's '±' convention."""
    v = np.asarray(vals, dtype=float)
    sd = float(np.std(v, ddof=1)) if v.size > 1 else float("nan")
    return {"mean": float(np.mean(v)), "std": sd, "n": int(v.size), "per_seed": v.tolist()}


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("[saved] %s" % path)
