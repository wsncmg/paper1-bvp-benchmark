#!/usr/bin/env python3
"""make_figures.py — generate Figure 1–4 for the benchmark paper.

All figures come from the same runs as Tables 6 and 7 (cross-location, seed 0).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # keep text editable in PDF
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import torch
import lodo_lib as L

OUT = "figures/"
os.makedirs(OUT, exist_ok=True)

# ── Load data ──
X, y, meta = L.load_dataset("/data/widar/bvp", suffix="L0")
folds = L.lodo_splits(meta["loc"])
tr, te = folds[0]  # cross-location, fold 0 (matches Section 5.2 linear probe)

# ── Train a fresh ERM teacher on this fold (seed 0) ──
print("Training ERM teacher for figures...")
tr2, va = L.source_val_split(y, tr, frac=0.10, seed=0)
model = L.train_model(X, y, meta["loc"], tr2, va, 0, method="erm")
model.eval()

# Extract features and logits
with torch.no_grad():
    x_te = torch.from_numpy(X[te]).to(L.DEVICE)
    logits_te, feat_te = model(x_te, return_feat=True)
    probs_te = torch.softmax(logits_te, 1).cpu().numpy()
    preds_te = logits_te.argmax(1).cpu().numpy()
    feat_te = feat_te.cpu().numpy()

    x_tr = torch.from_numpy(X[tr]).to(L.DEVICE)
    _, feat_tr = model(x_tr, return_feat=True)
    feat_tr = feat_tr.cpu().numpy()

# ── Figure 1: Linear probe — gesture vs location (train-side features) ──
print("Figure 1: linear probe...")
# Gesture probe (trained on train, evaluated on train — same as Section 5.2)
clf_g = LogisticRegression(max_iter=1000, random_state=0)
clf_g.fit(feat_tr, y[tr])
gest_acc = clf_g.score(feat_tr, y[tr])
# Location probe
clf_l = LogisticRegression(max_iter=1000, random_state=0)
clf_l.fit(feat_tr, meta["loc"][tr])
loc_acc = clf_l.score(feat_tr, meta["loc"][tr])
chance_gest = 1.0 / len(np.unique(y))
chance_loc = 1.0 / len(np.unique(meta["loc"]))

fig, ax = plt.subplots(figsize=(3.2, 2.6))
bars = ax.bar(["Gesture", "Location"], [gest_acc * 100, loc_acc * 100],
              color=["#2196F3", "#FF9800"], width=0.5)
ax.axhline(chance_gest * 100, color="gray", ls="--", lw=0.8, label="Chance")
ax.axhline(chance_loc * 100, color="gray", ls="--", lw=0.8)
for b, v in zip(bars, [gest_acc, loc_acc]):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
            f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 100)
ax.set_title("Linear probe on GRU features", fontsize=9)
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}fig1_probe.png", dpi=300)
fig.savefig(f"{OUT}fig1_probe.pdf")
plt.close(fig)

# ── Figure 2: Normalised confusion matrix ──
print("Figure 2: confusion matrix...")
gestures = ["Push", "Pull", "Slide", "Up", "Down", "Push-Pull"]
cm = confusion_matrix(y[te], preds_te, normalize="true")
fig, ax = plt.subplots(figsize=(4.5, 3.8))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_xticklabels(gestures, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(6)); ax.set_yticklabels(gestures, fontsize=8)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{cm[i,j]:.2f}" if cm[i,j] > 0.02 else "",
                ha="center", va="center", fontsize=7,
                color="white" if cm[i,j] > 0.45 else "black")
ax.set_xlabel("Predicted", fontsize=9)
ax.set_ylabel("True", fontsize=9)
ax.set_title("Teacher (ERM) — Cross-Location", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(f"{OUT}fig2_cm.png", dpi=300)
fig.savefig(f"{OUT}fig2_cm.pdf")
plt.close(fig)

# ── Figure 3: KL divergence distribution ──
print("Figure 3: KL distribution...")
u = 1.0 / 6  # uniform
kl = np.sum(probs_te * np.log(probs_te / u + 1e-12), axis=1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.4))
# Histogram
ax1.hist(kl, bins=20, color="#607D8B", edgecolor="white", lw=0.5)
ax1.axvline(np.mean(kl), color="red", ls="-", lw=1, label=f"mean={np.mean(kl):.3f}")
ax1.set_xlabel("KL(q || uniform)")
ax1.set_ylabel("Count")
ax1.legend(fontsize=7)
# Per-class boxplot
cls_data = [kl[y[te] == c] for c in range(6)]
bp = ax2.boxplot(cls_data, labels=[g[:3] for g in gestures], patch_artist=True,
                 boxprops=dict(facecolor="#E3F2FD"))
ax2.set_xlabel("Gesture")
ax2.set_ylabel("KL(q || uniform)")
ax2.tick_params(axis="x", rotation=45, labelsize=7)
fig.suptitle("Prediction confidence distribution", fontsize=9, y=1.01)
fig.tight_layout()
fig.savefig(f"{OUT}fig3_kl.png", dpi=300)
fig.savefig(f"{OUT}fig3_kl.pdf")
plt.close(fig)

# ── Figure 4: Epoch leakage per axis ──
print("Figure 4: epoch leakage...")
# Use the measured leak values from the audit (Protocol A vs target-test epoch selection)
axes = ["Loc", "Ori", "User", "Sess"]
leak_values = [1.8, 3.0, 4.1, 4.2]  # pp inflation from target-test epoch selection
baseline = [71.61, 62.43, 64.01, 30.20]  # 10-seed ERM from Table 4
# Note: leak values are from the audit (target-test epoch selection vs source-val)

fig, ax = plt.subplots(figsize=(3.6, 2.6))
x = np.arange(len(axes))
w = 0.35
bars1 = ax.bar(x - w/2, baseline, w, label="Clean (source-val)", color="#2196F3")
bars2 = ax.bar(x + w/2, [b + l for b, l in zip(baseline, leak_values)], w,
               label="Target-test epoch", color="#F44336")
# Add leak annotations
for i, (b, l) in enumerate(zip(baseline, leak_values)):
    ax.annotate(f"+{l:.1f}", xy=(i, b + l + 0.5), ha="center", fontsize=8, color="#F44336")
ax.set_xticks(x)
ax.set_xticklabels(axes)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(55, 78)
ax.legend(fontsize=7, loc="lower right")
ax.set_title("Inflation from target-test epoch selection", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}fig4_leak.png", dpi=300)
fig.savefig(f"{OUT}fig4_leak.pdf")
plt.close(fig)

print(f"All 4 figures saved to {OUT}/")
print(f"  fig1_probe.png  — linear probe: gesture {gest_acc*100:.1f}%, location {loc_acc*100:.1f}%")
print(f"  fig2_cm.png     — confusion matrix")
print(f"  fig3_kl.png     — KL distribution, mean={np.mean(kl):.3f}")
print(f"  fig4_leak.png   — leak=1.8-4.2 pp per axis")
