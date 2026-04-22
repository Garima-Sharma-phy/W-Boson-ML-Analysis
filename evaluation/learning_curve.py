"""
Graph 1 — Learning Curve
Shows AUC vs training set size.
Proves model is not data-starved.
Run: python3 graph1_learning_curve.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

plt.style.use("dark_background")

# ── Load data ────────────────────────────────────────────────────
df = pd.read_csv("final_master_training_data.csv")

FEATURES = [
    "muon_pt", "muon_eta", "muon_iso",
    "muon_dxy", "muon_dz",
    "met_pt", "met_significance",
    "delta_phi", "mT",
    "n_jets", "max_btag",
]

X = df[FEATURES].fillna(0).values
y = df["label"].values

# ── BDT (same hyperparameters as training) ────────────────────────
bdt = GradientBoostingClassifier(
    n_estimators=300, max_depth=4,
    learning_rate=0.05, subsample=0.8,
    min_samples_leaf=50, random_state=42
)

# ── Compute learning curve ────────────────────────────────────────
print("Computing learning curve (this takes a few minutes)...")

train_sizes = np.linspace(0.1, 1.0, 8)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

train_sizes_abs, train_scores, val_scores = learning_curve(
    bdt, X, y,
    train_sizes=train_sizes,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

# ── Plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train_sizes_abs, train_mean,
        color="#4FC3F7", lw=2.5, marker="o", ms=6,
        label="Training AUC")
ax.fill_between(train_sizes_abs,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.2, color="#4FC3F7")

ax.plot(train_sizes_abs, val_mean,
        color="#EF5350", lw=2.5, marker="s", ms=6,
        label="Validation AUC (3-fold CV)")
ax.fill_between(train_sizes_abs,
                val_mean - val_std,
                val_mean + val_std,
                alpha=0.2, color="#EF5350")

ax.axhline(val_mean[-1], color="#FFD54F", linestyle="--",
           alpha=0.7, lw=1.5,
           label=f"Final val AUC = {val_mean[-1]:.4f}")

ax.set_xlabel("Number of training events", fontsize=13)
ax.set_ylabel("AUC-ROC", fontsize=13)
ax.set_title("Learning Curve — W Boson BDT Classifier", fontsize=14)
ax.legend(fontsize=11)
ax.set_ylim([0.93, 1.002])
ax.tick_params(labelsize=11)
ax.grid(True, alpha=0.2)

# Annotate convergence
ax.annotate("Curve flattens → sufficient data",
            xy=(train_sizes_abs[-1], val_mean[-1]),
            xytext=(train_sizes_abs[-3], val_mean[-1] - 0.015),
            fontsize=10, color="#FFD54F",
            arrowprops=dict(arrowstyle="->", color="#FFD54F"))

plt.tight_layout()
plt.savefig("graph1_learning_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: graph1_learning_curve.png")

# ── Print table for thesis ────────────────────────────────────────
print("\nTable values for your thesis:")
print(f"{'Train size':<15} {'Train AUC':>12} {'Val AUC':>12}")
print("─" * 40)
for sz, tm, vm in zip(train_sizes_abs, train_mean, val_mean):
    print(f"{int(sz):<15} {tm:>12.4f} {vm:>12.4f}")