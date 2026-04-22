"""
Graph 2 — Cross-Validation Box Plot
Shows stability of AUC, Accuracy, F1 across 5 folds.
Run: python3 graph2_crossval.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

plt.style.use("dark_background")

df = pd.read_csv("final_master_training_data.csv")
FEATURES = [
    "muon_pt", "muon_eta", "muon_iso",
    "muon_dxy", "muon_dz",
    "met_pt", "met_significance",
    "delta_phi", "mT", "n_jets", "max_btag",
]
X = df[FEATURES].fillna(0).values
y = df["label"].values

bdt = GradientBoostingClassifier(
    n_estimators=300, max_depth=4,
    learning_rate=0.05, subsample=0.8,
    min_samples_leaf=50, random_state=42
)

print("Running 5-fold cross-validation...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "AUC"      : "roc_auc",
    "Accuracy" : "accuracy",
    "F1"       : "f1",
}

cv_results = cross_validate(bdt, X, y, cv=kfold,
                            scoring=scoring, n_jobs=-1)

auc_scores = cv_results["test_AUC"]
acc_scores = cv_results["test_Accuracy"]
f1_scores  = cv_results["test_F1"]

# ── Print table values for thesis ────────────────────────────────
print("\nFold-by-fold results:")
print(f"{'Fold':<8} {'AUC':>8} {'Accuracy':>10} {'F1':>8}")
print("─" * 36)
for i, (a, ac, f) in enumerate(zip(auc_scores, acc_scores,
                                    f1_scores), 1):
    print(f"{i:<8} {a:>8.4f} {ac:>10.4f} {f:>8.4f}")
print("─" * 36)
print(f"{'Mean':<8} {auc_scores.mean():>8.4f} "
      f"{acc_scores.mean():>10.4f} {f1_scores.mean():>8.4f}")
print(f"{'Std':<8} {auc_scores.std():>8.4f} "
      f"{acc_scores.std():>10.4f} {f1_scores.std():>8.4f}")

# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("5-Fold Cross-Validation — Classifier Stability",
             fontsize=15)

# Left: Box plot of all three metrics
ax = axes[0]
data   = [auc_scores, acc_scores, f1_scores]
labels = ["AUC-ROC", "Accuracy", "F1 Score"]
colors = ["#4FC3F7", "#EF5350", "#FFD54F"]

bp = ax.boxplot(data, patch_artist=True, notch=False,
                medianprops=dict(color="white", lw=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for whisker in bp["whiskers"]:
    whisker.set(color="gray", lw=1.5)
for cap in bp["caps"]:
    cap.set(color="gray", lw=1.5)
for flier in bp["fliers"]:
    flier.set(marker="o", color="white", alpha=0.5)

ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Distribution across 5 folds", fontsize=12)
ax.set_ylim([0.93, 1.01])
ax.grid(True, alpha=0.2, axis="y")

for i, (scores, color) in enumerate(zip(data, colors), 1):
    ax.annotate(f"{scores.mean():.4f}\n±{scores.std():.4f}",
                xy=(i, scores.mean()),
                xytext=(i + 0.25, scores.mean()),
                fontsize=9, color=color, va="center")

# Right: AUC per fold bar chart
ax = axes[1]
folds = [f"Fold {i}" for i in range(1, 6)]
bars = ax.bar(folds, auc_scores, color="#4FC3F7",
              alpha=0.8, edgecolor="none", width=0.5)
ax.axhline(auc_scores.mean(), color="#FFD54F", linestyle="--",
           lw=2, label=f"Mean AUC = {auc_scores.mean():.4f}")
ax.fill_between([-0.5, 4.5],
                auc_scores.mean() - auc_scores.std(),
                auc_scores.mean() + auc_scores.std(),
                alpha=0.15, color="#FFD54F",
                label=f"±1σ = {auc_scores.std():.4f}")

for bar, val in zip(bars, auc_scores):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.0005,
            f"{val:.4f}", ha="center", va="bottom",
            fontsize=10, color="white")

ax.set_ylabel("AUC-ROC", fontsize=12)
ax.set_title("AUC per fold", fontsize=12)
ax.set_ylim([0.98, 0.995])
ax.legend(fontsize=10)
ax.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig("graph2_crossval.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: graph2_crossval.png")