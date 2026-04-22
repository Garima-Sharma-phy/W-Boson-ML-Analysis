"""
Graph 5 — Feature Removal Impact Study
Removes each feature one at a time and measures AUC drop.
More rigorous than Gini importance alone.
Run: python3 graph5_feature_removal.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import warnings
warnings.filterwarnings("ignore")

plt.style.use("dark_background")
SIGNAL_COLOR = "#4FC3F7"
ACCENT       = "#FFD54F"

df = pd.read_csv("final_master_training_data.csv")
FEATURES = [
    "muon_pt", "muon_eta", "muon_iso",
    "muon_dxy", "muon_dz",
    "met_pt", "met_significance",
    "delta_phi", "mT", "n_jets", "max_btag",
]
X = df[FEATURES].fillna(0).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Try to load existing model first
try:
    model_data = joblib.load("w_boson_classifier.pkl")
    bdt = model_data["model"]
    print("Loaded existing model.")
except Exception:
    print("Training new model...")
    bdt = GradientBoostingClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        min_samples_leaf=50, random_state=42
    )
    bdt.fit(X_train, y_train)

# ── Baseline AUC ─────────────────────────────────────────────────
base_auc = roc_auc_score(y_test,
               bdt.predict_proba(X_test)[:, 1])
print(f"\nBaseline AUC (all features): {base_auc:.4f}")

# ── Remove each feature one at a time ────────────────────────────
print("\nFeature removal study:")
print(f"{'Feature':<22} {'AUC w/o feature':>18} "
      f"{'AUC drop':>12} {'Impact':>10}")
print("─" * 64)

auc_drops = []
auc_without = []

for i, feat in enumerate(FEATURES):
    X_test_mod = X_test.copy()
    X_test_mod[:, i] = np.mean(X_train[:, i])
    auc_mod = roc_auc_score(y_test,
                  bdt.predict_proba(X_test_mod)[:, 1])
    drop  = base_auc - auc_mod
    auc_drops.append(drop)
    auc_without.append(auc_mod)

    impact = "★★★" if drop > 0.01 else "★★" if drop > 0.003 else "★"
    print(f"{feat:<22} {auc_mod:>18.4f} {drop:>12.4f} {impact:>10}")

# ── Plot ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Feature Importance Analysis", fontsize=15)

# Left: AUC drop when feature is removed
ax = axes[0]
sorted_idx  = np.argsort(auc_drops)[::-1]
sorted_feat = [FEATURES[i] for i in sorted_idx]
sorted_drop = [auc_drops[i] for i in sorted_idx]

colors = [ACCENT if d > 0.01 else
          SIGNAL_COLOR if d > 0.003 else
          "gray" for d in sorted_drop]

bars = ax.barh(sorted_feat[::-1], sorted_drop[::-1],
               color=colors[::-1], height=0.6,
               edgecolor="none")
ax.axvline(0, color="white", lw=0.5)
ax.set_xlabel("AUC drop when feature removed", fontsize=12)
ax.set_title("AUC Drop Method\n"
             "(larger drop = more important)", fontsize=12)
ax.tick_params(labelsize=10)

for bar, val in zip(bars, sorted_drop[::-1]):
    ax.text(val + 0.0002,
            bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)

# Right: Compare Gini importance vs AUC drop
ax = axes[1]
gini_imp = bdt.feature_importances_
x = np.arange(len(FEATURES))
width = 0.35

bars1 = ax.bar(x - width/2, gini_imp,
               width, label="Gini importance",
               color=SIGNAL_COLOR, alpha=0.8,
               edgecolor="none")
bars2 = ax.bar(x + width/2,
               [d / max(auc_drops) for d in auc_drops],
               width,
               label="AUC drop (normalised)",
               color=ACCENT, alpha=0.8,
               edgecolor="none")

ax.set_xticks(x)
ax.set_xticklabels(FEATURES, rotation=45, ha="right",
                   fontsize=9)
ax.set_ylabel("Importance (normalised)", fontsize=12)
ax.set_title("Gini Importance vs AUC Drop Method", fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig("graph5_feature_removal.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("\nSaved: graph5_feature_removal.png")

# ── Table for thesis ──────────────────────────────────────────────
print("\nTable for thesis:")
print(f"{'Feature':<22} {'Gini':>8} {'AUC drop':>10} "
      f"{'AUC w/o':>10}")
print("─" * 52)
for feat, gi, drop, awf in zip(
        FEATURES, gini_imp, auc_drops, auc_without):
    print(f"{feat:<22} {gi:>8.4f} {drop:>10.4f} {awf:>10.4f}")