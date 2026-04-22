"""
W Boson Event Classifier
=========================
Train a classifier on MC simulation using truth labels,
then apply to real collision data to identify W events.

ALL training features are available in real data too.
Only the LABEL comes from MC truth (not available in real data).

Goal: Given an event, predict P(event is W boson) → 0 or 1

Requirements:
    pip install pandas numpy matplotlib seaborn scikit-learn shap joblib

Usage:
    python train_classifier.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_curve, auc, roc_auc_score,
                             confusion_matrix, classification_report,
                             precision_recall_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

plt.style.use("dark_background")
SIGNAL_COLOR = "#4FC3F7"
BKG_COLOR    = "#EF5350"
ACCENT       = "#FFD54F"

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Load CSV and define features
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("  W Boson Event Classifier — Training")
print("=" * 60)

df = pd.read_csv("final_master_training_data.csv")

# ── Features: ONLY variables available in real collision data ──────
# DO NOT include: genWeight, genPartFlav, label, label_exact, label_simple
FEATURES = [
    # Muon kinematics
    "muon_pt",           # transverse momentum
    "muon_eta",          # pseudorapidity
    "muon_iso",          # isolation — very powerful for W vs QCD

    # Muon quality
    "muon_dxy",          # transverse impact parameter
    "muon_dz",           # longitudinal impact parameter

    # MET
    "met_pt",            # missing transverse energy (neutrino proxy)
    "met_significance",  # how significant the MET is

    # Derived (computed from reco quantities — available in real data)
    "delta_phi",         # angle between muon and MET
    "mT",                # transverse mass — peaks at W mass for signal

    # Jets
    "n_jets",            # number of jets (pT > 30 GeV)
    "max_btag",          # max b-tag score (high = likely ttbar)
]

# ── Target label ──────────────────────────────────────────────────
# Use EXACT label (W direct, tau vetoed) — from MC truth only
TARGET = "label"

print(f"\n[1/6] Dataset loaded")
print(f"      Total events : {len(df):,}")
print(f"      Signal (W)   : {df[TARGET].sum():,}  ({100*df[TARGET].mean():.1f}%)")
print(f"      Background   : {(df[TARGET]==0).sum():,}  ({100*(1-df[TARGET].mean()):.1f}%)")
print(f"      Features     : {len(FEATURES)}")
print(f"\n      Feature list:")
for f in FEATURES:
    print(f"        - {f}")

# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Prepare data
# ═══════════════════════════════════════════════════════════════════
print(f"\n[2/6] Preparing data...")

X = df[FEATURES].fillna(0).values
y = df[TARGET].values

# Train / Validation / Test split: 60 / 20 / 20
X_temp,  X_test,  y_temp,  y_test  = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val,   y_train, y_val   = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)

print(f"      Train set  : {len(X_train):,} events")
print(f"      Val set    : {len(X_val):,} events")
print(f"      Test set   : {len(X_test):,} events")

# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Train classifier
# ═══════════════════════════════════════════════════════════════════
print(f"\n[3/6] Training BDT classifier...")
print(f"      (Gradient Boosted Decision Tree)")

# BDT is the standard in HEP — robust, interpretable, no scaling needed
bdt = GradientBoostingClassifier(
    n_estimators   = 300,    # number of trees
    max_depth      = 4,      # depth per tree
    learning_rate  = 0.05,   # shrinkage
    subsample      = 0.8,    # stochastic gradient boosting
    min_samples_leaf = 50,   # regularisation
    random_state   = 42,
    verbose        = 0
)

bdt.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score

print("\n[+] Running 5-Fold Cross Validation...")
# Set up the 5 splits
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Run the evaluation on all 5 folds
cv_scores = cross_val_score(bdt, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

# Print the results exactly as they appear in your thesis table
print("      Fold Results (AUC-ROC):")
for i, score in enumerate(cv_scores, 1):
    print(f"        Fold {i}: {score:.4f}")
    
print(f"      Mean ± Std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
# Scores on all splits
train_score = bdt.score(X_train, y_train)
val_score   = bdt.score(X_val,   y_val)
test_score  = bdt.score(X_test,  y_test)

train_auc   = roc_auc_score(y_train, bdt.predict_proba(X_train)[:,1])
val_auc     = roc_auc_score(y_val,   bdt.predict_proba(X_val)[:,1])
test_auc    = roc_auc_score(y_test,  bdt.predict_proba(X_test)[:,1])

print(f"\n      Results:")
print(f"      {'Split':<12} {'Accuracy':>10} {'AUC':>10}")
print(f"      {'─'*34}")
print(f"      {'Train':<12} {train_score:>10.4f} {train_auc:>10.4f}")
print(f"      {'Validation':<12} {val_score:>10.4f} {val_auc:>10.4f}")
print(f"      {'Test':<12} {test_score:>10.4f} {test_auc:>10.4f}")

# Check for overfitting
if train_auc - test_auc > 0.05:
    print(f"\n  ⚠️  Possible overfitting! Train AUC >> Test AUC")
    print(f"      Consider: fewer trees, shallower depth, more regularisation")
else:
    print(f"\n  ✅  No significant overfitting detected")

# ═══════════════════════════════════════════════════════════════════
# STEP 4 — Evaluate and plot
# ═══════════════════════════════════════════════════════════════════
print(f"\n[4/6] Generating evaluation plots...")

y_score = bdt.predict_proba(X_test)[:, 1]  # P(W event)
y_pred  = bdt.predict(X_test)              # 0 or 1

# ── Figure 1: Core evaluation (2x2) ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("W Boson Classifier — Evaluation on Test Set", fontsize=16)

# ── Plot 1: BDT score distribution ───────────────────────────────
ax = axes[0, 0]
score_sig = y_score[y_test == 1]
score_bkg = y_score[y_test == 0]

ax.hist(score_bkg, bins=50, range=(0,1), color=BKG_COLOR,
        alpha=0.7, density=True, histtype="stepfilled", label=f"Background  (n={len(score_bkg):,})")
ax.hist(score_sig, bins=50, range=(0,1), color=SIGNAL_COLOR,
        alpha=0.7, density=True, histtype="stepfilled", label=f"Signal W→μν (n={len(score_sig):,})")

ax.axvline(0.5, color=ACCENT, linestyle="--", label="Default threshold (0.5)")
ax.set_xlabel("BDT score  P(W event)", fontsize=12)
ax.set_ylabel("Normalised events", fontsize=12)
ax.set_title("Classifier Score Distribution", fontsize=13)
ax.legend(fontsize=10)

# Annotate separation quality
separation = abs(score_sig.mean() - score_bkg.mean())
ax.text(0.05, 0.85, f"Mean separation: {separation:.3f}",
        transform=ax.transAxes, fontsize=10, color=ACCENT)

# ── Plot 2: ROC curve ─────────────────────────────────────────────
ax = axes[0, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc_val = auc(fpr, tpr)

ax.plot(fpr, tpr, color=SIGNAL_COLOR, lw=2.5,
        label=f"BDT  (AUC = {roc_auc_val:.4f})")
ax.plot([0,1], [0,1], color="gray", lw=1.5,
        linestyle="--", label="Random classifier")
ax.fill_between(fpr, tpr, alpha=0.1, color=SIGNAL_COLOR)

# Mark the working point at threshold=0.5
thresh_idx = np.argmin(np.abs(thresholds - 0.5))
ax.scatter(fpr[thresh_idx], tpr[thresh_idx],
           color=ACCENT, s=100, zorder=5, label="Threshold = 0.5")

ax.set_xlabel("False positive rate  (bkg efficiency)", fontsize=12)
ax.set_ylabel("True positive rate  (signal efficiency)", fontsize=12)
ax.set_title("ROC Curve", fontsize=13)
ax.legend(fontsize=10)
ax.text(0.55, 0.2, f"AUC = {roc_auc_val:.4f}",
        fontsize=14, color=ACCENT, transform=ax.transAxes)

# ── Plot 3: Confusion matrix ──────────────────────────────────────
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
            ax=ax, linewidths=0.5,
            xticklabels=["Pred: Bkg", "Pred: W"],
            yticklabels=["True: Bkg", "True: W"])
ax.set_title("Confusion Matrix (%)", fontsize=13)
ax.set_ylabel("True label", fontsize=12)
ax.set_xlabel("Predicted label", fontsize=12)

# Annotate
tn, fp, fn, tp = cm.ravel()
ax.text(0.5, -0.12,
        f"True Pos: {tp:,}   False Pos: {fp:,}   True Neg: {tn:,}   False Neg: {fn:,}",
        transform=ax.transAxes, ha="center", fontsize=9, color="gray")

# ── Plot 4: Feature importance ────────────────────────────────────
ax = axes[1, 1]
importances = pd.Series(bdt.feature_importances_, index=FEATURES)
importances_sorted = importances.sort_values(ascending=True)

colors = [ACCENT if v > importances.mean() else SIGNAL_COLOR
          for v in importances_sorted.values]
bars = ax.barh(importances_sorted.index, importances_sorted.values,
               color=colors, height=0.6, edgecolor="none")
ax.axvline(importances.mean(), color="gray", linestyle="--",
           alpha=0.7, label=f"Mean ({importances.mean():.3f})")
ax.set_xlabel("Importance (BDT Gini)", fontsize=12)
ax.set_title("Feature Importance", fontsize=13)
ax.legend(fontsize=10)

for bar, val in zip(bars, importances_sorted.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("classifier_evaluation2.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: classifier_evaluation2.png")

# ── Figure 2: Threshold analysis ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Threshold Analysis — Choose Your Working Point", fontsize=15)

# ── Plot: Precision, Recall, F1 vs threshold ──────────────────────
ax = axes[0]
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_score)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

# Align lengths
thr_plot = np.linspace(0, 1, 100)
prec_interp = np.interp(thr_plot, pr_thresholds,
                        precision[:-1])
rec_interp  = np.interp(thr_plot, pr_thresholds,
                        recall[:-1])
f1_interp   = np.interp(thr_plot, pr_thresholds,
                        f1[:-1])

ax.plot(thr_plot, prec_interp, color=SIGNAL_COLOR, lw=2, label="Precision")
ax.plot(thr_plot, rec_interp,  color=BKG_COLOR,    lw=2, label="Recall")
ax.plot(thr_plot, f1_interp,   color=ACCENT,       lw=2, label="F1 score")
ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7, label="Default 0.5")

best_f1_thresh = thr_plot[np.argmax(f1_interp)]
ax.axvline(best_f1_thresh, color="white", linestyle=":",
           alpha=0.7, label=f"Best F1 threshold ({best_f1_thresh:.2f})")

ax.set_xlabel("Classification threshold", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Precision / Recall / F1 vs Threshold", fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

# ── Plot: Signal efficiency vs background rejection ───────────────
ax = axes[1]
sig_eff = tpr                          # true positive rate
bkg_rej = 1 - fpr                     # 1 - false positive rate

ax.plot(sig_eff, bkg_rej, color=SIGNAL_COLOR, lw=2.5)

# Mark some working points
for target_eff in [0.7, 0.8, 0.9, 0.95]:
    idx = np.argmin(np.abs(sig_eff - target_eff))
    ax.scatter(sig_eff[idx], bkg_rej[idx],
               s=80, zorder=5, color=ACCENT)
    ax.annotate(f"ε_sig={target_eff:.0%}\nε_bkg_rej={bkg_rej[idx]:.1%}",
                xy=(sig_eff[idx], bkg_rej[idx]),
                xytext=(sig_eff[idx]-0.15, bkg_rej[idx]-0.08),
                fontsize=8, color=ACCENT,
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=0.8))

ax.set_xlabel("Signal efficiency  (true positive rate)", fontsize=12)
ax.set_ylabel("Background rejection  (1 - false positive rate)", fontsize=12)
ax.set_title("Signal Efficiency vs Background Rejection", fontsize=13)
ax.set_xlim([0.5, 1.0])
ax.set_ylim([0.5, 1.0])

plt.tight_layout()
plt.savefig("threshold_analysis2.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: threshold_analysis2.png")

# ── Figure 3: mT distribution after classifier cut ───────────────
print("  Plotting mT after classifier cut...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("mT Distribution After Classifier Cut\n"
             "(This is what you would see on real data)", fontsize=14)

# THE FIX: Rebuild df_test using the perfectly aligned test arrays
df_test = pd.DataFrame(X_test, columns=FEATURES)
df_test["label"] = y_test
df_test["bdt_score"] = bdt.predict_proba(X_test)[:, 1]

thresholds_to_test = [0.3, 0.5, 0.7]
for ax, thresh in zip(axes, thresholds_to_test):
    df_pass = df_test[df_test["bdt_score"] > thresh]
    sig_pass = df_pass[df_pass["label"] == 1]
    bkg_pass = df_pass[df_pass["label"] == 0]
    purity   = len(sig_pass) / len(df_pass) * 100 if len(df_pass) > 0 else 0

    ax.hist(sig_pass["mT"], bins=50, range=(0, 130),
            color=SIGNAL_COLOR, alpha=0.7, histtype="stepfilled",
            label=f"Signal  ({len(sig_pass):,})")
    ax.hist(bkg_pass["mT"], bins=50, range=(0, 130),
            color=BKG_COLOR, alpha=0.7, histtype="stepfilled",
            label=f"Background ({len(bkg_pass):,})")
    ax.axvspan(60, 100, alpha=0.08, color=ACCENT)
    ax.set_xlabel("mT [GeV]", fontsize=12)
    ax.set_ylabel("Events", fontsize=12)
    ax.set_title(f"BDT score > {thresh}\nPurity = {purity:.1f}%", fontsize=12)
    ax.legend(fontsize=10)
    ax.text(0.05, 0.88, f"Events passing: {len(df_pass):,}",
            transform=ax.transAxes, fontsize=9, color="gray")

plt.tight_layout()
plt.savefig("mt_after_cut2.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: mt_after_cut2.png")

# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Full classification report
# ═══════════════════════════════════════════════════════════════════
print(f"\n[5/6] Classification Report (threshold = 0.5):")
print(f"\n{classification_report(y_test, y_pred, target_names=['Background', 'W signal'])}")

print(f"Key metrics:")
print(f"  Signal efficiency   : {100*tpr[thresh_idx]:.1f}%  (% of true W events correctly identified)")
print(f"  Background rejection: {100*(1-fpr[thresh_idx]):.1f}%  (% of backgrounds correctly rejected)")
print(f"  Purity at 0.5 cut   : {100*cm[1,1]/(cm[0,1]+cm[1,1]):.1f}%  (% of predicted W that are truly W)")

# ═══════════════════════════════════════════════════════
# 6. THE CUTFLOW SUMMARY (EVENT TRACKING)
# ═══════════════════════════════════════════════════════

# 1. Total events originally in the ROOT file
n_total_initial = tree.num_entries

# 2. Events surviving after manual cuts (trigger, pT, mT, iso, etc.)
n_pass_manual = len(df_real)
n_removed_manual = n_total_initial - n_pass_manual

# 3. Events surviving after the AI model
n_pass_ai = len(df_pure_w)
n_removed_ai = n_pass_manual - n_pass_ai

print("\n" + "═"*50)
print(" 📊 CUTFLOW SUMMARY")
print("═"*50)
print(f"1. Total Raw Events in File      : {n_total_initial:,}")
print(f"2. Removed by Manual Kinematics  : -{n_removed_manual:,}")
print(f"3. Events Passed to AI           : {n_pass_manual:,}")
print(f"4. Removed by AI Model (Bkg)     : -{n_removed_ai:,}")
print("-" * 50)
print(f"🏆 FINAL PURE W BOSONS RETAINED  : {n_pass_ai:,}")
print("═"*50)

# Optional: Calculate percentages
pct_manual = (n_removed_manual / n_total_initial) * 100 if n_total_initial > 0 else 0
pct_ai = (n_removed_ai / n_pass_manual) * 100 if n_pass_manual > 0 else 0

print(f"\n* The manual cuts threw away {pct_manual:.1f}% of the raw data.")
print(f"* The AI threw away {pct_ai:.1f}% of the remaining background.")

# ═══════════════════════════════════════════════════════════════════
# STEP 6 — Save model
# ═══════════════════════════════════════════════════════════════════
print(f"\n[6/6] Saving trained model...")

model_data = {
    "model"    : bdt,
    "features" : FEATURES,
    "auc"      : roc_auc_val,
    "threshold": 0.5,
}
joblib.dump(model_data, "w_boson_classifier2.pkl")
print(f"  Saved: w_boson_classifier2.pkl")

