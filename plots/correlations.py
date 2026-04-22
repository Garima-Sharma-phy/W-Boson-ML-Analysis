"""
Graph 4 — Correlation Matrices
Signal vs background correlation structure.
Different structures = BDT exploits correlations
that cut-based analysis cannot.
Run: python3 graph4_correlations.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.style.use("dark_background")

df  = pd.read_csv("final_master_training_data.csv")
sig = df[df["label"] == 1]
bkg = df[df["label"] == 0]

FEATURES = [
    "muon_pt", "muon_eta", "muon_iso",
    "muon_dxy", "muon_dz",
    "met_pt", "met_significance",
    "delta_phi", "mT",
    "n_jets", "max_btag",
]

SHORT = {
    "muon_pt"          : "μ pT",
    "muon_eta"         : "μ η",
    "muon_iso"         : "μ iso",
    "muon_dxy"         : "dxy",
    "muon_dz"          : "dz",
    "met_pt"           : "MET",
    "met_significance" : "MET sig",
    "delta_phi"        : "Δφ",
    "mT"               : "mT",
    "n_jets"           : "n_jets",
    "max_btag"         : "btag",
}

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle("Correlation Matrices: Signal (left) vs "
             "Background (right)", fontsize=15)

for ax, subset, title in zip(
        axes,
        [sig[FEATURES].rename(columns=SHORT),
         bkg[FEATURES].rename(columns=SHORT)],
        ["Signal (W→μν)", "Background"]):

    corr = subset.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(corr, ax=ax, mask=mask,
                annot=True, fmt=".2f",
                cmap="RdBu_r", center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.3,
                annot_kws={"size": 8},
                cbar_kws={"shrink": 0.8,
                           "label": "Pearson r"})
    ax.set_title(title, fontsize=13, pad=10)
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("graph4_correlations.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("Saved: graph4_correlations.png")

# ── Print strongly correlated pairs ──────────────────────────────
print("\nStrongly correlated pairs (|r| > 0.3):")
corr_sig = sig[FEATURES].corr()
corr_bkg = bkg[FEATURES].corr()

print("\nIn SIGNAL:")
for i in range(len(FEATURES)):
    for j in range(i+1, len(FEATURES)):
        r = corr_sig.iloc[i, j]
        if abs(r) > 0.3:
            print(f"  {FEATURES[i]} — {FEATURES[j]}: r = {r:.3f}")

print("\nIn BACKGROUND:")
for i in range(len(FEATURES)):
    for j in range(i+1, len(FEATURES)):
        r = corr_bkg.iloc[i, j]
        if abs(r) > 0.3:
            print(f"  {FEATURES[i]} — {FEATURES[j]}: r = {r:.3f}")