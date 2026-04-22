"""
Graph 3 — Signal vs Background Distributions
Shows normalised distributions of all 11 features
for signal and background separately.
Run: python3 graph3_distributions.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.style.use("dark_background")
SIGNAL_COLOR = "#4FC3F7"
BKG_COLOR    = "#EF5350"

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

XLABELS = {
    "muon_pt"          : r"Muon $p_T$ [GeV]",
    "muon_eta"         : r"Muon $\eta$",
    "muon_iso"         : r"Muon isolation $I_{\rm rel}$",
    "muon_dxy"         : r"Muon $d_{xy}$ [cm]",
    "muon_dz"          : r"Muon $d_z$ [cm]",
    "met_pt"           : r"MET [GeV]",
    "met_significance" : r"MET significance",
    "delta_phi"        : r"$\Delta\phi(\mu, \mathrm{MET})$ [rad]",
    "mT"               : r"$m_T$ [GeV]",
    "n_jets"           : r"Jet multiplicity ($p_T > 30$ GeV)",
    "max_btag"         : r"Max b-tag score",
}

RANGES = {
    "muon_pt"          : (0, 150),
    "muon_eta"         : (-2.5, 2.5),
    "muon_iso"         : (0, 0.5),
    "muon_dxy"         : (-0.05, 0.05),
    "muon_dz"          : (-0.3, 0.3),
    "met_pt"           : (0, 150),
    "met_significance" : (0, 30),
    "delta_phi"        : (-np.pi, np.pi),
    "mT"               : (0, 130),
    "n_jets"           : (0, 8),
    "max_btag"         : (0, 1),
}

fig, axes = plt.subplots(3, 4, figsize=(22, 15))
fig.suptitle("Signal vs Background Feature Distributions\n"
             "(Normalised to unit area)",
             fontsize=16, y=1.01)

for ax, feat in zip(axes.flat, FEATURES):
    vmin, vmax = RANGES[feat]
    bins = 50 if feat != "n_jets" else 9

    ax.hist(sig[feat].dropna(), bins=bins,
            range=(vmin, vmax), density=True,
            color=SIGNAL_COLOR, alpha=0.7,
            histtype="stepfilled",
            label=f"Signal W→μν (n={len(sig):,})")
    ax.hist(bkg[feat].dropna(), bins=bins,
            range=(vmin, vmax), density=True,
            color=BKG_COLOR, alpha=0.7,
            histtype="stepfilled",
            label=f"Background (n={len(bkg):,})")

    # Compute KS separation metric
    from scipy.stats import ks_2samp
    ks_stat, ks_pval = ks_2samp(
        sig[feat].dropna().values,
        bkg[feat].dropna().values
    )

    ax.set_xlabel(XLABELS[feat], fontsize=11)
    ax.set_ylabel("Normalised events", fontsize=10)
    ax.set_title(f"{feat}\nKS = {ks_stat:.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=9)

# Hide the unused 12th subplot
axes.flat[-1].set_visible(False)

plt.tight_layout()
plt.savefig("graph3_distributions.png", dpi=150,
            bbox_inches="tight")
plt.show()
print("Saved: graph3_distributions.png")

# ── Print KS statistics table for thesis ─────────────────────────
print("\nKolmogorov-Smirnov separation statistics:")
print(f"{'Feature':<22} {'KS statistic':>14} {'p-value':>12}")
print("─" * 50)
from scipy.stats import ks_2samp
for feat in FEATURES:
    ks, pv = ks_2samp(sig[feat].dropna().values,
                       bkg[feat].dropna().values)
    print(f"{feat:<22} {ks:>14.4f} {pv:>12.4e}")